"""The FinDocGPT agent as a LangGraph state machine.

    plan ─┬─(company not in corpus)──────────────────────────────► abstain ─► END
          └► retrieve ─► assess ─┬─(weak evidence, 1 retry)─► rewrite ─► retrieve
                                 └► calculate? ─► generate ─► verify ─┬─(rejections, once)─► revise ─► verify
                                                                      └► END

* plan       LLM planner: route, companies, per-entity sub-queries (rule-based fallback).
* retrieve   hybrid BM25+dense search per sub-query with company filters, cross-encoder rerank.
* assess     corrective-RAG check: any sub-query whose best rerank score is below threshold is weak.
* rewrite    LLM rewrites only the weak sub-queries into 10-K vocabulary; retrieval runs once more.
* calculate  grounded calculator tool when the plan needs arithmetic.
* generate   cited answer from the assembled evidence.
* verify     claim-level check against all sources.
* revise     on rejections, the writer gets the objections and rewrites once; still-unsupported sentences are removed.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.config import settings
from src.retrieval.store import Hit, Reranker, Retriever

from .answer import AnswerWriter, Verifier, is_abstention
from .calculator import Calculation, Calculator
from .planner import Plan, Planner, SubQuery

REWRITE_SYSTEM = """Rewrite a search query for SEC 10-K retrieval. Use the vocabulary filings actually use
(e.g. "net sales", "segment operating income", "primarily due to", "remaining performance obligation").
Return JSON: {"query": "<rewritten query>"}"""
NOT_IN_CORPUS = "The filings in the corpus do not disclose this. FinDocGPT covers the latest 10-K of: "


class AgentState(TypedDict, total=False):
    question: str
    plan: Plan
    results: dict[int, list[Hit]]  # sub-query index -> reranked hits
    weak: list[int]
    retries: int
    evidence: list[Hit]
    calculation: Calculation | None
    draft: str
    verdicts: list[Any]
    rejected: list[Any]
    needs_revision: bool
    revisions: int
    answer: str
    trace: list[dict[str, Any]]


def _log(state: AgentState, step: str, started: float, **detail: Any) -> list[dict[str, Any]]:
    return [*state.get("trace", []), {"step": step, "ms": round((time.perf_counter() - started) * 1000), **detail}]


class FinDocAgent:
    def __init__(self, retriever: Retriever | None = None, reranker: Reranker | None = None, planner: Planner | None = None,
                 calculator: Calculator | None = None, writer: AnswerWriter | None = None, verifier: Verifier | None = None,
                 rewriter=None, enable_rewrite: bool = True, enable_verify: bool = True) -> None:
        from src.llm import agent

        self.retriever = retriever or Retriever()
        self.reranker = reranker or Reranker()
        self.planner = planner or Planner()
        self.calculator = calculator or Calculator()
        self.writer = writer or AnswerWriter()
        self.verifier = verifier or Verifier()
        self.rewriter = rewriter or agent()
        self.enable_rewrite = enable_rewrite
        self.enable_verify = enable_verify
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------------
    def plan(self, state: AgentState) -> AgentState:
        t = time.perf_counter()
        plan = self.planner.plan(state["question"])
        return {"plan": plan, "results": {}, "retries": 0,
                "trace": _log(state, "plan", t, route=plan.route, tickers=plan.tickers, planner=plan.planner,
                              sub_queries=[s.query for s in plan.sub_queries], needs_calculation=plan.needs_calculation)}

    def _search(self, sub: SubQuery) -> list[Hit]:
        tickers = [sub.ticker] if sub.ticker else None
        hits = self.retriever.search(sub.query, tickers=tickers)
        if sub.prefer_tables:
            seen = {h.chunk.chunk_id for h in hits}
            hits += [h for h in self.retriever.search(sub.query, limit=10, tickers=tickers, kinds=["table"])
                     if h.chunk.chunk_id not in seen]
        return self.reranker.rerank(sub.query, hits, keep=settings.rerank_keep)

    def retrieve(self, state: AgentState) -> AgentState:
        t = time.perf_counter()
        plan = state["plan"]
        results = dict(state.get("results", {}))
        todo = state.get("weak") if state.get("retries") else range(len(plan.sub_queries))
        for i in todo:
            fresh = self._search(plan.sub_queries[i])
            old = results.get(i, [])
            # On a retry keep whichever attempt found stronger evidence.
            if not old or (fresh and fresh[0].rerank_score > old[0].rerank_score):
                results[i] = fresh
        best = {plan.sub_queries[i].query: round(hits[0].rerank_score, 3) if hits else None for i, hits in results.items()}
        return {"results": results, "trace": _log(state, "retrieve", t, best_rerank_score=best)}

    def assess(self, state: AgentState) -> AgentState:
        weak = [i for i, hits in state["results"].items()
                if not hits or hits[0].rerank_score < settings.retry_threshold]
        return {"weak": weak}

    def rewrite(self, state: AgentState) -> AgentState:
        t = time.perf_counter()
        plan = state["plan"]
        rewrites = {}
        for i in state["weak"]:
            sub = plan.sub_queries[i]
            try:
                raw = self.rewriter.complete_json(REWRITE_SYSTEM, f"Company: {sub.ticker or 'any'}\nQuestion: {state['question']}\nWeak query: {sub.query}")
                new_query = str(raw.get("query", "")).strip()
            except Exception:  # a failed rewrite simply keeps the original query
                new_query = ""
            if new_query:
                rewrites[sub.query] = new_query
                plan.sub_queries[i] = SubQuery(new_query, sub.ticker, sub.prefer_tables)
        return {"plan": plan, "retries": state.get("retries", 0) + 1, "trace": _log(state, "rewrite", t, rewrites=rewrites)}

    def assemble(self, state: AgentState) -> AgentState:
        # Round-robin over sub-queries so every compared entity is represented in the context.
        ranked = [state["results"][i] for i in sorted(state["results"])]
        evidence: list[Hit] = []
        seen: set[str] = set()
        for depth in range(settings.rerank_keep):
            for hits in ranked:
                if depth < len(hits) and hits[depth].chunk.chunk_id not in seen and len(evidence) < settings.max_context_chunks:
                    seen.add(hits[depth].chunk.chunk_id)
                    evidence.append(hits[depth])
        return {"evidence": evidence}

    def calculate(self, state: AgentState) -> AgentState:
        t = time.perf_counter()
        pool = {h.chunk.chunk_id: h for hits in state["results"].values() for h in hits}
        calc, attempts = self.calculator.run(state["question"], list(pool.values()))
        return {"calculation": calc, "trace": _log(state, "calculate", t, attempts=attempts,
                                                   result=calc.formatted if calc else None)}

    def _sources(self, state: AgentState) -> list[tuple[str, str]]:
        sources = []
        if state.get("calculation"):
            sources.append(("CALCULATOR | deterministic arithmetic over cited filing figures", state["calculation"].as_evidence()))
        for hit in state["evidence"]:
            sources.append((f"{hit.chunk.chunk_id} | {hit.chunk.context}", hit.chunk.text[:1800]))
        return sources

    def generate(self, state: AgentState) -> AgentState:
        t = time.perf_counter()
        draft = self.writer.write(state["question"], self._sources(state))
        return {"draft": draft, "trace": _log(state, "generate", t, chars=len(draft))}

    def verify(self, state: AgentState) -> AgentState:
        t = time.perf_counter()
        if not self.enable_verify:
            return {"answer": state["draft"], "verdicts": []}
        verdicts = self.verifier.verify(state["draft"], self._sources(state))
        rejected = [v for v in verdicts if not v.supported]
        revisions = state.get("revisions", 0)
        if rejected and revisions < 1:
            # Self-correction: send the objections back to the writer once instead of deleting text immediately.
            return {"needs_revision": True, "rejected": rejected, "verdicts": verdicts,
                    "trace": _log(state, "verify", t, removed=len(rejected), kept=len(verdicts) - len(rejected), revising=True)}
        kept = [v.sentence for v in verdicts if v.supported]
        answer = " ".join(kept) if kept else "I could not verify an answer to this question against the filings."
        return {"answer": answer, "verdicts": verdicts, "needs_revision": False,
                "trace": _log(state, "verify", t, removed=len(rejected), kept=len(kept))}

    def revise(self, state: AgentState) -> AgentState:
        t = time.perf_counter()
        draft = self.writer.write(state["question"], self._sources(state), feedback=state["rejected"])
        return {"draft": draft, "revisions": state.get("revisions", 0) + 1, "needs_revision": False,
                "trace": _log(state, "revise", t, objections=len(state["rejected"]))}

    def abstain(self, state: AgentState) -> AgentState:
        from src.companies import COMPANIES

        return {"answer": NOT_IN_CORPUS + ", ".join(COMPANIES.values()) + ".", "draft": "", "evidence": [],
                "verdicts": [], "calculation": None}

    # ---- graph -------------------------------------------------------------------------------
    def _build(self):
        g = StateGraph(AgentState)
        # Node names must differ from state keys, hence "make_plan" for the plan() step.
        g.add_node("make_plan", self.plan)
        for name in ("retrieve", "assess", "rewrite", "assemble", "calculate", "generate", "verify", "revise", "abstain"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("make_plan")
        g.add_conditional_edges("make_plan", lambda s: "retrieve" if s["plan"].in_corpus else "abstain")
        g.add_edge("retrieve", "assess")
        g.add_conditional_edges("assess", lambda s: "rewrite" if s["weak"] and self.enable_rewrite and s["retries"] < 1 else "assemble")
        g.add_edge("rewrite", "retrieve")
        g.add_conditional_edges("assemble", lambda s: "calculate" if s["plan"].needs_calculation else "generate")
        g.add_edge("calculate", "generate")
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", lambda s: "revise" if s.get("needs_revision") else END)
        g.add_edge("revise", "verify")
        g.add_edge("abstain", END)
        return g.compile()

    def run(self, question: str) -> dict[str, Any]:
        state = self.graph.invoke({"question": question, "trace": []})
        plan: Plan = state["plan"]
        calc = state.get("calculation")
        return {
            "question": question,
            "route": plan.route,
            "plan": plan.to_dict(),
            "answer": state["answer"],
            "draft": state.get("draft", ""),
            "abstained": is_abstention(state["answer"]),
            "calculation": calc.to_dict() if calc else None,
            "verdicts": [v.to_dict() for v in state.get("verdicts", [])],
            "removed_claims": [v.sentence for v in state.get("verdicts", []) if not v.supported],
            "sources": ([{"label": "S1", "chunk_id": "CALCULATOR", "kind": "calculation", "text": calc.as_evidence()}] if calc else [])
                       + [{"label": f"S{i}", **h.to_dict()} for i, h in enumerate(state.get("evidence", []), 2 if calc else 1)],
            "trace": state.get("trace", []),
        }
