"""Query planning: route, companies, and per-entity sub-queries.

An LLM planner decomposes the question (e.g. a comparison becomes one focused sub-query
per segment/company). Its output is validated against the corpus registry, and a
deterministic rule-based planner is the fallback, so routing never hard-fails.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.companies import COMPANIES, extract_tickers
from src.llm import LLM, LLMError

ROUTES = ("factual", "comparison", "computation")
EXPLICIT_CALCULATION = re.compile(r"\b(calculate|compute)\b", re.I)
FISCAL_YEARS = {
    "AAPL": "FY2025 (ended Sep 27, 2025)", "MSFT": "FY2025 (ended Jun 30, 2025)", "NVDA": "FY2026 (ended Jan 25, 2026)",
    "CRM": "FY2026 (ended Jan 31, 2026)", "ADBE": "FY2025 (ended Nov 28, 2025)", "AMZN": "FY2025 (calendar)",
    "GOOGL": "FY2025 (calendar)", "META": "FY2025 (calendar)", "NFLX": "FY2025 (calendar)", "UBER": "FY2025 (calendar)",
}
CORPUS_DESCRIPTION = "\n".join(f"- {t}: {name}, latest 10-K {FISCAL_YEARS[t]}" for t, name in COMPANIES.items())


@dataclass
class SubQuery:
    query: str
    ticker: str | None
    prefer_tables: bool = False


@dataclass
class Plan:
    route: str
    tickers: list[str]
    sub_queries: list[SubQuery]
    needs_calculation: bool
    in_corpus: bool = True
    rationale: str = ""
    planner: str = "llm"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


PLANNER_SYSTEM = f"""You plan retrieval for a question-answering agent over SEC 10-K filings.
The corpus contains exactly one latest 10-K per company:
{CORPUS_DESCRIPTION}

Return JSON with keys:
- "route": "factual" (what a filing says/discloses/attributes, drivers, risks), "comparison" (contrasting two or more segments,
  companies, metrics or periods), or "computation" (the user explicitly asks to calculate/compute a number such as a growth rate,
  margin, or percentage-point change).
- "tickers": tickers from the list above that the question is about ([] if none of them).
- "in_corpus": false only if the question is about a company that is not in the list.
- "needs_calculation": true if answering requires arithmetic on reported figures (growth rates, margins, differences).
- "sub_queries": 1-4 search queries phrased in 10-K vocabulary, each {{"query": str, "ticker": str, "prefer_tables": bool}}.
  Decompose comparisons into one sub-query per item being compared. Set prefer_tables=true when the answer lives in
  financial statement tables (revenues, operating income, margins by segment or year).
  For questions about what drove or caused a change, add a sub-query in MD&A phrasing, e.g.
  "<metric> increased compared to prior year primarily due to", because 10-Ks explain changes that way.
- "rationale": one short sentence."""


def rule_based_plan(question: str) -> Plan:
    """Deterministic fallback; also the v1-style baseline router used for ablation."""
    q = " ".join(question.lower().split())
    tickers = extract_tickers(question)
    if re.search(r"\b(calculate|compute|growth rate|percentage[- ]point|year[- ]over[- ]year|yoy)\b", q):
        route = "computation"
    elif re.search(r"\b(compare|comparison|versus|vs\.?|relative to)\b", q):
        route = "comparison"
    else:
        route = "factual"
    needs_calc = route == "computation" or bool(re.search(r"\bmargin|growth|change\b", q) and route == "comparison")
    subs = [SubQuery(question, t, prefer_tables=needs_calc) for t in tickers] or [SubQuery(question, None)]
    return Plan(route, tickers, subs, needs_calc, in_corpus=True, rationale="rule-based fallback", planner="rules")


class Planner:
    def __init__(self, llm: LLM | None = None) -> None:
        from src.llm import agent

        self.llm = llm or agent()

    def plan(self, question: str) -> Plan:
        detected = extract_tickers(question)
        try:
            raw = self.llm.complete_json(PLANNER_SYSTEM, f"Question: {question}")
        except LLMError as exc:
            fallback = rule_based_plan(question)
            fallback.notes.append(f"LLM planner failed: {exc}")
            return fallback

        route = raw.get("route") if raw.get("route") in ROUTES else rule_based_plan(question).route
        if EXPLICIT_CALCULATION.search(question):
            route = "computation"  # an explicit instruction to calculate overrides the model's judgement
        tickers = [t for t in raw.get("tickers", []) if t in COMPANIES]
        # Deterministic name matching is a guard-rail: never drop a company the user explicitly named.
        tickers = list(dict.fromkeys(tickers + detected))
        subs: list[SubQuery] = []
        for item in raw.get("sub_queries", [])[:4]:
            if not isinstance(item, dict) or not str(item.get("query", "")).strip():
                continue
            ticker = item.get("ticker") if item.get("ticker") in COMPANIES else (tickers[0] if len(tickers) == 1 else None)
            subs.append(SubQuery(str(item["query"]).strip(), ticker, bool(item.get("prefer_tables"))))
        if not subs:
            subs = [SubQuery(question, t) for t in tickers] or [SubQuery(question, None)]
        in_corpus = bool(tickers) or bool(raw.get("in_corpus", True))
        needs_calc = bool(raw.get("needs_calculation")) or route == "computation"
        return Plan(route, tickers, subs, needs_calc, in_corpus, str(raw.get("rationale", ""))[:300])
