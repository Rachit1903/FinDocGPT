"""Retrieval-only ablation: how much does each retrieval component contribute?

Scores gold-evidence recall@k / hit@k / MRR for progressively richer retrieval on the 40
answerable gold questions. No generation is involved, so this costs no LLM tokens.
The "agent plan" row reuses the planner's sub-queries recorded in v2_predictions.json, and
the v1 row scores the contexts frozen in v1_predictions.json.

    python -m eval.retrieval_ablation
"""

from __future__ import annotations

import json
from collections.abc import Callable

from src.companies import extract_tickers
from src.config import settings
from src.retrieval.store import Hit, Reranker, Retriever

from .metrics import evidence_found
from .run_eval import RESULTS, _read, load_gold

K_VALUES = (5, 8)


def _found(evidence: list[str], texts: list[str]) -> list[bool]:
    return [any(evidence_found(e, t) for t in texts) for e in evidence]


def score(gold: list[dict], ranked_texts: Callable[[dict], list[str]]) -> dict[str, float]:
    out: dict[str, list[float]] = {f"recall@{k}": [] for k in K_VALUES} | {f"hit@{k}": [] for k in K_VALUES} | {"mrr": []}
    for item in gold:
        texts = ranked_texts(item)
        for k in K_VALUES:
            found = _found(item["evidence"], texts[:k])
            out[f"recall@{k}"].append(sum(found) / len(found))
            out[f"hit@{k}"].append(float(any(found)))
        first = next((rank for rank, t in enumerate(texts, 1) if any(_found(item["evidence"], [t]))), None)
        out["mrr"].append(1 / first if first else 0.0)
    return {name: round(sum(v) / len(v), 3) for name, v in out.items()}


def main() -> None:
    gold = [g for g in load_gold() if g["category"] != "unanswerable"]
    retriever, reranker = Retriever(), Reranker()
    depth = max(K_VALUES)
    texts = lambda hits: [h.chunk.text for h in hits]

    def reranked(item: dict, tickers: list[str] | None) -> list[Hit]:
        return reranker.rerank(item["question"], retriever.search(item["question"], tickers=tickers), keep=depth)

    v1 = {p["id"]: p for p in _read(RESULTS / "v1_predictions.json")}
    v2 = {p["id"]: p for p in _read(RESULTS / "v2_predictions.json")}
    rows = {
        "v1 pipeline (frozen contexts)": score(gold, lambda g: v1[g["id"]]["contexts"]),
        "BM25 only": score(gold, lambda g: [retriever.by_id[c].text for c in retriever.bm25_search(g["question"], depth)]),
        "Dense only": score(gold, lambda g: [retriever.by_id[c].text for c in retriever.dense_search(g["question"], depth)]),
        "Hybrid (RRF)": score(gold, lambda g: texts(retriever.search(g["question"], limit=depth))),
        "Hybrid + rerank": score(gold, lambda g: texts(reranked(g, None))),
        "Hybrid + rerank + company filter": score(gold, lambda g: texts(reranked(g, extract_tickers(g["question"]) or None))),
    }
    if v2:
        # Context the v2 agent actually handed the generator (planner sub-queries, rewrites, tables), calculator excluded.
        rows["v2 agent (plan + decomposition + rewrite)"] = score(
            gold, lambda g: [t for t, cid in zip(v2[g["id"]]["contexts"], v2[g["id"]]["source_ids"]) if cid != "CALCULATOR"])

    header = ["Configuration", *[f"recall@{k}" for k in K_VALUES], *[f"hit@{k}" for k in K_VALUES], "MRR"]
    lines = ["# Retrieval ablation", "",
             f"40 answerable gold questions; recall = share of gold evidence strings present in the top-k chunks. "
             f"Embeddings `{settings.embedding_model}`, reranker `{settings.reranker_model}`.", "",
             "| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[h]:.2f}" for h in header[1:-1]) + f" | {m['mrr']:.2f} |")
    (RESULTS / "RETRIEVAL_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "retrieval_ablation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
