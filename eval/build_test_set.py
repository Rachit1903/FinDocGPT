"""Create filing-grounded reference records from the approved evaluation draft."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.generation.groq import GroqGenerator
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "eval" / "test_set_draft.md"
OUTPUT = ROOT / "eval" / "test_set.json"
ITEM = re.compile(r"^(\d+)\. \*\*(Factual|Comparison|Computation)\*\* — (.+)$")
TICKER = re.compile(r"^## ([A-Z]+)$")


def draft_items() -> list[dict]:
    ticker = None
    items = []
    for line in DRAFT.read_text(encoding="utf-8").splitlines():
        ticker_match = TICKER.match(line)
        if ticker_match:
            ticker = ticker_match.group(1)
        item_match = ITEM.match(line)
        if item_match and ticker:
            number, category, question = item_match.groups()
            items.append({"id": f"eval-{int(number):02d}", "ticker": ticker, "category": category.lower(), "question": question})
    if len(items) != 40:
        raise ValueError(f"Expected 40 approved draft items, found {len(items)}.")
    return items


def main() -> None:
    retriever, reranker, generator = HybridRetriever(), CrossEncoderReranker(), GroqGenerator()
    records = []
    for item in draft_items():
        candidates = retriever.retrieve(item["question"], k=20, candidate_limit=20)
        ranked = reranker.rerank(item["question"], candidates, k=5)
        # The reference is constrained to real filing excerpts and records every source ID used.
        contexts = [result.chunk.text for result in ranked]
        reference = generator.answer(item["question"], contexts)
        records.append({**item, "reference_answer": reference, "reference_source_ids": [result.chunk.chunk_id for result in ranked]})
        print(f"Built {item['id']}: {item['ticker']} {item['category']}")
    OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
