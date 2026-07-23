"""Three Phase 4 examples proving distinct router branches execute."""

from __future__ import annotations

import os

from .pipeline import FinDocPipeline

EXAMPLES = [
    "What did Apple say about its services business?",
    "Compare Apple and Microsoft gross-margin discussion.",
    "Calculate how Apple's total gross margin percentage changed from 2024 to 2025.",
]


def main() -> None:
    # Preload the shared CPU model runtime once before the three routes run.
    # This also avoids repeated Torch DLL initialization on Windows.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from sentence_transformers import SentenceTransformer  # noqa: F401

    pipeline = FinDocPipeline()
    for query in EXAMPLES:
        response = pipeline.answer(query)
        print(f"\nQUERY: {query}\nROUTE: {response.route}\nWHY: {response.decision.reason}")
        if response.route.value == "computation":
            print(response.result.answer)
            print(f"CALCULATION: {response.result.calculation}")
            print(f"TABLE: {response.result.table_id} | {response.result.table_title}")
        else:
            for item in response.result:
                print(f"#{item.reranked_rank} (fused #{item.fused_rank}) {item.chunk.metadata['ticker']}: {item.chunk.text[:180]}")


if __name__ == "__main__":
    main()
