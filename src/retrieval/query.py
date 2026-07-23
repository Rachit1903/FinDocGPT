"""Small CLI for manually inspecting hybrid retrieval results."""

from __future__ import annotations

import argparse

from .hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval against indexed filings.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    for result in HybridRetriever().retrieve(args.query, k=args.k):
        print(f"\n[{result.rrf_score:.5f}] {result.metadata['ticker']} {result.metadata['filing_type']} | {result.metadata['section']}")
        print(f"BM25 rank: {result.bm25_rank}; dense rank: {result.dense_rank}")
        print(result.text[:600])


if __name__ == "__main__":
    main()
