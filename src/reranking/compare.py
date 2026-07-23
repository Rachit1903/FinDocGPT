"""CLI showing how reranking changes a hybrid-retrieval result order."""

from __future__ import annotations

import argparse

from src.retrieval.hybrid import HybridRetriever

from .reranker import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hybrid retrieval before and after cross-encoder reranking.")
    parser.add_argument("query")
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    candidates = HybridRetriever().retrieve(args.query, k=args.candidate_limit, candidate_limit=args.candidate_limit)
    reranked = CrossEncoderReranker().rerank(args.query, candidates, k=args.k)
    print("BEFORE: hybrid RRF order")
    for rank, item in enumerate(candidates, start=1):
        print(f"{rank:>2}. {item.metadata['ticker']} | {item.metadata['section']} | RRF={item.rrf_score:.5f}")
        print(f"    {item.text[:180]}")
    print("\nAFTER: BAAI/bge-reranker-base order")
    for item in reranked:
        print(f"{item.reranked_rank:>2}. (was {item.fused_rank:>2}) {item.chunk.metadata['ticker']} | score={item.reranker_score:.4f}")
        print(f"    {item.chunk.text[:180]}")


if __name__ == "__main__":
    main()
