"""Embed the chunk corpus into Qdrant (embedded on-disk by default, or QDRANT_URL if set).

Usage: python -m src.retrieval.build_index [--recreate] [--model BAAI/bge-base-en-v1.5]
"""

from __future__ import annotations

import argparse

from .store import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    retriever = Retriever(embedding_model=args.model)
    count = retriever.build_index(recreate=args.recreate)
    print(f"Collection {retriever.collection} holds {count} vectors.")


if __name__ == "__main__":
    main()
