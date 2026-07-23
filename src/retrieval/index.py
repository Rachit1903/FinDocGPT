"""Build or refresh Qdrant's dense index from Phase 1 prose chunks."""

from __future__ import annotations

import argparse

from qdrant_client.models import Distance, PointStruct, VectorParams

from .hybrid import DEFAULT_COLLECTION, DEFAULT_MODEL, HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Phase 1 prose chunks into Qdrant.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild this collection.")
    args = parser.parse_args()

    retriever = HybridRetriever(collection_name=args.collection, embedding_model=args.model)
    sample_vector = retriever.embedder.encode("dimension probe", normalize_embeddings=True)
    if args.recreate and retriever.client.collection_exists(args.collection):
        retriever.client.delete_collection(args.collection)
    if not retriever.client.collection_exists(args.collection):
        retriever.client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=len(sample_vector), distance=Distance.COSINE),
        )

    total = len(retriever._chunks)
    for start in range(0, total, args.batch_size):
        batch = retriever._chunks[start : start + args.batch_size]
        vectors = retriever.embedder.encode(
            [chunk["text"] for chunk in batch], normalize_embeddings=True, show_progress_bar=False
        )
        points = [
            PointStruct(id=retriever.point_id(chunk["chunk_id"]), vector=vector.tolist(), payload=chunk)
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        retriever.client.upsert(collection_name=args.collection, points=points, wait=True)
        print(f"Indexed {min(start + len(batch), total)}/{total} chunks")
    print(f"Qdrant collection '{args.collection}' is ready with {total} chunks.")


if __name__ == "__main__":
    main()
