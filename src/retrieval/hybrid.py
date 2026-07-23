"""Hybrid BM25 + Qdrant dense retrieval, fused with reciprocal-rank fusion."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from .fusion import reciprocal_rank_fusion

DEFAULT_COLLECTION = "findocgpt_prose_chunks"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)?")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    """Loads one canonical prose corpus and retrieves via sparse + dense rankings."""

    def __init__(
        self,
        chunks_path: Path = Path("data/processed/prose_chunks.jsonl"),
        collection_name: str = DEFAULT_COLLECTION,
        qdrant_url: str | None = None,
        embedding_model: str = DEFAULT_MODEL,
    ) -> None:
        self.chunks_path = chunks_path
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.embedding_model_name = embedding_model
        self._chunks = self._load_chunks(chunks_path)
        self._by_id = {chunk["chunk_id"]: chunk for chunk in self._chunks}
        self._bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in self._chunks])
        self._embedder: Any | None = None
        self._client: QdrantClient | None = None

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Processed chunk corpus not found: {path}")
        # Ingestion is append-only, so retain the current version of every chunk ID.
        records: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records[record["chunk_id"]] = record
        if not records:
            raise ValueError(f"No prose chunks found in {path}")
        return list(records.values())

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            # Import lazily so BM25/RRF tooling and tests do not require Torch.
            # FinDocGPT uses PyTorch only; this prevents Transformers from loading
            # an unrelated, broken TensorFlow installation on Windows.
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name, device="cpu")
        return self._embedder

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"findocgpt:{chunk_id}"))

    def bm25_search(self, query: str, limit: int = 20) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        indices = np.argsort(scores)[::-1][:limit]
        return [self._chunks[index]["chunk_id"] for index in indices if scores[index] > 0]

    def dense_search(self, query: str, limit: int = 20) -> list[str]:
        vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(
            collection_name=self.collection_name, query=vector, limit=limit, with_payload=True
        )
        return [point.payload["chunk_id"] for point in response.points]

    def retrieve(self, query: str, k: int = 5, candidate_limit: int = 20) -> list[RetrievedChunk]:
        if k < 1 or candidate_limit < k:
            raise ValueError("candidate_limit must be at least k, and k must be positive")
        bm25_ids = self.bm25_search(query, candidate_limit)
        dense_ids = self.dense_search(query, candidate_limit)
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[:k]
        bm25_ranks = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)}
        dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=self._by_id[chunk_id]["text"],
                metadata={key: value for key, value in self._by_id[chunk_id].items() if key not in {"chunk_id", "text"}},
                rrf_score=score,
                bm25_rank=bm25_ranks.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
            )
            for chunk_id, score in fused
        ]
