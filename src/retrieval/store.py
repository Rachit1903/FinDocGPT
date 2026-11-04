"""Hybrid retrieval over the chunk corpus: BM25 + dense (Qdrant), fused with RRF, then reranked.

Both first-stage retrievers support metadata filters (ticker, chunk kind) so a question
about Apple never competes with nine other filers' near-identical boilerplate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from src.config import settings
from src.ingestion.models import Chunk

from .fusion import reciprocal_rank_fusion

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has", "have", "how",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "which", "with", "compare", "describe", "disclose", "disclosed", "identify", "latest", "annual", "filing",
}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOP_WORDS and len(t) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float  # RRF score before reranking, cross-encoder probability after
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        c = self.chunk
        return {"chunk_id": c.chunk_id, "kind": c.kind, "ticker": c.ticker, "fiscal_year": c.fiscal_year,
                "item": c.item, "subsection": c.subsection, "text": c.text, "source_url": c.source_url,
                "bm25_rank": self.bm25_rank, "dense_rank": self.dense_rank, "fused_rank": self.fused_rank,
                "rerank_score": self.rerank_score}


def load_corpus(path=None) -> list[Chunk]:
    path = path or settings.corpus_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingestion.run_ingestion` first.")
    with path.open(encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"findocgpt:{chunk_id}"))


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class Retriever:
    def __init__(self, chunks: Sequence[Chunk] | None = None, embedding_model: str | None = None) -> None:
        self.chunks = list(chunks) if chunks is not None else load_corpus()
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.embedding_model = embedding_model or settings.embedding_model
        self._tickers = np.array([c.ticker for c in self.chunks])
        self._kinds = np.array([c.kind for c in self.chunks])
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(c.indexed_text) for c in self.chunks])

    @cached_property
    def embedder(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedding_model, device="cpu")
        model.max_seq_length = 512
        return model

    @cached_property
    def qdrant(self):
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url, timeout=60.0)
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_path))

    @property
    def collection(self) -> str:
        return f"{settings.collection}_{self.embedding_model.split('/')[-1]}"

    def embed_query(self, query: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.embedding_model.lower() else ""
        return self.embedder.encode(prefix + query, normalize_embeddings=True).tolist()

    def bm25_search(self, query: str, limit: int, tickers: Sequence[str] | None = None, kinds: Sequence[str] | None = None) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        mask = scores > 0
        if tickers:
            mask &= np.isin(self._tickers, list(tickers))
        if kinds:
            mask &= np.isin(self._kinds, list(kinds))
        indices = np.flatnonzero(mask)
        top = indices[np.argsort(-scores[indices])[:limit]]
        return [self.chunks[i].chunk_id for i in top]

    def dense_search(self, query: str, limit: int, tickers: Sequence[str] | None = None, kinds: Sequence[str] | None = None) -> list[str]:
        from qdrant_client import models

        conditions = []
        if tickers:
            conditions.append(models.FieldCondition(key="ticker", match=models.MatchAny(any=list(tickers))))
        if kinds:
            conditions.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
        response = self.qdrant.query_points(
            collection_name=self.collection, query=self.embed_query(query), limit=limit,
            query_filter=models.Filter(must=conditions) if conditions else None, with_payload=["chunk_id"],
        )
        return [point.payload["chunk_id"] for point in response.points]

    def search(self, query: str, limit: int | None = None, tickers: Sequence[str] | None = None,
               kinds: Sequence[str] | None = None) -> list[Hit]:
        limit = limit or settings.candidates_per_query
        bm25_ids = self.bm25_search(query, limit, tickers, kinds)
        dense_ids = self.dense_search(query, limit, tickers, kinds)
        bm25_rank = {cid: r for r, cid in enumerate(bm25_ids, 1)}
        dense_rank = {cid: r for r, cid in enumerate(dense_ids, 1)}
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[:limit]
        return [Hit(self.by_id[cid], score, bm25_rank.get(cid), dense_rank.get(cid), rank, queries=[query])
                for rank, (cid, score) in enumerate(fused, 1)]

    def _indexed_hashes(self) -> dict[str, str]:
        hashes, offset = {}, None
        while True:
            points, offset = self.qdrant.scroll(self.collection, limit=1024, offset=offset, with_payload=["chunk_id", "text_hash"])
            hashes.update({p.payload["chunk_id"]: p.payload.get("text_hash", "") for p in points})
            if offset is None:
                return hashes

    def build_index(self, batch_size: int = 32, recreate: bool = False) -> int:
        """Embed new or changed chunks (by text hash) and drop vectors for chunks no longer in the corpus."""
        from qdrant_client.models import Distance, PayloadSchemaType, PointIdsList, PointStruct, VectorParams

        client = self.qdrant
        if recreate and client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        if not client.collection_exists(self.collection):
            dim = self.embedder.get_sentence_embedding_dimension()
            client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            if settings.qdrant_url:
                for key in ("ticker", "kind"):
                    client.create_payload_index(self.collection, key, PayloadSchemaType.KEYWORD)
        indexed = self._indexed_hashes()
        stale = [cid for cid in indexed if cid not in self.by_id]
        if stale:
            client.delete(self.collection, points_selector=PointIdsList(points=[point_id(cid) for cid in stale]))
        todo = [c for c in self.chunks if indexed.get(c.chunk_id) != text_hash(c.indexed_text)]
        print(f"{len(todo)} of {len(self.chunks)} chunks need embedding; {len(stale)} stale vectors removed", flush=True)
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            vectors = self.embedder.encode([c.indexed_text for c in batch], normalize_embeddings=True,
                                           batch_size=batch_size, show_progress_bar=False)
            client.upsert(self.collection, [
                PointStruct(id=point_id(c.chunk_id), vector=v.tolist(),
                            payload={"chunk_id": c.chunk_id, "ticker": c.ticker, "kind": c.kind,
                                     "fiscal_year": c.fiscal_year, "text_hash": text_hash(c.indexed_text)})
                for c, v in zip(batch, vectors, strict=True)
            ])
            print(f"indexed {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
        return client.count(self.collection).count


class Reranker:
    """Cross-encoder second stage; scores are sigmoid probabilities in [0, 1]."""

    def __init__(self, model_name: str | None = None, model: Any | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit], keep: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.indexed_text) for h in hits], show_progress_bar=False, batch_size=16)
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = float(score)
        return sorted(hits, key=lambda h: (-h.rerank_score, h.chunk.chunk_id))[:keep]
