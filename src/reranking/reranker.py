"""Cross-encoder reranking using BAAI/bge-reranker-base on CPU."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from src.retrieval.hybrid import RetrievedChunk

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


@dataclass(frozen=True)
class RerankedChunk:
    chunk: RetrievedChunk
    reranker_score: float
    fused_rank: int
    reranked_rank: int


class CrossEncoderReranker:
    """Rerank a bounded candidate set; never use the cross-encoder as first-stage retrieval."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            # Import lazily so deterministic reranking tests need no Torch runtime.
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, candidates: Iterable[RetrievedChunk], k: int = 5) -> list[RerankedChunk]:
        candidate_list = list(candidates)
        if k < 1:
            raise ValueError("k must be positive")
        if not candidate_list:
            return []
        pairs = [(query, candidate.text) for candidate in candidate_list]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            enumerate(zip(candidate_list, scores, strict=True), start=1),
            key=lambda item: (-float(item[1][1]), item[1][0].chunk_id),
        )
        return [
            RerankedChunk(
                chunk=candidate,
                reranker_score=float(score),
                fused_rank=fused_rank,
                reranked_rank=reranked_rank,
            )
            for reranked_rank, (fused_rank, (candidate, score)) in enumerate(ranked[:k], start=1)
        ]
