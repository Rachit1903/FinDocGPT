"""Application service layer for the FinDocGPT API."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.generation.groq import GroqGenerator
from src.router.pipeline import FinDocPipeline
from src.router.query_router import QueryRoute
from src.verification.citations import ClaimVerdict, SourceChunk, verify_citations


class FinDocService:
    def __init__(self, pipeline: FinDocPipeline | None = None, generator: GroqGenerator | None = None) -> None:
        self.pipeline = pipeline or FinDocPipeline()
        self.generator = generator or GroqGenerator()

    @staticmethod
    def _serialize_sources(items: list[Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for item in items:
            if hasattr(item, "chunk"):
                chunk = item.chunk
                sources.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "rrf_score": chunk.rrf_score,
                        "bm25_rank": chunk.bm25_rank,
                        "dense_rank": chunk.dense_rank,
                        "reranker_score": item.reranker_score,
                        "fused_rank": item.fused_rank,
                        "reranked_rank": item.reranked_rank,
                    }
                )
            else:
                sources.append(
                    {
                        "chunk_id": item.table_id,
                        "text": item.source_context,
                        "metadata": {
                            "ticker": item.ticker,
                            "table_id": item.table_id,
                            "table_title": item.table_title,
                            "metric": item.metric,
                        },
                        "rrf_score": None,
                        "bm25_rank": None,
                        "dense_rank": None,
                        "reranker_score": None,
                        "fused_rank": None,
                        "reranked_rank": None,
                    }
                )
        return sources

    @staticmethod
    def _to_verdicts(verdicts: list[ClaimVerdict]) -> list[dict[str, Any]]:
        return [asdict(verdict) for verdict in verdicts]

    def answer(self, query: str, k: int = 5) -> dict[str, Any]:
        pipeline_result = self.pipeline.answer(query, k=k)
        route = pipeline_result.route

        if route is QueryRoute.COMPUTATION:
            computation = pipeline_result.result
            sources = [
                SourceChunk(chunk_id=computation.table_id, text=computation.source_context),
            ]
            draft_answer = computation.answer
            verification = verify_citations(draft_answer, sources)
            return {
                "query": query,
                "route": route.value,
                "route_reason": pipeline_result.decision.reason,
                "draft_answer": draft_answer,
                "answer": verification.verified_answer,
                "unsupported_claims": verification.unsupported_claims,
                "sources": self._serialize_sources([computation]),
                "verdicts": self._to_verdicts(verification.verdicts),
                "computation": asdict(computation),
            }

        ranked = pipeline_result.result
        contexts = [result.chunk.text for result in ranked]
        draft_answer = self.generator.answer(query, contexts)
        source_chunks = [SourceChunk(chunk_id=item.chunk.chunk_id, text=item.chunk.text) for item in ranked]
        verification = verify_citations(draft_answer, source_chunks)
        return {
            "query": query,
            "route": route.value,
            "route_reason": pipeline_result.decision.reason,
            "draft_answer": draft_answer,
            "answer": verification.verified_answer,
            "unsupported_claims": verification.unsupported_claims,
            "sources": self._serialize_sources(ranked),
            "verdicts": self._to_verdicts(verification.verdicts),
            "computation": None,
        }
