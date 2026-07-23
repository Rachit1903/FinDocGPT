"""Hand-rolled state-machine pipeline joining router, retrieval, reranking, and tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.numeric import FinancialTableAgent
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.hybrid import HybridRetriever

from .query_router import QueryRoute, QueryRouter, RouteDecision


@dataclass(frozen=True)
class PipelineResult:
    route: QueryRoute
    decision: RouteDecision
    result: Any


class FinDocPipeline:
    def __init__(self, router: QueryRouter | None = None, retriever: Any | None = None, reranker: Any | None = None, numeric_agent: Any | None = None) -> None:
        self.router = router or QueryRouter()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.numeric_agent = numeric_agent or FinancialTableAgent()

    def answer(self, query: str, k: int = 5) -> PipelineResult:
        decision = self.router.classify(query)
        if decision.route is QueryRoute.COMPUTATION:
            return PipelineResult(decision.route, decision, self.numeric_agent.answer(query))
        candidates = self.retriever.retrieve(query, k=20, candidate_limit=20)
        return PipelineResult(decision.route, decision, self.reranker.rerank(query, candidates, k=k))
