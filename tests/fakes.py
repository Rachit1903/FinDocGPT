"""Test doubles: scripted LLMs and an in-memory retriever, so tests need no network or models."""

from __future__ import annotations

from src.ingestion.models import Chunk
from src.llm import LLMError
from src.retrieval.store import Hit


class FakeLLM:
    def __init__(self, json_responses=None, text_responses=None):
        self.json_responses = list(json_responses or [])
        self.text_responses = list(text_responses or [])
        self.json_prompts: list[str] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.json_prompts.append(user)
        if not self.json_responses:
            raise LLMError("no scripted response")
        response = self.json_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return self.text_responses.pop(0)


def make_chunk(chunk_id: str, text: str, ticker: str = "GOOGL", kind: str = "prose") -> Chunk:
    return Chunk(chunk_id=chunk_id, kind=kind, ticker=ticker, company=ticker, form="10-K", fiscal_year=2025,
                 report_date="2025-12-31", filing_date="2026-02-01", item="Item 7", subsection="", context=f"{ticker} 10-K",
                 text=text, table=None, source_url="https://sec.gov", token_estimate=len(text.split()))


def make_hit(chunk_id: str, text: str, ticker: str = "GOOGL", kind: str = "prose", score: float = 0.9) -> Hit:
    return Hit(make_chunk(chunk_id, text, ticker, kind), score, rerank_score=score)


class FakeRetriever:
    def __init__(self, hits_by_ticker: dict[str, list[Hit]]):
        self.hits_by_ticker = hits_by_ticker
        self.queries: list[tuple[str, tuple | None]] = []

    def search(self, query, limit=None, tickers=None, kinds=None):
        self.queries.append((query, tuple(tickers) if tickers else None))
        pool = [h for t, hits in self.hits_by_ticker.items() if not tickers or t in tickers for h in hits]
        return [h for h in pool if not kinds or h.chunk.kind in kinds]


class FakeReranker:
    """Keeps each hit's preset score, so tests control what counts as weak evidence."""

    def __init__(self, scores_by_query: dict[str, float] | None = None):
        self.scores_by_query = scores_by_query or {}

    def rerank(self, query, hits, keep):
        for hit in hits:
            hit.rerank_score = self.scores_by_query.get(query, hit.score)
        return sorted(hits, key=lambda h: -h.rerank_score)[:keep]
