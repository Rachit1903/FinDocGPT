import json

import pandas as pd

from src.agents.numeric import FinancialTableAgent
from src.router.pipeline import FinDocPipeline
from src.router.query_router import QueryRoute, QueryRouter


def test_router_selects_each_path() -> None:
    router = QueryRouter()
    assert router.classify("What did Apple disclose about services?").route is QueryRoute.FACTUAL_LOOKUP
    assert router.classify("Compare Apple versus Microsoft gross margin.").route is QueryRoute.COMPARISON
    assert router.classify("Calculate Apple's gross margin percentage change from 2024 to 2025.").route is QueryRoute.COMPUTATION


def test_numeric_agent_calculates_from_parquet(tmp_path) -> None:
    table_path = tmp_path / "table.parquet"
    pd.DataFrame([["Total gross margin percentage", "46.9", "%", "46.2", "%"]]).to_parquet(table_path, index=False)
    manifest_path = tmp_path / "tables.jsonl"
    manifest_path.write_text(json.dumps({
        "table_id": "aapl-table", "parquet_path": str(table_path), "ticker": "AAPL",
        "filing_date": "2025-10-31", "title": "2025 2024 Total gross margin percentage", "section": "Item 7",
    }) + "\n", encoding="utf-8")
    answer = FinancialTableAgent(manifest_path).answer("Calculate Apple's total gross margin percentage from 2024 to 2025.")
    assert answer.values_by_year == {2025: 46.9, 2024: 46.2}
    assert "+0.7 percentage points" in answer.answer


def test_numeric_agent_infers_years_from_table_metadata(tmp_path) -> None:
    table_path = tmp_path / "table.parquet"
    pd.DataFrame([['Operating margin', '12.0', '%', '10.0', '%']]).to_parquet(table_path, index=False)
    manifest_path = tmp_path / "tables.jsonl"
    manifest_path.write_text(json.dumps({
        "table_id": "meta-table", "parquet_path": str(table_path), "ticker": "META",
        "filing_date": "2025-10-31", "title": "2025 2024 Operating margin", "section": "Item 7",
    }) + "\n", encoding="utf-8")
    answer = FinancialTableAgent(manifest_path).answer("Calculate Meta's operating-margin change between the two latest annual periods.")
    assert answer.values_by_year == {2025: 12.0, 2024: 10.0}
    assert "+2.0 percentage points" in answer.answer


def test_pipeline_executes_retrieval_and_table_branches() -> None:
    class FakeRetriever:
        def retrieve(self, query, k, candidate_limit):
            return ["candidate"]

    class FakeReranker:
        def rerank(self, query, candidates, k):
            return ["reranked"]

    class FakeNumericAgent:
        def answer(self, query):
            return "computed"

    pipeline = FinDocPipeline(retriever=FakeRetriever(), reranker=FakeReranker(), numeric_agent=FakeNumericAgent())
    assert pipeline.answer("What did AAPL disclose?").result == ["reranked"]
    assert pipeline.answer("Compare AAPL versus MSFT.").result == ["reranked"]
    assert pipeline.answer("Calculate AAPL gross margin percentage from 2024 to 2025.").result == "computed"
