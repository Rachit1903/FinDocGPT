"""Typed records shared by ingestion, retrieval, and the agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: str
    accession_number: str
    form: str
    filing_date: str
    report_date: str | None
    primary_document: str
    source_url: str

    @property
    def fiscal_year(self) -> int:
        # The period-of-report year is the fiscal year label every filer uses in its 10-K.
        return int((self.report_date or self.filing_date)[:4])


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a section-aware prose passage or a (slice of a) normalized table."""

    chunk_id: str
    kind: str  # "prose" | "table"
    ticker: str
    company: str
    form: str
    fiscal_year: int
    report_date: str | None
    filing_date: str
    item: str
    subsection: str
    context: str
    text: str
    table: dict[str, Any] | None
    source_url: str
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def indexed_text(self) -> str:
        """What BM25, the embedder, and the reranker see: header + body."""
        return f"{self.context}\n{self.text}"

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Chunk":
        return cls(**record)
