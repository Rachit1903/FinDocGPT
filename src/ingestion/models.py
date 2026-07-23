"""Typed records shared by the Phase 1 ingestion pipeline."""

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


@dataclass(frozen=True)
class ProseChunk:
    chunk_id: str
    text: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    report_date: str | None
    section: str
    source_url: str
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TableManifest:
    table_id: str
    parquet_path: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    report_date: str | None
    title: str
    section: str
    source_url: str
    rows: int
    columns: int
    column_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

