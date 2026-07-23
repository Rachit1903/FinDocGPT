"""Separate filing prose from tables before either is used for retrieval."""

from __future__ import annotations

import hashlib
import re
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

from .models import Filing, ProseChunk, TableManifest

WHITESPACE = re.compile(r"\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def clean_text(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def estimate_tokens(text: str) -> int:
    """A stable approximation suitable for chunk sizing before model tokenization."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _chunk_text(text: str, target_tokens: int = 400, overlap_tokens: int = 60) -> list[str]:
    sentences = [clean_text(sentence) for sentence in SENTENCE_END.split(text) if clean_text(sentence)]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_count = 0
            for prior in reversed(current):
                overlap.insert(0, prior)
                overlap_count += estimate_tokens(prior)
                if overlap_count >= overlap_tokens:
                    break
            current, current_tokens = overlap, sum(estimate_tokens(part) for part in overlap)
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_prose_chunks(soup: BeautifulSoup, filing: Filing) -> list[ProseChunk]:
    """Walk visible narrative blocks while deliberately excluding every table subtree."""
    for unwanted in soup(["script", "style", "noscript", "table"]):
        unwanted.decompose()
    body = soup.body or soup
    section = "Filing narrative"
    blocks: list[tuple[str, str]] = []
    for element in body.find_all([*HEADING_TAGS, "p", "div", "li"]):
        if not isinstance(element, Tag) or element.find_parent(["table", "script", "style"]):
            continue
        text = clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in HEADING_TAGS or (len(text) < 160 and re.match(r"^(item\s+)?\d+[a-z]?\.?", text, re.I)):
            section = text[:240]
            continue
        if len(text) >= 80:
            blocks.append((section, text))
    chunks: list[ProseChunk] = []
    for block_number, (block_section, text) in enumerate(blocks):
        for part_number, piece in enumerate(_chunk_text(text)):
            digest = hashlib.sha1(f"{filing.accession_number}:{block_number}:{part_number}".encode()).hexdigest()[:14]
            chunks.append(ProseChunk(
                chunk_id=f"{filing.ticker}-{digest}", text=piece, ticker=filing.ticker, cik=filing.cik,
                filing_type=filing.form, filing_date=filing.filing_date, report_date=filing.report_date,
                section=block_section, source_url=filing.source_url, token_estimate=estimate_tokens(piece),
            ))
    return chunks


def _table_context(table: Tag) -> tuple[str, str]:
    caption = table.find("caption")
    if caption:
        return clean_text(caption.get_text(" ", strip=True)), "Table caption"
    preceding = table.find_all_previous([*HEADING_TAGS, "p", "div"], limit=8)
    candidates = [clean_text(item.get_text(" ", strip=True)) for item in preceding]
    candidates = [candidate for candidate in candidates if 3 < len(candidate) < 280]
    title = candidates[0] if candidates else "Untitled financial table"
    section = next((candidate for candidate in candidates if re.match(r"^(item\s+)?\d+", candidate, re.I)), "Filing table")
    return title, section


def _flatten_column_name(column: object) -> str:
    values = column if isinstance(column, tuple) else (column,)
    return clean_text(" ".join(str(value) for value in values if str(value).lower() != "nan"))


def extract_tables(soup: BeautifulSoup, filing: Filing, tables_dir: Path) -> list[TableManifest]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[TableManifest] = []
    for ordinal, table in enumerate(soup.find_all("table")):
        try:
            # Pin pandas to lxml, which is a declared project dependency. Without
            # an explicit flavor pandas may select optional html5lib instead.
            frame = pd.read_html(StringIO(str(table)), flavor="lxml", displayed_only=False)[0]
        except (ValueError, IndexError):
            continue
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        # A table with one data row is still structurally meaningful (for example,
        # a single-period balance or a small note disclosure).
        if frame.empty or frame.shape[0] < 1 or frame.shape[1] < 2:
            continue
        frame.columns = [_flatten_column_name(column) for column in frame.columns]
        title, section = _table_context(table)
        digest = hashlib.sha1(f"{filing.accession_number}:table:{ordinal}".encode()).hexdigest()[:14]
        table_id = f"{filing.ticker}-{digest}"
        parquet_path = tables_dir / f"{table_id}.parquet"
        frame.to_parquet(parquet_path, index=False)
        manifests.append(TableManifest(
            table_id=table_id, parquet_path=str(parquet_path).replace("\\", "/"), ticker=filing.ticker,
            cik=filing.cik, filing_type=filing.form, filing_date=filing.filing_date,
            report_date=filing.report_date, title=title, section=section, source_url=filing.source_url,
            rows=len(frame), columns=len(frame.columns), column_names=[str(column) for column in frame.columns],
        ))
    return manifests


def parse_filing(html_path: Path, filing: Filing, tables_dir: Path) -> tuple[list[ProseChunk], list[TableManifest]]:
    soup = BeautifulSoup(html_path.read_bytes(), "lxml")
    # Use independent trees: prose extraction removes tables, while table extraction needs them intact.
    prose_soup = BeautifulSoup(str(soup), "lxml")
    return extract_prose_chunks(prose_soup, filing), extract_tables(soup, filing, tables_dir)
