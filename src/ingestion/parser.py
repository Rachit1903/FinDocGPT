"""Turn one 10-K HTML document into section-aware prose chunks and normalized table chunks.

Two decisions matter most for retrieval quality:
  * Prose is packed into ~300-token chunks *within* an Item section, so a sentence like
    "Services net sales increased ... due to advertising" keeps its surrounding paragraph.
  * Every chunk carries a contextual header (company, filing, Item, subsection) that is
    indexed with the text. Filing sentences rarely name the company, so without the header
    "Services net sales increased" is indistinguishable across ten filers.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator

from bs4 import BeautifulSoup, NavigableString, Tag

from src.companies import COMPANIES

from .models import Chunk, Filing
from .tables import normalize_table

WHITESPACE = re.compile(r"\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")
ITEM_HEADING = re.compile(r"^item\s+\d+[a-z]?\s*[.:]", re.I)
BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
TARGET_TOKENS = 300
MAX_TOKENS = 420
TABLE_ROWS_PER_CHUNK = 45


QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "\xa0": " "})
YEAR_COLUMN = re.compile(r"^(19|20)\d{2}$")


def clean_text(value: str) -> str:
    return WHITESPACE.sub(" ", value.translate(QUOTES)).strip()


def block_text(tag: Tag) -> str:
    # Inline spans are joined exactly as a browser renders them; some filers split words across spans.
    return clean_text(tag.get_text(""))


def estimate_tokens(text: str) -> int:
    return max(1, round(len(re.findall(r"\w+|[^\w\s]", text)) * 0.85))


def _is_bold(element: Tag) -> bool:
    texts = [node for node in element.find_all(string=True) if clean_text(str(node))]
    if not texts:
        return False
    for node in texts:
        styled = node.parent
        style = (styled.get("style") or "").lower() if isinstance(styled, Tag) else ""
        if not (("font-weight:700" in style.replace(" ", "")) or ("font-weight:bold" in style.replace(" ", ""))
                or (isinstance(styled, Tag) and styled.name in {"b", "strong"})):
            return False
    return True


def _walk(node: Tag) -> Iterator[tuple[str, Tag]]:
    """Yield ("table", tag) and ("block", tag) in document order; blocks are leaf-level only."""
    for child in node.children:
        if isinstance(child, NavigableString) or not isinstance(child, Tag):
            continue
        if child.name in {"script", "style", "noscript", "head"}:
            continue
        if child.name == "table":
            yield "table", child
        elif child.name in BLOCK_TAGS and not child.find(list(BLOCK_TAGS | {"table"})):
            yield "block", child
        else:
            yield from _walk(child)


def _split_long(text: str) -> list[str]:
    sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
    pieces, current = [], []
    for sentence in sentences:
        if current and estimate_tokens(" ".join(current + [sentence])) > MAX_TOKENS:
            pieces.append(" ".join(current))
            current = current[-1:] if estimate_tokens(current[-1]) < 60 else []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


class _ChunkBuilder:
    def __init__(self, filing: Filing) -> None:
        self.filing = filing
        self.chunks: list[Chunk] = []
        self._buffer: list[str] = []
        self._buffer_subsection = ""
        self.item = "Cover and preamble"
        self.subsection = ""
        self._last_year_columns: list[str] = []

    @property
    def filing_id(self) -> str:
        return f"{self.filing.ticker}-{self.filing.form.replace('/', '')}-FY{self.filing.fiscal_year}"

    def context(self, subsection: str) -> str:
        company = COMPANIES.get(self.filing.ticker, self.filing.ticker)
        parts = [f"{company} ({self.filing.ticker}) {self.filing.form} for fiscal year {self.filing.fiscal_year}", self.item]
        if subsection:
            parts.append(subsection)
        return " | ".join(parts)

    def _emit(self, kind: str, text: str, subsection: str, table: dict | None = None) -> None:
        chunk_id = f"{self.filing_id}-{len(self.chunks):04d}"
        self.chunks.append(Chunk(
            chunk_id=chunk_id, kind=kind, ticker=self.filing.ticker, company=COMPANIES.get(self.filing.ticker, ""),
            form=self.filing.form, fiscal_year=self.filing.fiscal_year, report_date=self.filing.report_date,
            filing_date=self.filing.filing_date, item=self.item, subsection=subsection,
            context=self.context(subsection), text=text, table=table, source_url=self.filing.source_url,
            token_estimate=estimate_tokens(text),
        ))

    def flush(self) -> None:
        if self._buffer:
            text = "\n".join(self._buffer)
            if estimate_tokens(text) >= 25:
                self._emit("prose", text, self._buffer_subsection)
        self._buffer = []

    def add_paragraph(self, text: str) -> None:
        for piece in (_split_long(text) if estimate_tokens(text) > MAX_TOKENS else [text]):
            current = estimate_tokens("\n".join(self._buffer)) if self._buffer else 0
            if self._buffer and current + estimate_tokens(piece) > TARGET_TOKENS:
                last = self._buffer[-1]
                self.flush()
                # Carry a short preceding paragraph forward as overlap so no chunk starts mid-thought.
                if estimate_tokens(last) <= 60:
                    self._buffer = [last]
            if not self._buffer:
                self._buffer_subsection = self.subsection
            self._buffer.append(piece)

    def set_item(self, heading: str) -> None:
        self.flush()
        self.item, self.subsection = heading[:120], ""

    def set_subsection(self, heading: str) -> None:
        self.flush()
        self.subsection = heading[:120]

    def add_table(self, table: Tag, title: str) -> None:
        normalized = normalize_table(table)
        if normalized is None or not normalized.has_numbers or len(normalized.rows) < 2:
            return
        self.flush()
        width = max(len(values) for _, values in normalized.rows)
        if all(YEAR_COLUMN.match(c) for c in normalized.columns[:width]) and normalized.columns:
            self._last_year_columns = normalized.columns[:width]
        elif len(self._last_year_columns) == width:
            # Filers often split one visual table in two; the second half loses its year header row.
            normalized.columns = list(self._last_year_columns)
        for start in range(0, len(normalized.rows), TABLE_ROWS_PER_CHUNK):
            part = type(normalized)(normalized.columns, normalized.rows[start:start + TABLE_ROWS_PER_CHUNK], normalized.header_text)
            header = f"Table: {title}" + (f" ({normalized.header_text})" if normalized.header_text else "")
            text = f"{header}\n{part.to_markdown()}"
            self._emit("table", text, title, {"title": title, "columns": part.columns,
                                               "rows": [[label, values] for label, values in part.rows]})


def parse_filing(html: bytes, filing: Filing) -> list[Chunk]:
    soup = BeautifulSoup(html, "lxml")
    # iXBRL filings embed a hidden header with raw XBRL facts; it is not reader-visible text.
    for hidden in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        hidden.decompose()
    events = list(_walk(soup.body or soup))
    block_texts = [block_text(tag) for kind, tag in events if kind == "block"]
    # Running page headers/footers ("Apple Inc. | 2025 Form 10-K | 23") repeat on every page.
    repeated = {text for text, count in Counter(re.sub(r"\d+", "#", t) for t in block_texts).items() if count >= 3 and len(text) < 120}

    builder = _ChunkBuilder(filing)
    recent_context = ""
    for kind, tag in events:
        if kind == "table":
            table_text = clean_text(tag.get_text(" "))
            if len(table_text) < 200 and ITEM_HEADING.match(table_text):
                # Some filers lay out Item headings as a two-cell table ("Item 7." | "Management's ...").
                builder.set_item(table_text)
                recent_context = ""
            else:
                builder.add_table(tag, recent_context or builder.subsection or builder.item)
            continue
        text = block_text(tag)
        if not text or re.sub(r"\d+", "#", text) in repeated or re.fullmatch(r"[\d\s|]+", text):
            continue
        if len(text) < 160 and ITEM_HEADING.match(text):
            builder.set_item(text)
            recent_context = ""
        elif len(text) < 120 and not text.endswith(".") and _is_bold(tag):
            builder.set_subsection(text)
            recent_context = text
        else:
            if len(text) < 200:
                recent_context = text
            if len(text) >= 40:
                builder.add_paragraph(text)
    builder.flush()
    return builder.chunks
