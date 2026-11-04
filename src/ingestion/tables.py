"""Normalize SEC HTML tables into (label, values) rows with resolved column headers.

EDGAR tables use colspans, spacer cells, and split "$" / "%" / ")" cells for layout.
Reading them with pandas duplicates every spanned cell, so we walk the logical
<td> cells directly and rebuild a compact grid that both an LLM and pandas can read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import Tag

WHITESPACE = re.compile(r"\s+")
NUMERIC_CELL = re.compile(r"^\(?-?\$?\s*\d[\d,]*(\.\d+)?\s*\)?\s*%?\)?$")
YEAR_CELL = re.compile(r"^(19|20)\d{2}$")
UNITS_CELL = re.compile(r"\bin (millions|thousands|billions)\b|except per|dollars in", re.I)
ATTACH_LEFT = {"%", ")", "%)", ")%", "pts", "bps"}
DASHES = {"—", "–", "-", "— %", "—%"}


@dataclass
class NormalizedTable:
    columns: list[str]
    rows: list[tuple[str, list[str]]] = field(default_factory=list)
    header_text: str = ""

    @property
    def has_numbers(self) -> bool:
        return any(any(NUMERIC_CELL.match(v) for v in values) for _, values in self.rows)

    def to_markdown(self, max_rows: int | None = None) -> str:
        width = max([len(self.columns)] + [len(values) for _, values in self.rows] or [0])
        columns = self.columns + [""] * (width - len(self.columns))
        lines = ["| | " + " | ".join(columns) + " |"]
        for label, values in self.rows[:max_rows]:
            lines.append(f"| {label} | " + " | ".join(values + [""] * (width - len(values))) + " |")
        return "\n".join(lines)


def _clean(text: str) -> str:
    text = WHITESPACE.sub(" ", text.replace("\xa0", " ").replace("’", "'")).strip()
    # "( 2,656 )" -> "(2,656)" so a negative value parses as one numeric cell.
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))


def _logical_cells(tr: Tag) -> list[str]:
    raw = [_clean(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"], recursive=False)]
    cells: list[str] = []
    pending_dollar = False
    for cell in raw:
        if not cell:
            continue
        if cell == "$":
            pending_dollar = True
            continue
        if cells and cells[-1] in {"(", "$("}:
            cells[-1] = cells[-1] + cell
            continue
        squashed = re.sub(r"(\d\)?)\s+%", r"\1%", cell)
        if NUMERIC_CELL.match(squashed):
            cell = squashed
        if cell in ATTACH_LEFT and cells:
            if not (cell.startswith("%") and cells[-1].endswith("%")):
                cells[-1] = f"{cells[-1]}{cell}" if cell != "pts" else f"{cells[-1]} pts"
            continue
        if cell in DASHES:
            cell = "—"
        if pending_dollar and NUMERIC_CELL.match(cell):
            cell = f"${cell}"
        pending_dollar = False
        cells.append(cell)
    return cells


def _is_value(cell: str) -> bool:
    return bool(NUMERIC_CELL.match(cell)) or cell == "—"


def normalize_table(table: Tag) -> NormalizedTable | None:
    grid = [cells for tr in table.find_all("tr") if (cells := _logical_cells(tr))]
    if not grid:
        return None

    header_rows: list[list[str]] = []
    body: list[tuple[str, list[str]]] = []
    for cells in grid:
        label_first = not _is_value(cells[0]) and not YEAR_CELL.match(cells[0])
        label = cells[0] if label_first else ""
        values = cells[1:] if label_first else cells
        is_header = not body and (not values or all(not _is_value(v) or YEAR_CELL.match(v) for v in values))
        if is_header:
            header_rows.append(cells)
        else:
            body.append((label, values))
    if not body:
        return None

    width = max(len(values) for _, values in body)
    columns: list[str] = []
    # Prefer the deepest header row whose cell count matches the value columns (usually the year row).
    # Units cells ("(In millions, except percentages)") sit in header rows but are not columns.
    header_rows = [[c for c in cells if not UNITS_CELL.search(c)] for cells in header_rows]
    for cells in reversed(header_rows):
        years = [c for c in cells if YEAR_CELL.match(c)]
        match = next((c for c in (years, cells) if len(c) == width), None)
        if match:
            columns = match
            break
    if not columns:
        # Typical two-level header: a year row plus a "% Change" cell on the row above it.
        year_row = next((cells for cells in reversed(header_rows) if any(YEAR_CELL.match(c) for c in cells)), [])
        extras = [c for cells in header_rows if cells is not year_row for c in cells
                  if re.search(r"change|%|vs\.?", c, re.I) and not YEAR_CELL.match(c)]
        candidate = [c for c in year_row if YEAR_CELL.match(c)] + extras
        if len(candidate) == width:
            columns = candidate
    if not columns and header_rows:
        last = header_rows[-1]
        columns = (last[-width:] if len(last) >= width else last)
    if len(columns) < width:
        columns = columns + [f"col{i + 1}" for i in range(len(columns), width)]
    header_text = " ".join(" ".join(cells) for cells in header_rows)
    return NormalizedTable(columns=columns, rows=body, header_text=header_text)
