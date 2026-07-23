"""Pandas-backed computation over Phase 1 structured filing tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?\)?")


@dataclass(frozen=True)
class NumericAnswer:
    answer: str
    metric: str
    values_by_year: dict[int, float]
    calculation: str
    table_id: str
    table_title: str
    ticker: str
    source_context: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FinancialTableAgent:
    """Selects a stored table and calculates from its dataframe, never prose text."""

    def __init__(self, manifest_path: Path = Path("data/processed/tables_manifest.jsonl")) -> None:
        self.tables = self._load_manifest(manifest_path)

    @staticmethod
    def _load_manifest(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Table manifest not found: {path}")
        deduped: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                deduped[row["table_id"]] = row
        return list(deduped.values())

    @staticmethod
    def _ticker(query: str) -> str:
        known = {"AAPL", "AMZN", "GOOGL", "META", "NFLX", "MSFT", "NVDA", "CRM", "ADBE", "UBER"}
        aliases = {"apple": "AAPL", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "microsoft": "MSFT", "nvidia": "NVDA", "netflix": "NFLX", "salesforce": "CRM", "adobe": "ADBE", "uber": "UBER", "meta": "META"}
        upper = query.upper()
        for ticker in known:
            if re.search(rf"\b{ticker}\b", upper):
                return ticker
        lower = query.lower()
        for name, ticker in aliases.items():
            if re.search(rf"\b{re.escape(name)}\b", lower):
                return ticker
        raise ValueError("Computation questions must name one supported company ticker or company name.")

    @staticmethod
    def _metric(query: str) -> str:
        normalized = re.sub(r"[-_/]+", " ", query.lower())
        normalized = re.sub(r"\s+", " ", normalized)
        if "total gross margin percentage" in normalized or "gross margin percentage" in normalized:
            return "total gross margin percentage"
        if "operating margin percentage" in normalized or "operating margin" in normalized:
            return "operating margin percentage"
        if "gross margin" in normalized:
            return "gross margin"
        if "net sales" in normalized:
            return "net sales"
        if "revenue" in normalized:
            return "revenue"
        if "operating income" in normalized:
            return "operating income"
        if "net income" in normalized:
            return "net income"
        raise ValueError("Could not identify a supported table metric. Try gross margin percentage, gross margin, net sales, revenue, operating income, or net income.")

    @staticmethod
    def _years_from_table(table: dict[str, Any]) -> list[int]:
        years = [int(year) for year in YEAR_PATTERN.findall(f"{table['title']} {table['section']}")]
        return list(dict.fromkeys(years))

    @staticmethod
    def _years_from_frame(frame: pd.DataFrame) -> list[int]:
        extracted: list[int] = []
        for value in list(frame.columns) + frame.astype(str).head(5).to_numpy().ravel().tolist():
            extracted.extend(int(year) for year in YEAR_PATTERN.findall(str(value)))
        return list(dict.fromkeys(extracted))

    @staticmethod
    def _dedupe_adjacent(values: list[float]) -> list[float]:
        output: list[float] = []
        for value in values:
            if not output or value != output[-1]:
                output.append(value)
        return output

    @staticmethod
    def _numbers_from_row(row: pd.Series) -> list[float]:
        values: list[float] = []
        for cell in row.tolist():
            for token in NUMBER_PATTERN.findall(str(cell)):
                clean = token.replace(",", "").replace("(", "-").replace(")", "")
                try:
                    values.append(float(clean))
                except ValueError:
                    pass
        return FinancialTableAgent._dedupe_adjacent(values)

    def _select_table(self, ticker: str, metric: str, years: list[int]) -> dict[str, Any]:
        metric_terms = set(metric.split())
        candidates = [table for table in self.tables if table["ticker"] == ticker]
        scored: list[tuple[int, dict[str, Any]]] = []
        for table in candidates:
            haystack = f"{table['title']} {table['section']}".lower()
            score = sum(term in haystack for term in metric_terms)
            score += sum(str(year) in haystack for year in years)
            if score:
                scored.append((score, table))
        if not scored:
            raise ValueError(f"No structured {metric!r} table found for {ticker}.")
        return max(scored, key=lambda item: (item[0], item[1]["filing_date"]))[1]

    def answer(self, query: str) -> NumericAnswer:
        ticker = self._ticker(query)
        metric = self._metric(query)
        query_years = [int(year) for year in YEAR_PATTERN.findall(query)]
        table = self._select_table(ticker, metric, query_years)
        frame = pd.read_parquet(table["parquet_path"])
        table_years = self._years_from_table(table)
        frame_years = self._years_from_frame(frame)
        years = query_years if len(query_years) >= 2 else sorted(set(table_years + frame_years))[-2:]
        if len(years) < 2:
            raise ValueError("Computation questions must reference or imply two fiscal years.")
        old_year, new_year = years[0], years[1]
        row_matches = frame.apply(
            lambda row: metric.lower() in " ".join(str(value).lower() for value in row.tolist()), axis=1
        )
        if not row_matches.any() and metric == "total gross margin percentage":
            row_matches = frame.apply(lambda row: "gross margin percentage" in " ".join(str(v).lower() for v in row.tolist()), axis=1)
        if not row_matches.any() and metric == "operating margin percentage":
            row_matches = frame.apply(lambda row: "operating margin" in " ".join(str(v).lower() for v in row.tolist()), axis=1)
        if not row_matches.any():
            raise ValueError(f"The selected table does not contain a {metric!r} row.")
        numbers = self._numbers_from_row(frame.loc[row_matches].iloc[-1])
        title_years = table_years
        if len(title_years) < len(numbers):
            # SEC table extraction sometimes omits explicit column headers; use query years only when unambiguous.
            title_years = [new_year, old_year][:len(numbers)]
        values_by_year = dict(zip(title_years[:len(numbers)], numbers, strict=False))
        if old_year not in values_by_year or new_year not in values_by_year:
            raise ValueError(f"Could not map both requested years to values in table {table['table_id']}.")
        old_value, new_value = values_by_year[old_year], values_by_year[new_year]
        absolute_change = new_value - old_value
        relative_change = (absolute_change / old_value * 100) if old_value else None
        normalized_query = re.sub(r"[-_/]+", " ", query.lower())
        normalized_query = re.sub(r"\s+", " ", normalized_query)
        is_percent = "percentage" in metric or "percent" in table["title"].lower() or "margin" in metric
        is_growth = "growth" in normalized_query or "change" in normalized_query or "yoy" in normalized_query
        if is_percent:
            calculation = f"{new_value:.1f}% − {old_value:.1f}% = {absolute_change:+.1f} percentage points"
            suffix = f" ({relative_change:+.1f}% relative)" if relative_change is not None else ""
            answer = f"{ticker} {metric} changed from {old_value:.1f}% in {old_year} to {new_value:.1f}% in {new_year}: {absolute_change:+.1f} percentage points{suffix}."
        else:
            calculation = f"{new_value:,.2f} − {old_value:,.2f} = {absolute_change:+,.2f}; ({absolute_change:,.2f} / {old_value:,.2f}) × 100 = {relative_change:+.1f}%"
            if is_growth and relative_change is not None:
                answer = f"{ticker} {metric} growth from {old_year} to {new_year} was {relative_change:+.1f}% based on values of {old_value:,.2f} and {new_value:,.2f}."
            else:
                answer = f"{ticker} {metric} changed from {old_value:,.2f} in {old_year} to {new_value:,.2f} in {new_year}: {absolute_change:+,.2f} ({relative_change:+.1f}%)."
        source_context = f"[{table['table_id']}] {table['title']}\n{frame.loc[row_matches].to_string(index=False)}"
        return NumericAnswer(answer, metric, values_by_year, calculation, table["table_id"], table["title"], ticker, source_context)
