"""Explicit, inspectable routing for the three FinDocGPT query paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class QueryRoute(StrEnum):
    FACTUAL_LOOKUP = "factual_lookup"
    COMPARISON = "comparison"
    COMPUTATION = "computation"


@dataclass(frozen=True)
class RouteDecision:
    route: QueryRoute
    reason: str


class QueryRouter:
    """Rule-based by design: auditable and deterministic for filing-question routing."""

    COMPUTATION_PATTERNS = (
        r"\b(calculate|compute|math|ratio|percentage change|percent change|growth rate)\b",
        r"\b(change|increased|decreased)\b.{0,80}\bfrom\b.{0,80}\bto\b",
        r"\b(yoy|year.over.year|year over year|margin percentage|percentage points?)\b",
    )
    COMPARISON_PATTERNS = (
        r"\b(compare|comparison|versus|vs\.?|relative to|against)\b",
        r"\b(which|who)\b.{0,60}\b(higher|lower|more|less|better|worse)\b",
    )

    def classify(self, query: str) -> RouteDecision:
        normalized = " ".join(query.lower().split())
        if any(re.search(pattern, normalized) for pattern in self.COMPUTATION_PATTERNS):
            return RouteDecision(QueryRoute.COMPUTATION, "Calculation or period-over-period numeric language detected.")
        if any(re.search(pattern, normalized) for pattern in self.COMPARISON_PATTERNS):
            return RouteDecision(QueryRoute.COMPARISON, "Cross-entity or relative-comparison language detected.")
        return RouteDecision(QueryRoute.FACTUAL_LOOKUP, "No comparison or explicit numeric-computation signal detected.")
