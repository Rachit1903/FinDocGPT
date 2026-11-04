"""Registry of the companies in the corpus and deterministic name/ticker matching."""

from __future__ import annotations

import re

COMPANIES: dict[str, str] = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "META": "Meta Platforms",
    "NFLX": "Netflix",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "UBER": "Uber",
}

ALIASES: dict[str, str] = {
    "apple": "AAPL", "amazon": "AMZN", "aws": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META", "instagram": "META", "netflix": "NFLX", "microsoft": "MSFT",
    "azure": "MSFT", "nvidia": "NVDA", "salesforce": "CRM", "adobe": "ADBE", "uber": "UBER",
}


def extract_tickers(text: str) -> list[str]:
    """Return supported tickers mentioned by symbol or company/product name, in order of appearance."""
    hits: dict[str, int] = {}
    for ticker in COMPANIES:
        match = re.search(rf"\b{ticker}\b", text)
        if match:
            hits.setdefault(ticker, match.start())
    lower = text.lower()
    for alias, ticker in ALIASES.items():
        match = re.search(rf"\b{re.escape(alias)}\b", lower)
        if match:
            hits[ticker] = min(hits.get(ticker, match.start()), match.start())
    return sorted(hits, key=hits.get)
