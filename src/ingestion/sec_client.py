"""A small, fair-access SEC EDGAR client; no API key is required."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import requests

from .models import Filing

SEC_DATA_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"


class SecClient:
    def __init__(self, user_agent: str | None = None, pause_seconds: float = 0.15) -> None:
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if not self.user_agent or "@" not in self.user_agent:
            raise ValueError(
                "SEC_USER_AGENT must identify the application and include a contact email. "
                "Set it in .env before downloading filings."
            )
        self.pause_seconds = pause_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})

    def _get_json(self, url: str) -> dict:
        response = self.session.get(url, timeout=45)
        response.raise_for_status()
        time.sleep(self.pause_seconds)
        return response.json()

    def ticker_to_cik(self, tickers: Iterable[str]) -> dict[str, str]:
        # SEC publishes the ticker directory from www.sec.gov; filing-submission
        # JSON remains on data.sec.gov.
        payload = self._get_json(f"{SEC_WWW_URL}/files/company_tickers.json")
        lookup = {item["ticker"].upper(): str(item["cik_str"]).zfill(10) for item in payload.values()}
        missing = [ticker for ticker in tickers if ticker.upper() not in lookup]
        if missing:
            raise ValueError(f"Unknown SEC tickers: {', '.join(missing)}")
        return {ticker.upper(): lookup[ticker.upper()] for ticker in tickers}

    def latest_filings(self, ticker: str, cik: str, forms: set[str], per_form: int) -> list[Filing]:
        submissions = self._get_json(f"{SEC_DATA_URL}/submissions/CIK{cik}.json")
        recent = submissions["filings"]["recent"]
        found: dict[str, int] = {form: 0 for form in forms}
        filings: list[Filing] = []
        for index, form in enumerate(recent["form"]):
            if form not in forms or found[form] >= per_form:
                continue
            primary_document = recent["primaryDocument"][index]
            if not primary_document:
                continue
            accession = recent["accessionNumber"][index]
            archive_accession = accession.replace("-", "")
            source_url = f"{SEC_ARCHIVES_URL}/{int(cik)}/{archive_accession}/{primary_document}"
            filings.append(Filing(
                ticker=ticker,
                cik=cik,
                accession_number=accession,
                form=form,
                filing_date=recent["filingDate"][index],
                report_date=recent["reportDate"][index] or None,
                primary_document=primary_document,
                source_url=source_url,
            ))
            found[form] += 1
            if all(found[requested] >= per_form for requested in forms):
                break
        return filings

    def download_filing(self, filing: Filing, raw_dir: Path) -> Path:
        destination = raw_dir / filing.ticker / f"{filing.accession_number}.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return destination
        response = self.session.get(filing.source_url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        time.sleep(self.pause_seconds)
        return destination
