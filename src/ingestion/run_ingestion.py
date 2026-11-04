"""Download (or reuse) the latest 10-K for each company and rebuild the chunk corpus.

Usage:
    python -m src.ingestion.run_ingestion            # reuse cached raw HTML when present
    python -m src.ingestion.run_ingestion --refresh  # re-download from EDGAR
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from dotenv import load_dotenv

from src.companies import COMPANIES
from src.config import settings

from .models import Filing
from .parser import parse_filing
from .sec_client import SecClient


def _cached_filings() -> dict[str, Filing]:
    path = settings.raw_dir / "filings_manifest.jsonl"
    if not path.exists():
        return {}
    filings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record.pop("local_html_path", None)
        filings[record["ticker"]] = Filing(**record)
    return filings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(COMPANIES))
    parser.add_argument("--refresh", action="store_true", help="Ignore cached filings and query EDGAR again.")
    args = parser.parse_args()
    load_dotenv()

    cached = {} if args.refresh else _cached_filings()
    missing = [t for t in args.tickers if t not in cached or not (settings.raw_dir / t / f"{cached[t].accession_number}.html").exists()]
    if missing:
        client = SecClient()
        for ticker, cik in client.ticker_to_cik(missing).items():
            filing = client.latest_filings(ticker, cik, {"10-K"}, 1)[0]
            client.download_filing(filing, settings.raw_dir)
            cached[ticker] = filing

    chunks = []
    for ticker in args.tickers:
        filing = cached[ticker]
        html = (settings.raw_dir / ticker / f"{filing.accession_number}.html").read_bytes()
        filing_chunks = parse_filing(html, filing)
        kinds = Counter(chunk.kind for chunk in filing_chunks)
        print(f"{ticker} FY{filing.fiscal_year}: {kinds['prose']} prose chunks, {kinds['table']} table chunks")
        chunks.extend(filing_chunks)

    settings.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.corpus_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    with (settings.raw_dir / "filings_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for filing in cached.values():
            handle.write(json.dumps(filing.__dict__) + "\n")
    print(f"Wrote {len(chunks)} chunks to {settings.corpus_path}")


if __name__ == "__main__":
    main()
