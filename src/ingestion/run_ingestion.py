"""CLI entrypoint for repeatable SEC filing ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .parser import parse_filing
from .sec_client import SecClient

DEFAULT_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "NFLX", "MSFT", "NVDA", "CRM", "ADBE", "UBER"]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and table-aware parse recent SEC filings.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--filing-types", nargs="+", choices=["10-K", "10-Q"], default=["10-K", "10-Q"])
    parser.add_argument("--filings-per-type", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.filings_per_type < 1:
        parser.error("--filings-per-type must be at least 1")

    load_dotenv()
    client = SecClient()
    tickers = [ticker.upper() for ticker in args.tickers]
    raw_dir = args.data_dir / "raw_filings"
    processed_dir = args.data_dir / "processed"
    tables_dir = processed_dir / "tables"
    ciks = client.ticker_to_cik(tickers)

    all_filings = []
    all_chunks = []
    all_tables = []
    for ticker, cik in ciks.items():
        filings = client.latest_filings(ticker, cik, set(args.filing_types), args.filings_per_type)
        if not filings:
            print(f"WARNING: no requested filings found for {ticker}")
        for filing in filings:
            print(f"Downloading {filing.ticker} {filing.form} filed {filing.filing_date}")
            html_path = client.download_filing(filing, raw_dir)
            chunks, tables = parse_filing(html_path, filing, tables_dir)
            all_filings.append({**filing.__dict__, "local_html_path": str(html_path).replace("\\", "/")})
            all_chunks.extend(chunk.to_dict() for chunk in chunks)
            all_tables.extend(table.to_dict() for table in tables)
            print(f"  extracted {len(chunks)} prose chunks and {len(tables)} structured tables")

    write_jsonl(processed_dir / "filings_manifest.jsonl", all_filings)
    write_jsonl(processed_dir / "prose_chunks.jsonl", all_chunks)
    write_jsonl(processed_dir / "tables_manifest.jsonl", all_tables)
    print(f"Saved {len(all_filings)} filings, {len(all_chunks)} prose chunks, and {len(all_tables)} tables.")


if __name__ == "__main__":
    main()

