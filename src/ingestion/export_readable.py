"""Export the corpus as human-readable Markdown: exactly the chunks the agent retrieves from, in filing order.

    python -m src.ingestion.export_readable      # writes data/readable/INDEX.md and data/readable/<TICKER>.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from src.config import settings

OUT_DIR = settings.corpus_path.parent.parent / "readable"


def _table_markdown(text: str) -> str:
    """Chunk table text is a caption line plus pipe rows; add the separator row Markdown needs to render it."""
    lines = text.splitlines()
    rows = [line for line in lines if line.startswith("|")]
    if not rows:
        return text
    width = rows[0].count("|") - 1
    caption = [line for line in lines if not line.startswith("|")]
    table = [rows[0], "|" + "---|" * width, *rows[1:]]
    return "\n".join([f"*{' '.join(caption)}*", "", *table]) if caption else "\n".join(table)


def main() -> None:
    chunks = [json.loads(line) for line in settings.corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_ticker[chunk["ticker"]].append(chunk)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = ["# FinDocGPT corpus", "",
             f"{len(chunks)} chunks from {len(by_ticker)} 10-K filings. Each file below is one filing, in order, exactly as",
             "the agent sees it (chunk id, section, text; tables in their normalized form).", "",
             "| Company | Fiscal year | Period end | Prose chunks | Tables | File |", "|---|---|---|---|---|---|"]
    for ticker in sorted(by_ticker):
        items = by_ticker[ticker]
        first = items[0]
        kinds = Counter(c["kind"] for c in items)
        index.append(f"| {first['company']} ({ticker}) | FY{first['fiscal_year']} | {first['report_date']} | "
                     f"{kinds['prose']} | {kinds['table']} | [{ticker}.md]({ticker}.md) |")

        sections = list(dict.fromkeys(c["item"] for c in items))
        lines = [f"# {first['company']} ({ticker}) 10-K, fiscal year {first['fiscal_year']}", "",
                 f"Period ended {first['report_date']}, filed {first['filing_date']}. Source: {first['source_url']}", "",
                 "## Sections", ""]
        lines += [f"- {item} ({sum(c['item'] == item for c in items)} chunks)" for item in sections]
        current = None
        for chunk in items:
            if chunk["item"] != current:
                current = chunk["item"]
                lines += ["", "---", "", f"## {current}"]
            body = _table_markdown(chunk["text"]) if chunk["kind"] == "table" else chunk["text"]
            lines += ["", f"#### `{chunk['chunk_id']}` · {chunk['kind']} · {chunk['subsection']}", "", body]
        (OUT_DIR / f"{ticker}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT_DIR / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Wrote {len(by_ticker) + 1} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
