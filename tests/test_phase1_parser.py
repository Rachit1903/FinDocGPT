from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.models import Filing
from src.ingestion.parser import extract_prose_chunks, extract_tables


def test_prose_and_tables_are_separate(tmp_path: Path) -> None:
    filing = Filing("TEST", "0000000001", "0000000001-26-000001", "10-K", "2026-01-31", "2025-12-31", "test.htm", "https://example.test")
    html = """<html><body><h2>Item 7. Management's Discussion</h2><p>Revenue increased because subscriptions grew across enterprise customers during the year. This narrative is deliberately long enough to be retained as a prose block.</p><table><caption>Consolidated Statements of Operations</caption><tr><th>Year</th><th>Revenue</th></tr><tr><td>2025</td><td>100</td></tr></table></body></html>"""
    soup = BeautifulSoup(html, "lxml")
    prose = extract_prose_chunks(BeautifulSoup(str(soup), "lxml"), filing)
    tables = extract_tables(soup, filing, tmp_path)
    assert len(prose) == 1
    assert "100" not in prose[0].text
    assert len(tables) == 1
    assert (tmp_path / f"{tables[0].table_id}.parquet").exists()

