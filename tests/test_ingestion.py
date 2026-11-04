from bs4 import BeautifulSoup

from src.ingestion.models import Filing
from src.ingestion.parser import parse_filing
from src.ingestion.tables import normalize_table

SEGMENT_TABLE = """
<table>
  <tr><td></td><td colspan="3">Year Ended December 31,</td><td colspan="2">2024 to 2025 % Change</td></tr>
  <tr><td>(In millions, except percentages)</td><td colspan="2">2024</td><td colspan="2">2025</td><td></td></tr>
  <tr><td>Mobility</td><td>$</td><td>25,087</td><td>$</td><td>29,670</td><td>18</td><td>%</td></tr>
  <tr><td>Freight</td><td></td><td>5,141</td><td></td><td>5,099</td><td>( 1</td><td>)%</td></tr>
  <tr><td>Operating income (loss)</td><td>$</td><td>( 2,656 )</td><td></td><td>3,792</td><td>n/m</td></tr>
</table>
"""


def test_normalize_table_resolves_headers_and_cells():
    table = normalize_table(BeautifulSoup(SEGMENT_TABLE, "lxml").find("table"))
    assert table.columns == ["2024", "2025", "2024 to 2025 % Change"]
    rows = dict(table.rows)
    assert rows["Mobility"] == ["$25,087", "$29,670", "18%"]
    assert rows["Freight"][2] == "(1)%"
    assert rows["Operating income (loss)"][0] == "$(2,656)"


FILING_HTML = """
<html><body>
<div><span style="font-weight:700">Item 7. Management's Discussion and Analysis</span></div>
<div><span style="font-weight:700">Products and Services Performance</span></div>
<div><span>Services net sales increased during 2025 compared to 2024 primarily due to higher net sales from advertising, the App Store and cloud services.</span></div>
<div><span>Example Corp. | 2025 Form 10-K | 21</span></div>
<div><span>Example Corp. | 2025 Form 10-K | 22</span></div>
<div><span>Example Corp. | 2025 Form 10-K | 23</span></div>
<div><span>Gross margin was as follows (dollars in millions):</span></div>
<table>
  <tr><td></td><td>2025</td><td>2024</td></tr>
  <tr><td>Products</td><td>$</td><td>112,887</td><td>$</td><td>109,633</td></tr>
  <tr><td>Services</td><td></td><td>82,314</td><td></td><td>71,050</td></tr>
</table>
</body></html>
"""


def test_parse_filing_builds_contextual_prose_and_table_chunks():
    filing = Filing("AAPL", "0000320193", "x", "10-K", "2025-10-31", "2025-09-27", "a.htm", "https://sec.gov/a")
    chunks = parse_filing(FILING_HTML.encode(), filing)
    prose = [c for c in chunks if c.kind == "prose"]
    tables = [c for c in chunks if c.kind == "table"]
    assert len(prose) == 1 and len(tables) == 1
    assert prose[0].item.startswith("Item 7.")
    assert prose[0].subsection == "Products and Services Performance"
    # The header names the company even though the sentence does not.
    assert prose[0].context.startswith("Apple (AAPL) 10-K for fiscal year 2025")
    assert "Form 10-K | 2" not in prose[0].text  # repeated page footers are dropped
    assert "| Services | 82,314 | 71,050 |" in tables[0].text
    assert tables[0].table["columns"] == ["2025", "2024"]
