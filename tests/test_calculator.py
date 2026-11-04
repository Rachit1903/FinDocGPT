import pytest

from src.agent.calculator import Calculator, appears_in, parse_number, safe_eval
from tests.fakes import FakeLLM, make_hit

TABLE = "| | 2024 | 2025 |\n| Google Cloud revenue | 43,229 | 58,705 |\n| Google Cloud operating income | 6,112 | 13,910 |"


def test_parse_number_handles_filing_formats():
    assert parse_number("$(2,656)") == -2656
    assert parse_number("46.9%") == 46.9
    assert parse_number("128,725") == 128725


def test_appears_in_requires_whole_number_match():
    assert appears_in("13,910", TABLE)
    assert appears_in("13910", TABLE)
    assert not appears_in("3,910", TABLE)  # a suffix of 13,910 is not a match
    assert not appears_in("13,91", TABLE)


def test_safe_eval_allows_only_arithmetic():
    assert safe_eval("(a / b - c / d) * 100", {"a": 1, "b": 4, "c": 1, "d": 5}) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        safe_eval("__import__('os')", {})


def test_calculator_rejects_ungrounded_operand_then_self_corrects():
    hit = make_hit("GOOGL-T1", TABLE, kind="table")
    bad = {"operands": [{"name": "a", "label": "x", "value_text": "14,000", "chunk_id": "GOOGL-T1"}], "expression": "a", "unit": ""}
    good = {
        "operands": [
            {"name": "a", "label": "Cloud OI 2025", "value_text": "13,910", "chunk_id": "GOOGL-T1"},
            {"name": "b", "label": "Cloud revenue 2025", "value_text": "58,705", "chunk_id": "GOOGL-T1"},
            {"name": "c", "label": "Cloud OI 2024", "value_text": "6,112", "chunk_id": "GOOGL-T1"},
            {"name": "d", "label": "Cloud revenue 2024", "value_text": "43,229", "chunk_id": "GOOGL-T1"},
        ],
        "expression": "(a / b - c / d) * 100", "unit": "pp", "label": "Google Cloud operating margin change 2024-2025",
    }
    llm = FakeLLM(json_responses=[bad, good])
    calc, log = Calculator(llm).run("Calculate Google Cloud's operating-margin change", [hit])
    assert calc is not None and calc.result == pytest.approx(9.556, abs=1e-3)
    assert "does not appear" in log[0] and log[1].startswith("attempt 2: ok")
    assert "rejected" in llm.json_prompts[1]  # the rejection reason was fed back to the model
