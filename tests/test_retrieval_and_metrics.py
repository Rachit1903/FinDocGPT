from eval.metrics import evidence_recall, numeric_correct
from src.companies import extract_tickers
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.store import tokenize


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])
    assert [doc for doc, _ in fused] == ["b", "a", "c"]


def test_tokenize_drops_question_boilerplate():
    assert tokenize("What did Apple disclose about AWS net-sales?") == ["apple", "about", "aws", "net-sales"]


def test_extract_tickers_by_symbol_name_and_product():
    assert extract_tickers("Compare AWS with Microsoft Azure") == ["AMZN", "MSFT"]
    assert extract_tickers("Tesla margins") == []


def test_numeric_correct_uses_tolerance_and_ignores_sign_wording():
    assert numeric_correct("Margin fell 0.94 percentage points.", {"value": -0.94, "unit": "pp"})
    assert numeric_correct("AWS grew 19.7% year over year.", {"value": 19.68, "unit": "%"})
    assert not numeric_correct("AWS grew 17% year over year.", {"value": 19.68, "unit": "%"})


def test_evidence_recall_is_whitespace_and_currency_insensitive():
    contexts = ["| AWS | $107,556 |  128,725 |"]
    assert evidence_recall(["| AWS | 107,556 | 128,725 |", "missing sentence"], contexts) == 0.5
