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


def test_judge_tolerates_misspelled_keys_and_rejects_malformed_verdicts():
    from eval.metrics import judge
    from tests.fakes import FakeLLM

    gold = {"reference_answer": "Margin rose 2.8 pp.", "key_facts": ["26.7% in 2024", "29.5% in 2025", "+2.8 pp"]}
    misspelled = {"key_fats": [{"id": i, "covered": True} for i in (1, 2, 3)], "contradicts_gold": False,
                  "sentences": [{"id": 1, "supported_by_context": "true"}]}
    verdict = judge(FakeLLM(json_responses=[misspelled]), "q", gold, "Margin rose from 26.7% to 29.5%.", ["ctx"])
    assert verdict["fact_coverage"] == 1.0 and verdict["faithfulness"] == 1.0

    verdict = judge(FakeLLM(json_responses=[{"contradicts_gold": False}]), "q", gold, "Answer.", ["ctx"])
    assert "error" in verdict and "fact_coverage" not in verdict
