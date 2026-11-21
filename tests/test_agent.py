from src.agent.answer import AnswerWriter, Verifier
from src.agent.calculator import Calculator
from src.agent.graph import FinDocAgent
from src.agent.planner import Planner, rule_based_plan
from src.llm import LLMError
from tests.fakes import FakeLLM, FakeReranker, FakeRetriever, make_hit

CLOUD_TABLE = "| | 2024 | 2025 |\n| Google Cloud revenue | 43,229 | 58,705 |\n| Google Cloud operating income | 6,112 | 13,910 |"


def test_rule_based_plan_does_not_send_metric_questions_to_the_calculator():
    # v1's router sent "drivers of Services net sales" to the numeric agent because it mentions a metric.
    assert rule_based_plan("What did Apple identify as the main drivers of Services net sales?").route == "factual"
    assert rule_based_plan("Compare Meta's Family of Apps and Reality Labs").route == "comparison"
    assert rule_based_plan("Calculate Uber's year-over-year revenue growth").route == "computation"


def test_planner_keeps_named_companies_and_falls_back_on_llm_failure():
    plan = Planner(FakeLLM([{"route": "comparison", "tickers": [], "sub_queries": [{"query": "AWS growth", "ticker": "AMZN"}]}])).plan(
        "Compare Amazon and Microsoft cloud growth")
    assert plan.tickers == ["AMZN", "MSFT"]  # MSFT added by deterministic name matching
    fallback = Planner(FakeLLM([LLMError("down")])).plan("Calculate Apple's gross margin change")
    assert fallback.planner == "rules" and fallback.route == "computation"


def _agent(planner_json, generator_text, verify_json, calc_json=None, scores=None, rewrite_json=None):
    generator_text = generator_text if isinstance(generator_text, list) else [generator_text]
    verify_json = verify_json if isinstance(verify_json, list) else [verify_json]
    retriever = FakeRetriever({"GOOGL": [make_hit("GOOGL-T1", CLOUD_TABLE, kind="table"),
                                         make_hit("GOOGL-P1", "Google Cloud revenue grew 36% driven by AI infrastructure.")]})
    return FinDocAgent(
        retriever=retriever, reranker=FakeReranker(scores),
        planner=Planner(FakeLLM([planner_json])), calculator=Calculator(FakeLLM([calc_json] if calc_json else [])),
        writer=AnswerWriter(FakeLLM(text_responses=generator_text)), verifier=Verifier(FakeLLM(verify_json)),
        rewriter=FakeLLM([rewrite_json] if rewrite_json else []),
    ), retriever


def test_computation_runs_calculator_and_verifier_drops_unsupported_sentence():
    calc = {"operands": [{"name": "a", "label": "OI 2025", "value_text": "13,910", "chunk_id": "GOOGL-T1"},
                         {"name": "b", "label": "Rev 2025", "value_text": "58,705", "chunk_id": "GOOGL-T1"},
                         {"name": "c", "label": "OI 2024", "value_text": "6,112", "chunk_id": "GOOGL-T1"},
                         {"name": "d", "label": "Rev 2024", "value_text": "43,229", "chunk_id": "GOOGL-T1"}],
            "expression": "(a / b - c / d) * 100", "unit": "pp", "label": "Google Cloud operating margin change"}
    agent, _ = _agent(
        {"route": "computation", "tickers": ["GOOGL"], "needs_calculation": True,
         "sub_queries": [{"query": "Google Cloud operating income revenue", "ticker": "GOOGL", "prefer_tables": True}]},
        # First draft has an unsupported claim; the revision keeps it, so it is removed after re-verification.
        ["Google Cloud operating margin rose 9.56 percentage points [S1]. Cloud will double next year [S3].",
         "Google Cloud operating margin rose 9.56 percentage points [S1]. Cloud will double next year [S3]."],
        [{"verdicts": [{"id": 1, "supported": True}, {"id": 2, "supported": False, "reason": "not in sources"}]},
         {"verdicts": [{"id": 1, "supported": True}, {"id": 2, "supported": False, "reason": "not in sources"}]}],
        calc_json=calc,
    )
    result = agent.run("Calculate Google Cloud's operating-margin change between the two latest annual periods.")
    assert result["route"] == "computation"
    assert result["calculation"]["formatted"] == "+9.56 percentage points"
    assert result["answer"] == "Google Cloud operating margin rose 9.56 percentage points [S1]."
    assert result["removed_claims"] == ["Cloud will double next year [S3]."]
    assert result["sources"][0]["chunk_id"] == "CALCULATOR"
    assert [step["step"] for step in result["trace"]] == ["plan", "retrieve", "calculate", "generate", "verify", "revise", "verify"]


def test_weak_retrieval_triggers_one_query_rewrite():
    agent, retriever = _agent(
        {"route": "factual", "tickers": ["GOOGL"], "sub_queries": [{"query": "cloud stuff", "ticker": "GOOGL"}]},
        "Google Cloud revenue grew 36% [S1].", {"verdicts": [{"id": 1, "supported": True}]},
        scores={"cloud stuff": 0.05}, rewrite_json={"query": "Google Cloud revenues increased primarily due to"},
    )
    result = agent.run("What drove Google Cloud?")
    steps = [step["step"] for step in result["trace"]]
    assert steps == ["plan", "retrieve", "rewrite", "retrieve", "generate", "verify"]
    assert [q for q, _ in retriever.queries] == ["cloud stuff", "Google Cloud revenues increased primarily due to"]


def test_question_about_company_outside_corpus_abstains_without_retrieval():
    agent, retriever = _agent({"route": "factual", "tickers": [], "in_corpus": False, "sub_queries": []}, "", {})
    result = agent.run("What was Tesla's automotive gross margin?")
    assert result["abstained"] and retriever.queries == []


def test_verifier_rejects_direction_words_that_contradict_the_figures():
    from src.agent.answer import direction_conflict

    assert direction_conflict("Provisions were $7.2 billion in fiscal 2026 (down from $3.7 billion in fiscal 2025)[S1].")
    assert direction_conflict("Gross margin fell from 69.76% in fiscal 2024 to 68.82% in fiscal 2025[S1].") is None
    assert direction_conflict("Operating income increased from $(2,656) million to $3,792 million[S1].") is None

    # The LLM verifier says "supported", but the deterministic check overrides it.
    llm_says_ok = {"verdicts": [{"id": 1, "supported": True, "reason": "numbers appear in S1"}]}
    verdicts = Verifier(FakeLLM([llm_says_ok])).verify(
        "Provisions were $7.2 billion in fiscal 2026 (down from $3.7 billion in fiscal 2025)[S1].", [("S1", "...")])
    assert not verdicts[0].supported and "contradicts" in verdicts[0].reason
