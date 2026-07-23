from src.verification.citations import LexicalEntailmentJudge, SourceChunk, verify_citations


def test_verifier_removes_adversarial_unsupported_claim() -> None:
    sources = [SourceChunk("aapl-margin", "Apple's total gross margin percentage was 46.9% in fiscal 2025.")]
    answer = "Apple's total gross margin percentage was 46.9% in fiscal 2025. Apple's revenue was $500 billion in fiscal 2025."
    result = verify_citations(answer, sources, judge=LexicalEntailmentJudge())
    assert "46.9%" in result.verified_answer
    assert "500 billion" not in result.verified_answer
    assert result.unsupported_claims == ["Apple's revenue was $500 billion in fiscal 2025."]
