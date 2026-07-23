"""Run Groq citation verification against one supported and one adversarial claim."""

from .citations import SourceChunk, verify_citations


def main() -> None:
    sources = [SourceChunk("aapl-gross-margin", "Apple's total gross margin percentage was 46.9% in fiscal 2025.")]
    answer = "Apple's total gross margin percentage was 46.9% in fiscal 2025. Apple's revenue was $500 billion in fiscal 2025."
    result = verify_citations(answer, sources)
    print(result.verified_answer)
    print("Unsupported claims:", result.unsupported_claims)


if __name__ == "__main__":
    main()
