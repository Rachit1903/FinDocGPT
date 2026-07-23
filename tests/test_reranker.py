from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.hybrid import RetrievedChunk


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool) -> list[float]:
        assert show_progress_bar is False
        return [0.2, 0.9, 0.5]


def candidate(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, f"Text for {chunk_id}", {"ticker": "TEST", "section": "Item 7"}, 0.01, 1, 1)


def test_reranker_reorders_fused_candidates() -> None:
    output = CrossEncoderReranker(model=FakeCrossEncoder()).rerank(
        "query", [candidate("first"), candidate("second"), candidate("third")], k=2
    )
    assert [(item.chunk.chunk_id, item.fused_rank, item.reranked_rank) for item in output] == [
        ("second", 2, 1), ("third", 3, 2)
    ]
