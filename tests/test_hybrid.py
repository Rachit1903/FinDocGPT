from src.retrieval.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], rrf_k=10)
    assert [chunk_id for chunk_id, _ in result] == ["b", "a", "c"]
