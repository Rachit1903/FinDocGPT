"""Dependency-free rank-fusion primitives used by hybrid retrieval."""

from __future__ import annotations

from collections.abc import Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[str]], *, rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse rank lists without mixing incompatible relevance-score scales."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda result: (-result[1], result[0]))
