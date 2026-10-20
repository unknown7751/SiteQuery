"""Reciprocal Rank Fusion: merge ranked lists without comparing their incompatible scores."""

from collections.abc import Sequence

from backend.app.vectorstore.base import SearchHit, chunk_key


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], k: int = 60
) -> list[SearchHit]:
    """Fuse best-first `rankings` into one list, best first.

    A chunk scores sum(1 / (k + rank)) over every list that contains it (rank starts at 1),
    so agreement between retrievers beats a single high placement. The returned
    `relevance` is that score divided by its maximum possible value (rank 1 in every list),
    giving a 0..1 figure comparable across queries. It measures rank agreement, not cosine
    similarity.
    """
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = chunk_key(hit.document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            hits.setdefault(key, hit)  # first list wins; dense is passed first

    ceiling = len(rankings) / (k + 1)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [SearchHit(document=hits[key].document, relevance=scores[key] / ceiling) for key in ordered]
