"""Metrik retrieval. Peringkat dideduplikasi dulu agar mata kuliah kembar antar prodi tidak menghabiskan slot."""

import math
from collections.abc import Hashable, Iterable


def dedupe(keys: Iterable[Hashable]) -> list[Hashable]:
    return list(dict.fromkeys(keys))


def score_ranking(ranked_keys: Iterable[Hashable], targets: set[Hashable]) -> dict[str, float]:
    if not targets:
        raise ValueError("Kueri tanpa target tidak bisa dinilai")

    top5 = dedupe(ranked_keys)[:5]
    relevance = [key in targets for key in top5]
    first_rank = next((rank for rank, rel in enumerate(relevance, start=1) if rel), None)

    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1) if rel)
    ideal_hits = min(len(targets), 5)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return {
        "hit@1": float(any(relevance[:1])),
        "hit@3": float(any(relevance[:3])),
        "hit@5": float(any(relevance)),
        "mrr@5": 1.0 / first_rank if first_rank else 0.0,
        "recall@5": sum(relevance) / ideal_hits,
        "ndcg@5": dcg / idcg,
    }
