import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.bm25 import BM25Okapi, tokenize
from lib.metrics import dedupe, score_ranking


def test_dedupe_keeps_first_occurrence_order():
    assert dedupe(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]


def test_perfect_single_target():
    scores = score_ranking(["A", "B", "C"], {"A"})
    assert scores == {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "mrr@5": 1.0, "recall@5": 1.0, "ndcg@5": 1.0}


def test_target_at_rank_three():
    scores = score_ranking(["X", "Y", "A", "Z"], {"A"})
    assert scores["hit@1"] == 0.0
    assert scores["hit@3"] == 1.0
    assert scores["mrr@5"] == pytest.approx(1 / 3)
    assert scores["ndcg@5"] == pytest.approx(1 / math.log2(4))


def test_target_outside_top_five_is_a_miss():
    scores = score_ranking(["P", "Q", "R", "S", "T", "A"], {"A"})
    assert scores["hit@5"] == 0.0
    assert scores["mrr@5"] == 0.0
    assert scores["ndcg@5"] == 0.0


def test_duplicates_do_not_consume_slots():
    scores = score_ranking(["X", "X", "X", "X", "X", "A"], {"A"})
    assert scores["hit@3"] == 1.0
    assert scores["mrr@5"] == 0.5


def test_multi_target_recall_and_ndcg():
    scores = score_ranking(["A", "X", "B"], {"A", "B", "C"})
    assert scores["recall@5"] == pytest.approx(2 / 3)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert scores["ndcg@5"] == pytest.approx((1 + 1 / math.log2(4)) / idcg)


def test_no_targets_is_rejected():
    with pytest.raises(ValueError):
        score_ranking(["A"], set())


def test_kode_plus_prodi_rejects_twin_from_other_prodi():
    ranked = [("AGI101", "Teknik Elektro"), ("AGI101", "Teknologi Sains Data")]
    assert score_ranking(ranked, {("AGI101", "Teknologi Sains Data")})["hit@1"] == 0.0


def test_tokenize_matches_app_rules():
    assert tokenize("Apa itu SIA107, a?") == ["apa", "itu", "sia107"]


def test_bm25_ranks_rare_term_first_and_drops_non_matches():
    index = BM25Okapi(["statistika dasar", "fisika dasar", "kimia organik"])
    hits = index.search("statistika")
    assert [doc for doc, _ in hits] == [0]


def test_bm25_ties_follow_corpus_order():
    index = BM25Okapi(["fisika dasar", "fisika dasar", "kimia organik"])
    assert [doc for doc, _ in index.search("fisika")] == [0, 1]
