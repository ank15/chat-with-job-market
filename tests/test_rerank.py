"""Tests for the diversity re-ranking logic."""

from src.rerank import diversify, jaccard, token_set


class TestJaccard:
    def test_identical(self):
        assert jaccard(token_set("python sql aws"), token_set("python sql aws")) == 1.0

    def test_disjoint(self):
        assert jaccard(token_set("python"), token_set("java")) == 0.0

    def test_partial_overlap(self):
        # {a,b} vs {b,c} → intersection 1, union 3.
        assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3

    def test_both_empty(self):
        assert jaccard(set(), set()) == 1.0


class TestDiversify:
    def test_drops_near_duplicates(self):
        candidates = [
            {"id": "1", "text": "machine learning engineer at capital one, build models"},
            {"id": "2", "text": "machine learning engineer at capital one, build models"},  # dup
            {"id": "3", "text": "frontend react developer building user interfaces"},
        ]
        kept = diversify(candidates, top_k=5, threshold=0.6)
        ids = [c["id"] for c in kept]
        assert ids == ["1", "3"]  # the duplicate (#2) is dropped

    def test_respects_top_k(self):
        # Genuinely distinct texts so none are filtered as duplicates.
        texts = ["python sql", "java spring", "react css", "rust systems", "go cloud"]
        candidates = [{"id": str(i), "text": t} for i, t in enumerate(texts)]
        kept = diversify(candidates, top_k=3, threshold=0.6)
        assert len(kept) == 3

    def test_keeps_distinct_results(self):
        candidates = [
            {"id": "a", "text": "data analyst sql dashboards"},
            {"id": "b", "text": "data engineer pipelines etl"},
            {"id": "c", "text": "machine learning models training"},
        ]
        kept = diversify(candidates, top_k=5, threshold=0.6)
        assert len(kept) == 3  # all distinct, none dropped

    def test_order_preserved(self):
        candidates = [
            {"id": "x", "text": "alpha beta gamma"},
            {"id": "y", "text": "delta epsilon zeta"},
        ]
        kept = diversify(candidates, top_k=5, threshold=0.6)
        assert [c["id"] for c in kept] == ["x", "y"]


class TestGroupCap:
    def test_caps_results_per_company(self):
        # Three distinct roles, all at the same company; cap should keep only 2.
        candidates = [
            {"id": "1", "text": "data analyst sql dashboards", "co": "Acme"},
            {"id": "2", "text": "data engineer pipelines etl", "co": "Acme"},
            {"id": "3", "text": "machine learning models training", "co": "Acme"},
        ]
        kept = diversify(
            candidates, top_k=5, threshold=0.6,
            group_key=lambda c: c["co"], max_per_group=2,
        )
        assert [c["id"] for c in kept] == ["1", "2"]  # third Acme role dropped

    def test_other_companies_still_included(self):
        candidates = [
            {"id": "1", "text": "data analyst sql", "co": "Acme"},
            {"id": "2", "text": "data engineer etl", "co": "Acme"},
            {"id": "3", "text": "ml engineer models", "co": "Acme"},
            {"id": "4", "text": "frontend react developer", "co": "Globex"},
        ]
        kept = diversify(
            candidates, top_k=5, threshold=0.6,
            group_key=lambda c: c["co"], max_per_group=2,
        )
        assert [c["id"] for c in kept] == ["1", "2", "4"]  # 2 Acme + the Globex role
