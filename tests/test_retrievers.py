"""Tests for the TF-IDF retriever and the hybrid score-fusion logic.

The dense retriever needs a model, so it's not exercised here; the hybrid fusion is
tested with a stub retriever that returns canned scores.
"""

from src.retrievers.base import Retriever
from src.retrievers.hybrid import HybridRetriever, _minmax_normalize
from src.retrievers.tfidf import TfidfRetriever

DOCS = [
    {"id": "a", "text": "python machine learning aws"},
    {"id": "b", "text": "java spring backend services"},
    {"id": "c", "text": "python data analysis sql"},
]


class StubRetriever(Retriever):
    """Returns pre-set scores; lets us test hybrid fusion without a model."""

    def __init__(self, scores):
        self._scores = scores

    def index(self, documents):
        pass

    def score_all(self, query):
        return dict(self._scores)


class TestTfidfRetriever:
    def test_retrieves_lexically_relevant_doc_first(self):
        r = TfidfRetriever()
        r.index(DOCS)
        top = r.retrieve("python sql", top_k=1)
        assert top[0][0] == "c"  # the python+sql doc

    def test_scores_cover_all_docs(self):
        r = TfidfRetriever()
        r.index(DOCS)
        assert set(r.score_all("python")) == {"a", "b", "c"}


class TestMinMaxNormalize:
    def test_scales_to_unit_range(self):
        out = _minmax_normalize({"a": 0.0, "b": 5.0, "c": 10.0})
        assert out == {"a": 0.0, "b": 0.5, "c": 1.0}

    def test_flat_scores_become_zero(self):
        assert _minmax_normalize({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}

    def test_empty(self):
        assert _minmax_normalize({}) == {}


class TestHybridFusion:
    def test_weighted_combination(self):
        # Lexical favors 'a', dense favors 'b'; alpha=0.5 should balance them.
        lexical = StubRetriever({"a": 1.0, "b": 0.0})
        dense = StubRetriever({"a": 0.0, "b": 1.0})
        hybrid = HybridRetriever(lexical, dense, alpha=0.5)
        scores = hybrid.score_all("q")
        assert scores["a"] == 0.5
        assert scores["b"] == 0.5

    def test_alpha_weights_dense_more(self):
        lexical = StubRetriever({"a": 1.0, "b": 0.0})
        dense = StubRetriever({"a": 0.0, "b": 1.0})
        hybrid = HybridRetriever(lexical, dense, alpha=0.8)
        scores = hybrid.score_all("q")
        assert scores["b"] > scores["a"]  # dense-favored doc wins
