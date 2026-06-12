"""Hybrid retriever — fuse lexical (TF-IDF) and dense (embedding) scores.

TF-IDF catches exact terms (acronyms, product names); dense catches meaning. Their
raw scores live on different scales, so each is min-max normalized to [0, 1] before a
weighted sum: ``alpha * dense + (1 - alpha) * lexical``.

The fusion logic is pure and unit-tested; only the underlying dense retriever needs a
model.
"""

from .. import config
from .base import Retriever


def _minmax_normalize(scores):
    """Scale a {id: score} mapping into [0, 1]. Flat input → all zeros."""
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever(Retriever):
    def __init__(self, lexical, dense, alpha=config.HYBRID_ALPHA):
        self.lexical = lexical
        self.dense = dense
        self.alpha = alpha

    def index(self, documents):
        self.lexical.index(documents)
        self.dense.index(documents)

    def score_all(self, query):
        lex = _minmax_normalize(self.lexical.score_all(query))
        den = _minmax_normalize(self.dense.score_all(query))
        ids = set(lex) | set(den)
        return {
            doc_id: self.alpha * den.get(doc_id, 0.0)
            + (1 - self.alpha) * lex.get(doc_id, 0.0)
            for doc_id in ids
        }
