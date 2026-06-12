"""Retriever interface shared by the TF-IDF, dense, and hybrid retrievers.

Implementations provide ``index`` and ``score_all``; ``retrieve`` (top-k) is shared.
Returning a {doc_id: score} mapping from ``score_all`` is what lets the hybrid
retriever fuse two retrievers cleanly.
"""

from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def index(self, documents):
        """Build the internal index from a list of document dicts."""

    @abstractmethod
    def score_all(self, query):
        """Return a {doc_id: similarity_score} mapping over all indexed docs."""

    def retrieve(self, query, top_k=5):
        """Return the top-k ``(doc_id, score)`` pairs, highest score first."""
        scores = self.score_all(query)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
