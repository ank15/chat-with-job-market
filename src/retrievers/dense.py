"""Dense retriever — sentence-transformer embeddings + cosine similarity.

Mirrors the persisted-embedding pattern from the sister project's
`semantic_matcher.py`. The heavy `sentence-transformers` import is deferred into
`_get_model` so this module imports fast and offline (and so tests can monkeypatch
the encoder).

TODO: persist the document embeddings to disk (like the sister project's .npy cache)
so the corpus is only encoded once.
"""

import numpy as np

from .. import config
from .base import Retriever

_MODEL_CACHE = {}


def _get_model(model_name):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class DenseRetriever(Retriever):
    def __init__(self, model_name=config.EMBED_MODEL):
        self.model_name = model_name
        self.ids = []
        self.embeddings = None  # (n_docs, dim), L2-normalized

    def _encode(self, texts):
        model = _get_model(self.model_name)
        vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def index(self, documents):
        self.ids = [d["id"] for d in documents]
        self.embeddings = self._encode(d["text"] for d in documents)

    def score_all(self, query):
        # Normalized vectors → dot product == cosine similarity.
        query_vec = self._encode([query])[0]
        sims = self.embeddings @ query_vec
        return dict(zip(self.ids, sims.tolist()))
