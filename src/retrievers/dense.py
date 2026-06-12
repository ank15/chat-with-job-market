"""Dense retriever — sentence-transformer embeddings + cosine similarity.

Mirrors the persisted-embedding pattern from the sister project's
`semantic_matcher.py`. The heavy `sentence-transformers` import is deferred into
`_get_model` so this module imports fast and offline (and so tests can monkeypatch
the encoder).

The document embeddings can be persisted to disk (a `.npy` cache, like the sister
project) so the corpus is only encoded once — see `index(..., cache_path=...)`.
"""

from pathlib import Path

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

    def index(self, documents, cache_path=None):
        """Embed every document into the search matrix.

        If ``cache_path`` is given and a saved matrix with the same number of rows
        already exists, it's reused (skip the expensive re-encoding); otherwise we
        encode the corpus and save it for next time.
        """
        self.ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                cached = np.load(cache_path)
                if cached.shape[0] == len(texts):
                    self.embeddings = cached
                    return
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.embeddings = self._encode(texts)
            np.save(cache_path, self.embeddings)
            return

        self.embeddings = self._encode(texts)

    def score_all(self, query):
        # Normalized vectors → dot product == cosine similarity.
        query_vec = self._encode([query])[0]
        sims = self.embeddings @ query_vec
        return dict(zip(self.ids, sims.tolist()))
