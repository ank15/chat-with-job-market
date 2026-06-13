"""TF-IDF retriever — the lexical baseline.

Fully implemented (sklearn is light). This is the baseline the dense and hybrid
retrievers are benchmarked against in `eval/retrieval_eval.py`.
"""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .. import config
from .base import Retriever

# English stop words plus job-board boilerplate (see config.DOMAIN_STOPWORDS).
DEFAULT_STOP_WORDS = list(ENGLISH_STOP_WORDS | config.DOMAIN_STOPWORDS)


class TfidfRetriever(Retriever):
    def __init__(self, **vectorizer_kwargs):
        vectorizer_kwargs.setdefault("stop_words", DEFAULT_STOP_WORDS)
        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        self.ids = []
        self.matrix = None

    def index(self, documents):
        self.ids = [d["id"] for d in documents]
        self.matrix = self.vectorizer.fit_transform(d["text"] for d in documents)

    def score_all(self, query):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        return dict(zip(self.ids, sims.tolist()))
