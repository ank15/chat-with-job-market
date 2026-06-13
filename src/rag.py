"""RAG orchestration: retrieve relevant documents, then generate a grounded answer.

This is the glue layer the app and the eval harness both call. The retrieval half is
implemented; the answer half depends on `generate.generate_answer` (stubbed).
"""

from . import config
from .generate import generate_answer
from .rerank import diversify


class RAGPipeline:
    def __init__(self, retriever, documents, top_k=config.TOP_K):
        self.retriever = retriever
        self.top_k = top_k
        self._doc_by_id = {d["id"]: d for d in documents}

    def retrieve(self, question):
        """Return the top-k *distinct* documents (with scores attached).

        Over-fetches chunks, keeps the best chunk per posting, then applies diversity
        re-ranking so the results aren't near-duplicates of each other.
        """
        candidate_k = max(self.top_k * config.CANDIDATE_MULTIPLIER, 40)
        hits = self.retriever.retrieve(question, candidate_k)

        # Collapse to one (highest-scoring) chunk per posting, preserving rank order.
        seen_postings = set()
        candidates = []
        for doc_id, score in hits:
            posting_id = doc_id.rsplit("::", 1)[0]
            if posting_id in seen_postings:
                continue
            seen_postings.add(posting_id)
            doc = dict(self._doc_by_id[doc_id])
            doc["score"] = score
            candidates.append(doc)

        # Drop near-duplicate postings and cap per company, then take top-k.
        return diversify(
            candidates,
            self.top_k,
            config.DIVERSITY_THRESHOLD,
            group_key=lambda d: (d.get("metadata") or {}).get("company"),
            max_per_group=config.MAX_PER_COMPANY,
        )

    def answer(self, question):
        """Full RAG: retrieve → generate. Returns the answer plus its citations."""
        documents = self.retrieve(question)
        answer = generate_answer(question, documents)
        return {"answer": answer, "citations": documents}
