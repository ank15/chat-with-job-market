"""RAG orchestration: retrieve relevant documents, then generate a grounded answer.

This is the glue layer the app and the eval harness both call. The retrieval half is
implemented; the answer half depends on `generate.generate_answer` (stubbed).
"""

from . import config
from .generate import generate_answer


class RAGPipeline:
    def __init__(self, retriever, documents, top_k=config.TOP_K):
        self.retriever = retriever
        self.top_k = top_k
        self._doc_by_id = {d["id"]: d for d in documents}

    def retrieve(self, question):
        """Return the top-k retrieved documents (with their scores attached)."""
        hits = self.retriever.retrieve(question, self.top_k)
        results = []
        for doc_id, score in hits:
            doc = dict(self._doc_by_id[doc_id])
            doc["score"] = score
            results.append(doc)
        return results

    def answer(self, question):
        """Full RAG: retrieve → generate. Returns the answer plus its citations."""
        documents = self.retrieve(question)
        answer = generate_answer(question, documents)
        return {"answer": answer, "citations": documents}
