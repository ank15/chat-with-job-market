"""Retrieval evaluation — benchmark the three retrievers on a labeled set.

`precision_at_k` / `recall_at_k` are pure and unit-tested. Relevance uses a
transparent **proxy**: a posting is relevant to a question if its job title matches
the question's role (e.g. "data analyst"). Retrieval happens over *chunks*, so we
collapse retrieved chunks back to their parent postings before scoring — we care
whether the right *job* surfaced, not the exact passage.

Run: ``python -m eval.retrieval_eval``
"""

import json

from src import config, ingest
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.tfidf import TfidfRetriever

# Retrieve this many chunks, then collapse to the top-k unique postings.
CANDIDATE_CHUNKS = 50


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for doc_id in top_k if doc_id in relevant) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of all relevant items that appear in the top-k."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant) / len(relevant)


def load_eval_set(path=config.EVAL_SET_FILE):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_relevant_ids(df, title_contains):
    """All posting ids whose job title contains `title_contains` (the proxy label)."""
    titles = df["job_title"].fillna("").str.lower()
    mask = titles.str.contains(title_contains.lower())
    return set(df.loc[mask, config.ID_COLUMN])


def _posting_id(doc_id):
    """Map a chunk id ('job_link::3') back to its posting id ('job_link')."""
    return doc_id.rsplit("::", 1)[0]


def top_postings(retriever, question, k):
    """Retrieve chunks and collapse them to the top-k unique postings, in rank order."""
    hits = retriever.retrieve(question, top_k=CANDIDATE_CHUNKS)
    postings = []
    for doc_id, _ in hits:
        pid = _posting_id(doc_id)
        if pid not in postings:
            postings.append(pid)
    return postings[:k]


def evaluate(retriever, eval_set, k=config.TOP_K):
    """Mean precision@k for one retriever over the eval set."""
    precisions = [
        precision_at_k(top_postings(retriever, ex["question"], k), ex["relevant_ids"], k)
        for ex in eval_set
    ]
    return sum(precisions) / max(len(eval_set), 1)


def main():
    df = ingest.load_postings()
    documents = ingest.build_documents(df)
    print(f"Indexing {len(documents):,} chunks from {len(df):,} postings...\n")

    eval_set = load_eval_set()
    for ex in eval_set:
        ex["relevant_ids"] = resolve_relevant_ids(df, ex["relevant_title_contains"])

    # Index once; the hybrid reuses the already-indexed tfidf + dense instances.
    tfidf = TfidfRetriever()
    tfidf.index(documents)
    dense = DenseRetriever()
    dense.index(documents, cache_path=config.ARTIFACTS_DIR / "chunk_embeddings.npy")
    retrievers = {
        "TF-IDF": tfidf,
        "Dense": dense,
        "Hybrid": HybridRetriever(tfidf, dense),
    }

    print(f"{'Retriever':<10} | precision@{config.TOP_K}")
    print("-" * 26)
    for name, retriever in retrievers.items():
        print(f"{name:<10} | {evaluate(retriever, eval_set):>11.3f}")


if __name__ == "__main__":
    main()
