"""Retrieval evaluation — benchmark the three retrievers on a labeled set.

`precision_at_k` / `recall_at_k` are pure and unit-tested. Relevance uses a
transparent **proxy**: a posting is relevant to a question if its job title matches
the question's role (e.g. "data analyst"). Retrieval happens over *chunks*, so we
collapse retrieved chunks back to their parent postings before scoring — we care
whether the right *job* surfaced, not the exact passage.

Run: ``python -m eval.retrieval_eval``
"""

import json
import time

from src import config, ingest
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.tfidf import TfidfRetriever

# Retrieve this many chunks, then collapse to the top-k unique postings.
CANDIDATE_CHUNKS = 50
# Depth used for MRR (how far down we look for the first relevant result).
MRR_DEPTH = 10


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


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    """1.0 if at least one relevant item is in the top-k, else 0.0."""
    relevant = set(relevant_ids)
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    """1 / rank of the first relevant item (0 if none). Averaged over queries = MRR."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


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
    """Average retrieval metrics + mean query latency for one retriever.

    Returns precision@k, recall@k, hit_rate@k, MRR, and latency in milliseconds.
    """
    # One untimed warmup query so latency reflects steady state, not the
    # one-off model/thread warmup cost (esp. for the dense encoder).
    if eval_set:
        top_postings(retriever, eval_set[0]["question"], MRR_DEPTH)

    p = r = h = m = 0.0
    total_ms = 0.0
    for ex in eval_set:
        start = time.perf_counter()
        ranked = top_postings(retriever, ex["question"], MRR_DEPTH)
        total_ms += (time.perf_counter() - start) * 1000

        relevant = ex["relevant_ids"]
        top_k = ranked[:k]
        p += precision_at_k(top_k, relevant, k)
        r += recall_at_k(top_k, relevant, k)
        h += hit_rate_at_k(top_k, relevant, k)
        m += reciprocal_rank(ranked, relevant)

    n = max(len(eval_set), 1)
    return {
        "precision": p / n,
        "recall": r / n,
        "hit_rate": h / n,
        "mrr": m / n,
        "latency_ms": total_ms / n,
    }


def main():
    df = ingest.load_postings()
    documents = ingest.build_documents(df)
    print(f"Indexing {len(documents):,} chunks from {len(df):,} postings...\n")

    eval_set = load_eval_set()
    for ex in eval_set:
        ex["relevant_ids"] = resolve_relevant_ids(df, ex["relevant_title_contains"])

    # Index once (timed); the hybrid reuses the already-indexed tfidf + dense.
    tfidf = TfidfRetriever()
    dense = DenseRetriever()
    index_times = {}
    for name, retriever, kwargs in [
        ("TF-IDF", tfidf, {}),
        ("Dense", dense, {"cache_path": config.ARTIFACTS_DIR / "chunk_embeddings.npy"}),
    ]:
        start = time.perf_counter()
        retriever.index(documents, **kwargs)
        index_times[name] = time.perf_counter() - start
    index_times["Hybrid"] = index_times["TF-IDF"] + index_times["Dense"]
    retrievers = {"TF-IDF": tfidf, "Dense": dense, "Hybrid": HybridRetriever(tfidf, dense)}

    k = config.TOP_K
    print(
        f"{'Retriever':<10} | P@{k}   | R@{k}   | Hit@{k} |  MRR  | "
        f"latency/query | index time"
    )
    print("-" * 76)
    for name, retriever in retrievers.items():
        s = evaluate(retriever, eval_set, k=k)
        print(
            f"{name:<10} | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"{s['hit_rate']:.3f} | {s['mrr']:.3f} | "
            f"{s['latency_ms']:>8.1f} ms  | {index_times[name]:>6.1f} s"
        )


if __name__ == "__main__":
    main()
