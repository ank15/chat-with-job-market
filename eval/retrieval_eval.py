"""Retrieval evaluation — benchmark the three retrievers on a labeled set.

`precision_at_k` and `recall_at_k` are implemented and unit-tested. `evaluate` runs a
retriever over the eval questions and averages the metrics; `main` wires up all three
retrievers so you get a side-by-side table — the direct extension of the
TF-IDF-vs-semantic comparison from the sister project.
"""

import json

from src import config, ingest
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.tfidf import TfidfRetriever


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the top-k retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of all relevant docs that appear in the top-k."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def load_eval_set(path=config.EVAL_SET_FILE):
    """Load the labeled eval set (one JSON object per line)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retriever, eval_set, k=config.TOP_K):
    """Average precision@k / recall@k for one retriever over the eval set."""
    precisions, recalls = [], []
    for example in eval_set:
        hits = retriever.retrieve(example["question"], top_k=k)
        retrieved_ids = [doc_id for doc_id, _ in hits]
        precisions.append(precision_at_k(retrieved_ids, example["relevant_ids"], k))
        recalls.append(recall_at_k(retrieved_ids, example["relevant_ids"], k))
    n = max(len(eval_set), 1)
    return {"precision@k": sum(precisions) / n, "recall@k": sum(recalls) / n}


def main():
    documents = ingest.build_documents()
    eval_set = load_eval_set()

    dense = DenseRetriever()
    tfidf = TfidfRetriever()
    retrievers = {
        "TF-IDF": tfidf,
        "Dense": dense,
        "Hybrid": HybridRetriever(tfidf, dense),
    }

    print(f"{'Retriever':<10} | precision@{config.TOP_K} | recall@{config.TOP_K}")
    print("-" * 40)
    for name, retriever in retrievers.items():
        retriever.index(documents)
        scores = evaluate(retriever, eval_set)
        print(f"{name:<10} | {scores['precision@k']:>11.3f} | {scores['recall@k']:>8.3f}")


if __name__ == "__main__":
    main()
