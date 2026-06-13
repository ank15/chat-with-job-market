"""Diversity re-ranking: filter near-duplicate retrieved documents.

A relevance-only retriever can return several near-identical postings — e.g. the
same job reposted by an aggregator across cities. Feeding five copies of one job to
the LLM makes it over-generalize ("all these jobs are at Capital One").

This is a greedy, MMR-style pass: walk the candidates in relevance (score) order and
keep each one only if it isn't too similar (Jaccard token overlap) to a result
already kept — so the final top-k are k *distinct* jobs. Pure and unit-tested.
"""

import re


def token_set(text):
    """Lowercased set of word/number tokens — the unit of similarity comparison."""
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def jaccard(a, b):
    """Jaccard similarity of two token sets, in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def diversify(candidates, top_k, threshold, group_key=None, max_per_group=None):
    """Greedily pick up to ``top_k`` candidates, skipping redundant ones.

    ``candidates``: list of dicts in score order, each with a ``"text"`` key.

    A candidate is skipped if either:
    - its Jaccard token overlap with an already-kept candidate is ``>= threshold``
      (near-duplicate text), or
    - ``group_key`` is given and its group already has ``max_per_group`` results
      kept (so one employer/company can't dominate the list).

    Returns the kept subset, order preserved.
    """
    selected = []
    kept_tokens = []
    group_counts = {}
    for candidate in candidates:
        tokens = token_set(candidate["text"])
        if any(jaccard(tokens, kept) >= threshold for kept in kept_tokens):
            continue
        if group_key is not None and max_per_group is not None:
            group = group_key(candidate)
            if group_counts.get(group, 0) >= max_per_group:
                continue
            group_counts[group] = group_counts.get(group, 0) + 1
        selected.append(candidate)
        kept_tokens.append(tokens)
        if len(selected) >= top_k:
            break
    return selected
