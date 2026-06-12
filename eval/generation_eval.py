"""Generation evaluation — is the answer actually grounded in the retrieved context?

A grounded answer only makes claims supported by the sources. We start with a cheap
lexical-overlap proxy (implemented + testable) and leave the stronger LLM-as-judge
version as a TODO.
"""

import re


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def lexical_groundedness(answer, context_documents):
    """Proxy score in [0, 1]: fraction of answer tokens found in the context.

    High overlap doesn't prove correctness, but low overlap is a strong smell that the
    model went off-script (hallucinated). A fast first-pass signal.
    """
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for doc in context_documents:
        context_tokens |= _tokenize(doc["text"])
    grounded = answer_tokens & context_tokens
    return len(grounded) / len(answer_tokens)


def llm_groundedness(answer, context_documents, model=None):
    """Ask an LLM judge whether every claim in `answer` is supported by the context.

    TODO: prompt a judge model to return a 0-1 faithfulness score (and flag any
    unsupported claims). This is the stronger, but pricier, signal.
    """
    raise NotImplementedError("LLM-as-judge groundedness not implemented yet.")
