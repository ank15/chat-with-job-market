"""Generation: turn a question + retrieved documents into a grounded, cited answer.

`build_prompt` is implemented and tested (it's pure). `generate_answer` wraps the
Groq LLM call and is stubbed — the prompt forces the model to answer ONLY from the
provided context and to cite sources by number, which is the main defense against
hallucination.
"""

from . import config

SYSTEM_INSTRUCTION = (
    "You answer questions about the job market using ONLY the numbered job-posting "
    "excerpts provided. Cite the sources you use with bracketed numbers like [1], "
    "[2]. If the context does not contain the answer, say you don't know — do not "
    "use outside knowledge."
)


def format_context(documents):
    """Render retrieved documents as a numbered context block for the prompt."""
    lines = []
    for i, doc in enumerate(documents, start=1):
        lines.append(f"[{i}] {doc['text']}")
    return "\n\n".join(lines)


def build_prompt(question, documents):
    """Assemble the full user prompt (context + question)."""
    context = format_context(documents)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with citations):"
    )


def generate_answer(question, documents, model=config.LLM_MODEL, temperature=0.2):
    """Call the LLM to produce an answer grounded in the retrieved documents.

    Low ``temperature`` keeps the model factual/deterministic rather than creative —
    what we want for a grounded Q&A system. The ``groq``/``dotenv`` imports are kept
    local so the rest of the module stays importable (and testable) without them.
    """
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()  # read GROQ_API_KEY from the .env file into the environment
    client = Groq()

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": build_prompt(question, documents)}],
    )
    return response.choices[0].message.content
