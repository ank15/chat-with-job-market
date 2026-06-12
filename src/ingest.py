"""Ingestion: load job postings and split them into retrievable documents.

`chunk_text` is implemented (it's pure and unit-tested); `load_postings` is a thin
wrapper over pandas, and `build_documents` ties them together. A "document" is a
dict: ``{"id": str, "text": str, "metadata": dict}``.
"""

from . import config


def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into overlapping word-windows.

    Overlap preserves context that would otherwise be cut at a chunk boundary.
    Returns a list of chunk strings (empty list for empty/whitespace input).
    """
    words = str(text).split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_postings(path=config.POSTINGS_FILE):
    """Load the postings CSV into a DataFrame.

    TODO: dedupe on ID_COLUMN, drop rows with empty TEXT_COLUMN.
    """
    import pandas as pd

    return pd.read_csv(path)


def build_documents(df=None):
    """Turn postings into chunked documents ready for indexing.

    Each chunk becomes its own document with a stable id ``"{job_id}::{n}"`` and
    metadata (job title, company, location) carried through for citations.

    TODO: pull real metadata columns once the schema is wired up.
    """
    if df is None:
        df = load_postings()

    documents = []
    for _, row in df.iterrows():
        job_id = row[config.ID_COLUMN]
        for i, chunk in enumerate(chunk_text(row[config.TEXT_COLUMN])):
            documents.append(
                {
                    "id": f"{job_id}::{i}",
                    "text": chunk,
                    "metadata": {
                        "job_id": job_id,
                        # TODO: "title": row.get("job_title"), etc.
                    },
                }
            )
    return documents
