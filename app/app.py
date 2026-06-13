"""Streamlit chat UI for the RAG assistant.

Run with: streamlit run app/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, ingest
from src.generate import generate_answer
from src.logging_config import get_logger
from src.rag import RAGPipeline
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.tfidf import TfidfRetriever

logger = get_logger()


@st.cache_resource
def load_documents():
    """Load + chunk the postings once, shared across retrievers."""
    return ingest.build_documents()


@st.cache_resource
def build_pipeline(retriever_name):
    """Build (and cache) the chosen retriever + the RAG pipeline.

    The dense retriever loads its embeddings from the on-disk cache, so the corpus
    is encoded once (by the benchmark) and reused here — the app starts fast.
    """
    documents = load_documents()
    embed_cache = config.ARTIFACTS_DIR / "chunk_embeddings.npy"

    if retriever_name == "TF-IDF":
        retriever = TfidfRetriever()
        retriever.index(documents)
    elif retriever_name == "Dense":
        retriever = DenseRetriever()
        retriever.index(documents, cache_path=embed_cache)
    else:  # Hybrid
        tfidf = TfidfRetriever()
        tfidf.index(documents)
        dense = DenseRetriever()
        dense.index(documents, cache_path=embed_cache)
        retriever = HybridRetriever(tfidf, dense)

    return RAGPipeline(retriever, documents)


st.title("Chat with the Job Market")
st.caption("Ask about skills, roles, and hiring trends — answers grounded in real postings.")

retriever_name = st.sidebar.radio("Retriever", ("Hybrid", "Dense", "TF-IDF"))
question = st.text_input("Your question:")

if st.button("Ask") and question.strip():
    try:
        with st.spinner("Retrieving and generating..."):
            pipeline = build_pipeline(retriever_name)
            # Time retrieval and generation separately for runtime visibility.
            t0 = time.perf_counter()
            citations = pipeline.retrieve(question)
            retrieve_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            answer = generate_answer(question, citations)
            generate_s = time.perf_counter() - t1
            result = {"answer": answer, "citations": citations}
    except Exception as exc:  # log and surface a friendly message instead of crashing
        logger.exception("query=%r engine=%s status=error error=%s", question, retriever_name, exc)
        st.error("Something went wrong generating the answer — see logs/app.log for details.")
        st.stop()

    top_score = citations[0].get("score", 0.0) if citations else 0.0
    logger.info(
        "query=%r engine=%s retrieved=%d top_score=%.3f retrieve_ms=%.0f generate_s=%.1f status=ok",
        question, retriever_name, len(citations), top_score, retrieve_ms, generate_s,
    )

    st.caption(
        f"⏱ retrieval {retrieve_ms:.0f} ms · generation {generate_s:.1f} s · "
        f"engine: {retriever_name}"
    )
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for i, doc in enumerate(result["citations"], start=1):
        m = doc.get("metadata", {})
        header = f"**[{i}] {m.get('title', 'Posting')}** — {m.get('company', '')} ({m.get('location', '')})"
        st.markdown(f"{header}  ·  score {doc.get('score', 0):.3f}")
        with st.expander("excerpt"):
            st.write(doc["text"][:500] + "...")
