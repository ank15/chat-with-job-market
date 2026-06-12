"""Streamlit chat UI for the RAG assistant.

Skeleton: the layout and wiring are here; it depends on the dense retriever and the
LLM generation step being filled in. Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, ingest
from src.rag import RAGPipeline
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.tfidf import TfidfRetriever


@st.cache_resource
def build_pipeline(retriever_name):
    """Build (and cache) documents + the chosen retriever + the RAG pipeline."""
    documents = ingest.build_documents()
    tfidf = TfidfRetriever()
    dense = DenseRetriever()
    retriever = {
        "TF-IDF": tfidf,
        "Dense": dense,
        "Hybrid": HybridRetriever(tfidf, dense),
    }[retriever_name]
    retriever.index(documents)
    return RAGPipeline(retriever, documents)


st.title("Chat with the Job Market")
st.caption("Ask about skills, roles, and hiring trends — answers grounded in real postings.")

retriever_name = st.sidebar.radio("Retriever", ("Hybrid", "Dense", "TF-IDF"))
question = st.text_input("Your question:")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating..."):
        pipeline = build_pipeline(retriever_name)
        result = pipeline.answer(question)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for i, doc in enumerate(result["citations"], start=1):
        st.markdown(f"**[{i}]** (score {doc.get('score', 0):.3f}) — {doc['text'][:300]}...")
