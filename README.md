# Chat with the Job Market — a RAG assistant over 12k job postings

Ask natural-language questions about the job market — *"What skills do senior remote
ML roles ask for?"* — and get an answer **grounded in real job postings, with
citations**. Built as a **Retrieval-Augmented Generation (RAG)** system with a
first-class **evaluation harness**.

> **Status: scaffold / work in progress.** The structure, interfaces, and the cheap
> pure-logic pieces (chunking, metrics, score fusion) are implemented; the heavier
> pieces (embeddings, the LLM call) are stubbed with `TODO`s and filled in next.

This is the GenAI step-up from my
[job-market semantic search project](../job-market-intelligence-talent-matching):
that one *retrieves* relevant jobs; this one *retrieves **and** generates* a grounded
answer.

---

## Why this project

Most "chat with your docs" demos stop at "it returns an answer." The hard,
interview-worthy part is **knowing whether the answer is any good** — so this project
treats **evaluation as a first-class citizen**:

- **Retrieval eval** — `precision@k` / `recall@k` across three retrievers
  (TF-IDF, dense embeddings, hybrid), on the same labeled set. *Directly extends the
  TF-IDF-vs-semantic comparison from the sister project.*
- **Generation eval** — groundedness / faithfulness of the answer against the
  retrieved context (does the LLM stick to the sources, or hallucinate?).

---

## Architecture

```
Job postings ──► ingest (load + chunk) ──► documents
                                            │
                          ┌─────────────────┴─────────────────┐
                          ▼                                     │
                  Retriever (pluggable)                         │
            ┌──────────┬──────────┬──────────┐                  │
            │ TF-IDF   │  Dense   │  Hybrid  │  ◄── benchmarked  │
            └──────────┴──────────┴──────────┘     in eval/      │
                          │ top-k relevant chunks                │
                          ▼                                      │
              Generator (LLM + citations) ◄─────────────────────┘
                          │
                          ▼
              Grounded answer  +  cited postings
```

- **Retrieval layer** — comparable to TF-IDF/semantic search (same metric family).
- **Generation layer** — new capability; evaluated by groundedness, not ranking.

---

## Project structure

```
chat-with-job-market/
├── data/                       # job_postings.csv goes here (gitignored)
├── src/
│   ├── config.py               # paths, model names, constants
│   ├── ingest.py               # load postings + chunk into documents
│   ├── retrievers/
│   │   ├── base.py             # Retriever interface (index / score_all / retrieve)
│   │   ├── tfidf.py            # lexical baseline
│   │   ├── dense.py            # sentence-transformer embeddings
│   │   └── hybrid.py           # lexical + dense score fusion
│   ├── generate.py             # LLM answer with citations
│   └── rag.py                  # orchestrates retrieve → generate
├── app/
│   └── app.py                  # Streamlit chat UI
├── eval/
│   ├── datasets/qa_eval.jsonl  # small labeled eval set
│   ├── retrieval_eval.py       # precision@k / recall@k across retrievers
│   └── generation_eval.py      # groundedness / faithfulness
├── tests/                      # unit tests for the pure logic
├── requirements.txt
├── .env.example
└── README.md
```

---

## How to run (once filled in)

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your LLM key
cp .env.example .env          # then paste your GROQ_API_KEY

# 3. Add data: copy job_postings.csv into data/

# 4. Benchmark the retrievers
python -m eval.retrieval_eval

# 5. Launch the chat app
streamlit run app/app.py

# 6. Tests
pytest -q
```

---

## Tech stack (planned)

Python · sentence-transformers (embeddings) · scikit-learn (TF-IDF) · FAISS / numpy
(vector search) · Groq (LLM) · Streamlit (UI) · pytest.

---

## Roadmap

- [ ] Fill in dense retriever (sentence-transformers) + persist the index
- [ ] Implement the LLM generation step with enforced citations
- [ ] Complete the eval harness + publish the retriever benchmark numbers
- [ ] Add LLM-as-judge groundedness scoring
- [ ] Swap brute-force vector search for a FAISS index
- [ ] Deploy on Streamlit Community Cloud
