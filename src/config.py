"""Central configuration: paths, model names, and tunable constants.

Keeping these in one place means the app, the eval harness, and the tests all agree
on the same settings (no train/serve skew).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
POSTINGS_FILE = DATA_DIR / "final_jobs.csv"  # merged file (has job_summary text)
EVAL_SET_FILE = PROJECT_ROOT / "eval" / "datasets" / "qa_eval.jsonl"

# Which columns of the postings CSV to use.
TEXT_COLUMN = "job_summary"
ID_COLUMN = "job_link"

# Chunking
CHUNK_SIZE = 256      # words per chunk
CHUNK_OVERLAP = 40    # words of overlap between consecutive chunks

# Models
EMBED_MODEL = "all-MiniLM-L6-v2"          # sentence-transformer for dense retrieval
LLM_MODEL = "llama-3.3-70b-versatile"     # Groq model for generation

# Job-board boilerplate the TF-IDF retriever should ignore. Every posting is a
# "job opening", so these words carry no signal — and letting them count lets
# keyword-stuffed spam ("legal jobs, counsel jobs, lawyer jobs...") hijack results.
DOMAIN_STOPWORDS = {
    "job", "jobs", "opening", "openings", "hiring", "hire", "role", "roles",
    "position", "positions", "apply", "candidate", "candidates", "employment",
    "opportunity", "opportunities", "find", "looking", "seeking", "join",
    "team", "company",
}

# Retrieval / generation defaults
TOP_K = 5
HYBRID_ALPHA = 0.5    # weight on dense vs lexical in the hybrid retriever (0..1)

# Diversity re-ranking (drop near-duplicate results, e.g. the same job reposted)
CANDIDATE_MULTIPLIER = 8   # fetch TOP_K * this many chunks before diversifying
DIVERSITY_THRESHOLD = 0.6  # drop a result if its token-overlap (Jaccard) with a
                           # kept result is >= this; lower = stricter de-duplication
MAX_PER_COMPANY = 2        # at most this many results from the same company
