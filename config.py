"""
Configuration module for the Autonomous Literature Synthesizer.
All tunable parameters are centralized here.
Supports: Groq (cloud, free) and Ollama (local) as LLM backends.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma_db"
PARSED_DIR = DATA_DIR / "parsed"
UPLOAD_DIR = DATA_DIR / "uploads"

# Create directories on import
for _dir in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, PAPERS_DIR, CHROMA_DIR, PARSED_DIR, UPLOAD_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# LLM Backend Selection
# Options: "groq" (cloud, free) or "ollama" (local)
# ──────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")

# ── Groq (Cloud – Free tier) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

# Available Groq models (all free):
# - llama-3.3-70b-versatile   (best quality, recommended)
# - llama-3.1-8b-instant      (fastest)
# - mixtral-8x7b-32768        (good for long context)
# - gemma2-9b-it              (Google's model)

# ── Ollama (Local – fallback) ──
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

# ──────────────────────────────────────────────
# Embedding Model (local, free)
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# ──────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "research_papers"

# ──────────────────────────────────────────────
# ArXiv
# ──────────────────────────────────────────────
ARXIV_MAX_RESULTS_PER_QUERY = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
ARXIV_RATE_LIMIT_SECONDS = float(os.getenv("ARXIV_RATE_LIMIT", "3.0"))
ARXIV_NUM_RETRIES = int(os.getenv("ARXIV_NUM_RETRIES", "3"))

# ──────────────────────────────────────────────
# Plagiarism / Semantic Overlap
# ──────────────────────────────────────────────
SEMANTIC_OVERLAP_THRESHOLD = float(
    os.getenv("SEMANTIC_OVERLAP_THRESHOLD", "0.70")
)
MAX_REWRITE_ATTEMPTS = int(os.getenv("MAX_REWRITE_ATTEMPTS", "3"))

# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────
TARGET_REPORT_WORDS = int(os.getenv("TARGET_REPORT_WORDS", "1500"))

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
