# Autonomous Literature Synthesizer

### An Andrew Ng–Inspired Agentic Workflow for Local Research Paper Analysis

> **Zero paid APIs.** Everything runs locally with Ollama, ChromaDB, and SentenceTransformers.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
│  CLI + Orchestrator + Pre-flight Checks                      │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Planner  │ Searcher │ Extractor│  Critic  │     Writer      │
│  Agent   │  Agent   │  Agent   │  Agent   │     Agent       │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                       agents.py                              │
│  5 Specialized CrewAI Agents with Ollama LLM                 │
├─────────────────────────────────────────────────────────────┤
│                       tasks.py                               │
│  Sequential Task Chain with Context Dependencies             │
├─────────────────────────────────────────────────────────────┤
│                       tools.py                               │
│  8 Custom BaseTool Implementations                           │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  PDF Parser  │ ChromaDB     │ ArXiv Client  │  Plagiarism   │
│  (marker-pdf)│ VectorStore  │ (rate-limited)│  Guard (TF-IDF│
├──────────────┴──────────────┴───────────────┴───────────────┤
│                       utils.py                               │
│  Core Utilities + Error Handling                             │
├─────────────────────────────────────────────────────────────┤
│                       config.py                              │
│  Centralized Configuration with ENV Overrides                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Ollama → https://ollama.com
# Start the server
ollama serve

# Pull a model (choose one)
ollama pull llama3        # Recommended
ollama pull mistral       # Alternative
```

### 2. Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

### 3. Run

```bash
# Basic usage
python main.py path/to/paper.pdf

# With options
python main.py paper.pdf --model mistral --verbose --max-results 3

# Custom output name
python main.py paper.pdf --output my_analysis.md
```

## 📂 Project Structure

```
Research_Synthesizer/
├── main.py               # CLI entry point & orchestrator
├── agents.py             # 5 CrewAI agent definitions
├── tasks.py              # Sequential task chain
├── tools.py              # 8 custom CrewAI tools
├── utils.py              # Core utilities (PDF, DB, ArXiv, plagiarism)
├── config.py             # Centralized configuration
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── data/
    ├── input/            # Place input PDFs here
    ├── output/           # Generated reports
    ├── papers/           # Downloaded arXiv PDFs
    ├── parsed/           # Cached Markdown conversions
    └── chroma_db/        # ChromaDB persistent storage
```

## 🤖 The Agent Squad

| # | Agent | Role | Tools |
|---|-------|------|-------|
| 1 | **Planner** | Analyzes title/abstract → 5 arXiv queries | `parse_pdf`, `extract_title_abstract` |
| 2 | **Searcher** | Executes queries, downloads & stores papers | `arxiv_search`, `arxiv_download`, `parse_pdf`, `store_in_vectordb` |
| 3 | **Extractor** | Parses papers, extracts methodologies | `parse_pdf`, `query_vectordb`, `format_citation` |
| 4 | **Critic** | 7-dimension evaluation of target paper | `query_vectordb`, `format_citation` |
| 5 | **Writer** | Synthesizes 1,500-word report with citations | `query_vectordb`, `format_citation`, `check_plagiarism` |

## 📊 Evaluation Dimensions

The Critic Agent evaluates on 7 dimensions (scored 1-10):

1. **Originality** – Novelty of contributions
2. **Importance** – Significance of the problem
3. **Claim Support** – Evidence backing claims
4. **Experimental Soundness** – Rigor of experiments
5. **Clarity** – Writing quality and organization
6. **Community Value** – Usefulness to researchers
7. **Contextualization** – Positioning within literature

## 🛡️ Plagiarism & Ethics Guard

- **TF-IDF Cosine Similarity** at the sentence level
- Threshold: **70%** semantic overlap triggers rewrite
- The Writer Agent automatically rewrites flagged sentences
- All claims must have **in-text citations** `[Author et al., Year]`

## ⚙️ Configuration

All parameters can be set via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_TEMPERATURE` | `0.3` | LLM creativity (0-1) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `ARXIV_MAX_RESULTS` | `5` | Papers per query |
| `ARXIV_RATE_LIMIT` | `3.0` | Seconds between requests |
| `SEMANTIC_OVERLAP_THRESHOLD` | `0.70` | Plagiarism trigger |
| `TARGET_REPORT_WORDS` | `1500` | Report word count target |

## 🔧 Troubleshooting

### Ollama connection refused
```bash
# Make sure Ollama is running
ollama serve
# Verify model is available
ollama list
```

### PDF parsing fails
The system uses `marker-pdf` with a `PyMuPDF` fallback. If both fail:
```bash
pip install marker-pdf PyMuPDF
```

### ArXiv rate limiting
The system respects a 3-second delay between requests. If you hit limits:
```bash
ARXIV_RATE_LIMIT=5.0 python main.py paper.pdf
```

### Out of memory
Use a smaller model:
```bash
python main.py paper.pdf --model phi3
```

## 📜 License

MIT License – Use freely for research and education.
