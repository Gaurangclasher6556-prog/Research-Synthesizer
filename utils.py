"""
utils.py – Utility functions for the Autonomous Literature Synthesizer.

Modules:
  1. PDF Parsing      → marker-pdf based conversion
  2. Vector DB        → ChromaDB setup & operations
  3. ArXiv Helpers    → Search, download, rate-limiting
  4. Semantic Overlap → Plagiarism guard with TF-IDF cosine similarity
  5. Text Helpers     → Citation formatting, word counting
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import textwrap
from pathlib import Path
from typing import Any

import arxiv
import chromadb
import numpy as np
from rich.logging import RichHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("synthesizer")


# ╔══════════════════════════════════════════════╗
# ║  1. PDF PARSING                              ║
# ╚══════════════════════════════════════════════╝


def parse_pdf_to_markdown(pdf_path: str | Path) -> str:
    """
    Convert a PDF file to Markdown using the marker-pdf library.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Markdown string of the parsed document.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        RuntimeError: If marker-pdf conversion fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"📄 Parsing PDF: {pdf_path.name}")

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models, config={"output_format": "markdown"})
        rendered = converter(str(pdf_path))
        markdown_text = rendered.markdown

        if not markdown_text or len(markdown_text.strip()) < 50:
            raise RuntimeError("Marker returned empty or near-empty output.")

        # Cache the parsed result
        cache_path = config.PARSED_DIR / f"{pdf_path.stem}.md"
        cache_path.write_text(markdown_text, encoding="utf-8")
        logger.info(f"✅ Parsed {pdf_path.name} → {len(markdown_text)} chars")
        return markdown_text

    except ImportError:
        logger.warning(
            "marker-pdf not installed or failed – falling back to basic text extraction."
        )
        return _fallback_pdf_parse(pdf_path)
    except Exception as e:
        logger.error(f"❌ PDF parsing failed for {pdf_path.name}: {e}")
        return _fallback_pdf_parse(pdf_path)


def _fallback_pdf_parse(pdf_path: Path) -> str:
    """
    Minimal fallback using PyMuPDF (fitz) if marker is unavailable.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
        full_text = "\n\n".join(text_parts)
        cache_path = config.PARSED_DIR / f"{pdf_path.stem}.md"
        cache_path.write_text(full_text, encoding="utf-8")
        logger.info(f"📝 Fallback parsed {pdf_path.name} → {len(full_text)} chars")
        return full_text
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Cannot parse PDF.")
        raise RuntimeError(
            "Neither marker-pdf nor PyMuPDF is available. Install one:\n"
            "  pip install marker-pdf   OR   pip install PyMuPDF"
        )


def extract_title_and_abstract(markdown_text: str) -> dict[str, str]:
    """
    Heuristic extraction of title and abstract from parsed Markdown.

    Returns:
        dict with keys 'title' and 'abstract'.
    """
    lines = markdown_text.strip().split("\n")
    title = ""
    abstract = ""

    # Title: first non-empty line or first heading
    for line in lines:
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped
            break

    # Abstract: look for "Abstract" heading or keyword
    abstract_pattern = re.compile(r"^#+\s*abstract|^abstract", re.IGNORECASE)
    capturing = False
    abstract_lines: list[str] = []

    for line in lines:
        if abstract_pattern.match(line.strip()):
            capturing = True
            continue
        if capturing:
            # Stop at the next heading
            if line.strip().startswith("#") and abstract_lines:
                break
            abstract_lines.append(line)

    abstract = " ".join(abstract_lines).strip()

    # If no abstract section found, take first 500 chars after title
    if not abstract:
        body = markdown_text[len(title) :].strip()
        abstract = body[:500]

    return {"title": title, "abstract": abstract}


# ╔══════════════════════════════════════════════╗
# ║  2. VECTOR DATABASE (ChromaDB)               ║
# ╚══════════════════════════════════════════════╝


class VectorStore:
    """Wrapper around ChromaDB for local vector storage of research papers."""

    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._embedding_fn = self._get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
        logger.info(
            f"🗄️  ChromaDB ready | Collection: {config.CHROMA_COLLECTION_NAME} "
            f"| Docs: {self._collection.count()}"
        )

    def _get_embedding_function(self):
        """Get SentenceTransformer-based embedding function for ChromaDB."""
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )

    def add_paper(
        self,
        paper_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a paper's text to the vector store, chunked for retrieval.

        Args:
            paper_id: Unique identifier for the paper.
            text: Full text of the paper.
            metadata: Additional metadata (title, authors, etc.).
        """
        chunks = self._chunk_text(text, chunk_size=800, overlap=100)
        if not chunks:
            logger.warning(f"⚠️  No chunks generated for paper {paper_id}")
            return

        ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**(metadata or {}), "chunk_index": i, "paper_id": paper_id}
            for i in range(len(chunks))
        ]

        # Upsert to handle re-runs gracefully
        batch_size = 40  # ChromaDB recommends batching
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(f"📥 Stored {len(chunks)} chunks for paper '{paper_id}'")

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Query the vector store for relevant paper chunks.

        Returns:
            List of dicts with keys: document, metadata, distance.
        """
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        output = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append(
                    {"document": doc, "metadata": meta, "distance": dist}
                )
        return output

    def get_all_paper_ids(self) -> list[str]:
        """Return unique paper IDs stored in the collection."""
        all_meta = self._collection.get()["metadatas"]
        return list({m.get("paper_id", "unknown") for m in all_meta if m})

    @staticmethod
    def _chunk_text(
        text: str, chunk_size: int = 800, overlap: int = 100
    ) -> list[str]:
        """Split text into overlapping chunks of roughly `chunk_size` words."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks


# ╔══════════════════════════════════════════════╗
# ║  3. ARXIV HELPERS                            ║
# ╚══════════════════════════════════════════════╝


class ArxivSearcher:
    """Rate-limited arXiv search and download utility."""

    def __init__(self):
        self._client = arxiv.Client(
            page_size=config.ARXIV_MAX_RESULTS_PER_QUERY,
            delay_seconds=config.ARXIV_RATE_LIMIT_SECONDS,
            num_retries=config.ARXIV_NUM_RETRIES,
        )
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < config.ARXIV_RATE_LIMIT_SECONDS:
            sleep_time = config.ARXIV_RATE_LIMIT_SECONDS - elapsed
            logger.debug(f"⏳ Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """
        Search arXiv and return paper metadata.

        Args:
            query: Search query string.
            max_results: Override default max results.

        Returns:
            List of dicts with keys: id, title, authors, abstract,
            pdf_url, published, categories.
        """
        max_results = max_results or config.ARXIV_MAX_RESULTS_PER_QUERY
        self._rate_limit()

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        try:
            for paper in self._client.results(search):
                results.append(
                    {
                        "id": paper.entry_id.split("/")[-1],
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "abstract": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "published": paper.published.isoformat() if paper.published else "",
                        "categories": paper.categories,
                    }
                )
        except Exception as e:
            logger.error(f"❌ ArXiv search failed for '{query}': {e}")

        logger.info(f"🔍 ArXiv query '{query[:50]}…' → {len(results)} results")
        return results

    def download_pdf(self, paper_meta: dict) -> Path | None:
        """
        Download a paper PDF to the papers directory.

        Args:
            paper_meta: Dict with at least 'id' and 'pdf_url'.

        Returns:
            Path to downloaded PDF or None on failure.
        """
        self._rate_limit()
        paper_id = paper_meta["id"]
        filename = f"{self._sanitize_filename(paper_id)}.pdf"
        target = config.PAPERS_DIR / filename

        if target.exists():
            logger.debug(f"📁 PDF already cached: {filename}")
            return target

        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self._client.results(search))
            paper.download_pdf(dirpath=str(config.PAPERS_DIR), filename=filename)
            logger.info(f"⬇️  Downloaded: {filename}")
            return target
        except StopIteration:
            logger.error(f"❌ Paper not found on arXiv: {paper_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Download failed for {paper_id}: {e}")
            return None

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove characters unsafe for filenames."""
        return re.sub(r'[<>:"/\\|?*]', "_", name)


# ╔══════════════════════════════════════════════╗
# ║  4. SEMANTIC OVERLAP / PLAGIARISM GUARD      ║
# ╚══════════════════════════════════════════════╝


class PlagiarismGuard:
    """
    Detects high semantic overlap between generated text and sources.
    Uses TF-IDF cosine similarity at the sentence level.
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold or config.SEMANTIC_OVERLAP_THRESHOLD
        self._vectorizer = TfidfVectorizer(
            stop_words="english", max_features=10000
        )

    def check_overlap(
        self, generated_text: str, source_texts: list[str]
    ) -> list[dict]:
        """
        Check each sentence in generated_text against all source texts.

        Returns:
            List of dicts for flagged sentences:
            {sentence, source_index, similarity, needs_rewrite}.
        """
        gen_sentences = self._split_sentences(generated_text)
        if not gen_sentences or not source_texts:
            return []

        # Build one big corpus: sources + generated sentences
        all_texts = source_texts + gen_sentences

        try:
            tfidf_matrix = self._vectorizer.fit_transform(all_texts)
        except ValueError:
            logger.warning("⚠️  TF-IDF failed (empty vocabulary). Skipping overlap check.")
            return []

        n_sources = len(source_texts)
        source_vectors = tfidf_matrix[:n_sources]
        gen_vectors = tfidf_matrix[n_sources:]

        flagged = []
        sim_matrix = cosine_similarity(gen_vectors, source_vectors)

        for i, sentence in enumerate(gen_sentences):
            max_sim = float(np.max(sim_matrix[i]))
            max_source_idx = int(np.argmax(sim_matrix[i]))
            if max_sim >= self.threshold:
                flagged.append(
                    {
                        "sentence": sentence,
                        "source_index": max_source_idx,
                        "similarity": round(max_sim, 4),
                        "needs_rewrite": True,
                    }
                )

        logger.info(
            f"🛡️  Plagiarism check: {len(flagged)}/{len(gen_sentences)} "
            f"sentences flagged (threshold={self.threshold})"
        )
        return flagged

    def get_rewrite_instructions(self, flagged: list[dict]) -> str:
        """Generate a prompt asking the LLM to rewrite flagged sentences."""
        if not flagged:
            return ""

        instructions = [
            "The following sentences have ≥70% semantic overlap with source "
            "material and MUST be rewritten in your own words while preserving "
            "the meaning. Use in-text citations where appropriate.\n"
        ]
        for i, item in enumerate(flagged, 1):
            instructions.append(
                f"{i}. [Similarity: {item['similarity']:.0%}] "
                f"\"{item['sentence']}\""
            )
        return "\n".join(instructions)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]


# ╔══════════════════════════════════════════════╗
# ║  5. TEXT HELPERS                             ║
# ╚══════════════════════════════════════════════╝


def format_citation(paper_meta: dict) -> str:
    """
    Create an in-text citation like [Smith et al., 2024].

    Args:
        paper_meta: Dict with 'authors' (list) and 'published' (ISO string).
    """
    authors = paper_meta.get("authors", [])
    year = paper_meta.get("published", "")[:4] or "n.d."

    if len(authors) == 0:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = authors[0].split()[-1]  # Last name
    elif len(authors) == 2:
        author_str = (
            f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
        )
    else:
        author_str = f"{authors[0].split()[-1]} et al."

    return f"[{author_str}, {year}]"


def word_count(text: str) -> int:
    """Return approximate word count."""
    return len(text.split())


def truncate_text(text: str, max_words: int = 500) -> str:
    """Truncate text to max_words, appending '…' if truncated."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def generate_paper_hash(text: str) -> str:
    """Generate a short SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
