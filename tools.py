"""
tools.py – Custom CrewAI tools for the Autonomous Literature Synthesizer.

Each tool wraps a utility function so that agents can invoke them autonomously.
"""

from __future__ import annotations

import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import config
from utils import (
    ArxivSearcher,
    PlagiarismGuard,
    VectorStore,
    extract_title_and_abstract,
    format_citation,
    parse_pdf_to_markdown,
    logger,
)


# ──────────────────────────────────────────────
# Shared singletons (lazy-initialized)
# ──────────────────────────────────────────────
_arxiv_searcher: ArxivSearcher | None = None
_vector_store: VectorStore | None = None
_plagiarism_guard: PlagiarismGuard | None = None


def get_arxiv_searcher() -> ArxivSearcher:
    global _arxiv_searcher
    if _arxiv_searcher is None:
        _arxiv_searcher = ArxivSearcher()
    return _arxiv_searcher


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_plagiarism_guard() -> PlagiarismGuard:
    global _plagiarism_guard
    if _plagiarism_guard is None:
        _plagiarism_guard = PlagiarismGuard()
    return _plagiarism_guard


# ╔══════════════════════════════════════════════╗
# ║  TOOL 1: Parse PDF                          ║
# ╚══════════════════════════════════════════════╝


class ParsePDFInput(BaseModel):
    pdf_path: str = Field(
        ..., description="Absolute or relative path to the PDF file to parse."
    )


class ParsePDFTool(BaseTool):
    name: str = "parse_pdf"
    description: str = (
        "Converts a PDF research paper into Markdown text. "
        "Returns the full Markdown content of the paper."
    )
    args_schema: Type[BaseModel] = ParsePDFInput

    def _run(self, pdf_path: str) -> str:
        try:
            markdown = parse_pdf_to_markdown(pdf_path)
            return markdown[:15000]  # Limit output to avoid LLM context overflow
        except Exception as e:
            return f"ERROR: Failed to parse PDF '{pdf_path}': {e}"


# ╔══════════════════════════════════════════════╗
# ║  TOOL 2: Extract Title & Abstract           ║
# ╚══════════════════════════════════════════════╝


class ExtractMetaInput(BaseModel):
    markdown_text: str = Field(
        ..., description="Markdown text of a parsed research paper."
    )


class ExtractMetadataTool(BaseTool):
    name: str = "extract_title_abstract"
    description: str = (
        "Extracts the title and abstract from a parsed Markdown paper. "
        "Returns a JSON object with 'title' and 'abstract' keys."
    )
    args_schema: Type[BaseModel] = ExtractMetaInput

    def _run(self, markdown_text: str) -> str:
        try:
            result = extract_title_and_abstract(markdown_text)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"ERROR: Metadata extraction failed: {e}"


# ╔══════════════════════════════════════════════╗
# ║  TOOL 3: ArXiv Search                       ║
# ╚══════════════════════════════════════════════╝


class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="Search query for arXiv.")
    max_results: int = Field(
        default=5, description="Maximum number of results to return."
    )


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Searches arXiv for academic papers matching a query. "
        "Returns a JSON list of paper metadata including id, title, "
        "authors, abstract, pdf_url, published date, and categories."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            searcher = get_arxiv_searcher()
            results = searcher.search(query, max_results=max_results)
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"ERROR: ArXiv search failed: {e}"


# ╔══════════════════════════════════════════════╗
# ║  TOOL 4: ArXiv Download                     ║
# ╚══════════════════════════════════════════════╝


class ArxivDownloadInput(BaseModel):
    paper_id: str = Field(..., description="arXiv paper ID (e.g., '2401.12345').")


class ArxivDownloadTool(BaseTool):
    name: str = "arxiv_download"
    description: str = (
        "Downloads a PDF from arXiv given a paper ID. "
        "Returns the local file path of the downloaded PDF."
    )
    args_schema: Type[BaseModel] = ArxivDownloadInput

    def _run(self, paper_id: str) -> str:
        try:
            searcher = get_arxiv_searcher()
            paper_meta = {"id": paper_id, "pdf_url": f"https://arxiv.org/pdf/{paper_id}"}
            path = searcher.download_pdf(paper_meta)
            if path:
                return str(path)
            return f"ERROR: Could not download paper {paper_id}."
        except Exception as e:
            return f"ERROR: Download failed for {paper_id}: {e}"


# ╔══════════════════════════════════════════════╗
# ║  TOOL 5: Store in Vector DB                 ║
# ╚══════════════════════════════════════════════╝


class StoreInVectorDBInput(BaseModel):
    paper_id: str = Field(..., description="Unique identifier for the paper.")
    text: str = Field(..., description="Full text content of the paper.")
    title: str = Field(default="", description="Title of the paper.")
    authors: str = Field(default="", description="Comma-separated author names.")


class StoreInVectorDBTool(BaseTool):
    name: str = "store_in_vectordb"
    description: str = (
        "Stores a paper's text content in the ChromaDB vector database, "
        "chunked and embedded for similarity search. "
        "Returns a confirmation message."
    )
    args_schema: Type[BaseModel] = StoreInVectorDBInput

    def _run(
        self, paper_id: str, text: str, title: str = "", authors: str = ""
    ) -> str:
        try:
            vs = get_vector_store()
            metadata = {"title": title, "authors": authors}
            vs.add_paper(paper_id, text, metadata)
            return f"Successfully stored paper '{paper_id}' in vector database."
        except Exception as e:
            return f"ERROR: Failed to store paper: {e}"


# ╔══════════════════════════════════════════════╗
# ║  TOOL 6: Query Vector DB                    ║
# ╚══════════════════════════════════════════════╝


class QueryVectorDBInput(BaseModel):
    query: str = Field(..., description="Search query for the vector database.")
    n_results: int = Field(default=5, description="Number of results to return.")


class QueryVectorDBTool(BaseTool):
    name: str = "query_vectordb"
    description: str = (
        "Queries the ChromaDB vector database for paper chunks "
        "most relevant to a given query. Returns matching text chunks "
        "with metadata and similarity distances."
    )
    args_schema: Type[BaseModel] = QueryVectorDBInput

    def _run(self, query: str, n_results: int = 5) -> str:
        try:
            vs = get_vector_store()
            results = vs.query(query, n_results=n_results)
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            return f"ERROR: Vector DB query failed: {e}"


# ╔══════════════════════════════════════════════╗
# ║  TOOL 7: Plagiarism Check                   ║
# ╚══════════════════════════════════════════════╝


class PlagiarismCheckInput(BaseModel):
    generated_text: str = Field(
        ..., description="The generated report text to check."
    )
    source_texts: str = Field(
        ...,
        description=(
            "JSON array of source text strings to check against. "
            "Each element is a source document string."
        ),
    )


class PlagiarismCheckTool(BaseTool):
    name: str = "check_plagiarism"
    description: str = (
        "Checks a generated report for semantic overlap with source material. "
        "Flags sentences with ≥70% similarity and provides rewrite instructions. "
        "Returns flagged sentences and rewrite guidance."
    )
    args_schema: Type[BaseModel] = PlagiarismCheckInput

    def _run(self, generated_text: str, source_texts: str) -> str:
        try:
            guard = get_plagiarism_guard()
            sources = json.loads(source_texts)
            flagged = guard.check_overlap(generated_text, sources)
            if not flagged:
                return "✅ No sentences exceed the semantic overlap threshold. Report is clean."
            rewrite_instructions = guard.get_rewrite_instructions(flagged)
            return (
                f"⚠️ {len(flagged)} sentence(s) flagged for high overlap.\n\n"
                f"{rewrite_instructions}"
            )
        except json.JSONDecodeError:
            return "ERROR: source_texts must be a valid JSON array of strings."
        except Exception as e:
            return f"ERROR: Plagiarism check failed: {e}"


# ╔══════════════════════════════════════════════╗
# ║  TOOL 8: Citation Formatter                 ║
# ╚══════════════════════════════════════════════╝


class CitationInput(BaseModel):
    paper_metadata_json: str = Field(
        ...,
        description=(
            "JSON string of paper metadata with 'authors' (list) "
            "and 'published' (ISO date string) fields."
        ),
    )


class CitationFormatterTool(BaseTool):
    name: str = "format_citation"
    description: str = (
        "Formats a paper's metadata into an in-text citation like "
        "[Smith et al., 2024]. Input is a JSON string of paper metadata."
    )
    args_schema: Type[BaseModel] = CitationInput

    def _run(self, paper_metadata_json: str) -> str:
        try:
            meta = json.loads(paper_metadata_json)
            return format_citation(meta)
        except Exception as e:
            return f"ERROR: Citation formatting failed: {e}"
