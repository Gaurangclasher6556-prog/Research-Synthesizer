"""
agents.py – Agent definitions for the Autonomous Literature Synthesizer.

Supports two LLM backends:
  - Groq (cloud, free) – default, no install needed
  - Ollama (local) – fallback for offline use

Five specialized agents:
  1. Planner Agent       → Generates search queries from the target paper
  2. Searcher Agent      → Executes arXiv queries, downloads & stores papers
  3. Extraction Agent    → Parses related papers, extracts methodologies
  4. Critic Agent        → 7-dimension evaluation of the target paper
  5. Writer Agent        → Synthesizes the final research report with citations
"""

from __future__ import annotations

import os
import time
from crewai import Agent, LLM

import config
from tools import (
    ArxivDownloadTool,
    ArxivSearchTool,
    CitationFormatterTool,
    ExtractMetadataTool,
    ParsePDFTool,
    PlagiarismCheckTool,
    QueryVectorDBTool,
    StoreInVectorDBTool,
)


def _get_llm() -> LLM:
    """Create LLM – prefers Gemini (1M TPM free), falls back to Groq, then Ollama."""
    # Priority 1: Google Gemini (1,000,000 TPM on free tier – best choice)
    if config.GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = config.GEMINI_API_KEY
        # Prefix with "gemini/" so LiteLLM routes to Google AI Studio (API key)
        # instead of Vertex AI (which requires GCE metadata credentials).
        model_name = config.GEMINI_MODEL
        if not model_name.startswith("gemini/"):
            model_name = f"gemini/{model_name}"
        return LLM(
            model=model_name,
            temperature=config.GEMINI_TEMPERATURE,
            max_retries=3,
            timeout=120,
        )
    # Priority 2: Groq (very rate-limited on free tier – 6k TPM)
    if config.LLM_BACKEND == "groq" and config.GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
        return LLM(
            model=f"groq/{config.GROQ_MODEL}",
            temperature=config.GROQ_TEMPERATURE,
            max_retries=6,
            timeout=120,
        )
    # Priority 3: Ollama (local)
    return LLM(
        model=f"ollama/{config.OLLAMA_MODEL}",
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.OLLAMA_TEMPERATURE,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 1: PLANNER                           ║
# ╚══════════════════════════════════════════════╝


def create_planner_agent() -> Agent:
    return Agent(
        role="Query Planner",
        goal="Produce 3 arXiv search queries from the paper abstract.",
        backstory="Expert research strategist specializing in literature search.",
        llm=_get_llm(),
        tools=[ExtractMetadataTool()],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=3,
        memory=False,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 2: SEARCHER                          ║
# ╚══════════════════════════════════════════════╝


def create_searcher_agent() -> Agent:
    return Agent(
        role="Paper Searcher",
        goal="Search arXiv with the given queries and return paper titles and abstracts.",
        backstory="Academic librarian skilled in systematic literature search.",
        llm=_get_llm(),
        tools=[ArxivSearchTool()],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=5,
        memory=False,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 3: EXTRACTION                        ║
# ╚══════════════════════════════════════════════╝


def create_extraction_agent() -> Agent:
    return Agent(
        role="Methodology Extractor",
        goal="From the found papers, extract: methodology, results, and limitations in bullet points.",
        backstory="Research analyst specializing in systematic reviews.",
        llm=_get_llm(),
        tools=[QueryVectorDBTool()],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=4,
        memory=False,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 4: CRITIC                            ║
# ╚══════════════════════════════════════════════╝


def create_critic_agent() -> Agent:
    return Agent(
        role="Paper Critic",
        goal="Score the paper 1-10 on: Originality, Importance, Evidence, Clarity, Novelty. Give one sentence per dimension.",
        backstory="Senior NeurIPS reviewer with expertise in rigorous evaluation.",
        llm=_get_llm(),
        tools=[],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=3,
        memory=False,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 5: WRITER                            ║
# ╚══════════════════════════════════════════════╝


def create_writer_agent() -> Agent:
    return Agent(
        role="Synthesis Writer",
        goal="Write a 600-word academic synthesis report with sections: Overview, Related Work, Critique, Conclusion.",
        backstory="Science writer with PhD in computational linguistics.",
        llm=_get_llm(),
        tools=[],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=4,
        memory=False,
    )
