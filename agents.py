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
    """Create LLM with automatic retry/backoff on Groq rate-limit errors."""
    if config.LLM_BACKEND == "groq":
        if config.GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
        return LLM(
            model=f"groq/{config.GROQ_MODEL}",
            temperature=config.GROQ_TEMPERATURE,
            max_retries=6,        # retry up to 6× on any error
            timeout=120,
        )
    else:
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
