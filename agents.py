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
    """
    Create an LLM instance based on the configured backend.

    Groq (default): Free cloud API, runs Llama 3 70B, Mixtral, etc.
    Ollama (fallback): Local inference, requires Ollama installed.
    """
    if config.LLM_BACKEND == "groq":
        # Ensure API key is set in environment for LiteLLM
        if config.GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY

        return LLM(
            model=f"groq/{config.GROQ_MODEL}",
            temperature=config.GROQ_TEMPERATURE,
        )
    else:
        # Ollama (local)
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
        role="Research Query Planner",
        goal=(
            "Analyze a research paper's title and abstract to produce exactly 5 "
            "highly targeted, diverse search queries for arXiv that will surface "
            "the most relevant related work. Queries should cover: "
            "(1) core methodology, (2) specific problem domain, "
            "(3) alternative approaches, (4) foundational concepts, "
            "(5) recent advances in the field."
        ),
        backstory=(
            "You are a seasoned research strategist with a PhD in information "
            "retrieval and 15 years of experience in academic research. You have "
            "an exceptional ability to decompose a research paper's contributions "
            "into orthogonal search dimensions. You understand arXiv search syntax "
            "and know how to craft queries that balance precision and recall. "
            "Your queries consistently uncover hidden gems that other researchers miss."
        ),
        llm=_get_llm(),
        tools=[ExtractMetadataTool(), ParsePDFTool()],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 2: SEARCHER                          ║
# ╚══════════════════════════════════════════════╝


def create_searcher_agent() -> Agent:
    return Agent(
        role="Academic Paper Retrieval Specialist",
        goal=(
            "Execute the provided search queries on arXiv to find and download "
            "the most relevant related papers. For each query, retrieve up to "
            f"{config.ARXIV_MAX_RESULTS_PER_QUERY} papers. Download the PDFs, "
            "parse them into text, and store them in the vector database for "
            "later retrieval. Avoid duplicates. Return a structured summary of "
            "all retrieved papers with their metadata."
        ),
        backstory=(
            "You are a digital librarian and data engineer specializing in "
            "academic paper retrieval systems. You have built search pipelines "
            "for top university research labs. You are meticulous about "
            "rate-limiting API calls, deduplicating results, and maintaining "
            "clean metadata. You always verify that downloads complete "
            "successfully before moving on."
        ),
        llm=_get_llm(),
        tools=[
            ArxivSearchTool(),
            ArxivDownloadTool(),
            ParsePDFTool(),
            StoreInVectorDBTool(),
        ],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=15,
        memory=True,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 3: EXTRACTION                        ║
# ╚══════════════════════════════════════════════╝


def create_extraction_agent() -> Agent:
    return Agent(
        role="Research Methodology Extraction Specialist",
        goal=(
            "For each retrieved paper, extract and organize the key information "
            "into a structured Markdown format. Focus on: \n"
            "- Core methodology and techniques used\n"
            "- Key findings and results\n"
            "- Datasets and evaluation metrics\n"
            "- Limitations acknowledged by the authors\n"
            "- How it relates to or differs from the target paper\n\n"
            "Use the vector database to retrieve relevant chunks and "
            "synthesize comprehensive extraction notes."
        ),
        backstory=(
            "You are a research analyst with expertise in systematic literature "
            "reviews. You have conducted meta-analyses for Cochrane Reviews and "
            "top-tier AI conferences. You excel at distilling complex papers into "
            "their essential components while preserving technical accuracy. "
            "You always note the specific page or section where key information "
            "was found for traceability."
        ),
        llm=_get_llm(),
        tools=[ParsePDFTool(), QueryVectorDBTool(), CitationFormatterTool()],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=10,
        memory=True,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 4: CRITIC                            ║
# ╚══════════════════════════════════════════════╝


def create_critic_agent() -> Agent:
    return Agent(
        role="Senior Research Paper Critic & Reviewer",
        goal=(
            "Perform a rigorous critical evaluation of the target paper by "
            "comparing it against the retrieved related papers. Evaluate on "
            "exactly 7 dimensions, providing a score (1-10) and detailed "
            "justification for each:\n\n"
            "1. **Originality** – How novel are the contributions?\n"
            "2. **Importance** – How significant is the problem being solved?\n"
            "3. **Claim Support** – Are the claims well-supported by evidence?\n"
            "4. **Experimental Soundness** – Is the experimental design robust?\n"
            "5. **Clarity** – Is the paper well-written and well-organized?\n"
            "6. **Community Value** – How useful is this to the research community?\n"
            "7. **Contextualization** – How well does it position itself within "
            "existing literature?\n\n"
            "Also provide an overall assessment and key strengths/weaknesses."
        ),
        backstory=(
            "You are a distinguished professor who has served as Area Chair for "
            "NeurIPS, ICML, and ACL. You have reviewed over 500 papers and are "
            "known for your fair, thorough, and constructive reviews. You always "
            "back your evaluations with specific evidence from the paper and "
            "relevant context from the broader literature. You avoid superficial "
            "praise and focus on substantive analysis."
        ),
        llm=_get_llm(),
        tools=[QueryVectorDBTool(), CitationFormatterTool()],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=8,
        memory=True,
    )


# ╔══════════════════════════════════════════════╗
# ║  AGENT 5: WRITER                            ║
# ╚══════════════════════════════════════════════╝


def create_writer_agent() -> Agent:
    return Agent(
        role="Academic Research Synthesis Writer",
        goal=(
            f"Write a {config.TARGET_REPORT_WORDS}-word Research Synthesis Report "
            "that brings together the target paper's analysis, the critic's "
            "evaluation, and the extracted methodologies from related work. "
            "The report MUST:\n\n"
            "1. Use a professional academic tone throughout\n"
            "2. Include proper in-text citations (e.g., [Smith et al., 2024])\n"
            "3. Be structured with clear sections: Introduction, Methodology "
            "Comparison, Critical Analysis, Synthesis & Positioning, and Conclusion\n"
            "4. Synthesize information – do NOT copy-paste from sources\n"
            "5. Highlight gaps in the literature that the target paper addresses\n"
            "6. Provide actionable recommendations for the authors\n\n"
            "CRITICAL: Every claim must be attributed to a source. "
            "The report must pass a plagiarism check."
        ),
        backstory=(
            "You are a Pulitzer-nominated science writer who also holds a PhD "
            "in computational linguistics. You have written for Nature, Science, "
            "and The AI Index Report. You pride yourself on making complex "
            "research accessible while maintaining rigorous academic standards. "
            "You never plagiarize – you always synthesize ideas in your own words "
            "while properly crediting sources. Your writing is clear, compelling, "
            "and citation-rich."
        ),
        llm=_get_llm(),
        tools=[
            QueryVectorDBTool(),
            CitationFormatterTool(),
            PlagiarismCheckTool(),
        ],
        verbose=config.VERBOSE,
        allow_delegation=False,
        max_iter=10,
        memory=True,
    )
