"""
tasks.py – Task definitions for the Autonomous Literature Synthesizer.

Each task maps to a specific step in the Andrew Ng–inspired agentic workflow:
  1. Planning Task      → Generate arXiv search queries
  2. Searching Task     → Execute queries, download & store papers
  3. Extraction Task    → Extract methodologies from related papers
  4. Critique Task      → 7-dimension evaluation of target paper
  5. Writing Task       → Synthesize final research report
  6. Plagiarism Task    → Check and rewrite flagged passages
"""

from __future__ import annotations

from crewai import Task

from agents import (
    create_planner_agent,
    create_searcher_agent,
    create_extraction_agent,
    create_critic_agent,
    create_writer_agent,
)
import config


def create_planning_task(paper_markdown: str, paper_metadata: dict) -> Task:
    """
    Task 1: Analyze the target paper and generate 5 diverse arXiv search queries.
    """
    return Task(
        description=(
            f"You have been given a research paper to analyze.\n\n"
            f"**Paper Title:** {paper_metadata.get('title', 'Unknown')}\n\n"
            f"**Paper Abstract:**\n{paper_metadata.get('abstract', 'Not available')}\n\n"
            f"Your task is to generate exactly 5 diverse, high-quality search "
            f"queries for arXiv to find the most relevant related work. "
            f"Each query should target a different aspect:\n\n"
            f"1. **Core Methodology** – The primary technique or algorithm used\n"
            f"2. **Problem Domain** – The specific problem or application area\n"
            f"3. **Alternative Approaches** – Competing methods for the same problem\n"
            f"4. **Foundational Work** – Seminal papers the work builds upon\n"
            f"5. **Recent Advances** – The latest state-of-the-art in the field\n\n"
            f"Format your output as a numbered list of 5 queries. Each query "
            f"should be specific enough to return relevant results but broad "
            f"enough to capture important related work."
        ),
        expected_output=(
            "A numbered list of exactly 5 arXiv search queries, each on its "
            "own line, formatted as:\n"
            "1. <query text>\n"
            "2. <query text>\n"
            "3. <query text>\n"
            "4. <query text>\n"
            "5. <query text>\n\n"
            "Each query should be a well-crafted search string suitable for "
            "the arXiv API, targeting a distinct aspect of the paper."
        ),
        agent=create_planner_agent(),
    )


def create_searching_task(planning_task: Task) -> Task:
    """
    Task 2: Execute the Planner's queries on arXiv, download papers,
    and store them in ChromaDB.
    """
    return Task(
        description=(
            "You have received 5 search queries from the Research Query Planner. "
            "Your task is to:\n\n"
            "1. Execute each query on arXiv using the `arxiv_search` tool\n"
            "2. For each result, download the PDF using `arxiv_download`\n"
            "3. Parse each downloaded PDF using `parse_pdf`\n"
            "4. Store the parsed text in the vector database using `store_in_vectordb`\n\n"
            "**Important guidelines:**\n"
            "- Skip papers that fail to download or parse – do not crash\n"
            "- Track all unique papers to avoid duplicates across queries\n"
            "- Respect arXiv rate limits (the tools handle this internally)\n"
            f"- Target {config.ARXIV_MAX_RESULTS_PER_QUERY} papers per query\n"
            "- After processing all queries, provide a full summary of all "
            "retrieved papers with their metadata\n\n"
            "Work through the queries one at a time. For each paper found, "
            "go through the full pipeline: search → download → parse → store."
        ),
        expected_output=(
            "A structured Markdown summary of all retrieved papers, formatted as:\n\n"
            "## Retrieved Papers Summary\n\n"
            "**Total papers retrieved:** <N>\n"
            "**Queries executed:** 5\n\n"
            "### Paper 1\n"
            "- **Title:** ...\n"
            "- **Authors:** ...\n"
            "- **Year:** ...\n"
            "- **ArXiv ID:** ...\n"
            "- **Key Topics:** ...\n\n"
            "(repeat for each paper)\n\n"
            "### Retrieval Errors\n"
            "- List any papers that failed to download or parse."
        ),
        agent=create_searcher_agent(),
        context=[planning_task],
    )


def create_extraction_task(searching_task: Task) -> Task:
    """
    Task 3: Parse the related papers and extract key methodologies.
    """
    return Task(
        description=(
            "You have a collection of related papers stored in the vector database. "
            "Your task is to extract structured methodology information from each paper.\n\n"
            "For each paper, use the `query_vectordb` tool to retrieve relevant chunks, "
            "then synthesize the following information:\n\n"
            "1. **Core Methodology** – The primary technique, algorithm, or framework\n"
            "2. **Key Contributions** – What the paper claims as novel\n"
            "3. **Experimental Setup** – Datasets, baselines, evaluation metrics\n"
            "4. **Main Results** – Key quantitative/qualitative findings\n"
            "5. **Limitations** – Acknowledged weaknesses or constraints\n"
            "6. **Relevance** – How it relates to the target paper\n\n"
            "Use the `format_citation` tool to create proper in-text citations "
            "for each paper.\n\n"
            "Produce a clean Markdown document organizing all extracted information."
        ),
        expected_output=(
            "A structured Markdown document with extracted methodologies:\n\n"
            "# Methodology Extraction Report\n\n"
            "## Paper 1: [Title] [Citation]\n"
            "### Methodology\n...\n"
            "### Key Contributions\n...\n"
            "### Experimental Setup\n...\n"
            "### Main Results\n...\n"
            "### Limitations\n...\n"
            "### Relevance to Target Paper\n...\n\n"
            "(repeat for each paper)\n\n"
            "## Cross-Paper Methodology Comparison\n"
            "A brief synthesis comparing approaches across all papers."
        ),
        agent=create_extraction_agent(),
        context=[searching_task],
    )


def create_critique_task(
    paper_markdown: str,
    paper_metadata: dict,
    extraction_task: Task,
) -> Task:
    """
    Task 4: Evaluate the target paper on 7 dimensions.
    """
    title = paper_metadata.get("title", "Unknown")
    abstract = paper_metadata.get("abstract", "")

    return Task(
        description=(
            f"You are reviewing the paper: **{title}**\n\n"
            f"**Abstract:** {abstract[:800]}\n\n"
            "You have access to the methodology extractions from related papers "
            "(provided by the Extraction Agent). Use this context plus the vector "
            "database to perform a rigorous evaluation.\n\n"
            "Evaluate the target paper on exactly 7 dimensions:\n\n"
            "1. **Originality** (1-10) – How novel are the contributions compared "
            "to related work?\n"
            "2. **Importance** (1-10) – How significant is the problem and the "
            "proposed solution?\n"
            "3. **Claim Support** (1-10) – Are claims well-supported by theoretical "
            "analysis or empirical evidence?\n"
            "4. **Experimental Soundness** (1-10) – Is the experimental design "
            "rigorous, reproducible, and fair?\n"
            "5. **Clarity** (1-10) – Is the paper well-structured, clearly written, "
            "and easy to follow?\n"
            "6. **Community Value** (1-10) – Will this paper be useful to other "
            "researchers? Does it release code/data?\n"
            "7. **Contextualization** (1-10) – How well does it position itself "
            "within the existing literature?\n\n"
            "For each dimension, provide:\n"
            "- A numerical score (1-10)\n"
            "- A 2-3 sentence justification with specific evidence\n"
            "- Comparison to at least one related paper where relevant\n\n"
            "End with an Overall Assessment section summarizing key strengths "
            "and weaknesses."
        ),
        expected_output=(
            "A structured evaluation report in Markdown:\n\n"
            "# Critical Evaluation Report\n\n"
            "**Paper:** [Title]\n"
            "**Overall Score:** X.X/10\n\n"
            "## Dimension Scores\n\n"
            "### 1. Originality: X/10\n"
            "[Justification with evidence and comparison]\n\n"
            "(repeat for all 7 dimensions)\n\n"
            "## Overall Assessment\n"
            "### Key Strengths\n- ...\n\n"
            "### Key Weaknesses\n- ...\n\n"
            "### Recommendations for Authors\n- ..."
        ),
        agent=create_critic_agent(),
        context=[extraction_task],
    )


def create_writing_task(
    paper_metadata: dict,
    critique_task: Task,
    extraction_task: Task,
) -> Task:
    """
    Task 5: Synthesize the final 1,500-word Research Synthesis Report.
    """
    title = paper_metadata.get("title", "Unknown")
    target_words = config.TARGET_REPORT_WORDS

    return Task(
        description=(
            f"Write a comprehensive {target_words}-word Research Synthesis Report "
            f"for the paper: **{title}**.\n\n"
            "You have access to:\n"
            "- The Critic's 7-dimension evaluation (from the Critique Task)\n"
            "- Extracted methodologies from related papers (from the Extraction Task)\n"
            "- The vector database for additional context\n\n"
            "**Report Structure:**\n"
            "1. **Introduction** (~300 words) – Present the target paper and "
            "its main contributions in the broader research context\n"
            "2. **Methodology Comparison** (~400 words) – Compare the target paper's "
            "approach with related work, highlighting similarities and differences\n"
            "3. **Critical Analysis** (~350 words) – Summarize the 7-dimension "
            "evaluation with supporting evidence\n"
            "4. **Synthesis & Positioning** (~300 words) – Place the paper within "
            "the broader research landscape and identify gaps it addresses\n"
            "5. **Conclusion** (~150 words) – Final verdict and recommendations\n\n"
            "**MANDATORY REQUIREMENTS:**\n"
            "- Use in-text citations throughout: [Author et al., Year]\n"
            "- Cite at least 5 different sources\n"
            "- Do NOT copy-paste from sources – synthesize in your own words\n"
            "- Use the `format_citation` tool to create proper citations\n"
            "- After writing, use `check_plagiarism` to verify the report passes "
            "the semantic overlap check\n"
            "- If any sentences are flagged (≥70% overlap), rewrite them\n"
            f"- Target word count: {target_words} words (±10%)\n\n"
            "Produce a polished, publication-ready report."
        ),
        expected_output=(
            f"A {target_words}-word Research Synthesis Report in Markdown:\n\n"
            "# Research Synthesis Report\n\n"
            "**Target Paper:** [Title]\n"
            "**Date:** [Current Date]\n\n"
            "## 1. Introduction\n...\n\n"
            "## 2. Methodology Comparison\n...\n\n"
            "## 3. Critical Analysis\n...\n\n"
            "## 4. Synthesis & Positioning\n...\n\n"
            "## 5. Conclusion\n...\n\n"
            "## References\n"
            "- [1] Author, Title, Year, ArXiv ID\n"
            "(list all cited works)"
        ),
        agent=create_writer_agent(),
        context=[critique_task, extraction_task],
        output_file=str(config.OUTPUT_DIR / "synthesis_report.md"),
    )
