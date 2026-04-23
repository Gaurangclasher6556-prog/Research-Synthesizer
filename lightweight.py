"""
lightweight.py – Lightweight 2-call pipeline for the Autonomous Literature Synthesizer.

Bypasses CrewAI entirely. Uses only 2 direct LiteLLM API calls:
  Call 1: Generate 3 arXiv search queries from the paper
  Call 2: Generate the full synthesis report from paper + found papers

This is designed for free-tier LLM APIs (Gemini, Groq) where rate limits
make the 5-agent CrewAI pipeline impractical.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import litellm

import config
from utils import (
    ArxivSearcher,
    VectorStore,
    extract_title_and_abstract,
    format_citation,
    truncate_text,
    word_count,
    logger,
)


def _get_model_name() -> str:
    """Return the LiteLLM-compatible model name."""
    if config.GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = config.GEMINI_API_KEY
        model = config.GEMINI_MODEL
        if not model.startswith("gemini/"):
            model = f"gemini/{model}"
        return model
    if config.GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
        return f"groq/{config.GROQ_MODEL}"
    return f"ollama/{config.OLLAMA_MODEL}"


def _llm_call(model: str, prompt: str, max_tokens: int = 4096) -> str:
    """Make a single LLM call with built-in retry and backoff."""
    for attempt in range(8):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=180,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            is_rate_limit = any(p in err for p in [
                "rate_limit", "RateLimit", "429", "ResourceExhausted",
                "quota", "Too Many Requests", "RATE_LIMIT",
            ])
            if is_rate_limit and attempt < 7:
                wait = 30 + (attempt * 20)  # 30s, 50s, 70s, 90s...
                # Try to parse server-suggested wait time
                match = re.search(r"in (\d+\.?\d*)\s*s", err)
                if match:
                    wait = float(match.group(1)) + 5
                wait = min(wait, 120)
                logger.warning(f"⏳ Rate limit (attempt {attempt+1}/8) – waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("LLM call failed after 8 retries.")


def run_lightweight_pipeline(
    paper_markdown: str,
    paper_metadata: dict,
    add_log=None,
) -> str:
    """
    Run the full synthesis pipeline using only 2 LLM calls.

    Args:
        paper_markdown: Parsed markdown text of the target paper.
        paper_metadata: Dict with 'title' and 'abstract' keys.
        add_log: Optional callback for progress logging.

    Returns:
        The synthesis report as a markdown string.
    """
    def log(msg):
        if add_log:
            add_log(msg)
        logger.info(msg)

    model = _get_model_name()
    title = paper_metadata.get("title", "Unknown")
    abstract = paper_metadata.get("abstract", "")[:1500]

    # ════════════════════════════════════════════
    # CALL 1: Generate arXiv search queries
    # ════════════════════════════════════════════
    log("🔍 Generating search queries...")

    query_prompt = f"""You are a research assistant. Given this paper, generate exactly 3 arXiv search queries to find related work.

**Paper Title:** {title}

**Abstract:** {abstract}

Return ONLY a JSON array of 3 query strings. Example:
["transformer attention mechanisms", "large language model training", "neural network scaling laws"]

Return ONLY the JSON array, no other text."""

    queries_raw = _llm_call(model, query_prompt, max_tokens=500)
    log(f"✓ Query generation complete")

    # Parse queries from LLM response
    try:
        # Try to extract JSON array from the response
        json_match = re.search(r'\[.*?\]', queries_raw, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group())
        else:
            # Fallback: split by newlines and clean up
            queries = [q.strip().strip('"\'').strip() for q in queries_raw.strip().split('\n') if q.strip()]
    except (json.JSONDecodeError, Exception):
        # Ultimate fallback: use title keywords
        queries = [title, abstract[:100]]

    queries = queries[:3]  # Cap at 3
    log(f"📋 Queries: {queries}")

    # ════════════════════════════════════════════
    # ARXIV SEARCH (no LLM needed)
    # ════════════════════════════════════════════
    log("🔎 Searching arXiv...")
    searcher = ArxivSearcher()
    all_papers = []
    seen_ids = set()

    for i, query in enumerate(queries):
        try:
            results = searcher.search(query, max_results=config.ARXIV_MAX_RESULTS_PER_QUERY)
            for paper in results:
                pid = paper.get("id", "")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(paper)
            log(f"   Query {i+1}: '{query[:50]}' → {len(results)} results")
        except Exception as e:
            log(f"   Query {i+1} failed: {str(e)[:80]}")

    log(f"✓ Found {len(all_papers)} unique papers")

    # ════════════════════════════════════════════
    # BUILD CONTEXT from found papers
    # ════════════════════════════════════════════
    related_work_text = ""
    citations = []
    for i, paper in enumerate(all_papers[:8]):  # Cap at 8 papers
        citation = format_citation(paper)
        citations.append(citation)
        related_work_text += f"""
### Paper {i+1}: {paper.get('title', 'Unknown')} {citation}
**Authors:** {', '.join(paper.get('authors', [])[:3])}
**Year:** {paper.get('published', '')[:4]}
**Abstract:** {paper.get('abstract', '')[:400]}
---
"""

    if not related_work_text:
        related_work_text = "(No related papers found on arXiv. Provide analysis based on the paper alone.)"

    # ════════════════════════════════════════════
    # CALL 2: Generate the full synthesis report
    # ════════════════════════════════════════════
    log("📝 Generating synthesis report...")

    # Wait a bit before the second call to avoid rate limits
    time.sleep(5)

    target_words = config.TARGET_REPORT_WORDS
    report_prompt = f"""You are a senior academic reviewer. Write a comprehensive {target_words}-word Research Synthesis Report for the following paper, using the related work provided.

═══ TARGET PAPER ═══
**Title:** {title}
**Abstract:** {abstract}

═══ RELATED WORK FOUND ON ARXIV ═══
{related_work_text}

═══ INSTRUCTIONS ═══
Write a polished Markdown report with these sections:

# Research Synthesis Report

**Target Paper:** {title}
**Date:** {datetime.now().strftime('%B %d, %Y')}
**Papers Analyzed:** {len(all_papers)}

## 1. Introduction (~200 words)
Present the target paper and its main contributions in the broader research context.

## 2. Related Work Comparison (~200 words)
Compare the target paper with the related work found. Highlight similarities and differences in approach.

## 3. Critical Analysis (~200 words)
Evaluate the paper on: Originality (1-10), Importance (1-10), Evidence Quality (1-10), Clarity (1-10), Novelty (1-10).
Give a score and brief justification for each.

## 4. Synthesis & Positioning (~150 words)
Place the paper within the broader research landscape.

## 5. Conclusion (~50 words)
Final verdict and recommendations.

## References
List all cited papers in proper academic format.

REQUIREMENTS:
- Use in-text citations like {citations[0] if citations else '[Author et al., Year]'} throughout
- Write in your OWN words – do NOT copy from abstracts
- Be specific and evidence-based in your analysis
- Target word count: {target_words} words (±20%)"""

    report = _llm_call(model, report_prompt, max_tokens=4096)
    log(f"✓ Report generated – {word_count(report)} words")

    return report
