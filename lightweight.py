"""
lightweight.py – RAG-augmented pipeline with multi-provider failover.

Uses only 2 direct LLM calls with automatic provider fallback:
  Gemini → Groq → OpenRouter → Ollama

  Call 1: Generate 3 arXiv search queries from the paper
  Call 2: Generate the full RAG-augmented synthesis report

If one provider is rate-limited (429), the system automatically
falls back to the next available provider — no manual intervention.
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
import httpx
from utils import (
    ArxivSearcher,
    VectorStore,
    extract_title_and_abstract,
    format_citation,
    truncate_text,
    word_count,
    logger,
)


# ════════════════════════════════════════════
# MULTI-PROVIDER LLM LAYER
# ════════════════════════════════════════════


def _ollama_is_reachable() -> bool:
    """Quick health‑check for an Ollama server.
    Returns True if we can get a 200 response from the base URL.
    """
    try:
        resp = httpx.get(config.OLLAMA_BASE_URL, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _get_providers() -> list[dict]:
    """Return ordered list of available LLM providers.
    Ollama is only added when a reachable server is detected – this avoids the
    "Cannot assign requested address" error in cloud environments.
    """
    providers = []

    if config.GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = config.GEMINI_API_KEY
        model = config.GEMINI_MODEL
        if not model.startswith("gemini/"):
            model = f"gemini/{model}"
        providers.append({"name": "Gemini", "model": model})

    if config.GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
        providers.append({"name": "Groq", "model": f"groq/{config.GROQ_MODEL}"})

    if config.OPENROUTER_API_KEY:
        os.environ["OPENROUTER_API_KEY"] = config.OPENROUTER_API_KEY
        providers.append({"name": "OpenRouter", "model": f"openrouter/{config.OPENROUTER_MODEL}"})

    # Only add Ollama if a local server is reachable (e.g., running on your dev machine)
    if _ollama_is_reachable():
        providers.append({"name": "Ollama", "model": f"ollama/{config.OLLAMA_MODEL}"})
    else:
        # In cloud deployments we just skip Ollama – the error message will be clearer.
        pass
    return providers


def _llm_call(prompt: str, max_tokens: int = 4096, add_log=None) -> str:
    """Make an LLM call with automatic multi-provider fallback.

    If a provider hits its daily quota (limit: 0), skip immediately.
    If a provider hits per-minute limits, retry with backoff up to 5 times.
    """
    providers = _get_providers()
    last_error = None

    def log(msg):
        if add_log:
            add_log(msg)
        logger.info(msg)

    for provider in providers:
        model_name = provider["model"]
        pname = provider["name"]

        for attempt in range(5):
            try:
                log(f"🔗 {pname} (attempt {attempt + 1})...")
                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    timeout=180,
                )
                result = response.choices[0].message.content.strip()
                log(f"✓ {pname} responded ({len(result)} chars)")
                return result
            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                is_rate = any(k in err_lower for k in [
                    "rate_limit", "ratelimit", "429", "resourceexhausted",
                    "quota", "too many requests", "resource_exhausted",
                ])
                if is_rate:
                    # Daily quota completely gone → skip provider
                    if "limit: 0" in err_str or "perdayperproject" in err_lower.replace(" ", "").replace("_", ""):
                        log(f"⛔ {pname} daily quota exhausted → next provider")
                        last_error = e
                        break
                    if attempt < 4:
                        wait = 30 + (attempt * 15)
                        match = re.search(r"in (\d+\.?\d*)\s*s", err_str)
                        if match:
                            wait = float(match.group(1)) + 5
                        wait = min(wait, 90)
                        log(f"⏳ {pname} rate limit – waiting {wait:.0f}s...")
                        time.sleep(wait)
                    else:
                        log(f"⛔ {pname} retries exhausted → next provider")
                        last_error = e
                        break
                else:
                    log(f"❌ {pname} error: {err_str[:150]}")
                    last_error = e
                    break

    raise RuntimeError(
        f"All LLM providers failed. Last error: {last_error}\n\n"
        "💡 Ensure you have at least one valid provider key set (Gemini, Groq, or OpenRouter).\n"
        "   • Gemini: https://aistudio.google.com/apikey\n"
        "   • OpenRouter: https://openrouter.ai/keys\n"
        "   • Groq: https://console.groq.com\n"
        "If running locally and you want to use Ollama, start the Ollama server and ensure \n"
        "`config.OLLAMA_BASE_URL` points to it (default http://localhost:11434)."
    )


# ════════════════════════════════════════════
# RAG HELPERS
# ════════════════════════════════════════════


def _store_papers_in_vectordb(papers: list[dict], vs: VectorStore) -> int:
    """Store found papers' abstracts in ChromaDB for RAG retrieval."""
    stored = 0
    for paper in papers:
        pid = paper.get("id", "unknown")
        abstract = paper.get("abstract", "")
        title = paper.get("title", "")
        authors = ", ".join(paper.get("authors", [])[:5])
        if not abstract:
            continue
        text = f"Title: {title}\nAuthors: {authors}\n\nAbstract:\n{abstract}"
        try:
            vs.add_paper(pid, text, {"title": title, "authors": authors})
            stored += 1
        except Exception:
            pass
    return stored


def _rag_retrieve(vs: VectorStore, queries: list[str], n_per_query: int = 3) -> str:
    """Query vector DB and format retrieved chunks as context."""
    seen = set()
    chunks = []
    for query in queries:
        try:
            for r in vs.query(query, n_results=n_per_query):
                doc = r.get("document", "").strip()
                doc_key = doc[:200]
                if doc_key in seen or not doc:
                    continue
                seen.add(doc_key)
                chunks.append({
                    "text": doc[:600],
                    "source": r.get("metadata", {}).get("title", "Unknown"),
                    "distance": r.get("distance", 999),
                })
        except Exception:
            pass

    if not chunks:
        return ""

    chunks.sort(key=lambda x: x["distance"])
    parts = ["\n═══ RETRIEVED CONTEXT (RAG) ═══"]
    for i, c in enumerate(chunks[:8], 1):
        parts.append(f"\n[Chunk {i} | {c['source']}]\n{c['text']}")
    return "\n".join(parts)


# ════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════


def run_lightweight_pipeline(
    paper_markdown: str,
    paper_metadata: dict,
    add_log=None,
) -> str:
    """
    2-call RAG pipeline with multi-provider failover.

    Args:
        paper_markdown: Parsed markdown of the target paper.
        paper_metadata: Dict with 'title' and 'abstract'.
        add_log: Optional progress callback.

    Returns:
        Synthesis report as markdown string.
    """
    def log(msg):
        if add_log:
            add_log(msg)
        logger.info(msg)

    title = paper_metadata.get("title", "Unknown")
    abstract = paper_metadata.get("abstract", "")[:1500]

    providers = _get_providers()
    log(f"🔌 Providers: {' → '.join(p['name'] for p in providers)}")

    # ── CALL 1: Generate search queries ──
    log("🔍 Generating search queries...")
    query_prompt = (
        "You are a research assistant. Given this paper, generate exactly 3 "
        "arXiv search queries to find related work.\n\n"
        f"**Paper Title:** {title}\n\n"
        f"**Abstract:** {abstract}\n\n"
        "Return ONLY a JSON array of 3 query strings. Example:\n"
        '["transformer attention mechanisms", "large language model training", '
        '"neural network scaling laws"]\n\n'
        "Return ONLY the JSON array, no other text."
    )

    queries_raw = _llm_call(query_prompt, max_tokens=500, add_log=add_log)
    log("✓ Query generation complete")

    try:
        m = re.search(r'\[.*?\]', queries_raw, re.DOTALL)
        queries = json.loads(m.group()) if m else [title]
    except Exception:
        queries = [title, abstract[:100]]
    queries = queries[:3]
    log(f"📋 Queries: {queries}")

    # ── ARXIV SEARCH (no LLM) ──
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

    # ── RAG: Store & Retrieve ──
    log("🧠 Building RAG knowledge base...")
    rag_context = ""
    try:
        vs = VectorStore()
        stored = _store_papers_in_vectordb(all_papers[:8], vs)
        log(f"📦 Stored {stored} papers in vector DB")

        rag_queries = [
            f"methodology and approach in {title}",
            "key results and contributions",
            "limitations and future work",
            abstract[:200] if abstract else title,
        ]
        rag_context = _rag_retrieve(vs, rag_queries)
        if rag_context:
            log("✓ RAG context retrieved")
        else:
            log("⚠ No RAG context — continuing without")
    except Exception as e:
        log(f"⚠ RAG failed ({str(e)[:80]}) — continuing without")

    # ── Build related work context ──
    related_work_text = ""
    citations = []
    for i, paper in enumerate(all_papers[:8]):
        citation = format_citation(paper)
        citations.append(citation)
        related_work_text += (
            f"\n### Paper {i+1}: {paper.get('title', 'Unknown')} {citation}\n"
            f"**Authors:** {', '.join(paper.get('authors', [])[:3])}\n"
            f"**Year:** {paper.get('published', '')[:4]}\n"
            f"**Abstract:** {paper.get('abstract', '')[:400]}\n---\n"
        )
    if not related_work_text:
        related_work_text = "(No related papers found. Analyze based on the paper alone.)"

    # ── CALL 2: RAG-augmented synthesis report ──
    log("📝 Generating RAG-augmented synthesis report...")
    time.sleep(5)  # Brief pause to avoid back-to-back rate limits

    target_words = config.TARGET_REPORT_WORDS
    cite_example = citations[0] if citations else "[Author et al., Year]"

    report_prompt = (
        f"You are a senior academic reviewer. Write a comprehensive "
        f"{target_words}-word Research Synthesis Report for the following paper, "
        f"using the related work and retrieved context provided.\n\n"
        f"═══ TARGET PAPER ═══\n"
        f"**Title:** {title}\n**Abstract:** {abstract}\n\n"
        f"═══ RELATED WORK ═══\n{related_work_text}\n"
        f"{rag_context}\n\n"
        f"═══ INSTRUCTIONS ═══\n"
        f"Write a polished Markdown report:\n\n"
        f"# Research Synthesis Report\n\n"
        f"**Target Paper:** {title}\n"
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n"
        f"**Papers Analyzed:** {len(all_papers)}\n\n"
        f"## 1. Introduction (~200 words)\n"
        f"Present the paper and contributions in broader context.\n\n"
        f"## 2. Related Work Comparison (~200 words)\n"
        f"Compare with related work. Use retrieved context for specifics.\n\n"
        f"## 3. Critical Analysis (~200 words)\n"
        f"Score: Originality, Importance, Evidence, Clarity, Novelty (1-10 each).\n\n"
        f"## 4. Synthesis & Positioning (~150 words)\n"
        f"Place paper in broader landscape using retrieved evidence.\n\n"
        f"## 5. Conclusion (~50 words)\n"
        f"Final verdict.\n\n"
        f"## References\n"
        f"List all cited papers.\n\n"
        f"REQUIREMENTS:\n"
        f"- Use citations like {cite_example} throughout\n"
        f"- Write in your OWN words\n"
        f"- Ground claims in retrieved context where possible\n"
        f"- Target: {target_words} words (±20%)"
    )

    report = _llm_call(report_prompt, max_tokens=4096, add_log=add_log)
    log(f"✓ Report generated – {word_count(report)} words")
    return report
