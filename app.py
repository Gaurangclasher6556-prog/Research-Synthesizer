"""
app.py – Streamlit Frontend for the Autonomous Literature Synthesizer.

Launch: streamlit run app.py
Design: MIT/Harvard dark academia aesthetic
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

# ══════════════════════════════════════════════
# Streamlit Cloud ChromaDB Fix (SQLite3 version issue)
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ══════════════════════════════════════════════

try:
    import config
    from utils import (
        VectorStore,
        extract_title_and_abstract,
        parse_pdf_to_markdown,
        truncate_text,
        word_count,
        logger,
    )
except Exception as e:
    st.error(f"Failed to boot! Error during imports: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

# ══════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="Research Synthesizer · Autonomous Literature Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# Academic Dark Theme – MIT/Harvard Inspired CSS
# ══════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,600&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ═══ GLOBAL ═══ */
    .stApp {
        background: #0d1117;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    #MainMenu, footer { visibility: hidden; }

    header[data-testid="stHeader"] {
        background: rgba(13,17,23,0.92) !important;
        backdrop-filter: blur(24px) saturate(180%);
        border-bottom: 1px solid rgba(163,31,52,0.15);
    }

    /* ═══ SIDEBAR ═══ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #131820 100%) !important;
        border-right: 1px solid rgba(163,31,52,0.12);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #8b949e;
        font-size: 14px;
    }

    /* ═══ ANIMATED BG ═══ */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background:
            radial-gradient(ellipse 70% 50% at 15% 15%, rgba(163,31,52,0.06), transparent),
            radial-gradient(ellipse 50% 70% at 85% 85%, rgba(197,165,90,0.04), transparent);
        pointer-events: none;
        z-index: 0;
    }

    /* ═══ TYPOGRAPHY ═══ */
    .stMarkdown h1 {
        font-family: 'Crimson Pro', 'Georgia', serif !important;
        color: #f0f0f0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    .stMarkdown h2 {
        font-family: 'Crimson Pro', 'Georgia', serif !important;
        color: #e6e6e6 !important;
        font-weight: 600 !important;
    }
    .stMarkdown h3 {
        font-family: 'Inter', sans-serif !important;
        color: #d0d0d0 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .stMarkdown p { color: #8b949e; line-height: 1.7; }
    .stMarkdown a { color: #C5A55A !important; }

    /* ═══ BUTTONS ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #A31F34 0%, #c42a47 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 2px 12px rgba(163,31,52,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(163,31,52,0.4) !important;
        background: linear-gradient(135deg, #b8243e 0%, #d43454 100%) !important;
    }

    /* ═══ DOWNLOAD BUTTONS ═══ */
    .stDownloadButton > button {
        background: transparent !important;
        color: #C5A55A !important;
        border: 1px solid rgba(197,165,90,0.3) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(197,165,90,0.08) !important;
        border-color: rgba(197,165,90,0.6) !important;
    }

    /* ═══ FILE UPLOADER ═══ */
    .stFileUploader {
        border: 2px dashed rgba(163,31,52,0.2) !important;
        border-radius: 12px !important;
        background: rgba(13,17,23,0.6) !important;
    }
    .stFileUploader:hover {
        border-color: rgba(163,31,52,0.45) !important;
        box-shadow: 0 0 40px rgba(163,31,52,0.08);
    }
    .stFileUploader label { color: #C5A55A !important; font-weight: 500 !important; }

    /* ═══ SELECTBOX ═══ */
    .stSelectbox > div > div {
        background: #161b22 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
    }

    /* ═══ SLIDER ═══ */
    .stSlider > div > div > div {
        color: #C5A55A !important;
    }

    /* ═══ METRICS ═══ */
    [data-testid="stMetric"] {
        background: rgba(13,17,23,0.7);
        border: 1px solid rgba(163,31,52,0.1);
        border-radius: 12px;
        padding: 20px;
        border-left: 3px solid #A31F34;
    }
    [data-testid="stMetricValue"] {
        color: #C5A55A !important;
        font-family: 'Crimson Pro', serif !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6e7681 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
    }

    /* ═══ EXPANDERS ═══ */
    .streamlit-expanderHeader {
        background: rgba(22,27,34,0.8) !important;
        border: 1px solid rgba(163,31,52,0.1) !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
        font-weight: 600 !important;
    }

    /* ═══ PROGRESS ═══ */
    .stProgress > div > div {
        background: linear-gradient(90deg, #A31F34, #C5A55A) !important;
        border-radius: 4px;
    }
    .stProgress > div {
        background: rgba(22,27,34,0.8) !important;
        border-radius: 4px;
    }

    /* ═══ TABS ═══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(22,27,34,0.6);
        border-radius: 8px;
        padding: 3px;
        border: 1px solid rgba(163,31,52,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        color: #8b949e !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #A31F34 !important;
        color: white !important;
    }

    /* ═══ CODE ═══ */
    .stCodeBlock, pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stCodeBlock {
        background: rgba(0,0,0,0.35) !important;
        border: 1px solid rgba(163,31,52,0.08) !important;
        border-radius: 8px !important;
    }

    /* ═══ DIVIDERS ═══ */
    hr {
        border: none;
        border-top: 1px solid rgba(163,31,52,0.12);
        margin: 32px 0;
    }

    /* ═══ CUSTOM COMPONENTS ═══ */
    .inst-header {
        text-align: center;
        padding: 48px 0 32px;
    }
    .inst-crest {
        font-size: 48px;
        margin-bottom: 12px;
    }
    .inst-name {
        font-family: 'Crimson Pro', 'Georgia', serif;
        font-size: clamp(32px, 4.5vw, 52px);
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.08;
        color: #f0f0f0;
        margin-bottom: 4px;
    }
    .inst-dept {
        font-family: 'Crimson Pro', 'Georgia', serif;
        font-size: 20px;
        font-weight: 400;
        font-style: italic;
        color: #C5A55A;
        margin-bottom: 16px;
    }
    .inst-desc {
        font-size: 15px;
        color: #8b949e;
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.7;
    }
    .inst-rule {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #A31F34, transparent);
        margin: 24px auto;
    }
    .inst-badges {
        display: flex;
        justify-content: center;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 16px;
    }
    .badge {
        font-size: 11px;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 4px;
        background: rgba(163,31,52,0.08);
        color: #C5A55A;
        border: 1px solid rgba(163,31,52,0.15);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ═══ PIPELINE STEPS ═══ */
    .pipeline-step {
        padding: 14px 18px;
        margin: 6px 0;
        border-radius: 8px;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .step-pending {
        background: rgba(22,27,34,0.5);
        border: 1px solid rgba(255,255,255,0.04);
        color: #484f58;
    }
    .step-active {
        background: rgba(163,31,52,0.08);
        border: 1px solid rgba(163,31,52,0.25);
        color: #c9d1d9;
        animation: activePulse 2.5s ease-in-out infinite;
    }
    @keyframes activePulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(163,31,52,0.1); }
        50% { box-shadow: 0 0 20px 0 rgba(163,31,52,0.15); }
    }
    .step-done {
        background: rgba(46,160,67,0.06);
        border: 1px solid rgba(46,160,67,0.15);
        color: #7ee787;
    }
    .step-icon { font-size: 18px; flex-shrink: 0; }
    .step-text { flex: 1; font-weight: 500; }
    .step-status {
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ═══ REPORT STYLES ═══ */
    .report-frame {
        background: rgba(13,17,23,0.85);
        border: 1px solid rgba(163,31,52,0.1);
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
    }
    .report-header {
        background: linear-gradient(135deg, rgba(163,31,52,0.06), rgba(197,165,90,0.04));
        border-bottom: 1px solid rgba(163,31,52,0.1);
        padding: 20px 28px;
    }
    .report-header-title {
        font-family: 'Crimson Pro', serif;
        font-size: 18px;
        font-weight: 700;
        color: #f0f0f0;
    }
    .report-header-meta {
        font-size: 12px;
        color: #6e7681;
        margin-top: 4px;
    }
    .report-body {
        padding: 28px;
        font-size: 15px;
        line-height: 1.85;
        color: #c9d1d9;
    }
    .report-body h1 {
        font-family: 'Crimson Pro', serif;
        font-size: 26px;
        font-weight: 800;
        color: #f0f0f0;
        margin: 28px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(163,31,52,0.15);
    }
    .report-body h2 {
        font-family: 'Crimson Pro', serif;
        font-size: 21px;
        font-weight: 700;
        color: #C5A55A;
        margin: 24px 0 10px;
    }
    .report-body h3 {
        font-size: 16px;
        font-weight: 600;
        color: #c9d1d9;
        margin: 18px 0 8px;
    }
    .report-body p { margin-bottom: 14px; }
    .report-body strong { color: #f0f0f0; }
    .report-body blockquote {
        border-left: 3px solid #A31F34;
        padding-left: 16px;
        color: #8b949e;
        font-style: italic;
    }
    .report-body code {
        background: rgba(163,31,52,0.08);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 13px;
        color: #C5A55A;
    }
    .report-body ul, .report-body ol { padding-left: 24px; }
    .report-body li { margin-bottom: 4px; color: #8b949e; }

    /* ═══ SIDEBAR LOGO ═══ */
    .sidebar-logo {
        text-align: center;
        padding: 20px 0 8px;
    }
    .sidebar-logo-icon { font-size: 36px; }
    .sidebar-logo-text {
        font-family: 'Crimson Pro', serif;
        font-size: 17px;
        font-weight: 700;
        color: #f0f0f0;
        letter-spacing: -0.01em;
        margin-top: 4px;
    }
    .sidebar-logo-sub {
        font-size: 11px;
        color: #C5A55A;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════

defaults = {
    "pipeline_running": False,
    "pipeline_step": 0,
    "pipeline_logs": [],
    "report": "",
    "paper_metadata": {},
    "pipeline_complete": False,
    "pipeline_error": "",
    "elapsed_time": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════


def _load_groq_key() -> str:
    """Load Groq API key from: Streamlit secrets > .env > manual input."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    key = config.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    return key


def _load_gemini_key() -> str:
    """Load Gemini API key from: Streamlit secrets > .env > manual input."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")


def check_groq_key() -> bool:
    key = _load_groq_key()
    return bool(key and len(key) > 10)


def add_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.pipeline_logs.append(f"[{ts}] {msg}")


def get_past_reports() -> list[dict]:
    reports = []
    if config.OUTPUT_DIR.exists():
        for f in sorted(config.OUTPUT_DIR.glob("*.md"), reverse=True):
            reports.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_mtime),
                "path": f,
            })
    return reports


def run_full_pipeline(pdf_path: Path, progress_bar, status_text) -> str:
    start_time = time.time()

    try:
        # Step 1: Parse PDF
        st.session_state.pipeline_step = 1
        status_text.markdown("**Step 1 of 5** — Parsing target paper with PDF engine...")
        progress_bar.progress(10)
        add_log("📄 Parsing PDF...")

        paper_markdown = parse_pdf_to_markdown(pdf_path)
        
        # 🛡️ TRUNCATE LARGE PAPERS to prevent Groq RateLimitError 
        # Groq slashed limits to 6000 tokens/min!
        # 1200 words ≈ 1600 tokens (Leaves room for agent thoughts + context)
        paper_markdown = truncate_text(paper_markdown, max_words=1200)
        
        paper_metadata = extract_title_and_abstract(paper_markdown)
        st.session_state.paper_metadata = paper_metadata
        add_log(f"✓ Parsed: {paper_metadata.get('title', 'Unknown')[:80]}")

        # Step 2: Vector DB
        st.session_state.pipeline_step = 2
        status_text.markdown("**Step 2 of 5** — Embedding and storing in vector database...")
        progress_bar.progress(25)
        add_log("🗄️ Storing in ChromaDB...")

        vs = VectorStore()
        vs.add_paper("target_paper", paper_markdown,
                      {"title": paper_metadata["title"], "is_target": "true"})
        add_log("✓ Paper embedded and stored")

        # Step 3: Agent Crew
        st.session_state.pipeline_step = 3
        status_text.markdown("**Step 3 of 5** — Running 5-agent research crew... *(this takes several minutes)*")
        progress_bar.progress(40)
        add_log("🤖 Launching CrewAI crew...")
        add_log("   · Planner → generating arXiv queries")
        add_log("   · Searcher → finding related papers")
        add_log("   · Extractor → extracting methodologies")
        add_log("   · Critic → 7-dimension evaluation")
        add_log("   · Writer → synthesizing report")

        from crewai import Crew, Process
        from tasks import (
            create_planning_task, create_searching_task,
            create_extraction_task, create_critique_task, create_writing_task,
        )

        planning_task = create_planning_task(paper_markdown, paper_metadata)
        searching_task = create_searching_task(planning_task)
        extraction_task = create_extraction_task(searching_task)
        critique_task = create_critique_task(paper_markdown, paper_metadata, extraction_task)
        writing_task = create_writing_task(paper_metadata, critique_task, extraction_task)

        crew = Crew(
            agents=[planning_task.agent, searching_task.agent, extraction_task.agent,
                    critique_task.agent, writing_task.agent],
            tasks=[planning_task, searching_task, extraction_task,
                   critique_task, writing_task],
            process=Process.sequential,
            verbose=config.VERBOSE,
            memory=False,       # Disable crew memory to save tokens
            full_output=True,
            max_rpm=1,          # Limit to 1 request/minute to safely pace free-tier
        )

        # ── Retry loop: auto-wait on Groq rate-limit errors ──────────────
        import re as _re
        result = None
        for _attempt in range(6):
            try:
                result = crew.kickoff()
                break
            except Exception as _e:
                err_str = str(_e)
                if "rate_limit_exceeded" in err_str or "RateLimitError" in err_str:
                    # Parse wait time from error message e.g. "try again in 36.06s"
                    _wait_match = _re.search(r"in (\d+\.?\d*)\s*s", err_str)
                    _wait = float(_wait_match.group(1)) + 5 if _wait_match else 65
                    _wait = min(_wait, 90)  # cap at 90s
                    add_log(f"⏳ Rate limit hit – waiting {_wait:.0f}s then retrying (attempt {_attempt+1}/6)...")
                    time.sleep(_wait)
                else:
                    raise
        if result is None:
            raise RuntimeError("Pipeline failed after 6 retries due to persistent rate limits.")

        report = str(result)
        add_log("✓ Agent crew completed")

        # Step 4: Save
        st.session_state.pipeline_step = 4
        status_text.markdown("**Step 4 of 5** — Saving report...")
        progress_bar.progress(90)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = config.OUTPUT_DIR / f"synthesis_report_{timestamp}.md"
        report_path.write_text(report, encoding="utf-8")
        add_log(f"💾 Saved: {report_path.name}")

        # Step 5: Done
        st.session_state.pipeline_step = 5
        progress_bar.progress(100)
        elapsed = time.time() - start_time
        st.session_state.elapsed_time = round(elapsed, 1)
        add_log(f"✓ Complete — {word_count(report)} words in {elapsed:.1f}s")

        return report

    except Exception as e:
        add_log(f"✗ Error: {str(e)}")
        raise


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🏛️</div>
        <div class="sidebar-logo-text">Research Synthesizer</div>
        <div class="sidebar-logo-sub">Autonomous Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # API Key section – Gemini preferred (1M TPM free), Groq as fallback
    st.markdown("### 🔑 AI Engine")

    gemini_secret = _load_gemini_key()
    groq_secret = _load_groq_key()

    if gemini_secret:
        config.GEMINI_API_KEY = gemini_secret
        os.environ["GEMINI_API_KEY"] = gemini_secret
        has_key = True
        st.success("✓ Gemini AI ready (1M TPM)", icon="✅")
    elif groq_secret:
        config.GROQ_API_KEY = groq_secret
        os.environ["GROQ_API_KEY"] = groq_secret
        has_key = True
        st.success("✓ Groq AI ready", icon="✅")
    else:
        # Manual entry — prefer Gemini
        st.info(
            "**Recommended:** Use Google Gemini (1M tokens/min free).\n\n"
            "Get a free key → [aistudio.google.com](https://aistudio.google.com/apikey)",
            icon="🌟"
        )
        gemini_key = st.text_input(
            "Gemini API Key (Recommended)",
            value="", type="password",
            help="Free from aistudio.google.com — 1,000,000 tokens/min!",
        )
        groq_key_input = st.text_input(
            "Groq API Key (Fallback)",
            value="", type="password",
            help="Free from console.groq.com — but very rate-limited (6k TPM)",
        )
        if gemini_key:
            config.GEMINI_API_KEY = gemini_key
            os.environ["GEMINI_API_KEY"] = gemini_key
            has_key = True
            st.success("✓ Gemini key set!", icon="✅")
        elif groq_key_input:
            config.GROQ_API_KEY = groq_key_input
            os.environ["GROQ_API_KEY"] = groq_key_input
            has_key = bool(groq_key_input and len(groq_key_input) > 10)
            if has_key:
                st.warning("⚠️ Groq has very low rate limits. Gemini is recommended.", icon="⚠️")
        else:
            has_key = False

    st.divider()

    # Model Selection — Gemini models first
    st.markdown("### ⚙️ Model & Parameters")
    using_gemini = bool(config.GEMINI_API_KEY)
    if using_gemini:
        model_options = {
            "gemini-2.0-flash": "✨ Gemini 2.0 Flash (Recommended)",
            "gemini-1.5-flash": "Gemini 1.5 Flash (Stable)",
            "gemini-1.5-pro": "Gemini 1.5 Pro (High Quality)",
        }
        selected_model = st.selectbox(
            "Language Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
        )
        config.GEMINI_MODEL = selected_model
    else:
        model_options = {
            "llama-3.1-8b-instant": "Llama 3.1 8B (Fastest)",
            "llama-3.3-70b-versatile": "Llama 3.3 70B (Best Quality)",
            "gemma2-9b-it": "Gemma 2 9B (Google)",
        }
        selected_model = st.selectbox(
            "Language Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
        )
        config.GROQ_MODEL = selected_model

    max_results = st.slider("ArXiv papers per query", 2, 10, config.ARXIV_MAX_RESULTS_PER_QUERY)
    config.ARXIV_MAX_RESULTS_PER_QUERY = max_results

    overlap = st.slider("Plagiarism threshold", 0.50, 0.95, config.SEMANTIC_OVERLAP_THRESHOLD, 0.05)
    config.SEMANTIC_OVERLAP_THRESHOLD = overlap

    st.divider()

    # Past Reports
    reports = get_past_reports()
    st.markdown(f"### 📚 Report Archive ({len(reports)})")
    if reports:
        for r in reports[:8]:
            size_str = f"{r['size']/1024:.0f}KB"
            st.caption(f"📄 {r['filename']}  ·  {size_str}  ·  {r['created'].strftime('%b %d')}")
    else:
        st.caption("No reports yet")

    st.divider()
    st.markdown("""
    <div style="text-align:center; font-size:11px; color:#484f58; padding:4px 0;">
        Powered by CrewAI · Groq · ChromaDB<br>
        Free & Open Source
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════

# ── Institution-Style Header ──
st.markdown("""
<div class="inst-header">
    <div class="inst-crest">🏛️</div>
    <div class="inst-name">Research Synthesizer</div>
    <div class="inst-dept">Autonomous Literature Analysis Platform</div>
    <div class="inst-rule"></div>
    <div class="inst-desc">
        Upload a research paper. Our five-agent AI system will find related work on arXiv,
        perform a rigorous seven-dimension critique, and produce a comprehensive
        synthesis report with proper scholarly citations.
    </div>
    <div class="inst-badges">
        <span class="badge">Multi-Agent AI</span>
        <span class="badge">arXiv Integration</span>
        <span class="badge">7-Dimension Critique</span>
        <span class="badge">Plagiarism Guard</span>
        <span class="badge">Free & Cloud-Based</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats ──
stat_cols = st.columns(4)
with stat_cols[0]:
    st.metric("Model", model_options.get(config.GROQ_MODEL, config.GROQ_MODEL).split("(")[0].strip())
with stat_cols[1]:
    st.metric("Agent Crew", "5 Agents")
with stat_cols[2]:
    st.metric("Reports", str(len(reports)))
with stat_cols[3]:
    st.metric("Overlap Guard", f"{config.SEMANTIC_OVERLAP_THRESHOLD:.0%}")

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════

if not st.session_state.pipeline_running and not st.session_state.pipeline_complete:

    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        st.markdown("### Submit Paper for Analysis")
        uploaded_file = st.file_uploader(
            "Upload research paper (PDF)",
            type=["pdf"],
            help="PDF will be parsed, analyzed, critiqued, and synthesized automatically.",
        )

        if uploaded_file:
            st.markdown(f"""
            <div style="background:rgba(163,31,52,0.06); border:1px solid rgba(163,31,52,0.12);
                        border-radius:8px; padding:14px 18px; margin:8px 0;">
                <strong style="color:#f0f0f0;">📎 {uploaded_file.name}</strong>
                <span style="color:#6e7681; margin-left:12px;">
                    {uploaded_file.size/1024:.0f} KB · {model_options.get(selected_model, selected_model).split("(")[0].strip()}
                </span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▸ Begin Analysis", use_container_width=True, type="primary"):
                if not has_key:
                    st.error("Please enter your Groq API key in the sidebar.")
                else:
                    upload_path = config.UPLOAD_DIR / uploaded_file.name
                    upload_path.write_bytes(uploaded_file.getvalue())
                    st.session_state.pipeline_running = True
                    st.session_state.pipeline_logs = []
                    st.session_state.pipeline_step = 0
                    st.session_state.report = ""
                    st.session_state.pipeline_complete = False
                    st.session_state.pipeline_error = ""
                    st.session_state.upload_path = str(upload_path)
                    st.rerun()

    with col_info:
        st.markdown("### How It Works")
        st.markdown("""
        <div class="pipeline-step step-pending">
            <span class="step-icon">①</span>
            <span class="step-text">PDF parsed to structured text</span>
        </div>
        <div class="pipeline-step step-pending">
            <span class="step-icon">②</span>
            <span class="step-text">Embedded in vector database</span>
        </div>
        <div class="pipeline-step step-pending">
            <span class="step-icon">③</span>
            <span class="step-text">5 AI agents analyze & critique</span>
        </div>
        <div class="pipeline-step step-pending">
            <span class="step-icon">④</span>
            <span class="step-text">Plagiarism check & citation pass</span>
        </div>
        <div class="pipeline-step step-pending">
            <span class="step-icon">⑤</span>
            <span class="step-text">1,500-word synthesis report</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════

if st.session_state.pipeline_running:
    st.markdown("### Agent Pipeline — In Progress")

    steps_data = [
        ("📄", "Parse PDF", "Extracting text from document"),
        ("🗄️", "Vector Storage", "Embedding into ChromaDB"),
        ("🤖", "Agent Crew", "5-agent sequential analysis"),
        ("💾", "Save Report", "Persisting results"),
        ("✓", "Complete", "Report ready"),
    ]

    for i, (icon, name, desc) in enumerate(steps_data):
        step = i + 1
        if step < st.session_state.pipeline_step:
            cls = "step-done"
            status = '<span class="step-status" style="background:rgba(46,160,67,0.1);color:#7ee787;">Done</span>'
        elif step == st.session_state.pipeline_step:
            cls = "step-active"
            status = '<span class="step-status" style="background:rgba(163,31,52,0.1);color:#ff7b72;">Running</span>'
        else:
            cls = "step-pending"
            status = '<span class="step-status" style="color:#484f58;">Pending</span>'

        st.markdown(f"""
        <div class="pipeline-step {cls}">
            <span class="step-icon">{icon}</span>
            <span class="step-text">{name} <span style="font-weight:400;color:#6e7681;">— {desc}</span></span>
            {status}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        report = run_full_pipeline(Path(st.session_state.upload_path), progress_bar, status_text)
        st.session_state.report = report
        st.session_state.pipeline_complete = True
        st.session_state.pipeline_running = False

        with st.expander("📋 Pipeline Log", expanded=False):
            for log in st.session_state.pipeline_logs:
                st.text(log)

        st.balloons()
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.session_state.pipeline_error = str(e)
        st.session_state.pipeline_running = False
        status_text.error(f"Pipeline failed: {e}")
        with st.expander("📋 Error Log", expanded=True):
            for log in st.session_state.pipeline_logs:
                st.text(log)
            st.code(traceback.format_exc())
        if st.button("↻ Retry"):
            st.session_state.pipeline_complete = False
            st.session_state.pipeline_error = ""
            st.rerun()


# ══════════════════════════════════════════════
# REPORT DISPLAY
# ══════════════════════════════════════════════

if st.session_state.pipeline_complete and st.session_state.report:
    st.markdown("<hr>", unsafe_allow_html=True)

    meta = st.session_state.paper_metadata
    title = meta.get("title", "Research Paper")
    wc = word_count(st.session_state.report)
    elapsed = st.session_state.elapsed_time

    # Header
    hdr_cols = st.columns([4, 1, 1])
    with hdr_cols[0]:
        st.markdown(f"### Synthesis Report")
        st.caption(f"**{title[:100]}** · {wc} words · {elapsed}s · {model_options.get(config.GROQ_MODEL, '').split('(')[0].strip()}")
    with hdr_cols[1]:
        st.download_button(
            "⬇ Download",
            data=st.session_state.report,
            file_name=f"synthesis_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )
    with hdr_cols[2]:
        if st.button("✦ New Analysis"):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

    # Report in tabs
    tab_report, tab_raw, tab_log = st.tabs(["📖 Report", "📄 Source", "📋 Log"])

    with tab_report:
        st.markdown(st.session_state.report)

    with tab_raw:
        st.code(st.session_state.report, language="markdown")

    with tab_log:
        for log in st.session_state.pipeline_logs:
            st.text(log)


# ══════════════════════════════════════════════
# HISTORY (idle state)
# ══════════════════════════════════════════════

if not st.session_state.pipeline_running and not st.session_state.pipeline_complete:
    if reports:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📚 Report Archive")
        for r in reports[:5]:
            with st.expander(f"📄 {r['filename']}  ·  {r['created'].strftime('%B %d, %Y at %H:%M')}"):
                content = r["path"].read_text(encoding="utf-8")
                st.markdown(content)
                st.download_button("⬇ Download", data=content, file_name=r["filename"],
                                   mime="text/markdown", key=f"dl_{r['filename']}")
