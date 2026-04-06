"""
server.py – FastAPI web server for the Autonomous Literature Synthesizer.

Provides:
  - File upload endpoint for PDFs
  - WebSocket for real-time pipeline progress
  - REST API for past reports and system status
  - Static file serving for the frontend

Usage:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from utils import (
    ArxivSearcher,
    PlagiarismGuard,
    VectorStore,
    extract_title_and_abstract,
    parse_pdf_to_markdown,
    word_count,
    logger,
)

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Research Synthesizer",
    description="Autonomous Literature Synthesizer – Andrew Ng–inspired Agentic Workflow",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Upload directory
UPLOAD_DIR = config.DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Active jobs tracking
active_jobs: dict[str, dict[str, Any]] = {}

# WebSocket connections
ws_connections: dict[str, list[WebSocket]] = {}


# ──────────────────────────────────────────────
# WebSocket Manager
# ──────────────────────────────────────────────


async def send_progress(job_id: str, data: dict) -> None:
    """Send progress update to all WebSocket clients for a job."""
    if job_id in ws_connections:
        dead = []
        for ws in ws_connections[job_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_connections[job_id].remove(ws)


# ──────────────────────────────────────────────
# Pipeline Runner (async wrapper)
# ──────────────────────────────────────────────


async def run_pipeline_async(job_id: str, pdf_path: Path, model: str | None = None) -> None:
    """Run the multi-agent pipeline in background with progress updates."""
    job = active_jobs[job_id]

    try:
        # Apply model override
        if model:
            config.OLLAMA_MODEL = model

        # ── Step 1: Parse PDF ──
        job["status"] = "parsing"
        job["step"] = 1
        job["step_name"] = "Parsing PDF"
        await send_progress(job_id, {
            "type": "progress", "step": 1, "total_steps": 5,
            "status": "parsing", "message": "📄 Parsing target paper..."
        })

        paper_markdown = await asyncio.to_thread(parse_pdf_to_markdown, pdf_path)
        paper_metadata = extract_title_and_abstract(paper_markdown)
        job["paper_title"] = paper_metadata.get("title", "Unknown")
        job["paper_abstract"] = paper_metadata.get("abstract", "")

        await send_progress(job_id, {
            "type": "paper_info",
            "title": job["paper_title"],
            "abstract": job["paper_abstract"][:500],
        })

        # ── Step 2: Store in Vector DB ──
        job["status"] = "storing"
        job["step"] = 2
        job["step_name"] = "Storing in Vector DB"
        await send_progress(job_id, {
            "type": "progress", "step": 2, "total_steps": 5,
            "status": "storing", "message": "🗄️ Storing paper in vector database..."
        })

        vs = await asyncio.to_thread(VectorStore)
        await asyncio.to_thread(
            vs.add_paper, "target_paper", paper_markdown,
            {"title": paper_metadata["title"], "is_target": "true"}
        )

        # ── Step 3: Build & Run CrewAI Pipeline ──
        job["status"] = "agents_running"
        job["step"] = 3
        job["step_name"] = "Running Agent Squad"
        await send_progress(job_id, {
            "type": "progress", "step": 3, "total_steps": 5,
            "status": "agents_running",
            "message": "🤖 Launching 5-agent crew... This may take several minutes."
        })

        # Run the CrewAI pipeline in a thread
        report = await asyncio.to_thread(
            _run_crew_pipeline, paper_markdown, paper_metadata, job_id
        )

        # ── Step 4: Save Report ──
        job["status"] = "saving"
        job["step"] = 4
        job["step_name"] = "Saving Report"
        await send_progress(job_id, {
            "type": "progress", "step": 4, "total_steps": 5,
            "status": "saving", "message": "💾 Saving synthesis report..."
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"synthesis_report_{timestamp}.md"
        report_path = config.OUTPUT_DIR / report_filename
        report_path.write_text(report, encoding="utf-8")

        # ── Step 5: Complete ──
        job["status"] = "completed"
        job["step"] = 5
        job["step_name"] = "Complete"
        job["report"] = report
        job["report_file"] = report_filename
        job["completed_at"] = datetime.now().isoformat()
        job["word_count"] = word_count(report)
        job["elapsed"] = time.time() - job["started_at_ts"]

        await send_progress(job_id, {
            "type": "complete",
            "report": report,
            "word_count": job["word_count"],
            "elapsed": round(job["elapsed"], 1),
            "report_file": report_filename,
        })

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
        logger.exception(f"Pipeline failed for job {job_id}")

        await send_progress(job_id, {
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })


def _run_crew_pipeline(paper_markdown: str, paper_metadata: dict, job_id: str) -> str:
    """Synchronous CrewAI pipeline execution (runs in thread)."""
    from crewai import Crew, Process
    from tasks import (
        create_planning_task,
        create_searching_task,
        create_extraction_task,
        create_critique_task,
        create_writing_task,
    )

    planning_task = create_planning_task(paper_markdown, paper_metadata)
    searching_task = create_searching_task(planning_task)
    extraction_task = create_extraction_task(searching_task)
    critique_task = create_critique_task(paper_markdown, paper_metadata, extraction_task)
    writing_task = create_writing_task(paper_metadata, critique_task, extraction_task)

    crew = Crew(
        agents=[
            planning_task.agent,
            searching_task.agent,
            extraction_task.agent,
            critique_task.agent,
            writing_task.agent,
        ],
        tasks=[
            planning_task,
            searching_task,
            extraction_task,
            critique_task,
            writing_task,
        ],
        process=Process.sequential,
        verbose=config.VERBOSE,
        memory=True,
        full_output=True,
    )

    result = crew.kickoff()
    return str(result)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Place index.html in /static/</h1>")


@app.get("/api/status")
async def system_status():
    """System health check and configuration info."""
    # Check Ollama
    ollama_ok = False
    try:
        import urllib.request
        req = urllib.request.Request(f"{config.OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ollama_ok = resp.status == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "ollama": {
            "connected": ollama_ok,
            "url": config.OLLAMA_BASE_URL,
            "model": config.OLLAMA_MODEL,
        },
        "config": {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "arxiv_max_results": config.ARXIV_MAX_RESULTS_PER_QUERY,
            "overlap_threshold": config.SEMANTIC_OVERLAP_THRESHOLD,
            "target_words": config.TARGET_REPORT_WORDS,
        },
        "active_jobs": len([j for j in active_jobs.values() if j["status"] not in ("completed", "failed")]),
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), model: str = Form(default="")):
    """Upload a PDF and start the synthesis pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Please upload a PDF file."}, status_code=400)

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Initialize job
    active_jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "pdf_path": str(upload_path),
        "status": "queued",
        "step": 0,
        "step_name": "Queued",
        "started_at": datetime.now().isoformat(),
        "started_at_ts": time.time(),
        "model": model or config.OLLAMA_MODEL,
    }

    # Launch pipeline in background
    asyncio.create_task(
        run_pipeline_async(job_id, upload_path, model=model or None)
    )

    return {"job_id": job_id, "status": "queued", "filename": file.filename}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = []
    for jid, job in active_jobs.items():
        jobs.append({
            "id": jid,
            "filename": job.get("filename"),
            "status": job.get("status"),
            "step": job.get("step"),
            "step_name": job.get("step_name"),
            "paper_title": job.get("paper_title"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "word_count": job.get("word_count"),
        })
    return {"jobs": sorted(jobs, key=lambda x: x.get("started_at", ""), reverse=True)}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a specific job's status and report."""
    if job_id not in active_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    job = active_jobs[job_id]
    return {
        "id": job_id,
        "filename": job.get("filename"),
        "status": job.get("status"),
        "step": job.get("step"),
        "step_name": job.get("step_name"),
        "paper_title": job.get("paper_title"),
        "paper_abstract": job.get("paper_abstract"),
        "report": job.get("report"),
        "word_count": job.get("word_count"),
        "elapsed": job.get("elapsed"),
        "error": job.get("error"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }


@app.get("/api/reports")
async def list_reports():
    """List all generated reports."""
    reports = []
    for f in sorted(config.OUTPUT_DIR.glob("*.md"), reverse=True):
        reports.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return {"reports": reports}


@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    """Get a specific report's content."""
    report_path = config.OUTPUT_DIR / filename
    if not report_path.exists():
        return JSONResponse({"error": "Report not found"}, status_code=404)

    content = report_path.read_text(encoding="utf-8")

    # Convert markdown to HTML
    try:
        import markdown as md
        html = md.markdown(content, extensions=["tables", "fenced_code", "nl2br"])
    except Exception:
        html = f"<pre>{content}</pre>"

    return {"filename": filename, "markdown": content, "html": html}


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time pipeline progress."""
    await websocket.accept()

    if job_id not in ws_connections:
        ws_connections[job_id] = []
    ws_connections[job_id].append(websocket)

    # Send current job state if exists
    if job_id in active_jobs:
        job = active_jobs[job_id]
        await websocket.send_json({
            "type": "status",
            "status": job.get("status"),
            "step": job.get("step"),
            "step_name": job.get("step_name"),
        })

        # If already completed, send the report
        if job.get("status") == "completed" and job.get("report"):
            await websocket.send_json({
                "type": "complete",
                "report": job["report"],
                "word_count": job.get("word_count"),
                "elapsed": job.get("elapsed"),
            })

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if job_id in ws_connections:
            ws_connections[job_id] = [
                ws for ws in ws_connections[job_id] if ws != websocket
            ]


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  🔬 Research Synthesizer – Web Server")
    print("  http://localhost:8000")
    print("=" * 60 + "\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
