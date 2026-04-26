"""
main.py – Entry point for the Autonomous Literature Synthesizer.

Usage:
    python main.py <path_to_pdf>
    python main.py --help

This orchestrator:
  1. Parses the input PDF
  2. Builds a 5-agent CrewAI crew
  3. Runs the sequential pipeline
  4. Saves the final report to data/output/
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from crewai import Crew, Process

import config
from tasks import (
    create_planning_task,
    create_searching_task,
    create_extraction_task,
    create_critique_task,
    create_writing_task,
)
from utils import (
    extract_title_and_abstract,
    parse_pdf_to_markdown,
    word_count,
    logger,
)

console = Console()


# ──────────────────────────────────────────────
# CLI Argument Parsing
# ──────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Research Synthesizer",
        description=(
            "🔬 Autonomous Literature Synthesizer – "
            "An agentic workflow that critiques a "
            "research paper and finds related work, all locally."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py paper.pdf\n"
            "  python main.py paper.pdf --model mistral\n"
            "  python main.py paper.pdf --output my_report.md\n"
        ),
    )
    parser.add_argument(
        "pdf",
        type=str,
        help="Path to the target research paper (PDF).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file name for the synthesis report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=config.VERBOSE,
        help="Enable verbose agent logging.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=config.ARXIV_MAX_RESULTS_PER_QUERY,
        help="Max arXiv results per query.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Display Functions
# ──────────────────────────────────────────────


def display_banner() -> None:
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║   🔬  AUTONOMOUS LITERATURE SYNTHESIZER                  ║
    ║   ─────────────────────────────────────────────────────   ║
    ║   Stanford Agentic Reviewer × Local LLMs                 ║
    ║   CrewAI · Ollama · ChromaDB · arXiv                     ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan", border_style="bright_cyan"))


def display_config_table() -> None:
    table = Table(title="⚙️  Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("LLM Model", f"ollama/{config.OLLAMA_MODEL}")
    table.add_row("LLM Temperature", str(config.OLLAMA_TEMPERATURE))
    table.add_row("Embedding Model", config.EMBEDDING_MODEL_NAME)
    table.add_row("ArXiv Max Results/Query", str(config.ARXIV_MAX_RESULTS_PER_QUERY))
    table.add_row("Overlap Threshold", f"{config.SEMANTIC_OVERLAP_THRESHOLD:.0%}")
    table.add_row("Target Report Words", str(config.TARGET_REPORT_WORDS))
    table.add_row("Output Directory", str(config.OUTPUT_DIR))
    console.print(table)
    console.print()


def display_paper_info(metadata: dict) -> None:
    console.print(
        Panel(
            f"[bold]{metadata['title']}[/bold]\n\n"
            f"{metadata['abstract'][:500]}{'…' if len(metadata['abstract']) > 500 else ''}",
            title="📄 Target Paper",
            border_style="yellow",
        )
    )


def display_results(output: str, elapsed: float) -> None:
    console.print("\n")
    console.print(
        Panel(
            f"[bold green]Pipeline completed successfully![/bold green]\n\n"
            f"⏱️  Total time: {elapsed:.1f} seconds\n"
            f"📝 Report word count: ~{word_count(str(output))} words\n"
            f"📁 Report saved to: {config.OUTPUT_DIR / 'synthesis_report.md'}",
            title="✅ Results",
            border_style="green",
        )
    )


# ──────────────────────────────────────────────
# Pre-Flight Checks
# ──────────────────────────────────────────────


def preflight_check(pdf_path: Path) -> None:
    """Validate inputs before running the pipeline."""
    errors = []

    # Check PDF exists
    if not pdf_path.exists():
        errors.append(f"PDF not found: {pdf_path}")
    elif pdf_path.suffix.lower() != ".pdf":
        errors.append(f"File is not a PDF: {pdf_path}")

    # Check Ollama is running
    try:
        import urllib.request

        req = urllib.request.Request(f"{config.OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                errors.append(
                    f"Ollama returned status {resp.status}. "
                    "Ensure Ollama is running: `ollama serve`"
                )
    except Exception:
        errors.append(
            f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
            "Start it with: `ollama serve`\n"
            f"Then pull the model: `ollama pull {config.OLLAMA_MODEL}`"
        )

    if errors:
        console.print(Panel("\n".join(f"❌ {e}" for e in errors), title="Pre-flight Check Failed", border_style="red"))
        sys.exit(1)

    console.print("[green]✅ Pre-flight checks passed[/green]\n")


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────


def run_pipeline(pdf_path: Path, output_name: str | None = None) -> str:
    """
    Execute the full multi-agent literature synthesis pipeline.

    Args:
        pdf_path: Path to the target PDF.
        output_name: Optional custom output filename.

    Returns:
        The final synthesis report as a string.
    """
    start_time = time.time()

    # ── Step 1: Parse the target paper ──
    console.print("[bold cyan]📄 Step 1/5: Parsing target paper...[/bold cyan]")
    paper_markdown = parse_pdf_to_markdown(pdf_path)
    paper_metadata = extract_title_and_abstract(paper_markdown)
    display_paper_info(paper_metadata)

    # Store target paper in vector DB for cross-referencing
    from utils import VectorStore

    vs = VectorStore()
    vs.add_paper(
        paper_id="target_paper",
        text=paper_markdown,
        metadata={"title": paper_metadata["title"], "is_target": "true"},
    )

    # ── Step 2: Create Tasks (sequential dependencies) ──
    console.print("\n[bold cyan]🏗️  Step 2/5: Building agent squad...[/bold cyan]")

    planning_task = create_planning_task(paper_markdown, paper_metadata)
    searching_task = create_searching_task(planning_task)
    extraction_task = create_extraction_task(searching_task)
    critique_task = create_critique_task(paper_markdown, paper_metadata, extraction_task)
    writing_task = create_writing_task(paper_metadata, critique_task, extraction_task)

    # ── Step 3: Assemble the crew ──
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

    # ── Step 4: Kick off ──
    console.print("\n[bold cyan]🚀 Step 3/5: Launching agent crew...[/bold cyan]")
    console.print(
        "[dim]This may take several minutes depending on your hardware "
        "and the paper length.[/dim]\n"
    )

    result = crew.kickoff()

    # ── Step 5: Save and display results ──
    console.print("\n[bold cyan]💾 Step 5/5: Saving report...[/bold cyan]")

    report_text = str(result)

    # Save with custom name if provided
    if output_name:
        output_path = config.OUTPUT_DIR / output_name
        output_path.write_text(report_text, encoding="utf-8")
        logger.info(f"📁 Report saved to: {output_path}")

    # Also save a timestamped copy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = config.OUTPUT_DIR / f"synthesis_report_{timestamp}.md"
    timestamped_path.write_text(report_text, encoding="utf-8")

    elapsed = time.time() - start_time
    display_results(report_text, elapsed)

    return report_text


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────


def main() -> None:
    display_banner()

    args = parse_args()

    # Apply CLI overrides
    if args.model:
        config.OLLAMA_MODEL = args.model
    config.VERBOSE = args.verbose
    config.ARXIV_MAX_RESULTS_PER_QUERY = args.max_results

    display_config_table()

    pdf_path = Path(args.pdf).resolve()
    preflight_check(pdf_path)

    try:
        report = run_pipeline(pdf_path, output_name=args.output)
        console.print("\n[bold green]🎉 Done! Your synthesis report is ready.[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Pipeline interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Pipeline failed:[/bold red]\n\n{e}",
                title="💥 Error",
                border_style="red",
            )
        )
        logger.exception("Pipeline failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
