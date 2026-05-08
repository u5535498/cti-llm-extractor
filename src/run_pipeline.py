"""
run_pipeline.py — Orchestrates V0 through V3 extraction across all documents.

Loads the GGUF model once, then runs each active variant in sequence.
Writes results to outputs/v0/, outputs/v1/, outputs/v2/, outputs/v3/.

Usage
-----
    # Run all four variants on the dev set
    python -m src.run_pipeline \\
        --reports  raw_reports/ \\
        --variants v0 v1 v2 v3 \\
        --model    models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf

    # Run only V1 on a single document
    python -m src.run_pipeline \\
        --reports  raw_reports/DOC001.pdf \\
        --variants v1 \\
        --model    models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
"""

import argparse
import json
import logging
import time
from pathlib import Path

from llama_cpp import Llama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.utils_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
console = Console()

VARIANT_MAP = {
    "v0": "src.v0_extractor.V0Extractor",
    "v1": "src.v1_baseline_extractor.BaselineExtractor",
    "v2": "src.v2_evidence_grounder.EvidenceGrounder",
    "v3": "src.v3_cove_verifier.CoVeVerifier",
}

# Phrases that indicate the model was cut off mid-generation due to the
# context ceiling rather than a logic error in the extraction code.
_CONTEXT_OVERFLOW_MARKERS = (
    "json repair failed after four strategies",
    "context window",
    "context overflow",
    "max_tokens",
)


def _is_context_overflow(exc: Exception) -> bool:
    """
    Return True when an exception is most likely caused by the LLM running
    out of context tokens rather than a code-level bug.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONTEXT_OVERFLOW_MARKERS)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_path: str, n_ctx: int = 32768, verbose: bool = False) -> Llama:
    """
    Load the GGUF model with full GPU offload.

    Parameters
    ----------
    model_path : path to the .gguf file
    n_ctx      : context window size.  32768 is the default, sized to fit
                 the full prompt budget comfortably:
                   ~600  system message + schema
                   ~500  few-shot example
                   ~7500 document text (MAX_DOC_CHARS = 42_000 chars)
                   ~2048 generated JSON output
                 means ~10648 total
    verbose    : if True, llama.cpp prints layer-by-layer loading info
    """
    console.print(f"[bold cyan]Loading model:[/bold cyan] {model_path}")
    t0  = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        n_batch=512,
        verbose=verbose,
    )
    console.print(f"[green]Model loaded in {time.time() - t0:.1f}s[/green]")
    return llm


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def _import_extractor(variant: str):
    """Dynamically import and return the extractor class for a variant."""
    import importlib
    module_path, class_name = VARIANT_MAP[variant].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_report_paths(reports_arg: str) -> list:
    """
    Resolve --reports argument to a list of Path objects.
    Accepts either a directory (all PDFs/HTMLs) or a single file.
    """
    p = Path(reports_arg)
    if p.is_dir():
        paths = sorted(list(p.glob("*.pdf")) + list(p.glob("*.html")))
        if not paths:
            raise FileNotFoundError(f"No PDF or HTML files found in {p}")
        return paths
    elif p.is_file():
        return [p]
    else:
        raise FileNotFoundError(f"--reports path not found: {p}")


def run_variant(
    variant: str,
    llm: Llama,
    report_paths: list,
    output_root: Path,
):
    """
    Run one variant across all report_paths.
    V0 outputs are raw dicts; V1-V3 outputs are CTIDocument JSON.

    On failure the exception is caught and a sentinel JSON is written:
      {"doc_id": "...", "error": "...", "context_overflow": true/false}

    The `context_overflow` flag is set when the error message matches known
    llama.cpp truncation patterns.
    """
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    ExtractorClass = _import_extractor(variant)
    extractor      = ExtractorClass(llm)

    console.print(f"\n[bold]Running {variant.upper()}[/bold] → {output_dir}")
    successes       = 0
    failures        = 0
    ctx_overflows   = 0
    t0              = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"{variant.upper()}", total=len(report_paths))

        for report_path in report_paths:
            progress.update(task, description=f"{variant.upper()} — {report_path.name}")
            output_path = output_dir / f"{report_path.stem}.json"

            try:
                result = extractor.run_extraction(report_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    if hasattr(result, "model_dump_json"):
                        f.write(result.model_dump_json(indent=2))
                    else:
                        json.dump(result, f, indent=2)
                successes += 1

            except Exception as e:
                overflow = _is_context_overflow(e)
                if overflow:
                    logger.warning(
                        f"[{variant}] Context overflow on {report_path.name} — "
                        f"document exceeds available token budget. "
                        f"Writing sentinel and continuing."
                    )
                    ctx_overflows += 1
                else:
                    logger.error(
                        f"[{variant}] Failed on {report_path.name}: {e}",
                        exc_info=True,
                    )
                failures += 1
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "doc_id":           report_path.stem,
                            "error":            str(e),
                            "context_overflow": overflow,
                        },
                        f,
                        indent=2,
                    )

            progress.advance(task)

    elapsed = time.time() - t0
    overflow_note = (
        f", {ctx_overflows} context overflow(s)" if ctx_overflows else ""
    )
    console.print(
        f"[bold]{variant.upper()}[/bold] complete: "
        f"{successes} succeeded, {failures} failed{overflow_note}, "
        f"{elapsed:.1f}s total ({elapsed / len(report_paths):.1f}s/doc)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run V0-V3 CTI extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reports", required=True,
        help="Path to directory of PDF/HTML reports, or a single report file.",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=["v0", "v1", "v2", "v3"], default=["v1"],
        help="Which variants to run.",
    )
    parser.add_argument(
        "--model", default="models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Root directory for all variant outputs.",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=32768,
        help="Context window size passed to llama-cpp-python.",
    )
    parser.add_argument(
        "--verbose-model", action="store_true",
        help="Print llama.cpp layer loading output.",
    )
    args = parser.parse_args()

    report_paths = _get_report_paths(args.reports)
    console.print(f"Found [bold]{len(report_paths)}[/bold] report(s).")

    llm = load_model(args.model, n_ctx=args.n_ctx, verbose=args.verbose_model)

    for variant in args.variants:
        run_variant(variant, llm, report_paths, Path(args.output_dir))

    console.print("\n[bold green]All variants complete.[/bold green]")


if __name__ == "__main__":
    main()
