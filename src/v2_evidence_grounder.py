"""
evidence_grounder.py — V2 post-processing filter.

Applies document-provenance grounding to V1 extraction outputs.
For every atomic extraction that carries an evidence_snippet, the snippet is
matched against the source document text using token-level Jaccard overlap.
Extractions whose snippets cannot be localised in the document above a
configurable similarity threshold are flagged and removed, producing a V2
output with a lower unsupported rate than V1 at the cost of potentially
reduced recall.

Similarity metric
-----------------
Token-level Jaccard overlap is used for all matching.  Content words from
the snippet are compared against a sliding window of comparable length over
the source document text.  The window-maximum Jaccard score is returned as
the similarity value. 

Threshold guidance
------------------
DEFAULT_THRESHOLD = 0.65 is calibrated to pass legitimate paraphrase whilst
rejecting fully hallucinated snippets (fabricated numbers, wrong entity names,
sentences not related to any passage in the source document).

Minimum snippet token guard
---------------------------
MIN_SNIPPET_TOKENS = 5 defines the minimum number of content tokens (after
stop-word removal) that an evidence_snippet must contain before the Jaccard
similarity check is applied.  Snippets shorter than this floor cannot carry
enough signal to distinguish a grounded claim from a hallucination and are
therefore treated as absent.

KV-cache reset
--------------
After the V1 sub-call completes inside run_extraction(), the KV cache is
explicitly reset via self._v1.llm.reset().  This is necessary because
llama-cpp-python retains the KV cache between calls on the same Llama
instance.  With MAX_DOC_CHARS = 42_000 the V1 call consumes ~10,600 tokens;
without a reset the residual context would exhaust the window before the
next document is processed.  

Usage
-----
    from src.evidence_grounder import EvidenceGrounder
    grounder = EvidenceGrounder(threshold=0.65)
    grounded_doc, report = grounder.ground(cti_doc, source_text)

Or batch via CLI:
    python -m src.evidence_grounder \\
        --v1-outputs outputs/v1 \\
        --reports    raw_reports \\
        --output-dir outputs/v2
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_cpp import Llama
from pydantic import ValidationError
from rich.console import Console

from src.document_loader import load_report
from src.schema_models import CTIDocument
from src.utils_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
console = Console()

DEFAULT_THRESHOLD: float = 0.65

# Minimum number of content tokens (after stop-word removal) that an
# evidence_snippet must contain before the Jaccard similarity check is
# applied.  Snippets shorter than this floor are treated as absent.
MIN_SNIPPET_TOKENS: int = 5

# ---------------------------------------------------------------------------
# Stop-words (excluded from token overlap to avoid false positives on common
# function words that appear in virtually every document).
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these",
    "those", "it", "its", "as", "not", "no", "so", "if", "then", "than",
    "also", "such", "into", "about", "which", "who", "use", "used", "using",
})


def _tokenise(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, split on whitespace, remove stop-words.
    Returns a list of content tokens.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t and t not in _STOP_WORDS]


def _is_snippet_too_short(snippet: str) -> bool:
    """
    Return True if the snippet contains fewer than MIN_SNIPPET_TOKENS content
    tokens after stop-word removal.  Such snippets cannot carry enough signal
    for the Jaccard filter and are treated as absent.
    """
    return len(_tokenise(snippet)) < MIN_SNIPPET_TOKENS


# ---------------------------------------------------------------------------
# Core similarity function
# ---------------------------------------------------------------------------

def _snippet_similarity(snippet: str, document: str) -> float:
    """
    Return the highest token-level Jaccard overlap between the content words
    of `snippet` and any same-length sliding window of tokens in `document`.

    Jaccard overlap = |snippet_tokens ∩ window_tokens| / |snippet_tokens ∪ window_tokens|

    Using window_size = len(snippet_tokens) * 2 gives the window enough
    tokens to contain a paraphrase of the snippet whilst remaining specific
    enough to reject random coincidental overlap with unrelated passages.

    Returns 0.0 if snippet is empty or contains no content tokens.
    """
    snippet = snippet.strip()
    if not snippet:
        return 0.0

    # Fast path: exact substring match
    if snippet.lower() in document.lower():
        return 1.0

    snippet_tokens = _tokenise(snippet)
    if not snippet_tokens:
        return 0.0

    doc_tokens = _tokenise(document)
    if not doc_tokens:
        return 0.0

    snippet_set = set(snippet_tokens)
    window_size = max(len(snippet_tokens) * 2, len(snippet_tokens) + 5)
    best = 0.0

    for start in range(0, max(1, len(doc_tokens) - window_size + 1)):
        window_set = set(doc_tokens[start : start + window_size])
        intersection = len(snippet_set & window_set)
        union = len(snippet_set | window_set)
        if union == 0:
            continue
        jaccard = intersection / union
        if jaccard > best:
            best = jaccard
        if best >= 1.0:
            break

    return round(best, 4)


# ---------------------------------------------------------------------------
# Grounding logic applied to Pydantic objects
# ---------------------------------------------------------------------------

def _check_snippet(
    item: Any,
    field: str,
    document: str,
    threshold: float,
) -> Tuple[bool, float]:
    snippet = getattr(item, "evidence_snippet", None) or ""
    if not snippet.strip():
        return False, 0.0
    if _is_snippet_too_short(snippet):
        # Treat sub-threshold-length snippets as absent rather than failed
        return False, 0.0
    sim = _snippet_similarity(snippet, document)
    return sim >= threshold, sim


class GroundingReport:
    """Collects per-document grounding statistics for reporting."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.total_items = 0
        self.removed_items = 0
        self.removals: List[Dict[str, Any]] = []

    def record(self, category: str, value: str, snippet: str, similarity: float):
        self.total_items += 1
        self.removed_items += 1
        self.removals.append({
            "category": category,
            "value": value,
            "evidence_snippet": snippet,
            "similarity": similarity,
        })

    def record_pass(self):
        self.total_items += 1

    @property
    def removal_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return round(self.removed_items / self.total_items, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id":        self.doc_id,
            "total_items":   self.total_items,
            "removed_items": self.removed_items,
            "removal_rate":  self.removal_rate,
            "removals":      self.removals,
        }


# ---------------------------------------------------------------------------
# EvidenceGrounder
# ---------------------------------------------------------------------------

class EvidenceGrounder:
    """
    V2 pipeline: runs V1 extraction then applies evidence-grounding filter.

    Accepts a shared Llama instance so it can be instantiated by run_pipeline.py
    with the same interface as V0Extractor and BaselineExtractor.

    Parameters
    ----------
    llm : Llama
        Shared llama-cpp-python Llama instance.
    threshold : float
        Minimum token-level Jaccard overlap for a snippet to be accepted.
        Default is 0.65 (calibrated for paraphrase robustness).
    strict_mode : bool
        If True, items with empty/null/too-short evidence_snippet are also
        removed.  If False (default), such items are passed through.
    """

    def __init__(
        self,
        llm: Llama,
        threshold: float = DEFAULT_THRESHOLD,
        strict_mode: bool = False,
    ):
        self.llm = llm          # stored for external introspection / reset
        self.threshold = threshold
        self.strict_mode = strict_mode
        from src.v1_baseline_extractor import BaselineExtractor
        self._v1 = BaselineExtractor(llm)
        logger.info(
            f"EvidenceGrounder initialised: threshold={threshold}, "
            f"strict_mode={strict_mode}"
        )

    def _filter_list(
        self,
        items: List[Any],
        category: str,
        value_field: str,
        document: str,
        report: GroundingReport,
    ) -> List[Any]:
        """Filter a list of Pydantic sub-objects by grounding their snippets."""
        kept = []
        for item in items:
            snippet = getattr(item, "evidence_snippet", None) or ""

            # Treat absent or too-short snippets as missing evidence
            if not snippet.strip() or _is_snippet_too_short(snippet):
                if self.strict_mode:
                    value = getattr(item, value_field, str(item))
                    report.record(category, str(value), snippet, 0.0)
                else:
                    report.record_pass()
                    kept.append(item)
                continue

            sim = _snippet_similarity(snippet, document)
            if sim >= self.threshold:
                report.record_pass()
                kept.append(item)
            else:
                value = getattr(item, value_field, str(item))
                report.record(category, str(value), snippet, sim)
                logger.debug(
                    f"[{category}] REMOVED (sim={sim:.3f} < {self.threshold}): "
                    f"{value!r} | snippet: {snippet[:80]!r}"
                )
        return kept

    def ground(
        self,
        doc: CTIDocument,
        source_text: str,
    ) -> Tuple[CTIDocument, GroundingReport]:
        """
        Apply evidence grounding to a CTIDocument.
        Returns a new CTIDocument with unverifiable extractions removed.
        """
        doc_id = (
            doc.report_metadata.report_title
            if doc.report_metadata
            else "unknown"
        )
        report = GroundingReport(doc_id=doc_id)
        iocs = doc.indicators_of_compromise

        doc_dict = doc.model_dump()

        field_map = {
            "ip_addresses": ("ip_address", "value"),
            "domains":      ("domain",     "value"),
            "file_hashes":  ("file_hash",  "value"),
            "urls":         ("url",        "value"),
            "http_paths":   ("http_path",  "value"),
            "cves":         ("cve",        "value"),
            "tools":        ("tool",       "name"),
        }

        for attr, (cat, val_field) in field_map.items():
            original = getattr(iocs, attr, []) or []
            filtered = self._filter_list(
                original, cat, val_field, source_text, report
            )
            doc_dict["indicators_of_compromise"][attr] = [
                item.model_dump() for item in filtered
            ]

        filtered_ttps = self._filter_list(
            doc.ttps or [], "ttp", "technique", source_text, report
        )
        doc_dict["ttps"] = [t.model_dump() for t in filtered_ttps]

        try:
            grounded_doc = CTIDocument.model_validate(doc_dict)
        except ValidationError as e:
            logger.error(f"Re-validation failed after grounding {doc_id}: {e}")
            raise

        logger.info(
            f"Grounded {doc_id}: removed {report.removed_items}/{report.total_items} "
            f"items (removal_rate={report.removal_rate:.3f})"
        )
        return grounded_doc, report

    def run_extraction(self, report_path: Path) -> CTIDocument:
        """
        Full V2 pipeline for a single document.

        Step 1 — V1 extraction: calls BaselineExtractor.run_extraction(),
                  which fills the KV cache with ~10,600 tokens.
        Step 2 — KV-cache reset: calls self._v1.llm.reset() to free the
                  cache before the next document is processed.  The grounding
                  step makes no LLM calls, so this has no effect on output
                  quality.
        Step 3 — Load source text for grounding (pure I/O, no LLM).
        Step 4 — Apply Jaccard-based grounding filter (pure Python, no LLM).

        Returns a CTIDocument (Pydantic model), consistent with the V1
        interface expected by run_pipeline.py.
        """
        console.print(f"[V2] Processing: {report_path.name}")
        t0 = time.time()

        # Step 1: V1 extraction
        v1_result = self._v1.run_extraction(report_path)

        # Step 2: reset KV cache — prevents residual V1 context from
        # exhausting the window on subsequent documents in the batch.
        # The grounding step below makes no LLM calls, so this is safe.
        self._v1.llm.reset()
        logger.debug(f"[V2] KV cache reset after V1 extraction for {report_path.stem}")

        # Step 3: load source text for grounding
        doc_data = load_report(report_path)
        source_text = doc_data["text"]

        # Step 4: apply grounding filter
        grounded_doc, grnd_report = self.ground(v1_result, source_text)

        elapsed = round(time.time() - t0, 2)
        console.print(
            f"[V2] Done in {elapsed}s — removed "
            f"{grnd_report.removed_items}/{grnd_report.total_items} items "
            f"(removal_rate={grnd_report.removal_rate:.3f})"
        )
        logger.info(f"[V2] {report_path.stem} done in {elapsed}s")
        return grounded_doc


# ---------------------------------------------------------------------------
# Standalone batch runner (bypasses run_pipeline.py)
# ---------------------------------------------------------------------------

def batch_ground(
    v1_dir: Path,
    reports_dir: Path,
    output_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    strict_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Apply evidence grounding to all V1 output JSONs in v1_dir, writing V2
    outputs to output_dir.  Returns a list of GroundingReport dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grounder_only = _GrounderOnly(threshold=threshold, strict_mode=strict_mode)
    all_reports = []

    for json_path in sorted(v1_dir.glob("*.json")):
        doc_id = json_path.stem

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                v1_doc = CTIDocument.model_validate(json.load(f))
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Skipping {json_path.name}: failed to load/validate: {e}")
            continue

        report_path: Optional[Path] = None
        for ext in (".pdf", ".html", ".txt"):
            candidate = reports_dir / f"{doc_id}{ext}"
            if candidate.exists():
                report_path = candidate
                break

        if report_path is None:
            logger.warning(f"No raw report found for {doc_id} in {reports_dir} — skipping")
            continue

        source_data = load_report(report_path)
        source_text = source_data["text"]

        try:
            grounded_doc, grnd_report = grounder_only.ground(v1_doc, source_text)
        except Exception as e:
            logger.error(f"Grounding failed for {doc_id}: {e}", exc_info=True)
            continue

        out_path = output_dir / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(grounded_doc.model_dump_json(indent=2))

        all_reports.append(grnd_report.to_dict())
        logger.info(f"V2 written: {out_path}")

    report_path_out = output_dir / "_grounding_report.json"
    with open(report_path_out, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    logger.info(f"Grounding report written to {report_path_out}")

    return all_reports


class _GrounderOnly:
    """Lightweight grounder used by batch_ground() without a Llama instance."""
    def __init__(self, threshold=DEFAULT_THRESHOLD, strict_mode=False):
        self.threshold = threshold
        self.strict_mode = strict_mode

    def _filter_list(self, items, category, value_field, document, report):
        kept = []
        for item in items:
            snippet = getattr(item, "evidence_snippet", None) or ""
            if not snippet.strip() or _is_snippet_too_short(snippet):
                if self.strict_mode:
                    value = getattr(item, value_field, str(item))
                    report.record(category, str(value), snippet, 0.0)
                else:
                    report.record_pass()
                    kept.append(item)
                continue
            sim = _snippet_similarity(snippet, document)
            if sim >= self.threshold:
                report.record_pass()
                kept.append(item)
            else:
                value = getattr(item, value_field, str(item))
                report.record(category, str(value), snippet, sim)
        return kept

    def ground(self, doc, source_text):
        doc_id = (doc.report_metadata.report_title if doc.report_metadata else "unknown")
        report = GroundingReport(doc_id=doc_id)
        iocs = doc.indicators_of_compromise
        doc_dict = doc.model_dump()
        field_map = {
            "ip_addresses": ("ip_address", "value"),
            "domains":      ("domain",     "value"),
            "file_hashes":  ("file_hash",  "value"),
            "urls":         ("url",        "value"),
            "http_paths":   ("http_path",  "value"),
            "cves":         ("cve",        "value"),
            "tools":        ("tool",       "name"),
        }
        for attr, (cat, val_field) in field_map.items():
            original = getattr(iocs, attr, []) or []
            filtered = self._filter_list(original, cat, val_field, source_text, report)
            doc_dict["indicators_of_compromise"][attr] = [item.model_dump() for item in filtered]
        filtered_ttps = self._filter_list(doc.ttps or [], "ttp", "technique", source_text, report)
        doc_dict["ttps"] = [t.model_dump() for t in filtered_ttps]
        grounded_doc = CTIDocument.model_validate(doc_dict)
        return grounded_doc, report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V2: Apply evidence-grounding filter to V1 outputs."
    )
    parser.add_argument("--v1-outputs", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--output-dir", default="outputs/v2")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    batch_ground(
        v1_dir=Path(args.v1_outputs),
        reports_dir=Path(args.reports),
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        strict_mode=args.strict,
    )
