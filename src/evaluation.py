"""
evaluation.py — Full evaluation framework for V0–V3 CTI extraction variants.

Metrics implemented
-------------------
1.  IoC Precision / Recall / F1  (macro-averaged across documents)
2.  TTP Precision / Recall / F1  (macro-averaged across documents)
3.  Unsupported rate              (proportion of extractions with empty evidence_snippet)
4.  Schema compliance rate        (proportion of output files passing Pydantic validation)
5.  Fabrication rate              (proportion of extracted IoCs absent from gold set)
6.  Attribution error rate        (proportion of extracted TTPs absent from gold set)
7.  Threat summary ROUGE-L        (macro-averaged ROUGE-L F1 over the `summary` field)
8.  Wilcoxon signed-rank test     (paired, N=8 test docs, α=0.05) with rank-biserial r

Usage
-----
    # Quick dev-set dry-run (uses data/dev/ and outputs/)
    python -m src.evaluation --dev

    # Dev-set with a custom V2 directory (e.g. threshold sensitivity)
    python -m src.evaluation --dev --v2 outputs/v2_t055

    # Full test-set evaluation
    python -m src.evaluation \\
        --gold   data/test \\
        --v0     outputs/v0 \\
        --v1     outputs/v1 \\
        --v2     outputs/v2 \\
        --v3     outputs/v3 \\
        --output outputs/evaluation_results.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ValidationError
from scipy import stats

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "rouge-score not installed. ROUGE-L metric will return 0.0. "
        "Install with: pip install rouge-score"
    )

from src.schema_models import CTIDocument, IndicatorsOfCompromise

logger = logging.getLogger(__name__)

# Top-level keys that every real CTIDocument output must contain.
# Used to distinguish old-style error sentinels (which lack these keys)
# from genuine—but partially malformed—extraction outputs.
_CTI_DOC_KEYS = frozenset({"report_metadata", "indicators_of_compromise", "ttps"})

# Stem prefixes that identify ancillary / side-car files written alongside
# extraction outputs.  Files whose stems start with any of these strings are
# silently skipped in load_system_outputs so they never reach Pydantic
# validation and cannot distort the schema-compliance count.
_ANCILLARY_STEM_PREFIXES = ("_",)  # covers _cove_log, _grounding_report, etc.


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gold_outputs(gold_dir: Path) -> Dict[str, CTIDocument]:
    """
    Load gold-standard annotation JSON files into validated CTIDocument objects.
    Raises ValueError if any file fails validation, gold annotations must be clean.
    """
    outputs: Dict[str, CTIDocument] = {}
    for path in sorted(gold_dir.glob("*.json")):
        doc_id = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            outputs[doc_id] = CTIDocument.model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"Gold annotation {path.name} failed schema validation: {e}"
            ) from e
    logger.info(f"Loaded {len(outputs)} gold documents from {gold_dir}")
    return outputs


def _is_overflow_sentinel(data) -> bool:
    """
    Return True when a JSON file is an error/overflow sentinel written by
    run_pipeline.py or CoVeVerifier rather than a real extraction output.

    Non-dict roots (e.g. a bare JSON array produced by a malformed model
    output) are not sentinels so return False immediately so the caller
    lets Pydantic handle them as a compliance failure.
    """
    if not isinstance(data, dict):
        return False
    if data.get("context_overflow", False):
        return True
    if "error" in data and "doc_id" in data:
        if not _CTI_DOC_KEYS.intersection(data.keys()):
            return True
    return False


def _is_ancillary_file(path: Path) -> bool:
    """
    Return True for ancillary / side-car files that must never be treated as
    CTI extraction outputs.  Matches files whose stems begin with '_' (e.g.
    _cove_log.json, _grounding_report.json) or end with '_cove_log'.
    """
    stem = path.stem
    if stem.startswith(_ANCILLARY_STEM_PREFIXES):
        return True
    if stem.endswith("_cove_log"):
        return True
    return False


def load_system_outputs(
    system_dir: Path,
) -> Tuple[Dict[str, CTIDocument], Dict[str, bool], int]:
    """
    Load system output JSON files.  Returns:
      - validated        : dict[doc_id -> CTIDocument]  (files that pass Pydantic)
      - compliance       : dict[doc_id -> bool]          (True = passed, False = failed)
      - n_overflow       : count of context-overflow sentinels (excluded from metrics)

    The following categories of file are silently excluded so they cannot
    distort the schema-compliance count or inflate n_overflow:
      1. Ancillary / side-car files detected by _is_ancillary_file() —
         stems beginning with '_' (_cove_log, _grounding_report, etc.).
      2. Context-overflow / error sentinels detected by _is_overflow_sentinel().
    """
    validated: Dict[str, CTIDocument] = {}
    compliance: Dict[str, bool] = {}
    n_overflow = 0

    for path in sorted(system_dir.glob("*.json")):
        if _is_ancillary_file(path):
            logger.debug(f"Skipping ancillary file: {path.name}")
            continue

        doc_id = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if _is_overflow_sentinel(data):
            logger.warning(
                f"Skipping error/overflow sentinel for {doc_id} — "
                f"document exceeded token budget or JSON repair failed."
            )
            n_overflow += 1
            continue

        try:
            validated[doc_id] = CTIDocument.model_validate(data)
            compliance[doc_id] = True
        except ValidationError as e:
            logger.warning(f"Schema validation failed for {path.name}: {e}")
            compliance[doc_id] = False

    logger.info(
        f"Loaded {len(validated)}/{len(compliance) + n_overflow} valid outputs from "
        f"{system_dir} ({n_overflow} context-overflow sentinel(s) excluded)"
    )
    return validated, compliance, n_overflow


def load_v0_outputs(
    v0_dir: Path,
) -> Tuple[Dict[str, dict], Dict[str, bool], int]:
    """
    Load V0 outputs as raw dicts.  V0 never passes Pydantic because
    evidence_snippet is intentionally empty, compliance is False for all V0 docs.
    Context-overflow sentinels are excluded and counted separately.
    """
    outputs: Dict[str, dict] = {}
    compliance: Dict[str, bool] = {}
    n_overflow = 0

    for path in sorted(v0_dir.glob("*.json")):
        if _is_ancillary_file(path):
            logger.debug(f"Skipping ancillary file: {path.name}")
            continue

        doc_id = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if _is_overflow_sentinel(data):
            logger.warning(f"Skipping context-overflow sentinel for {doc_id} (V0).")
            n_overflow += 1
            continue

        outputs[doc_id] = data
        compliance[doc_id] = False  # By design

    logger.info(
        f"Loaded {len(outputs)} V0 raw outputs from {v0_dir} "
        f"({n_overflow} context-overflow sentinel(s) excluded)"
    )
    return outputs, compliance, n_overflow


# ---------------------------------------------------------------------------
# IoC flattening helpers
# ---------------------------------------------------------------------------

def _flatten_iocs_from_model(doc: CTIDocument) -> set:
    """
    Flatten all IoC sub-lists from a validated CTIDocument into a set of
    (type_label, normalised_value) tuples for set-based P/R/F1 computation.
    """
    ioc_map: IndicatorsOfCompromise = doc.indicators_of_compromise
    iocs: set = set()

    for item in ioc_map.ip_addresses:
        iocs.add(("ip", item.value.strip().lower()))
    for item in ioc_map.domains:
        iocs.add(("domain", item.value.strip().lower()))
    for item in ioc_map.file_hashes:
        iocs.add(("hash", item.value.strip().lower()))
    for item in ioc_map.urls:
        iocs.add(("url", item.value.strip().lower()))
    for item in ioc_map.http_paths:
        iocs.add(("http_path", item.value.strip().lower()))
    for item in ioc_map.cves:
        iocs.add(("cve", item.value.strip().upper()))
    for item in ioc_map.tools:
        iocs.add(("tool", item.name.strip().lower()))

    return iocs


def _flatten_iocs_from_dict(doc: dict) -> set:
    """
    Flatten IoCs from a raw V0 dict (not a CTIDocument object).
    Keys mirror the schema structure produced by v0_extractor.serialise_to_schema().
    """
    iocs: set = set()
    ioc_map = doc.get("indicators_of_compromise", {})

    for item in ioc_map.get("ip_addresses", []):
        iocs.add(("ip", item.get("value", "").strip().lower()))
    for item in ioc_map.get("domains", []):
        iocs.add(("domain", item.get("value", "").strip().lower()))
    for item in ioc_map.get("file_hashes", []):
        iocs.add(("hash", item.get("value", "").strip().lower()))
    for item in ioc_map.get("urls", []):
        iocs.add(("url", item.get("value", "").strip().lower()))
    for item in ioc_map.get("http_paths", []):
        iocs.add(("http_path", item.get("value", "").strip().lower()))
    for item in ioc_map.get("cves", []):
        iocs.add(("cve", item.get("value", "").strip().upper()))
    for item in ioc_map.get("tools", []):
        iocs.add(("tool", item.get("name", "").strip().lower()))

    for label in ("ip", "domain", "hash", "url", "http_path", "cve", "tool"):
        iocs.discard((label, ""))

    return iocs


# ---------------------------------------------------------------------------
# TTP flattening helpers
# ---------------------------------------------------------------------------

def _flatten_ttps_from_model(doc: CTIDocument) -> set:
    """
    Flatten TTPs from a CTIDocument into a set of normalised MITRE IDs.
    Falls back to (tactic, technique) tuple where mitre_id is absent.
    """
    ttps: set = set()
    for ttp in doc.ttps:
        if ttp.mitre_id:
            ttps.add(ttp.mitre_id.upper().strip())
        else:
            ttps.add((ttp.tactic.strip().lower(), ttp.technique.strip().lower()))
    return ttps


def _flatten_ttps_from_dict(doc: dict) -> set:
    """Flatten TTPs from a raw V0 dict."""
    ttps: set = set()
    for ttp in doc.get("ttps", []):
        mid = ttp.get("mitre_id")
        if mid:
            ttps.add(mid.upper().strip())
        else:
            ttps.add((
                ttp.get("tactic", "").strip().lower(),
                ttp.get("technique", "").strip().lower(),
            ))
    return ttps


# ---------------------------------------------------------------------------
# ROUGE-L for threat summary
# ---------------------------------------------------------------------------

def compute_rouge_l_per_doc(
    sys_summary: str,
    gold_summary: str,
) -> float:
    """
    Compute ROUGE-L F-measure between a system summary and gold summary.

    Returns 0.0 when either string is absent or empty, or when the
    rouge-score library is not installed.
    """
    if not _ROUGE_AVAILABLE:
        return 0.0
    sys_summary  = (sys_summary  or "").strip()
    gold_summary = (gold_summary or "").strip()
    if not sys_summary or not gold_summary:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score  = scorer.score(gold_summary, sys_summary)
    return round(score["rougeL"].fmeasure, 4)


def compute_threat_summary_rouge(
    system_outputs,
    gold_outputs: Dict[str, CTIDocument],
    is_v0: bool = False,
) -> Dict[str, Any]:
    """
    Compute macro-averaged ROUGE-L F1 for the `summary` field across all
    matched (system, gold) document pairs.

    For V0 raw dicts the summary is read from doc["summary"].
    For V1-V3 CTIDocument objects it is read from doc.summary.

    Returns a dict with:
      - macro_rouge_l   : float  — macro-averaged ROUGE-L F1
      - per_doc_rouge_l : dict   — {doc_id: rouge_l_score}
      - n_evaluated     : int    — documents with non-empty summaries in both
      - n_skipped       : int    — documents with an empty system or gold summary
    """
    per_doc: Dict[str, float] = {}
    n_skipped = 0

    for doc_id, gold_doc in gold_outputs.items():
        if doc_id not in system_outputs:
            continue

        sys_doc      = system_outputs[doc_id]
        gold_summary = (gold_doc.summary or "").strip()
        sys_summary  = (
            (sys_doc.get("summary", "") or "").strip()
            if is_v0
            else (getattr(sys_doc, "summary", "") or "").strip()
        )

        if not sys_summary or not gold_summary:
            n_skipped += 1
            per_doc[doc_id] = 0.0
            continue

        per_doc[doc_id] = compute_rouge_l_per_doc(sys_summary, gold_summary)

    macro = round(float(np.mean(list(per_doc.values()))), 4) if per_doc else 0.0

    return {
        "macro_rouge_l":   macro,
        "per_doc_rouge_l": per_doc,
        "n_evaluated":     len(per_doc) - n_skipped,
        "n_skipped":       n_skipped,
    }


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _prf1_from_sets(sys_set: set, gold_set: set) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F1 from two sets using standard set-overlap.
    This avoids sklearn's label-alignment fragility when gold/system label spaces
    differ per document.
    """
    if not gold_set and not sys_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not sys_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    tp = len(sys_set & gold_set)
    precision = tp / len(sys_set)
    recall    = tp / len(gold_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_ioc_metrics_per_doc(
    sys_doc,
    gold_doc: CTIDocument,
    is_v0: bool = False,
) -> Dict[str, float]:
    """Compute IoC P/R/F1 for a single (system, gold) document pair."""
    gold_iocs = _flatten_iocs_from_model(gold_doc)
    sys_iocs  = (
        _flatten_iocs_from_dict(sys_doc)
        if is_v0
        else _flatten_iocs_from_model(sys_doc)
    )
    return _prf1_from_sets(sys_iocs, gold_iocs)


def compute_ttp_metrics_per_doc(
    sys_doc,
    gold_doc: CTIDocument,
    is_v0: bool = False,
) -> Dict[str, float]:
    """Compute TTP P/R/F1 for a single (system, gold) document pair."""
    gold_ttps = _flatten_ttps_from_model(gold_doc)
    sys_ttps  = (
        _flatten_ttps_from_dict(sys_doc)
        if is_v0
        else _flatten_ttps_from_model(sys_doc)
    )
    return _prf1_from_sets(sys_ttps, gold_ttps)


def compute_unsupported_rate(
    system_outputs,
    is_v0: bool = False,
) -> float:
    """
    Corpus-level unsupported rate: proportion of all extracted atomic claims
    (IoCs + TTPs) that have an empty evidence_snippet.

    For V0 this will always return 1.0 by construction — the intended ablation signal.
    """
    total = 0
    unsupported = 0

    for doc in system_outputs.values():
        if is_v0:
            ioc_map = doc.get("indicators_of_compromise", {})
            all_items = (
                ioc_map.get("ip_addresses", [])
                + ioc_map.get("domains", [])
                + ioc_map.get("file_hashes", [])
                + ioc_map.get("urls", [])
                + ioc_map.get("http_paths", [])
                + ioc_map.get("cves", [])
            )
            for item in all_items:
                total += 1
                snippet = item.get("evidence_snippet", "")
                if not snippet or not snippet.strip():
                    unsupported += 1
            for ttp in doc.get("ttps", []):
                total += 1
                snippet = ttp.get("evidence_snippet", "")
                if not snippet or not snippet.strip():
                    unsupported += 1
        else:
            ioc_map = doc.indicators_of_compromise
            all_items = (
                ioc_map.ip_addresses
                + ioc_map.domains
                + ioc_map.file_hashes
                + ioc_map.urls
                + ioc_map.http_paths
                + ioc_map.cves
            )
            for item in all_items:
                total += 1
                snippet = getattr(item, "evidence_snippet", "") or ""
                if not snippet.strip():
                    unsupported += 1
            for ttp in doc.ttps:
                total += 1
                if not ttp.evidence_snippet.strip():
                    unsupported += 1

    return round(unsupported / total, 4) if total > 0 else 0.0


def compute_fabrication_rate(
    system_outputs,
    gold_outputs: Dict[str, CTIDocument],
    is_v0: bool = False,
) -> float:
    """
    Corpus-level fabrication rate: macro-average of per-document IoC
    false-positive proportions.
    """
    per_doc_rates: List[float] = []

    for doc_id, gold_doc in gold_outputs.items():
        if doc_id not in system_outputs:
            continue
        sys_doc = system_outputs[doc_id]
        gold_iocs = _flatten_iocs_from_model(gold_doc)
        sys_iocs  = (
            _flatten_iocs_from_dict(sys_doc) if is_v0
            else _flatten_iocs_from_model(sys_doc)
        )
        if not sys_iocs:
            per_doc_rates.append(0.0)
        else:
            fabricated = len(sys_iocs - gold_iocs)
            per_doc_rates.append(fabricated / len(sys_iocs))

    if not per_doc_rates:
        return 0.0
    return round(float(np.mean(per_doc_rates)), 4)


def compute_attribution_error_rate(
    system_outputs,
    gold_outputs: Dict[str, CTIDocument],
    is_v0: bool = False,
) -> float:
    """
    Corpus-level attribution error rate: macro-average of per-document TTP
    false-positive proportions.          
    """
    per_doc_rates: List[float] = []

    for doc_id, gold_doc in gold_outputs.items():
        if doc_id not in system_outputs:
            continue
        sys_doc = system_outputs[doc_id]
        gold_ttps = _flatten_ttps_from_model(gold_doc)
        sys_ttps  = (
            _flatten_ttps_from_dict(sys_doc) if is_v0
            else _flatten_ttps_from_model(sys_doc)
        )
        if not sys_ttps:
            per_doc_rates.append(0.0)
        else:
            misattributed = len(sys_ttps - gold_ttps)
            per_doc_rates.append(misattributed / len(sys_ttps))

    if not per_doc_rates:
        return 0.0
    return round(float(np.mean(per_doc_rates)), 4)


def compute_schema_compliance_rate(compliance: Dict[str, bool]) -> float:
    """Proportion of output documents that pass Pydantic schema validation."""
    if not compliance:
        return 0.0
    return round(sum(compliance.values()) / len(compliance), 4)


# ---------------------------------------------------------------------------
# Per-document aggregation
# ---------------------------------------------------------------------------

def evaluate_variant(
    system_outputs,
    gold_outputs: Dict[str, CTIDocument],
    compliance: Dict[str, bool],
    n_overflow: int = 0,
    is_v0: bool = False,
    variant_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Evaluate one system variant against the gold standard.

    Returns a results dict containing:
      - per_doc               : per-document metrics (IoC + TTP + ROUGE-L)
      - macro_ioc_*           : macro-averaged IoC P/R/F1
      - macro_ttp_*           : macro-averaged TTP P/R/F1
      - macro_rouge_l         : macro-averaged ROUGE-L F1 for the summary field
      - std_ioc_f1            : standard deviation of per-doc IoC F1
      - std_ttp_f1            : standard deviation of per-doc TTP F1
      - ioc_f1_per_doc        : ordered list of per-doc IoC F1 (for Wilcoxon)
      - ttp_f1_per_doc        : ordered list of per-doc TTP F1 (for Wilcoxon)
      - unsupported_rate      : corpus-level unsupported rate
      - fabrication_rate      : corpus-level IoC fabrication rate
      - attribution_error_rate: corpus-level TTP misattribution rate
      - schema_compliance     : proportion of files passing Pydantic validation
      - n_docs_evaluated      : number of matched (system, gold) pairs
      - n_context_overflow    : number of documents skipped due to context overflow
    """
    per_doc: Dict[str, Any] = {}

    for doc_id, gold_doc in gold_outputs.items():
        if doc_id not in system_outputs:
            logger.warning(f"[{variant_name}] Missing output for {doc_id} — skipping")
            continue

        sys_doc = system_outputs[doc_id]
        ioc_m = compute_ioc_metrics_per_doc(sys_doc, gold_doc, is_v0=is_v0)
        ttp_m = compute_ttp_metrics_per_doc(sys_doc, gold_doc, is_v0=is_v0)

        # ROUGE-L for summary field
        gold_summary = (gold_doc.summary or "").strip()
        sys_summary  = (
            (sys_doc.get("summary", "") or "").strip()
            if is_v0
            else (getattr(sys_doc, "summary", "") or "").strip()
        )
        rouge_l = compute_rouge_l_per_doc(sys_summary, gold_summary)

        per_doc[doc_id] = {
            "ioc_precision": round(ioc_m["precision"], 4),
            "ioc_recall":    round(ioc_m["recall"],    4),
            "ioc_f1":        round(ioc_m["f1"],        4),
            "ttp_precision": round(ttp_m["precision"], 4),
            "ttp_recall":    round(ttp_m["recall"],    4),
            "ttp_f1":        round(ttp_m["f1"],        4),
            "rouge_l":       rouge_l,
        }

    if not per_doc:
        logger.error(f"[{variant_name}] No matched documents — check output directory.")
        return {}

    def _macro(key: str) -> float:
        return round(float(np.mean([per_doc[d][key] for d in per_doc])), 4)

    def _std(key: str) -> float:
        vals = [per_doc[d][key] for d in per_doc]
        return round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)

    # Full ROUGE-L summary (includes n_evaluated / n_skipped breakdown)
    rouge_summary = compute_threat_summary_rouge(
        system_outputs, gold_outputs, is_v0=is_v0
    )

    return {
        "variant":               variant_name,
        "n_docs_evaluated":      len(per_doc),
        "n_context_overflow":    n_overflow,
        "macro_ioc_precision":   _macro("ioc_precision"),
        "macro_ioc_recall":      _macro("ioc_recall"),
        "macro_ioc_f1":          _macro("ioc_f1"),
        "std_ioc_f1":            _std("ioc_f1"),
        "macro_ttp_precision":   _macro("ttp_precision"),
        "macro_ttp_recall":      _macro("ttp_recall"),
        "macro_ttp_f1":          _macro("ttp_f1"),
        "std_ttp_f1":            _std("ttp_f1"),
        "macro_rouge_l":         rouge_summary["macro_rouge_l"],
        "rouge_l_n_evaluated":   rouge_summary["n_evaluated"],
        "rouge_l_n_skipped":     rouge_summary["n_skipped"],
        "unsupported_rate":      compute_unsupported_rate(system_outputs, is_v0=is_v0),
        "fabrication_rate":      compute_fabrication_rate(
                                     system_outputs, gold_outputs, is_v0=is_v0
                                 ),
        "attribution_error_rate": compute_attribution_error_rate(
                                     system_outputs, gold_outputs, is_v0=is_v0
                                 ),
        "schema_compliance":     compute_schema_compliance_rate(compliance),
        # Ordered per-doc lists for paired Wilcoxon tests
        "ioc_f1_per_doc": [per_doc[d]["ioc_f1"] for d in sorted(per_doc)],
        "ttp_f1_per_doc": [per_doc[d]["ttp_f1"] for d in sorted(per_doc)],
        "per_doc": per_doc,
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def rank_biserial_r(n: int, w_stat: float) -> float:
    """
    Compute rank-biserial correlation r from a Wilcoxon W statistic.
    """
    max_w = n * (n + 1) / 2
    return round(1.0 - (2 * w_stat / max_w), 4)


def wilcoxon_test(
    sys_a_scores: List[float],
    sys_b_scores: List[float],
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test comparing two paired systems over N test documents.
    """
    if len(sys_a_scores) != len(sys_b_scores):
        raise ValueError(
            f"Score lists must be the same length: "
            f"{len(sys_a_scores)} vs {len(sys_b_scores)}"
        )

    n = len(sys_a_scores)
    differences = [a - b for a, b in zip(sys_a_scores, sys_b_scores)]

    if all(d == 0 for d in differences):
        return {
            "comparison":    f"{label_a} vs {label_b}",
            "n":             n,
            "statistic":     None,
            "p_value":       1.0,
            "significant":   False,
            "effect_size_r": 0.0,
            "effect_label":  "none",
            "direction":     "no difference",
            "interpretation": (
                f"No difference observed between {label_a} and {label_b} "
                f"(all per-document scores identical, N={n})."
            ),
        }

    stat, p_value = stats.wilcoxon(sys_a_scores, sys_b_scores, alternative="two-sided")
    significant = bool(p_value < alpha)
    r = rank_biserial_r(n, stat)

    abs_r = abs(r)
    effect_label = "large" if abs_r >= 0.5 else "medium" if abs_r >= 0.3 else "small"

    direction = (
        f"{label_a} > {label_b}"
        if np.mean(sys_a_scores) > np.mean(sys_b_scores)
        else f"{label_b} > {label_a}"
    )

    interpretation = (
        f"Wilcoxon signed-rank test ({label_a} vs {label_b}, N={n}): "
        f"W={stat:.1f}, p={p_value:.4f} "
        f"({'significant' if significant else 'not significant'} at \u03b1={alpha}). "
        f"Effect size r={r:.3f} ({effect_label}). Direction: {direction}."
    )

    return {
        "comparison":    f"{label_a} vs {label_b}",
        "n":             n,
        "statistic":     float(stat),
        "p_value":       float(p_value),
        "significant":   significant,
        "effect_size_r": r,
        "effect_label":  effect_label,
        "direction":     direction,
        "interpretation": interpretation,
    }


def run_all_wilcoxon_tests(
    results: Dict[str, Dict],
    metric: str = "ioc_f1_per_doc",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Run all theory-motivated pairwise Wilcoxon tests across variants.

    Comparisons:
      V0 vs V1  — value of schema-guided JSON
      V1 vs V2  — value of document grounding
      V2 vs V3  — value of CoVe self-verification
      V0 vs V3  — overall ablation span
    """
    comparisons = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v0", "v3")]
    tests = []

    for a, b in comparisons:
        if a not in results or b not in results:
            logger.warning(f"Skipping Wilcoxon {a} vs {b}: variant(s) missing from results")
            continue
        per_doc_a = results[a].get("per_doc", {})
        per_doc_b = results[b].get("per_doc", {})
        shared_docs = sorted(set(per_doc_a) & set(per_doc_b))
        if len(shared_docs) < 2:
            logger.warning(
                f"Skipping Wilcoxon {a} vs {b}: fewer than 2 shared documents."
            )
            continue
        metric_key = metric.replace("_per_doc", "")
        scores_a = [per_doc_a[d][metric_key] for d in shared_docs]
        scores_b = [per_doc_b[d][metric_key] for d in shared_docs]
        tests.append(
            wilcoxon_test(
                scores_a, scores_b,
                label_a=a.upper(), label_b=b.upper(),
                alpha=alpha,
            )
        )

    return tests


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_results_table(results: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """Build a flat list of summary rows (one per variant) for display or CSV export."""
    keys = [
        "variant", "n_docs_evaluated", "n_context_overflow",
        "macro_ioc_precision", "macro_ioc_recall", "macro_ioc_f1", "std_ioc_f1",
        "macro_ttp_precision", "macro_ttp_recall", "macro_ttp_f1", "std_ttp_f1",
        "unsupported_rate", "fabrication_rate", "attribution_error_rate",
        "schema_compliance", "macro_rouge_l",
    ]
    return [
        {k: results[v].get(k) for k in keys}
        for v in ["v0", "v1", "v2", "v3"]
        if v in results
    ]


def print_results_table(results: Dict[str, Dict]) -> None:
    """Print a formatted summary table to stdout."""
    rows = build_results_table(results)
    if not rows:
        print("No results to display.")
        return

    col_w = 10
    headers = [
        "Variant", "N", "Overflow",
        "IoC-P", "IoC-R", "IoC-F1", "\u00b1IoC",
        "TTP-P", "TTP-R", "TTP-F1", "\u00b1TTP",
        "Unsupp.", "Fab.", "AttErr", "Comply", "ROUGE-L",
    ]
    sep = "-" * (col_w * len(headers))
    print(f"\n{sep}")
    print("".join(h.ljust(col_w) for h in headers))
    print(sep)

    for row in rows:
        vals = [
            str(row.get("variant", "")),
            str(row.get("n_docs_evaluated", "")),
            str(row.get("n_context_overflow", 0)),
            f"{row.get('macro_ioc_precision', 0):.3f}",
            f"{row.get('macro_ioc_recall', 0):.3f}",
            f"{row.get('macro_ioc_f1', 0):.3f}",
            f"{row.get('std_ioc_f1', 0):.3f}",
            f"{row.get('macro_ttp_precision', 0):.3f}",
            f"{row.get('macro_ttp_recall', 0):.3f}",
            f"{row.get('macro_ttp_f1', 0):.3f}",
            f"{row.get('std_ttp_f1', 0):.3f}",
            f"{row.get('unsupported_rate', 0):.3f}",
            f"{row.get('fabrication_rate', 0):.3f}",
            f"{row.get('attribution_error_rate', 0):.3f}",
            f"{row.get('schema_compliance', 0):.3f}",
            f"{row.get('macro_rouge_l', 0):.3f}",
        ]
        print("".join(v.ljust(col_w) for v in vals))

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_full_evaluation(
    gold_dir: Path,
    output_dirs: Dict[str, Path],
    output_path: Optional[Path] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Run evaluation for all provided variants and return a consolidated results dict.
    """
    gold = load_gold_outputs(gold_dir)
    variant_results: Dict[str, Dict] = {}

    for variant_name, out_dir in output_dirs.items():
        if not out_dir.exists():
            logger.warning(f"Output directory not found: {out_dir} — skipping {variant_name}")
            continue

        if variant_name == "v0":
            sys_outs, compliance, n_overflow = load_v0_outputs(out_dir)
            result = evaluate_variant(
                sys_outs, gold, compliance,
                n_overflow=n_overflow, is_v0=True, variant_name="v0",
            )
        else:
            sys_outs, compliance, n_overflow = load_system_outputs(out_dir)
            result = evaluate_variant(
                sys_outs, gold, compliance,
                n_overflow=n_overflow, is_v0=False, variant_name=variant_name,
            )

        if result:
            variant_results[variant_name] = result

    ioc_tests = run_all_wilcoxon_tests(variant_results, metric="ioc_f1_per_doc", alpha=alpha)
    ttp_tests = run_all_wilcoxon_tests(variant_results, metric="ttp_f1_per_doc", alpha=alpha)

    full_results = {
        "variants":         variant_results,
        "wilcoxon_ioc_f1":  ioc_tests,
        "wilcoxon_ttp_f1":  ttp_tests,
    }

    print_results_table(variant_results)
    for test in ioc_tests + ttp_tests:
        print(test["interpretation"])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, default=str)
        logger.info(f"Results written to {output_path}")

    return full_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate V0–V3 CTI extraction variants against gold annotations."
    )
    parser.add_argument(
        "--dev", action="store_true",
        help=(
            "Shortcut: evaluate against data/dev/ gold annotations using outputs/ dirs. "
            "Equivalent to --gold data/dev --v0 outputs/v0 --v1 outputs/v1 "
            "--v2 outputs/v2 --v3 outputs/v3 --output outputs/dev_evaluation_results.json. "
            "Individual --v0/v1/v2/v3 flags override the defaults when provided."
        ),
    )
    parser.add_argument("--gold",   default=None, help="Gold annotation directory")
    parser.add_argument("--v0",     default=None, help="V0 outputs directory")
    parser.add_argument("--v1",     default=None, help="V1 outputs directory")
    parser.add_argument("--v2",     default=None, help="V2 outputs directory")
    parser.add_argument("--v3",     default=None, help="V3 outputs directory")
    parser.add_argument(
        "--output", default=None,
        help="Path for consolidated results JSON (default: outputs/evaluation_results.json)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for Wilcoxon tests (default: 0.05)",
    )
    args = parser.parse_args()

    if args.dev:
        gold_dir = Path(args.gold) if args.gold else Path("data/dev")
        output_dirs = {
            "v0": Path(args.v0) if args.v0 else Path("outputs/v0"),
            "v1": Path(args.v1) if args.v1 else Path("outputs/v1"),
            "v2": Path(args.v2) if args.v2 else Path("outputs/v2"),
            "v3": Path(args.v3) if args.v3 else Path("outputs/v3"),
        }
        output_path = Path(args.output) if args.output else Path("outputs/dev_evaluation_results.json")
    else:
        if not args.gold:
            parser.error("--gold is required unless --dev is specified.")
        gold_dir = Path(args.gold)
        output_dirs = {
            v: Path(getattr(args, v))
            for v in ["v0", "v1", "v2", "v3"]
            if getattr(args, v) is not None
        }
        output_path = Path(args.output) if args.output else Path("outputs/evaluation_results.json")

    run_full_evaluation(
        gold_dir=gold_dir,
        output_dirs=output_dirs,
        output_path=output_path,
        alpha=args.alpha,
    )
