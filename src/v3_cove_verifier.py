"""
cove_verifier.py — V3 Chain-of-Verification (CoVe) pipeline.

Implements a three-step self-verification loop.

Adapted for CTI structured extraction as follows:

  Step 1 — Initial extraction
      Uses BaselineExtractor.run_extraction() directly.  This is identical
      to V1 by construction: same prompt, same few-shot examples, same
      JSON repair logic.  The V3 Step 1 output is therefore comparable to
      V1 on a document-by-document basis, so all V1→V3 differences are
      attributable solely to the CoVe verification pass.

  Step 2 — Generate and answer verification questions
      For each extracted IoC and TTP, one targeted attribute-level yes/no
      question is generated deterministically.  Questions verify not just
      entity presence but also the correctness of the extracted attribute
      (e.g. tactic assignment, malware family classification, CVE context).
      Each question is answered by a SEPARATE, independent LLM call using
      only the source text, so that the model cannot anchor on earlier
      answers (per the original CoVe paper).

  Step 3 — Targeted deletion
      Items whose verification question was answered NOT SUPPORTED are
      removed directly from the Pydantic model.  No second LLM generation
      call is made — deletions are deterministic and fully traceable to
      the Step 2 Q&A log.  This prevents the over-pruning artefact
      observed when asking the model to re-generate the full JSON.

Usage
-----
    from src.cove_verifier import CoVeVerifier
    verifier = CoVeVerifier(llm)          # llm = Llama instance from run_pipeline
    result = verifier.run_extraction(report_path)
"""

import json
import logging
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_cpp import Llama
from pydantic import ValidationError
from rich.console import Console

from src.document_loader import load_report
from src.schema_models import CTIDocument
from src.prompting import build_cove_verification_prompt
from src.utils_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
console = Console()

# Token budget for Step 2 per-claim yes/no verification answers.
# 256 tokens is sufficient for a single explanatory sentence.
_MAX_TOKENS_VERIFY  = 256

# Token budget for Step 1 initial extraction when running in standalone
# batch_verify mode (no pre-built initial_doc available).
_MAX_TOKENS_EXTRACT = 3072


# ---------------------------------------------------------------------------
# Question generation (deterministic, no extra LLM call)
# ---------------------------------------------------------------------------

def _generate_verification_questions(
    doc: CTIDocument,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Produce one targeted attribute-level yes/no question per extracted IoC
    and TTP.  Questions verify not just entity presence but also the
    correctness of the extracted attribute.

    Returns
    -------
    questions : list of question strings (one per claim)
    question_index : list of dicts mapping each question back to the
                     claim it covers, so that targeted deletion in Step 3
                     can identify exactly which Pydantic list entry to remove.
                     Each dict has keys: category, list_attr, item_index.
    """
    MAX_QUESTIONS = 30
    questions: List[str] = []
    question_index: List[Dict[str, Any]] = []
    iocs = doc.indicators_of_compromise

    for i, item in enumerate(iocs.ip_addresses or []):
        questions.append(
            f'Does the document describe the IP address "{item.value}" as a'
            f' malicious or threat-actor-controlled address (not benign infrastructure)?'
        )
        question_index.append({"category": "ioc", "list_attr": "ip_addresses", "item_index": i})

    for i, item in enumerate(iocs.domains or []):
        questions.append(
            f'Does the document associate the domain "{item.value}" with'
            f' threat-actor activity (e.g. phishing, C2, or impersonation)?'
        )
        question_index.append({"category": "ioc", "list_attr": "domains", "item_index": i})

    for i, item in enumerate(iocs.file_hashes or []):
        questions.append(
            f'Does the document attribute the file hash "{item.value}" to a'
            f' specific malware sample, dropper, or malicious tool?'
        )
        question_index.append({"category": "ioc", "list_attr": "file_hashes", "item_index": i})

    for i, item in enumerate(iocs.urls or []):
        questions.append(
            f'Does the document identify the URL "{item.value}" as being used'
            f' for command-and-control, payload delivery, or phishing?'
        )
        question_index.append({"category": "ioc", "list_attr": "urls", "item_index": i})

    for i, item in enumerate(iocs.cves or []):
        questions.append(
            f'Does the document describe "{item.value}" as being actively'
            f' exploited by the threat actor(s) covered in this report'
            f' (not merely mentioned as background context)?'
        )
        question_index.append({"category": "ioc", "list_attr": "cves", "item_index": i})

    for i, item in enumerate(iocs.tools or []):
        questions.append(
            f'Does the document describe "{item.name}" as a tool, malware, or'
            f' utility used BY the threat actor(s) in this report (not a'
            f' defensive or victim tool)?'
        )
        question_index.append({"category": "ioc", "list_attr": "tools", "item_index": i})

    for i, ttp in enumerate(doc.ttps or []):
        mid = ttp.mitre_id or ttp.technique
        tactic = ttp.tactic or "unspecified tactic"
        questions.append(
            f'Does the document describe the technique "{mid}" ({ttp.technique})'
            f' as being used specifically for "{tactic}" — not a different'
            f' tactic or phase of the attack?'
        )
        question_index.append({"category": "ttp", "list_attr": "ttps", "item_index": i})

    # Deduplicate while preserving index alignment
    seen: set = set()
    deduped_q: List[str] = []
    deduped_idx: List[Dict[str, Any]] = []
    for q, idx in zip(questions, question_index):
        if q not in seen:
            seen.add(q)
            deduped_q.append(q)
            deduped_idx.append(idx)

    if len(deduped_q) > MAX_QUESTIONS:
        logger.warning(
            f"Truncating verification questions from {len(deduped_q)} to "
            f"{MAX_QUESTIONS} to stay within context window."
        )
        deduped_q   = deduped_q[:MAX_QUESTIONS]
        deduped_idx = deduped_idx[:MAX_QUESTIONS]

    return deduped_q, deduped_idx


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def _parse_single_answer(raw_answer: str) -> bool:
    """
    Parse a single yes/no verification answer into a boolean.
    Errs on the side of retention (returns True) when the answer is
    ambiguous, to avoid over-pruning.
    """
    lower = raw_answer.lower()
    not_found = "not found" in lower or "not present" in lower or "not explicitly" in lower
    supported = (
        "yes" in lower
        or "true" in lower
        or "found" in lower
        or "present" in lower
        or "mentions" in lower
        or "explicitly" in lower
    )
    return supported and not not_found


def _format_qa_pairs(
    questions: List[str],
    answers: List[bool],
    raw_answers: List[str],
) -> str:
    """
    Format the per-claim Q&A results into a human-readable block
    for inclusion in the CoVe log.
    """
    lines = []
    for i, (q, supported) in enumerate(zip(questions, answers)):
        label = "SUPPORTED" if supported else "NOT SUPPORTED"
        raw = raw_answers[i] if i < len(raw_answers) else ""
        lines.append(f"{i+1}. [{label}] Q: {q}\n   A: {raw}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Targeted deletion (replaces LLM-based revision)
# ---------------------------------------------------------------------------

def _apply_targeted_deletion(
    initial_doc: CTIDocument,
    answers: List[bool],
    question_index: List[Dict[str, Any]],
) -> Tuple[CTIDocument, List[Dict[str, Any]]]:
    """
    Remove from the CTIDocument exactly the items whose verification question
    was answered NOT SUPPORTED.

    Works by building a set of (list_attr, item_index) pairs to delete, then
    reconstructing each sub-list with those indices omitted.  Operates on a
    deep copy of the model dict so the original is preserved for logging.

    Returns
    -------
    revised_doc   : new CTIDocument with flagged items removed
    deleted_items : list of dicts describing each removed item (for the log)
    """
    doc_dict = initial_doc.model_dump()
    ioc_dict = doc_dict["indicators_of_compromise"]

    # Collect indices to remove per list
    to_remove: Dict[str, set] = {}
    deleted_items: List[Dict[str, Any]] = []

    for answer, qidx in zip(answers, question_index):
        if not answer:
            attr = qidx["list_attr"]
            idx  = qidx["item_index"]
            to_remove.setdefault(attr, set()).add(idx)

    # IoC sub-lists
    for attr in ("ip_addresses", "domains", "file_hashes", "urls", "http_paths", "cves", "tools"):
        if attr not in to_remove:
            continue
        original = ioc_dict.get(attr) or []
        removed_indices = to_remove[attr]
        filtered = []
        for i, item in enumerate(original):
            if i in removed_indices:
                deleted_items.append({"list": attr, "item_index": i, "item": item})
                logger.info(f"[CoVe] Targeted deletion: {attr}[{i}] = {item}")
            else:
                filtered.append(item)
        ioc_dict[attr] = filtered

    # TTPs
    if "ttps" in to_remove:
        original_ttps = doc_dict.get("ttps") or []
        removed_indices = to_remove["ttps"]
        filtered_ttps = []
        for i, ttp in enumerate(original_ttps):
            if i in removed_indices:
                deleted_items.append({"list": "ttps", "item_index": i, "item": ttp})
                logger.info(f"[CoVe] Targeted deletion: ttps[{i}] = {ttp}")
            else:
                filtered_ttps.append(ttp)
        doc_dict["ttps"] = filtered_ttps

    doc_dict["indicators_of_compromise"] = ioc_dict

    try:
        revised_doc = CTIDocument.model_validate(doc_dict)
    except ValidationError as e:
        logger.error(f"[CoVe] Targeted deletion produced invalid document: {e}")
        raise

    return revised_doc, deleted_items


# ---------------------------------------------------------------------------
# CoVeVerifier
# ---------------------------------------------------------------------------

class CoVeVerifier:
    """
    V3 Chain-of-Verification verifier.

    Accepts a shared Llama instance so it can be instantiated by
    run_pipeline.py with the same interface as V0Extractor and
    EvidenceGrounder.  Internally wraps a BaselineExtractor for the
    initial extraction step.

    Parameters
    ----------
    llm : Llama
        Shared llama-cpp-python Llama instance.
    max_new_tokens_verify : int
        Token budget for each individual verification call (Step 2).
        Single-sentence yes/no answers; 256 is sufficient.
    max_new_tokens_extract : int
        Token budget for the initial extraction when running in standalone
        mode (Step 1 fallback).  3072 accommodates IOC-dense documents.
    """

    def __init__(
        self,
        llm: Llama,
        max_new_tokens_verify: int = _MAX_TOKENS_VERIFY,
        max_new_tokens_extract: int = _MAX_TOKENS_EXTRACT,
    ):
        self.max_new_tokens_verify  = max_new_tokens_verify
        self.max_new_tokens_extract = max_new_tokens_extract
        from src.v1_baseline_extractor import BaselineExtractor
        self.extractor = BaselineExtractor(llm)

    def _generate(self, prompt: str, max_tokens: int = _MAX_TOKENS_EXTRACT) -> str:
        """Thin wrapper that delegates to the shared BaselineExtractor._generate."""
        return self.extractor._generate(prompt, max_tokens=max_tokens)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        from src.v1_baseline_extractor import _try_parse_json
        return _try_parse_json(raw)

    def _verify_claim(self, doc_text: str, question: str) -> Tuple[bool, str]:
        """
        Ask the model a single yes/no verification question about one claim.
        Returns (supported: bool, raw_answer: str).

        Using independent per-claim prompts (rather than one batched prompt)
        prevents the model from stopping after the first "no" answer and
        missing subsequent failures.  This is the approach recommended by
        Dhuliawala et al. (2023, §3.2) for faithful CoVe evaluation.
        """
        from src.prompting import MAX_DOC_CHARS, SYS_OPEN, USER_OPEN, ASS_OPEN, EOT
        doc_text_trunc = doc_text[:MAX_DOC_CHARS]
        system = (
            SYS_OPEN
            + "You are a fact-checker for cyber threat intelligence reports. "
            + "Answer the question using ONLY the document text provided. "
            + "Reply with a single sentence starting with Yes or No."
            + EOT
        )
        user = (
            USER_OPEN
            + "Document text:\n" + doc_text_trunc
            + "\n\nQuestion: " + question
            + "\n\nAnswer:"
            + EOT + ASS_OPEN
        )
        prompt = system + user
        raw = self._generate(prompt, max_tokens=self.max_new_tokens_verify)
        supported = _parse_single_answer(raw)
        return supported, raw.strip()

    def verify(
        self,
        doc_data: Dict[str, Any],
        initial_doc: Optional[CTIDocument] = None,
    ) -> Tuple[CTIDocument, Dict[str, Any]]:
        """
        Run the CoVe Steps 2 and 3 for a single document.

        Parameters
        ----------
        doc_data : dict with at least keys 'doc_id' and 'text'
        initial_doc : CTIDocument, optional
            Pre-built initial extraction (from BaselineExtractor).  When
            provided, Step 1 is skipped and verification runs directly on
            this document.  When None, Step 1 is run internally using
            build_cove_initial_prompt (standalone / batch_verify mode).

        Returns
        -------
        revised_doc : CTIDocument
        cove_log : dict — structured log of all steps
        """
        from src.v1_baseline_extractor import _coerce_parsed

        doc_id = doc_data.get("doc_id", "unknown")
        source_text = doc_data["text"]
        cove_log: Dict[str, Any] = {"doc_id": doc_id, "steps": {}}
        t0 = time.time()

        # -------------------------------------------------------------------
        # Step 1: Initial extraction
        # If initial_doc is provided (called from run_extraction), use it
        # directly so V3 Step 1 == V1 by construction.
        # If not provided (standalone batch_verify), run extraction internally.
        # -------------------------------------------------------------------
        if initial_doc is not None:
            logger.info(f"[CoVe] Step 1 — Using pre-built V1 extraction: {doc_id}")
        else:
            logger.info(f"[CoVe] Step 1 — Running internal extraction: {doc_id}")
            from src.prompting import build_cove_initial_prompt
            prompt_1 = build_cove_initial_prompt(doc_data)
            raw_1 = self._generate(prompt_1, max_tokens=_MAX_TOKENS_EXTRACT)
            try:
                parsed_1 = _coerce_parsed(self._parse_json(raw_1))
                initial_doc = CTIDocument.model_validate(parsed_1)
            except (ValueError, ValidationError) as e:
                logger.error(f"[CoVe] Step 1 failed for {doc_id}: {e}")
                raise

        def _count_iocs(doc: CTIDocument) -> int:
            iocs = doc.indicators_of_compromise
            return len(
                (iocs.ip_addresses or [])
                + (iocs.domains or [])
                + (iocs.file_hashes or [])
                + (iocs.urls or [])
                + (iocs.cves or [])
                + (iocs.tools or [])
            )

        cove_log["steps"]["step1_initial"] = {
            "n_iocs": _count_iocs(initial_doc),
            "n_ttps": len(initial_doc.ttps or []),
            "source": "pre-built V1" if initial_doc is not None else "internal",
        }

        # -------------------------------------------------------------------
        # Step 2: Independent per-claim attribute-level verification
        # -------------------------------------------------------------------
        logger.info(f"[CoVe] Step 2 — Attribute-level verification: {doc_id}")
        questions, question_index = _generate_verification_questions(initial_doc)

        if not questions:
            logger.warning(
                f"[CoVe] No verification questions generated for {doc_id} — "
                f"returning initial extraction unchanged."
            )
            cove_log["steps"]["step2_verification"] = {"n_questions": 0}
            cove_log["steps"]["step3_deletion"] = {"changed": False, "deleted_items": []}
            cove_log["total_time_s"] = round(time.time() - t0, 2)
            return initial_doc, cove_log

        answers: List[bool] = []
        raw_answers: List[str] = []
        for idx, question in enumerate(questions):
            supported, raw_ans = self._verify_claim(source_text, question)
            answers.append(supported)
            raw_answers.append(raw_ans)
            logger.debug(
                f"[CoVe] {doc_id} Q{idx+1}/{len(questions)}: "
                f"{'SUPPORTED' if supported else 'NOT SUPPORTED'} — {question[:80]}"
            )

        qa_formatted = _format_qa_pairs(questions, answers, raw_answers)
        n_rejected = sum(1 for a in answers if not a)

        cove_log["steps"]["step2_verification"] = {
            "n_questions":     len(questions),
            "n_supported":     len(questions) - n_rejected,
            "n_not_supported": n_rejected,
            "qa_pairs":        qa_formatted,
        }
        logger.info(
            f"[CoVe] {doc_id}: {n_rejected}/{len(questions)} claims failed verification"
        )

        if n_rejected == 0:
            logger.info(f"[CoVe] All claims verified for {doc_id} — no deletions needed.")
            cove_log["steps"]["step3_deletion"] = {"changed": False, "deleted_items": []}
            cove_log["total_time_s"] = round(time.time() - t0, 2)
            return initial_doc, cove_log

        # -------------------------------------------------------------------
        # Step 3: Targeted deletion (deterministic — no LLM call)
        # Only items flagged NOT SUPPORTED in Step 2 are removed.
        # -------------------------------------------------------------------
        logger.info(f"[CoVe] Step 3 — Targeted deletion: {doc_id} ({n_rejected} items)")
        try:
            revised_doc, deleted_items = _apply_targeted_deletion(
                initial_doc, answers, question_index
            )
        except ValidationError as e:
            logger.warning(
                f"[CoVe] Targeted deletion produced invalid document for {doc_id}: {e}. "
                f"Falling back to initial extraction."
            )
            cove_log["steps"]["step3_deletion"] = {
                "changed": False, "fallback": True, "error": str(e), "deleted_items": [],
            }
            cove_log["total_time_s"] = round(time.time() - t0, 2)
            return initial_doc, cove_log

        cove_log["steps"]["step3_deletion"] = {
            "changed":       True,
            "n_iocs_before": cove_log["steps"]["step1_initial"]["n_iocs"],
            "n_iocs_after":  _count_iocs(revised_doc),
            "n_ttps_before": cove_log["steps"]["step1_initial"]["n_ttps"],
            "n_ttps_after":  len(revised_doc.ttps or []),
            "deleted_items": deleted_items,
        }
        cove_log["total_time_s"] = round(time.time() - t0, 2)

        logger.info(
            f"[CoVe] Deletion complete for {doc_id}: "
            f"removed {len(deleted_items)} item(s) in {cove_log['total_time_s']}s"
        )
        return revised_doc, cove_log

    def run_extraction(self, report_path: Path) -> CTIDocument:
        """
        Full V3 pipeline for a single document.

        Loads the document, runs V1 extraction (Step 1), passes the result
        directly to verify() as initial_doc so that V3 Step 1 == V1 by
        construction.  Then writes the CoVe log and returns the final
        CTIDocument.
        """
        console.print(f"[V3] Processing: {report_path.name}")
        t0 = time.time()

        # Step 1: run V1 extraction (identical prompt and logic to BaselineExtractor)
        console.print(f"[V3] Step 1 — V1 extraction (shared with BaselineExtractor)")
        initial_doc = self.extractor.run_extraction(report_path)

        # Steps 2-3: CoVe verification and targeted deletion
        doc_data = load_report(report_path)
        revised_doc, cove_log = self.verify(doc_data, initial_doc=initial_doc)

        log_dir = Path("outputs/v3")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{report_path.stem}_cove_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(cove_log, f, indent=2)

        elapsed = round(time.time() - t0, 2)
        n_deleted = len(
            cove_log["steps"].get("step3_deletion", {}).get("deleted_items", [])
        )
        console.print(
            f"[V3] Done in {elapsed}s — {n_deleted} item(s) removed by targeted deletion"
        )
        logger.info(f"[V3] {report_path.stem} done in {elapsed}s")
        return revised_doc


# ---------------------------------------------------------------------------
# Standalone batch runner (bypasses run_pipeline.py)
# ---------------------------------------------------------------------------

def batch_verify(
    reports_dir: Path,
    output_dir: Path,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> None:
    from src.v1_baseline_extractor import BaselineExtractor
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = BaselineExtractor(model_name=model_name)
    verifier  = CoVeVerifier.__new__(CoVeVerifier)
    verifier.extractor = extractor
    verifier.max_new_tokens_verify  = _MAX_TOKENS_VERIFY
    verifier.max_new_tokens_extract = _MAX_TOKENS_EXTRACT
    all_logs: List[Dict[str, Any]] = []

    report_paths = sorted(
        list(reports_dir.glob("*.pdf"))
        + list(reports_dir.glob("*.html"))
        + list(reports_dir.glob("*.txt"))
    )

    if not report_paths:
        logger.warning(f"No reports found in {reports_dir}")
        return

    for report_path in report_paths:
        doc_id = report_path.stem
        try:
            doc_data = load_report(report_path)
            # Standalone mode: no pre-built initial_doc; verify() runs Step 1 internally
            revised_doc, cove_log = verifier.verify(doc_data, initial_doc=None)
        except Exception as e:
            logger.error(f"[CoVe batch] Failed on {doc_id}: {e}", exc_info=True)
            continue

        out_path = output_dir / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(revised_doc.model_dump_json(indent=2))

        all_logs.append(cove_log)
        logger.info(f"[CoVe batch] Written: {out_path}")

    log_path = output_dir / "_cove_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2)
    logger.info(f"[CoVe batch] Log written to {log_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V3: Chain-of-Verification CTI extraction."
    )
    parser.add_argument("--reports", required=True)
    parser.add_argument("--output-dir", default="outputs/v3")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    batch_verify(
        reports_dir=Path(args.reports),
        output_dir=Path(args.output_dir),
        model_name=args.model,
    )
