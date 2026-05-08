"""
v1_baseline_extractor.py — V1 schema-guided JSON baseline extractor.

Uses llama-cpp-python for inference against a GGUF-quantised
Llama-3.1-8B-Instruct model. Accepts a shared Llama instance so
the model is loaded only once across V0/V1/V2/V3 in run_pipeline.py.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_cpp import Llama
from rich.console import Console

from src.document_loader import load_report
from src.prompting import build_baseline_prompt
from src.schema_models import CTIDocument
from src.utils_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
console = Console()

_VALID_REPORT_TYPES = {"incident", "campaign", "strategic"}
_VALID_HASH_TYPES   = {"MD5", "SHA1", "SHA256", "SHA512", "SSDEEP"}
_MITRE_ID_RE        = re.compile(r"^T[0-9]{4}(\.[0-9]{1,3})?$")
_DATE_RE            = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")

# ---------------------------------------------------------------------------
# Pattern for Strategy 2b: missing comma between two adjacent JSON values.
#
# The model occasionally writes two consecutive object properties without
# the required comma separator, e.g.:
#
#   "technique": "Phishing"
#   "evidence_snippet": "..."
#
# instead of:
#
#   "technique": "Phishing",
#   "evidence_snippet": "..."
#
# This regex matches a position between:
#   - a closing token: " (end of string), digit, true, false, null, } or ]
#   - optional whitespace / newlines
#   - an opening token: " (start of key/string), { or [
#
# and inserts a comma there.  The replacement is safe on valid JSON because
# a comma is never legal between those exact token pairs without one already
# being present.
# ---------------------------------------------------------------------------
_MISSING_COMMA_RE = re.compile(
    r'(["\d\]}\btrue\bfalse\bnull\b])'  # closing token
    r'(\s*\n\s*)'                         # newline-separated whitespace
    r'(["\[{])',                           # opening token
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_substring(text: str) -> str:
    """
    Isolate the outermost JSON object from model output.
    Robust to leading/trailing prose around the JSON blob.
    """
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in output. Head: {text[:200]}")
    return text[start:end + 1]


def _sanitise_json_string(text: str) -> str:
    """
    Best-effort repair of common JSON syntax errors produced by the model
    before handing the string to json.loads().

    Repairs applied (in order):
    1. Bare backslashes that are not part of a recognised JSON escape sequence
       (e.g. Windows paths like C:\\Users or LaTeX strings) are doubled so
       they become valid JSON string escapes.
    2. Trailing commas before a closing brace or bracket (]} ) are stripped.
       Some models emit these even though JSON does not permit them.
    """
    # --- 1. Fix invalid backslash escapes ---
    # Valid JSON escape sequences after a backslash: " \ / b f n r t u
    # Any other character following a backslash is illegal; double the slash.
    repaired = re.sub(
        r'\\(?!["\\/bfnrtu])',
        r'\\\\',
        text,
    )

    # --- 2. Remove trailing commas before ] or } ---
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    return repaired


def _insert_missing_commas(text: str) -> str:
    """
    Insert commas between adjacent JSON values that are separated only by
    a newline, with no existing comma.  
    """
    # Run iteratively until no further substitutions are possible, because a
    # single pass may miss consecutive missing commas.
    prev = None
    result = text
    while result != prev:
        prev = result
        result = _MISSING_COMMA_RE.sub(r'\1,\2\3', result)
    return result


def _try_parse_json(raw: str) -> dict:
    """
    Attempt to parse JSON from raw model output using up to four strategies:
    1. Direct parse after extracting the outermost object.
    2. Parse after applying _sanitise_json_string() repairs
    2b. Parse after additionally applying _insert_missing_commas()
    3. Parse after truncating to the last valid closing brace

    Raises ValueError if all strategies fail.
    """
    substring = _extract_json_substring(raw)

    # Strategy 1: direct parse
    try:
        return json.loads(substring)
    except json.JSONDecodeError:
        pass

    # Strategy 2: sanitise then parse
    sanitised = _sanitise_json_string(substring)
    try:
        return json.loads(sanitised)
    except json.JSONDecodeError:
        pass

    # Strategy 2b: additionally insert missing commas then parse
    comma_repaired = _insert_missing_commas(sanitised)
    try:
        return json.loads(comma_repaired)
    except json.JSONDecodeError:
        pass

    # Strategy 3: truncate to last complete top-level object.
    # Walk backwards from the end to find the deepest valid prefix.
    for end in range(len(comma_repaired), 0, -1):
        candidate = comma_repaired[:end]
        if candidate.rstrip().endswith(('}', ']', '"')) or candidate.rstrip()[-1:].isdigit():
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"JSON repair failed after four strategies. "
        f"First 300 chars: {raw[:300]}"
    )


def _normalise_ttp_id(raw_id: Optional[str]) -> Optional[str]:
    """Normalise T1566.1 -> T1566.001."""
    if not raw_id:
        return raw_id
    raw_id = raw_id.strip().upper()
    m = re.match(r"^(T\d{4})\.(\d{1,2})$", raw_id)
    if m:
        return f"{m.group(1)}.{int(m.group(2)):03d}"
    return raw_id


def _is_non_empty(value: Any) -> bool:
    """Return True iff value is a non-None, non-empty string after stripping."""
    return isinstance(value, str) and bool(value.strip())


def _prune_ioc_list(items: List[Any], label: str) -> List[Any]:
    """
    Remove IOC-like dicts that are missing, None, or empty in their
    required 'value' and 'evidence_snippet' fields.
    """
    kept = []
    for item in items:
        if not isinstance(item, dict):
            logger.debug(f"_prune_ioc_list [{label}]: dropped non-dict item {item!r}")
            continue
        if not _is_non_empty(item.get("value")):
            logger.debug(f"_prune_ioc_list [{label}]: dropped item with empty value: {item!r}")
            continue
        if not _is_non_empty(item.get("evidence_snippet")):
            logger.debug(f"_prune_ioc_list [{label}]: dropped item with empty evidence_snippet: {item!r}")
            continue
        kept.append(item)
    return kept


def _prune_hash_list(items: List[Any]) -> List[Any]:
    """
    Remove HashIOC dicts that fail the value/evidence_snippet checks
    or whose hash_type is not one of the five allowed literals.
    """
    kept = []
    for item in _prune_ioc_list(items, "file_hashes"):
        ht = item.get("hash_type")
        if not isinstance(ht, str) or ht.strip().upper() not in _VALID_HASH_TYPES:
            logger.debug(
                f"_prune_hash_list: dropped item with invalid hash_type {ht!r}: {item!r}"
            )
            continue
        item["hash_type"] = ht.strip().upper()
        kept.append(item)
    return kept


def _prune_tool_list(items: List[Any]) -> List[Any]:
    """
    Remove Tool dicts that are missing a non-empty 'name' or
    'evidence_snippet'.
    """
    kept = []
    for item in items:
        if not isinstance(item, dict):
            logger.debug(f"_prune_tool_list: dropped non-dict item {item!r}")
            continue
        if not _is_non_empty(item.get("name")):
            logger.debug(f"_prune_tool_list: dropped item with empty name: {item!r}")
            continue
        if not _is_non_empty(item.get("evidence_snippet")):
            logger.debug(f"_prune_tool_list: dropped item with empty evidence_snippet: {item!r}")
            continue
        kept.append(item)
    return kept


def _prune_ttp_list(items: List[Any]) -> List[Any]:
    """
    Remove TTP dicts where any of tactic, technique, or evidence_snippet
    is absent/None/empty, or where mitre_id is present but does not match
    the T####(.###)? pattern.
    """
    kept = []
    for item in items:
        if not isinstance(item, dict):
            logger.debug(f"_prune_ttp_list: dropped non-dict item {item!r}")
            continue
        if not _is_non_empty(item.get("tactic")):
            logger.debug(f"_prune_ttp_list: dropped TTP with empty tactic: {item!r}")
            continue
        if not _is_non_empty(item.get("technique")):
            logger.debug(f"_prune_ttp_list: dropped TTP with empty technique: {item!r}")
            continue
        if not _is_non_empty(item.get("evidence_snippet")):
            logger.debug(f"_prune_ttp_list: dropped TTP with empty evidence_snippet: {item!r}")
            continue
        mitre_id = item.get("mitre_id")
        if mitre_id is not None:
            # Attempt short-form normalisation first (e.g. T1566.1 -> T1566.001)
            mitre_id = _normalise_ttp_id(str(mitre_id))
            if not _MITRE_ID_RE.match(mitre_id):
                logger.debug(
                    f"_prune_ttp_list: cleared invalid mitre_id {item['mitre_id']!r}"
                )
                mitre_id = None
            item["mitre_id"] = mitre_id
        kept.append(item)
    return kept


def _prune_timeline_list(items: List[Any]) -> List[Any]:
    """
    Remove TimelineEvent dicts where date, event, or evidence_snippet are
    absent/None/empty, or where date does not match YYYY-MM-DD.
    """
    kept = []
    for item in items:
        if not isinstance(item, dict):
            logger.debug(f"_prune_timeline_list: dropped non-dict item {item!r}")
            continue
        if not _is_non_empty(item.get("event")):
            logger.debug(f"_prune_timeline_list: dropped event with empty event: {item!r}")
            continue
        if not _is_non_empty(item.get("evidence_snippet")):
            logger.debug(f"_prune_timeline_list: dropped event with empty evidence_snippet: {item!r}")
            continue
        date_val = item.get("date")
        if not isinstance(date_val, str) or not _DATE_RE.match(date_val.strip()):
            logger.debug(
                f"_prune_timeline_list: dropped event with invalid date {date_val!r}: {item!r}"
            )
            continue
        kept.append(item)
    return kept


def _prune_target_sectors(items: List[Any]) -> List[Any]:
    """Remove blank or non-string sector entries."""
    kept = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            logger.debug(f"_prune_target_sectors: dropped invalid sector {item!r}")
            continue
        kept.append(item)
    return kept


def _coerce_parsed(parsed: dict) -> dict:
    """
    Structural coercions applied after JSON parsing but before Pydantic
    validation. 
    """
    # --- report_metadata ---
    if not isinstance(parsed.get("report_metadata"), dict):
        parsed["report_metadata"] = {}
    rt = parsed["report_metadata"].get("report_type")
    if rt not in _VALID_REPORT_TYPES:
        parsed["report_metadata"]["report_type"] = "campaign"
        logger.debug(f"report_type coerced from {rt!r} to 'campaign'")

    # --- summary (required str, max 1500 chars) ---
    if not isinstance(parsed.get("summary"), str) or not parsed["summary"].strip():
        parsed["summary"] = "No summary provided by model."
        logger.debug("summary coerced to placeholder")
    elif len(parsed["summary"]) > 1500:
        parsed["summary"] = parsed["summary"][:1500]
        logger.debug("summary truncated to 1500 chars")

    # --- indicators_of_compromise ---
    if not isinstance(parsed.get("indicators_of_compromise"), dict):
        parsed["indicators_of_compromise"] = {}
    ioc = parsed["indicators_of_compromise"]

    ioc_defaults: Dict[str, Any] = {
        "ip_addresses": [],
        "domains": [],
        "file_hashes": [],
        "urls": [],
        "http_paths": [],
        "tools": [],
        "cves": [],
        "detection_signatures": {"yara_rules": [], "sigma_rules": []},
    }
    for k, v in ioc_defaults.items():
        ioc.setdefault(k, v)

    # Prune malformed list entries inside IOC sub-lists
    ioc["ip_addresses"]  = _prune_ioc_list(ioc.get("ip_addresses") or [], "ip_addresses")
    ioc["domains"]       = _prune_ioc_list(ioc.get("domains") or [], "domains")
    ioc["file_hashes"]   = _prune_hash_list(ioc.get("file_hashes") or [])
    ioc["urls"]          = _prune_ioc_list(ioc.get("urls") or [], "urls")
    ioc["http_paths"]    = _prune_ioc_list(ioc.get("http_paths") or [], "http_paths")
    ioc["cves"]          = _prune_ioc_list(ioc.get("cves") or [], "cves")
    ioc["tools"]         = _prune_tool_list(ioc.get("tools") or [])

    # detection_signatures sub-lists
    if isinstance(ioc.get("detection_signatures"), dict):
        ds = ioc["detection_signatures"]
        ds["yara_rules"]  = _prune_tool_list(ds.get("yara_rules") or [])
        ds["sigma_rules"] = _prune_tool_list(ds.get("sigma_rules") or [])
    else:
        ioc["detection_signatures"] = {"yara_rules": [], "sigma_rules": []}

    # --- ttps ---
    if not isinstance(parsed.get("ttps"), list):
        parsed["ttps"] = []
    parsed["ttps"] = _prune_ttp_list(parsed["ttps"])

    # --- top-level optional lists ---
    if not isinstance(parsed.get("target_sectors"), list):
        parsed["target_sectors"] = []
    parsed["target_sectors"] = _prune_target_sectors(parsed["target_sectors"])

    if not isinstance(parsed.get("timeline"), list):
        parsed["timeline"] = []
    parsed["timeline"] = _prune_timeline_list(parsed["timeline"])

    return parsed


def normalise_document(doc: CTIDocument) -> CTIDocument:
    """Light normalisation for evaluation consistency."""
    iocs = doc.indicators_of_compromise
    for item in iocs.ip_addresses:
        item.value = item.value.strip().lower()
    for item in iocs.domains:
        item.value = item.value.strip().lower()
    for item in iocs.file_hashes:
        item.value     = item.value.strip().lower()
        item.hash_type = item.hash_type.strip().upper()
    for item in iocs.urls:
        item.value = item.value.strip()
    for item in iocs.http_paths:
        item.value = item.value.strip()
    for item in iocs.cves:
        item.value = item.value.strip().upper()
    for item in iocs.tools:
        item.name = item.name.strip()
    for ttp in doc.ttps:
        if ttp.mitre_id:
            ttp.mitre_id = _normalise_ttp_id(ttp.mitre_id)
        ttp.tactic    = ttp.tactic.strip()
        ttp.technique = ttp.technique.strip()
    if doc.threat_actor:
        doc.threat_actor.name    = doc.threat_actor.name.strip()
        doc.threat_actor.aliases = [a.strip() for a in doc.threat_actor.aliases]
    doc.target_sectors = [s.strip() for s in doc.target_sectors]
    if doc.campaign_name:
        doc.campaign_name = doc.campaign_name.strip()
    if doc.summary:
        doc.summary = doc.summary.strip()
    return doc


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class BaselineExtractor:
    """
    V1 schema-guided JSON extraction baseline.

    Parameters
    ----------
    llm : Llama
        Shared llama-cpp-python Llama instance loaded once in run_pipeline.py.
    """

    def __init__(self, llm: Llama):
        self.llm = llm

    def _generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Run a single completion against the shared Llama instance."""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|eot_id|>"],
        )
        return response["choices"][0]["text"]

    def _parse_with_retry(self, prompt: str, raw_output: str) -> dict:
        """
        Parse JSON from raw_output using _try_parse_json() which applies up
        to four local repair strategies before touching the LLM.

        If all local repairs fail, send a SHORT corrective prompt containing
        only the broken JSON fragment (not the full original prompt) to avoid
        overflowing the context window on a retry.
        """
        try:
            return _try_parse_json(raw_output)
        except Exception as e:
            logger.warning(f"JSON parse failed after local repairs ({e}), retrying with LLM.")
            # Use a compact repair prompt rather than the full original prompt
            # to avoid exceeding the 16 384-token context window on retry.
            broken_fragment = _extract_json_substring(raw_output) if "{" in raw_output else raw_output
            repair_prompt = (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                "You are a JSON repair assistant. "
                "Fix the malformed JSON below so it is valid. "
                "Output ONLY the corrected JSON object with no prose.\n"
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"Malformed JSON:\n{broken_fragment[:3000]}\n\n"
                f"Parse error: {e}\n\n"
                "Corrected JSON:"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            retry_output = self._generate(repair_prompt, max_tokens=2048)
            return _try_parse_json(retry_output)

    def run_extraction(self, report_path: Path) -> CTIDocument:
        """Process one report and return a validated, normalised CTIDocument."""
        console.print(f"[V1] Processing: {report_path.name}")
        t0 = time.time()

        doc_data   = load_report(report_path)
        prompt     = build_baseline_prompt(doc_data)
        raw_output = self._generate(prompt)

        logger.debug(f"Raw output (first 500 chars):\n{raw_output[:500]}")

        parsed    = self._parse_with_retry(prompt, raw_output)
        parsed    = _coerce_parsed(parsed)
        validated = CTIDocument.model_validate(parsed)
        validated = normalise_document(validated)

        logger.info(f"{doc_data['doc_id']} done in {time.time() - t0:.1f}s")
        return validated

    def batch_extract(self, input_dir: Path, output_dir: Path):
        """Process all PDFs/HTMLs in input_dir; write validated JSON to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.html"))
        )
        if not paths:
            logger.warning(f"No reports found in {input_dir}")
            return
        for report_path in paths:
            try:
                result      = self.run_extraction(report_path)
                output_path = output_dir / f"{report_path.stem}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.model_dump_json(indent=2))
                console.print(f"  [green]✓[/green] {report_path.stem}")
            except Exception as e:
                logger.error(f"Failed on {report_path.name}: {e}", exc_info=True)
                console.print(f"  [red]✗[/red] {report_path.name}: {e}")
