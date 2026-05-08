"""
V0 Ablation: Free-text extraction + deterministic parsing.

Purpose: Ablation baseline to quantify the value of schema-guided JSON output (V1).
Design:  The LLM is given NO JSON instruction. Its prose output is parsed
         deterministically via regex. Because no evidence_snippet is ever
         populated, the unsupported rate is 1.0 by construction. This is the
         intended experimental signal, not a bug.

Outputs a CTIDocument-compatible dict written to outputs/v0/<doc_id>.json.
NOTE: Because V0 outputs cannot pass full Pydantic validation (missing required
      evidence_snippet fields), outputs are saved as raw dicts rather than
      CTIDocument objects. evaluation.py handles this via a separate V0 loader.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_cpp import Llama
from rich.console import Console

from src.document_loader import load_report
from src.utils_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
console = Console()

# Maximum characters of document text passed to the model.
# Dense technical CTI content (IPs, hashes, URLs) tokenises at ~1 char/token
# rather than the typical 4:1 ratio, so 10 000 chars keeps us safely within
# the 32 768-token context window once the system prompt is included.
MAX_DOC_CHARS = 10_000

# ---------------------------------------------------------------------------
# Prompt — deliberately NO JSON instruction, NO schema reference
# ---------------------------------------------------------------------------

V0_SYSTEM_MESSAGE = """\
You are a cyber threat intelligence analyst. Read the threat report below and \
extract the following in plain prose (do NOT use JSON or code blocks):

1. THREAT ACTOR: Name and any known aliases.
2. TARGET SECTORS: Industries or sectors targeted.
3. IP ADDRESSES: List each observed IP address on a new line prefixed "IP: ".
4. DOMAINS: List each observed domain on a new line prefixed "DOMAIN: ".
5. FILE HASHES: List each hash on a new line prefixed "HASH: <type> <value>" \
   where type is MD5, SHA1, SHA256, SHA512, or SSDEEP.
6. URLS: List each observed URL on a new line prefixed "URL: ".
7. CVEs: List each CVE on a new line prefixed "CVE: ".
8. TOOLS/MALWARE: List each tool or malware family prefixed "TOOL: ".
9. TTPS: List each MITRE ATT&CK technique on a new line as \
   "TTP: <technique_id> <tactic> - <technique_name>".
10. SUMMARY: Write 2-4 sentences summarising the campaign.

Only include information EXPLICITLY present in the document. \
Do NOT invent or infer. If a category has no entries, write "None found."
"""


def build_v0_messages(doc_data: dict) -> list:
    """Build free-text extraction messages list (no schema, no JSON).
    """
    doc_text = doc_data["text"][:MAX_DOC_CHARS]
    if len(doc_data["text"]) > MAX_DOC_CHARS:
        logger.debug(
            f"[V0] Truncated {doc_data['doc_id']} from "
            f"{len(doc_data['text'])} to {MAX_DOC_CHARS} chars"
        )
    return [
        {"role": "system", "content": V0_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                f"Document ID: {doc_data['doc_id']}\n"
                f"Title: {doc_data['title']}\n\n"
                f"Document text:\n{doc_text}\n\n"
                "Extract from this document."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Deterministic parser
# ---------------------------------------------------------------------------

_TTP_ID_RE = re.compile(r"^(T\d{4})\.(\d{1,2})$", re.IGNORECASE)

def _normalise_ttp_id(raw_id: str) -> str:
    """Normalise T1566.1 -> T1566.001."""
    m = _TTP_ID_RE.match(raw_id.strip())
    if m:
        return f"{m.group(1)}.{int(m.group(2)):03d}"
    return raw_id.strip().upper()


_PATTERNS: Dict[str, re.Pattern] = {
    "ip": re.compile(
        r"\b(?:IP:\s*)?((?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "domain": re.compile(
        r"\b(?:DOMAIN:\s*)?([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z]{2,})+)\b"
    ),
    "hash_sha256": re.compile(r"\b([A-Fa-f0-9]{64})\b"),
    "hash_sha1":   re.compile(r"\b([A-Fa-f0-9]{40})\b"),
    "hash_md5":    re.compile(r"\b([A-Fa-f0-9]{32})\b"),
    "url": re.compile(r"https?://[^\s\"'<>]+"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
    "tool": re.compile(r"(?:TOOL|MALWARE):\s*(.+)", re.IGNORECASE),
    "ttp": re.compile(
        r"TTP:\s*(T\d{4}(?:\.\d{1,3})?)\s+([A-Za-z /\-]+?)\s*-\s*(.+)",
        re.IGNORECASE
    ),
    "ttp_fallback": re.compile(
        r"\b(T\d{4}(?:\.\d{3})?)\b.*?([A-Z][a-z /\-]{3,40})",
    ),
}

_HASH_TYPES = {32: "MD5", 40: "SHA1", 64: "SHA256", 128: "SHA512"}

_DOMAIN_EXCLUDE = re.compile(
    r"(microsoft\.com|google\.com|github\.com|example\.com|localhost)", re.IGNORECASE
)


@dataclass
class V0ParseResult:
    """Structured output of the V0 deterministic parser."""
    ip_addresses:  List[str] = field(default_factory=list)
    domains:       List[str] = field(default_factory=list)
    file_hashes:   List[Tuple[str, str]] = field(default_factory=list)
    urls:          List[str] = field(default_factory=list)
    cves:          List[str] = field(default_factory=list)
    tools:         List[str] = field(default_factory=list)
    ttps:          List[Dict[str, str]] = field(default_factory=list)
    summary:       Optional[str] = None
    parse_errors:  List[str] = field(default_factory=list)
    raw_text:      str = ""


def parse_freetext(raw_output: str) -> V0ParseResult:
    """
    Deterministically parse the LLM free-text output into structured fields.
    Uses line-by-line prefix matching first; falls back to global regex scan.
    """
    result = V0ParseResult(raw_text=raw_output)
    seen_ips, seen_domains, seen_hashes, seen_urls, seen_cves, seen_tools = (
        set(), set(), set(), set(), set(), set()
    )

    lines = raw_output.splitlines()
    summary_lines: List[str] = []
    in_summary = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        upper = stripped.upper()

        if upper.startswith("SUMMARY:") or upper.startswith("10."):
            in_summary = True
            tail = stripped.split(":", 1)[-1].strip()
            if tail:
                summary_lines.append(tail)
            continue
        if in_summary:
            if re.match(r"^\d+\.", stripped) or re.match(
                r"^(IP|DOMAIN|HASH|URL|CVE|TOOL|TTP|THREAT|TARGET):", upper
            ):
                in_summary = False
            else:
                summary_lines.append(stripped)
                continue

        if upper.startswith("IP:"):
            val = stripped[3:].strip()
            m = _PATTERNS["ip"].search(val)
            if m and val not in seen_ips:
                seen_ips.add(val)
                result.ip_addresses.append(val)
            continue

        if upper.startswith("DOMAIN:"):
            val = stripped[7:].strip().lower()
            if val and val not in seen_domains and not _DOMAIN_EXCLUDE.search(val):
                seen_domains.add(val)
                result.domains.append(val)
            continue

        if upper.startswith("HASH:"):
            parts = stripped[5:].strip().split(None, 1)
            if len(parts) == 2:
                htype, hval = parts[0].upper(), parts[1].strip()
                if htype not in {"MD5", "SHA1", "SHA256", "SHA512", "SSDEEP"}:
                    htype = _HASH_TYPES.get(len(hval), "SHA256")
                key = (htype, hval)
                if key not in seen_hashes:
                    seen_hashes.add(key)
                    result.file_hashes.append(key)
            elif len(parts) == 1:
                hval = parts[0].strip()
                htype = _HASH_TYPES.get(len(hval), "SHA256")
                key = (htype, hval)
                if key not in seen_hashes:
                    seen_hashes.add(key)
                    result.file_hashes.append(key)
            continue

        if upper.startswith("URL:"):
            val = stripped[4:].strip()
            if val and val not in seen_urls:
                seen_urls.add(val)
                result.urls.append(val)
            continue

        if upper.startswith("CVE:"):
            val = stripped[4:].strip().upper()
            if val and val not in seen_cves:
                seen_cves.add(val)
                result.cves.append(val)
            continue

        if upper.startswith("TOOL:") or upper.startswith("MALWARE:"):
            val = re.sub(r"^(TOOL|MALWARE):\s*", "", stripped, flags=re.IGNORECASE).strip()
            if val and val not in seen_tools:
                seen_tools.add(val)
                result.tools.append(val)
            continue

        m_ttp = _PATTERNS["ttp"].search(stripped)
        if m_ttp:
            raw_id, tactic, technique = (
                m_ttp.group(1), m_ttp.group(2).strip(), m_ttp.group(3).strip()
            )
            result.ttps.append({
                "mitre_id":  _normalise_ttp_id(raw_id),
                "tactic":    tactic,
                "technique": technique,
            })
            continue

    # Fallback: global regex scan
    for m in _PATTERNS["ip"].finditer(raw_output):
        val = m.group(0).strip()
        if val not in seen_ips:
            seen_ips.add(val)
            result.ip_addresses.append(val)

    for m in _PATTERNS["url"].finditer(raw_output):
        val = m.group(0).strip()
        if val not in seen_urls:
            seen_urls.add(val)
            result.urls.append(val)

    for m in _PATTERNS["cve"].finditer(raw_output):
        val = m.group(0).upper()
        if val not in seen_cves:
            seen_cves.add(val)
            result.cves.append(val)

    for pattern_key in ("hash_sha256", "hash_sha1", "hash_md5"):
        for m in _PATTERNS[pattern_key].finditer(raw_output):
            val = m.group(1)
            htype = _HASH_TYPES.get(len(val), "SHA256")
            key = (htype, val)
            if key not in seen_hashes:
                seen_hashes.add(key)
                result.file_hashes.append(key)

    if not result.ttps:
        for m in _PATTERNS["ttp_fallback"].finditer(raw_output):
            result.ttps.append({
                "mitre_id":  _normalise_ttp_id(m.group(1)),
                "tactic":    "unknown",
                "technique": m.group(2).strip(),
            })

    result.summary = " ".join(summary_lines).strip() or None
    return result


# ---------------------------------------------------------------------------
# Serialiser: V0ParseResult -> CTIDocument-compatible dict
# ---------------------------------------------------------------------------

def serialise_to_schema(parsed: V0ParseResult, doc_id: str) -> Dict[str, Any]:
    """
    Convert V0ParseResult into a dict mirroring the CTIDocument schema.
    """
    EMPTY_SNIPPET = ""

    iocs: Dict[str, Any] = {
        "ip_addresses": [
            {"value": ip, "evidence_snippet": EMPTY_SNIPPET}
            for ip in parsed.ip_addresses
        ],
        "domains": [
            {"value": d, "evidence_snippet": EMPTY_SNIPPET,
             "masked": False, "phishing_template": False}
            for d in parsed.domains
        ],
        "file_hashes": [
            {"value": hval, "hash_type": htype, "evidence_snippet": EMPTY_SNIPPET}
            for htype, hval in parsed.file_hashes
        ],
        "urls": [
            {"value": url, "evidence_snippet": EMPTY_SNIPPET}
            for url in parsed.urls
        ],
        "http_paths": [],
        "tools": [
            {"name": t, "evidence_snippet": EMPTY_SNIPPET}
            for t in parsed.tools
        ],
        "cves": [
            {"value": cve, "evidence_snippet": EMPTY_SNIPPET}
            for cve in parsed.cves
        ],
        "detection_signatures": {"yara_rules": [], "sigma_rules": []},
    }

    ttps = [
        {
            "mitre_id":         ttp.get("mitre_id"),
            "tactic":           ttp.get("tactic", "unknown"),
            "technique":        ttp.get("technique", "unknown"),
            "evidence_snippet": EMPTY_SNIPPET,
            "mapping_type":     "explicit",
        }
        for ttp in parsed.ttps
    ]

    return {
        "doc_id":                    doc_id,
        "variant":                   "v0",
        "report_metadata":           {"report_type": "campaign"},
        "threat_actor":              None,
        "campaign_name":             None,
        "target_sectors":            [],
        "indicators_of_compromise":  iocs,
        "ttps":                      ttps,
        "summary":                   (parsed.summary or "")[:1500],
        "timeline":                  [],
        "parse_errors":              parsed.parse_errors,
    }


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class V0Extractor:
    """
    V0 ablation pipeline.

    Accepts a shared Llama instance from run_pipeline.py so the model
    is loaded only once across all variants.

    Parameters
    ----------
    llm : Llama
        Shared llama-cpp-python Llama instance.
    """

    def __init__(self, llm: Llama):
        self.llm = llm

    def _generate(self, messages: list, max_tokens: int = 1024) -> str:
        """
        V0 uses lower max_tokens than V1 (1024 vs 2048) because free-text
        output is more compact than structured JSON.
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        return response["choices"][0]["message"]["content"]

    def run_extraction(self, report_path: Path) -> Dict[str, Any]:
        """Full V0 pipeline for a single document. Returns schema-compatible dict."""
        console.print(f"[V0] Processing: {report_path.name}")
        t0 = time.time()

        doc_data = load_report(report_path)
        messages = build_v0_messages(doc_data)

        raw_output = self._generate(messages)
        logger.debug(f"V0 raw output ({len(raw_output)} chars):\n{raw_output[:500]}")

        parsed = parse_freetext(raw_output)

        if parsed.parse_errors:
            logger.warning(f"V0 parse errors for {doc_data['doc_id']}: {parsed.parse_errors}")

        result = serialise_to_schema(parsed, doc_data["doc_id"])
        result["inference_time_s"] = round(time.time() - t0, 2)

        console.print(
            f"[V0] Done: {len(parsed.ip_addresses)} IPs, "
            f"{len(parsed.domains)} domains, "
            f"{len(parsed.ttps)} TTPs"
        )
        return result

    def batch_extract(self, input_dir: Path, output_dir: Path):
        """Process all PDFs/HTMLs in input_dir, writing JSON to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_paths = sorted(
            list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.html"))
        )
        if not report_paths:
            logger.warning(f"No reports found in {input_dir}")
            return
        for report_path in report_paths:
            try:
                result    = self.run_extraction(report_path)
                out_path  = output_dir / f"{report_path.stem}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                console.print(f"  [green]✓[/green] {report_path.stem}")
            except Exception as e:
                logger.error(f"V0 failed on {report_path.name}: {e}", exc_info=True)
                console.print(f"  [red]✗[/red] {report_path.name}")


if __name__ == "__main__":
    import argparse
    from src.run_pipeline import load_model

    parser = argparse.ArgumentParser(description="V0 free-text CTI extractor")
    parser.add_argument("--input",      required=True, help="Path to PDF or HTML report")
    parser.add_argument("--output",     help="Output JSON path (optional)")
    parser.add_argument("--batch-dir",  help="Directory of reports for batch mode")
    parser.add_argument("--output-dir", default="outputs/v0", help="Batch output dir")
    parser.add_argument("--model",      default="models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf")
    args = parser.parse_args()

    llm       = load_model(args.model)
    extractor = V0Extractor(llm)

    if args.batch_dir:
        extractor.batch_extract(Path(args.batch_dir), Path(args.output_dir))
    else:
        result = extractor.run_extraction(Path(args.input))
        out    = args.output or f"outputs/v0/{Path(args.input).stem}.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out}")
