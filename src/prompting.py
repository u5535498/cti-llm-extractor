"""
prompting.py — Prompt templates for all pipeline variants (V0, V1, V2, V3).

Few-shot example uses DOC007 (Mandiant M-Trends 2024, TTP-rich) as the single
representative example. One example rather than two keeps the prompt compact
and well within the 32768-token context window even for long documents.

Document text is truncated to MAX_DOC_CHARS before insertion into the prompt
to prevent context-window overflow on verbose PDF reports.

Note: BOS (<|begin_of_text|>) is intentionally omitted from all prompt strings
because llama-cpp-python prepends it automatically when calling llm().
Including it manually causes a duplicate-BOS warning and degrades output quality.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).parent.parent / "schema_v0.2.json"
with open(SCHEMA_PATH, "r", encoding="utf-8") as _f:
    SCHEMA_JSON = json.dumps(json.load(_f), indent=2)

MAX_DOC_CHARS = 42_000

# ---------------------------------------------------------------------------
# Llama-3.1 special tokens
# BOS is deliberately excluded — llama-cpp-python adds it automatically.
# ---------------------------------------------------------------------------

SYS_OPEN  = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_OPEN = "<|start_header_id|>user<|end_header_id|>\n\n"
ASS_OPEN  = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT       = "<|eot_id|>"

# ---------------------------------------------------------------------------
# System message (shared by V1, V2, V3)
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    SYS_OPEN
    + "You are a precise cyber threat intelligence (CTI) analyst assistant. "
    + "Extract structured threat intelligence from the document and return it "
    + "as valid JSON that strictly conforms to the schema below.\n\n"
    + "RULES:\n"
    + "1. Only extract information EXPLICITLY stated in the document.\n"
    + "2. For every evidence_snippet field, copy a VERBATIM sentence, which"
    + "MUST contain and immediately surround the extracted value. Do NOT paraphrase, "
    + "summarise, or miss out the evidence item — copy the full "
    + "snippet as it appears in the source.\n"
    + "3. If a field cannot be populated, use null or [] — never invent values.\n"
    + "4. Output ONLY the JSON object. No prose, no markdown fences.\n"
    + "5. report_metadata.report_type MUST be exactly one of: incident, campaign, strategic.\n"
    + "6. summary is REQUIRED — write 1–3 sentences summarising the document.\n\n"
    + "SCHEMA:\n{SCHEMA_JSON}\n"
    + EOT
)

# ---------------------------------------------------------------------------
# Single few-shot example — DOC007 (Mandiant M-Trends 2024, TTP-rich)
# Compact subset: 3 CVEs, 3 tools, 4 TTPs, 1 timeline entry.
# All evidence_snippets are ≥15 words and contain the extracted value.
# ---------------------------------------------------------------------------

FEW_SHOT_DOC007_TEXT = """Mandiant M-Trends 2024 — Annual Threat Intelligence Report

This report covers Mandiant Consulting investigations between January 1 2023
and December 31 2023. The global median dwell time fell to 10 days.
Ransomware accounted for 23 percent of all investigations.

The most prevalent vulnerability Mandiant investigators observed in 2023 was
CVE-2023-34362, an SQL injection flaw in MOVEit Transfer exploited by FIN11.
The second most exploited was CVE-2022-21587, a critical unauthenticated file
upload vulnerability in Oracle E-Business Suite. The third was CVE-2023-2868,
a critical command injection vulnerability in Barracuda Email Security Gateways.

BEACON remains the most frequently observed malware family in Mandiant
investigations globally, appearing in 10 percent of cases.
ALPHV ransomware was identified in 5% of Mandiant led investigations in 2023.
LEMURLOOT is a web shell written in C# tailored to interact with the MOVEit
Transfer platform and was deployed following MOVEit exploitation.

Top ATT&CK techniques observed: T1059 Command and Scripting Interpreter,
T1027 Obfuscated Files or Information 46.5%, T1190 Exploit Public-Facing
Application 28.7%, T1566 Phishing 16.3%."""

FEW_SHOT_DOC007_JSON = json.dumps({
    "report_metadata": {
        "report_title": "M-Trends 2024",
        "report_type": "strategic"
    },
    "target_sectors": ["Financial Services", "High Tech", "Healthcare", "Government"],
    "indicators_of_compromise": {
        "ip_addresses": [], "domains": [], "file_hashes": [],
        "urls": [], "http_paths": [],
        "detection_signatures": {"yara_rules": [], "sigma_rules": []},
        "cves": [
            {"value": "CVE-2023-34362",
             "evidence_snippet": "The most prevalent vulnerability Mandiant investigators observed in 2023 was CVE-2023-34362, an SQL injection flaw in MOVEit Transfer exploited by FIN11."},
            {"value": "CVE-2022-21587",
             "evidence_snippet": "The second most exploited was CVE-2022-21587, a critical unauthenticated file upload vulnerability in Oracle E-Business Suite."},
            {"value": "CVE-2023-2868",
             "evidence_snippet": "The third was CVE-2023-2868, a critical command injection vulnerability in Barracuda Email Security Gateways."}
        ],
        "tools": [
            {"name": "BEACON",
             "evidence_snippet": "BEACON remains the most frequently observed malware family in Mandiant investigations globally, appearing in 10 percent of cases."},
            {"name": "ALPHV",
             "evidence_snippet": "ALPHV ransomware was identified in 5% of Mandiant led investigations in 2023."},
            {"name": "LEMURLOOT",
             "evidence_snippet": "LEMURLOOT is a web shell written in C# tailored to interact with the MOVEit Transfer platform and was deployed following MOVEit exploitation."}
        ]
    },
    "ttps": [
        {"mitre_id": "T1059", "tactic": "Execution",
         "technique": "Command and Scripting Interpreter",
         "evidence_snippet": "Top ATT&CK techniques observed: T1059 Command and Scripting Interpreter, T1027 Obfuscated Files or Information 46.5%."},
        {"mitre_id": "T1027", "tactic": "Defense Evasion",
         "technique": "Obfuscated Files or Information",
         "evidence_snippet": "T1027 Obfuscated Files or Information 46.5% was among the top ATT&CK techniques observed in Mandiant investigations."},
        {"mitre_id": "T1190", "tactic": "Initial Access",
         "technique": "Exploit Public-Facing Application",
         "evidence_snippet": "T1190 Exploit Public-Facing Application 28.7% was observed as a top initial access technique in Mandiant investigations."},
        {"mitre_id": "T1566", "tactic": "Initial Access",
         "technique": "Phishing",
         "evidence_snippet": "T1566 Phishing 16.3% was identified among the top ATT&CK techniques in Mandiant investigations during 2023."}
    ],
    "summary": "Mandiant M-Trends 2024 covers 2023 global investigations. Median dwell time fell to 10 days. Top exploited CVEs were CVE-2023-34362 (MOVEit), CVE-2022-21587 (Oracle), CVE-2023-2868 (Barracuda).",
    "timeline": [
        {"date": "2023-05-27",
         "event": "First known exploitation of CVE-2023-34362 (MOVEit)",
         "evidence_snippet": "Earliest evidence of exploitation of CVE-2023-34362 occurred on May 27, 2023, when threat actors began targeting MOVEit Transfer installations."}
    ]
}, indent=2)

FEW_SHOT_BLOCK = (
    "--- EXAMPLE ---\n"
    "Document:\n" + FEW_SHOT_DOC007_TEXT + "\n\n"
    "Output:\n" + FEW_SHOT_DOC007_JSON + "\n"
)

# ---------------------------------------------------------------------------
# V0 prompt  (free-text, no schema, no few-shot — ablation baseline)
# ---------------------------------------------------------------------------

def build_v0_prompt(doc_data: dict) -> str:
    """
    V0: unstructured free-text prompt. No JSON schema, no few-shot examples.
    Evidence snippets are not requested. This is the ablation condition.
    """
    doc_text = doc_data["text"][:MAX_DOC_CHARS]
    system = (
        SYS_OPEN
        + "You are a cyber threat intelligence analyst. "
        + "Read the threat report below and extract the key information, "
        + "including threat actors, indicators of compromise, tactics and "
        + "techniques, target sectors, and a brief summary.\n"
        + EOT
    )
    user = (
        USER_OPEN
        + f"Document ID: {doc_data['doc_id']}\n"
        + f"Title: {doc_data.get('title', '')}\n\n"
        + "Document text:\n"
        + doc_text
        + "\n\nExtract the threat intelligence from this document."
        + EOT
        + ASS_OPEN
    )
    return system + user


# ---------------------------------------------------------------------------
# V1 prompt  (schema-guided, one few-shot example)
# ---------------------------------------------------------------------------

def build_baseline_prompt(doc_data: dict) -> str:
    """
    V1 baseline: schema-guided single-pass extraction with one few-shot
    example. Document text is truncated to MAX_DOC_CHARS to prevent
    context-window overflow.
    """
    doc_text = doc_data["text"][:MAX_DOC_CHARS]
    if len(doc_data["text"]) > MAX_DOC_CHARS:
        logger.debug(
            f"{doc_data['doc_id']}: truncated from "
            f"{len(doc_data['text'])} to {MAX_DOC_CHARS} chars."
        )
    system = SYSTEM_MESSAGE.format(SCHEMA_JSON=SCHEMA_JSON)
    user = (
        USER_OPEN
        + FEW_SHOT_BLOCK
        + "--- EXTRACT FROM THIS DOCUMENT ---\n"
        + f"Document ID: {doc_data['doc_id']}\n"
        + f"Title: {doc_data.get('title', '')}\n\n"
        + "Document text:\n"
        + doc_text
        + "\n\nOutput JSON only:"
        + EOT
        + ASS_OPEN
    )
    return system + user


# ---------------------------------------------------------------------------
# V2 prompt  (identical to V1 — grounding is post-processing only)
# ---------------------------------------------------------------------------

def build_grounded_prompt(doc_data: dict) -> str:
    """V2 uses the same prompt as V1; evidence grounding is post-processing."""
    return build_baseline_prompt(doc_data)


# ---------------------------------------------------------------------------
# V3 — CoVe prompts  (three separate LLM calls per document)
# ---------------------------------------------------------------------------

def build_cove_initial_prompt(doc_data: dict) -> str:
    """V3 step 1: initial extraction — identical to V1 prompt."""
    return build_baseline_prompt(doc_data)


def build_cove_verification_prompt(
    doc_text: str,
    initial_json: str,
    questions: list,
) -> str:
    """V3 step 2: answer each verification question using only the source document."""
    doc_text = doc_text[:MAX_DOC_CHARS]
    q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    system = (
        SYS_OPEN
        + "You are a fact-checker for cyber threat intelligence reports. "
        + "Answer each question using ONLY the document text provided. "
        + "If the answer is not in the document, state 'NOT FOUND'. "
        + "One sentence per answer."
        + EOT
    )
    user = (
        USER_OPEN
        + "Document text:\n" + doc_text
        + "\n\nInitial extraction:\n" + initial_json
        + "\n\nVerification questions:\n" + q_block
        + "\n\nAnswer each question in order, numbered to match:"
        + EOT + ASS_OPEN
    )
    return system + user


def build_cove_revision_prompt(
    doc_text: str,
    initial_json: str,
    qa_pairs: str,
) -> str:
    """V3 step 3: revise the initial extraction given verification answers.

    IMPORTANT: The prompt explicitly instructs the model to copy ALL fields
    from the initial JSON verbatim and to modify ONLY the specific IoC/TTP
    items flagged NOT SUPPORTED.  This prevents the model from silently
    dropping required schema fields (e.g. threat_actor.confidence) when
    rewriting the JSON object.
    """
    doc_text = doc_text[:MAX_DOC_CHARS]
    system = (
        SYS_OPEN
        + "You are a cyber threat intelligence analyst. "
        + "Revise the JSON extraction below to remove or correct any claims "
        + "contradicted by the verification answers. "
        + "Output ONLY the corrected JSON object.\n\n"
        + "REVISION RULES:\n"
        + "1. Copy ALL fields from the initial extraction verbatim, including "
        + "report_metadata, summary, threat_actor (with ALL sub-fields such as "
        + "name, aliases, confidence, and evidence_snippet), target_sectors, "
        + "campaign_name, and timeline.\n"
        + "2. Only modify the specific indicators_of_compromise or ttps items "
        + "that are explicitly flagged NOT SUPPORTED in the verification Q&A.\n"
        + "3. Never add, rename, or remove any field from the JSON structure. "
        + "The output must conform to exactly the same schema as the input.\n"
        + "4. If threat_actor is present in the initial extraction, it MUST "
        + "appear in the output with every sub-field intact (name, aliases, "
        + "confidence, evidence_snippet)."
        + EOT
    )
    user = (
        USER_OPEN
        + "Source document:\n" + doc_text
        + "\n\nInitial extraction:\n" + initial_json
        + "\n\nVerification Q&A:\n" + qa_pairs
        + "\n\nRevised JSON:"
        + EOT + ASS_OPEN
    )
    return system + user
