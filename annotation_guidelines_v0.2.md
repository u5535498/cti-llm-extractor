# CTI Extraction Annotation Guidelines v0.2

**Date:** 23 February 2026  
**Schema version:** v0.2  
**Purpose:** These guidelines define how to apply `schema_v0.2.json` when annotating cyber threat intelligence (CTI) documents for the dissertation gold-standard dataset.

---

## 1. Overview and principles

### 1.1 General approach
- **Extract only what is explicitly stated or clearly implied** in the source document.
- **Anchor all extractions to evidence snippets** from the original text.
- **Distinguish report types** (incident, campaign, strategic) as they affect annotation expectations.
- **Empty arrays are acceptable** when the document does not provide information for a field (especially common in strategic reports).
- **Consistency over coverage**: extract systematically rather than exhaustively; follow the decision rules below.

### 1.2 Schema structure reminder
Each annotated document is a JSON object containing:
- `report_metadata` – report title and type (required)
- `threat_actor` – actor name, aliases, confidence (optional)
- `campaign_name` – operational campaign identifier (optional)
- `target_sectors` – list of victim sectors
- `indicators_of_compromise` – IPs, domains, hashes, URLs, HTTP paths, tools, CVEs, detection signatures
- `ttps` – tactics, techniques and evidence, with optional MITRE IDs
- `summary` – free-text overview (max 1500 characters)
- `timeline` – chronological events with dates and precision

---

## 2. Report metadata

### 2.1 report_type (REQUIRED)
**Purpose:** Categorise the document genre to enable stratified analysis.

**Enum values:**
- `"incident"` – detailed case study of a specific intrusion or compromise (e.g. "BlackByte ransomware intrusion at Organisation X").
- `"campaign"` – report covering a coordinated set of operations by an actor or multiple related incidents (e.g. "Scattered Spider phishing campaigns, July–October 2025").
- `"strategic"` – threat landscape, trend analysis or sector-wide overview (e.g. "ENISA Threat Landscape 2025").

**Decision rules:**
- If the report describes a single victim organisation's intrusion timeline → `"incident"`.
- If it describes a coordinated set of related intrusions by a single actor or across multiple targets → `"campaign"`.
- If it aggregates threats, techniques or trends across a sector, region or time period → `"strategic"`.

**Examples:**

| Report title / description | report_type | Rationale |
|----------------------------|-------------|-----------|
| "The five-day job: A BlackByte ransomware intrusion case study" | `"incident"` | Single intrusion, specific victim, detailed timeline |
| "Scattered Spider: Scattered Spider threat actors and their techniques" | `"campaign"` | Actor-centric, describes multiple phishing operations |
| "ENISA Threat Landscape 2025" | `"strategic"` | EU-wide threat overview, multiple actor types, trend analysis |
| "CISA Advisory: Malicious Listener for Ivanti Endpoint Manager Mobile" | `"incident"` or `"campaign"` | Choose `"incident"` if single exploitation case; `"campaign"` if multiple victims noted |

### 2.2 report_title (optional but recommended)
**Purpose:** Capture the formal or internal title of the document.

**Examples:**
- `"ENISA Threat Landscape 2025"`
- `"The five-day job: A BlackByte ransomware intrusion case study"`
- `"Malicious Listener for Ivanti Endpoint Manager Mobile (CISA AR25-261A)"`

---

## 3. Threat actor

### 3.1 When to populate
- Include `threat_actor` **only when the document explicitly attributes activity** to a named actor or group.
- If the report says "unattributed", "unknown actor" or does not mention an actor → **omit the entire `threat_actor` object**.

### 3.2 Fields
- `name` – the primary or most commonly used name in the document.
- `aliases` – array of all alternative names mentioned (e.g. vendor-specific designators).
- `confidence` – `"high"`, `"medium"` or `"low"` as stated or implied by the report.

**Positive examples:**

```json
{
  "threat_actor": {
    "name": "Scattered Spider",
    "aliases": ["UNC3944", "Scatter Swine", "Oktapus", "Octo Tempest", "Storm-0875", "Muddled Libra"],
    "confidence": "high"
  }
}
```

```json
{
  "threat_actor": {
    "name": "BlackByte",
    "aliases": ["BlackByte 2.0"],
    "confidence": "high"
  }
}
```

**Negative examples (omit threat_actor):**

- Report describes tactics without naming an actor.
- Strategic report covering many actors without singling one out as primary.
- Document states "attribution is ongoing" or "actor unknown".

---

## 4. Campaign name

### 4.1 When to populate
- Use `campaign_name` for **operational campaigns** only (e.g. named intrusion sets, attack waves).
- **Do not** use it for report titles or strategic documents.

**Positive examples:**
- `"Operation ShadowStrike"` (if the campaign is named in the report)
- `"Ivanti EPMM exploitation wave, May 2025"`

**Negative examples:**
- `"ENISA Threat Landscape 2025"` → this is a report title; use `report_metadata.report_title` instead.
- `"BlackByte ransomware campaign"` → too generic; omit unless the report uses a specific campaign name.

---

## 5. Target sectors

### 5.1 Approach
- Extract all sectors or industries explicitly mentioned as victims or targets.
- Use the terminology from the source document (e.g. "Finance", "Public administration", "Healthcare").
- Leave empty if no sectors are specified.

**Examples:**

```json
"target_sectors": ["Commercial Facilities"]
```

```json
"target_sectors": ["Public administration", "Transport", "Digital infrastructure and services", "Finance", "Manufacturing"]
```

---

## 6. Indicators of compromise (IoCs)

### 6.1 General principles
- Capture atomic, actionable indicators: IPs, domains, hashes, URLs, HTTP paths, tools, CVEs, detection rules.
- **Always provide an `evidence_snippet`** showing where the IoC appears in the text.
- Use optional `source_context` to distinguish `"observed"` (live telemetry), `"detection_rule_only"` (appears in YARA/SIGMA), or `"uncertain"`.

### 6.2 IP addresses

**Include:**
- IPs used for C2, exploitation, scanning, exfiltration.

**Optional `source_context`:**
- `"observed"` – seen in network logs, forensic evidence.
- `"detection_rule_only"` – IP appears only in a SIGMA rule or detection logic.

**Examples:**

```json
{
  "value": "185.225.73.244",
  "evidence_snippet": "The threat actor was observed operating from the following IP to exploit ProxyShell: 185.225.73[.]244"
}
```

```json
{
  "value": "82.132.235.212",
  "evidence_snippet": "82.132.235.212",
  "source_context": "detection_rule_only"
}
```

### 6.3 Domains

**Include:**
- C2 domains, phishing domains, hosting domains.

**Optional flags:**
- `masked`: `true` if the domain is written with defanging notation (e.g. `hxxps://`, `[.]`).
- `phishing_template`: `true` if the domain is a synthetic template (e.g. `targetsname-sso[.]com`) rather than a live, resolvable domain.

**Examples:**

```json
{
  "value": "myvisit.alteksecurity.org",
  "evidence_snippet": "hxxps://myvisit[.]alteksecurity[.]org/t",
  "masked": true
}
```

```json
{
  "value": "targetsname-sso.com",
  "evidence_snippet": "targetsname-sso[.]com targetsname-servicedesk[.]com targetsname-okta[.]com",
  "masked": true,
  "phishing_template": true
}
```

### 6.4 File hashes

**Include:**
- Hashes for malware samples, tools, dropper files.

**hash_type enum:**
- `"MD5"`, `"SHA1"`, `"SHA256"`, `"SHA512"`, `"SSDEEP"`.

**Examples:**

```json
{
  "value": "4a066569113a569a6feb8f44257ac8764ee8f2011765009fdfd82fe3f4b92d3e",
  "hash_type": "SHA256",
  "evidence_snippet": "api-msvc.dll (SHA-256: 4a066569113a569a6feb8f44257ac8764ee8f2011765009fdfd82fe3f4b92d3e)"
}
```

```json
{
  "value": "e33103767524879293d1b576a8b6257d",
  "hash_type": "MD5",
  "evidence_snippet": "MD5 e33103767524879293d1b576a8b6257d"
}
```

### 6.5 URLs

**Include:**
- Full URLs (protocol + domain + path) used for C2, downloads, exfiltration.

**Examples:**

```json
{
  "value": "https://www.live.com",
  "evidence_snippet": "Header Value https://www[.]live.com."
}
```

### 6.6 HTTP paths (new in v0.2)

**Include:**
- URI paths or API endpoints that are **not full URLs** (e.g. `/mifs/rs/api/v2/`).

**Examples:**

```json
{
  "value": "/mifs/rs/api/v2/",
  "evidence_snippet": "The cyber threat actors targeted the /mifs/rs/api/v2/ endpoint with HTTP GET requests"
}
```

### 6.7 Tools (new in v0.2)

**Include:**
- Named tools, utilities or frameworks mentioned (e.g. Mimikatz, Cobalt Strike, AdFind, NetScan, ngrok, AnyDesk, ExByte).

**Examples:**

```json
{
  "name": "Mimikatz",
  "evidence_snippet": "Evidence of likely usage of the credential theft tool Mimikatz... mimikatz.log"
}
```

```json
{
  "name": "NetScan",
  "evidence_snippet": "presence and execution of the network discovery tool NetScan"
}
```

### 6.8 CVEs (new in v0.2)

**Include:**
- CVE identifiers for vulnerabilities exploited or discussed.

**Examples:**

```json
{
  "value": "CVE-2021-34473",
  "evidence_snippet": "exploiting the ProxyShell vulnerabilities CVE-2021-34473, CVE-2021-34523, and CVE-2021-31207"
}
```

### 6.9 Detection signatures (new in v0.2)

**Include:**
- YARA rules and SIGMA rules mentioned in the report.

**Structure:**

```json
"detection_signatures": {
  "yara_rules": [
    {
      "name": "rule_name",
      "evidence_snippet": "YARA rule 'rule_name' detects..."
    }
  ],
  "sigma_rules": [
    {
      "name": "rule_name",
      "evidence_snippet": "SIGMA rule 'rule_name' identifies..."
    }
  ]
}
```

---

## 7. TTPs (Tactics, Techniques and Procedures)

### 7.1 General approach
- Extract tactics and techniques **explicitly mentioned or clearly described** in the narrative.
- Provide an `evidence_snippet` for each entry.
- Optionally include `mitre_id` (e.g. `T1566.001`) if stated; mark as `"explicit"` or `"inferred"` using `mapping_type`.

### 7.2 Selection rules by report type

| Report type | Selection rule |
|-------------|----------------|
| **Incident** | Extract all techniques mentioned in the intrusion narrative (initial access through impact). |
| **Campaign** | Extract techniques that recur or are emphasised across operations. |
| **Strategic** | Extract only the **top 10–15 most emphasised techniques** from narrative sections; do not exhaustively list every MITRE ID from appendix tables. |

### 7.3 Fields

- `tactic` – MITRE tactic name (e.g. "Initial Access", "Credential Access").
- `technique` – technique name or description.
- `mitre_id` – optional; MITRE ATT&CK ID (e.g. `T1566.001`).
- `mapping_type` – `"explicit"` if the report states the ID; `"inferred"` if you map it analytically.
- `mitre_confidence` – optional; `"high"`, `"medium"`, `"low"`.
- `evidence_location` – optional; page, section or table reference (e.g. "Table 3", "Appendix A").

**Positive examples:**

```json
{
  "mitre_id": "T1566.001",
  "tactic": "Initial Access",
  "technique": "Spearphishing Attachment",
  "evidence_snippet": "client execution T1203 remains prevalent ... almost always appearing alongside phishing T1566.001",
  "mapping_type": "explicit"
}
```

```json
{
  "tactic": "Credential Access",
  "technique": "Credential theft with Mimikatz",
  "evidence_snippet": "Evidence of likely usage of the credential theft tool Mimikatz... mimikatz.log",
  "mapping_type": "inferred",
  "mitre_confidence": "high"
}
```

**Negative examples (omit or restrict):**

- Strategic report lists 50+ MITRE IDs in a table without discussion → extract only those discussed in narrative sections or the top 10–15 most emphasised.
- Technique implied but very weakly → omit unless clearly supported by description.

---

## 8. Summary

### 8.1 Purpose
- Provide a concise overview of the document's key findings (max 1500 characters).

### 8.2 Required elements (where applicable)
- Primary threat actor (if attributed).
- High-level intrusion or threat narrative.
- Dominant TTP themes.
- Notable IoCs or tools.
- Target sector(s) and victim context.

**Example (incident):**

> "Case study of BlackByte ransomware intrusion exploiting ProxyShell on Exchange servers, deploying web shells, backdoors (api-msvc.dll), Cobalt Strike, AnyDesk, NetScan, AdFind, Mimikatz, ExByte exfiltration tool, and BlackByte 2.0 ransomware within five days. Includes IoCs like IPs, hashes, C2 domains."

**Example (strategic):**

> "ENISA Threat Landscape 2025 (October 2025) provides a strategic overview of cyber threats affecting the EU during the reporting period 1 July 2024 to 30 June 2025, highlighting phishing as the dominant initial vector, rapid exploitation of vulnerabilities, ransomware prominence, and recurring post-compromise discovery and persistence tradecraft mapped to MITRE ATT&CK."

---

## 9. Timeline

### 9.1 Purpose
- Capture key dates and events from the intrusion chronology or reporting period.

### 9.2 Fields

- `date` – ISO format `YYYY-MM-DD`.
- `date_precision` – `"day"`, `"month"`, `"quarter"`, `"year"`, `"approximate"`.
- `event` – brief description.
- `evidence_snippet` – supporting text.

### 9.3 Normalisation rules for date_precision

| Source text | Normalised date | date_precision |
|-------------|-----------------|----------------|
| "15 May 2025" | `2025-05-15` | `"day"` |
| "May 2025" | `2025-05-01` | `"month"` |
| "October 2025" | `2025-10-01` | `"month"` |
| "Around 15 May 2025" | `2025-05-15` | `"approximate"` |
| "Q2 2025" | `2025-04-01` | `"quarter"` |
| "2025" | `2025-01-01` | `"year"` |

### 9.4 What to include

- Intrusion start/end dates.
- Key compromise events (initial access, lateral movement, exfiltration, ransomware deployment).
- Publication dates.
- Reporting period boundaries (for strategic reports).

**Examples:**

```json
{
  "date": "2025-10-01",
  "date_precision": "month",
  "event": "Publication month of ENISA Threat Landscape 2025",
  "evidence_snippet": "ENISA THREAT LANDSCAPE 2025 ... October 2025"
}
```

```json
{
  "date": "2023-07-06",
  "date_precision": "day",
  "event": "Publication of BlackByte ransomware case study",
  "evidence_snippet": "The five-day job: A BlackByte ransomware intrusion case study July 6, 2023"
}
```

---

## 10. Common edge cases and decision rules

### 10.1 Strategic reports with no actor attribution
- Omit `threat_actor`; populate `report_metadata.report_type = "strategic"` and focus on TTPs, target_sectors and timeline.

### 10.2 Phishing templates vs. live domains
- Set `phishing_template: true` for synthetic/templated domains (e.g. `targetsname-sso[.]com`).
- Set `phishing_template: false` (default) for actual registered domains used in live campaigns.

### 10.3 IoCs appearing only in SIGMA/YARA rules
- Include the IoC (IP, domain, hash) but set `source_context: "detection_rule_only"`.
- Alternatively, if the rule name is more important than the IoC, record it under `detection_signatures`.

### 10.4 Approximate or incomplete dates
- Normalise to ISO `YYYY-MM-DD` and record `date_precision` accordingly (see § 9.3).

### 10.5 Numerous MITRE IDs in appendices (strategic reports)
- Extract only the top 10–15 most emphasised techniques from narrative sections.
- Do not exhaustively list every ID from tables unless the table is the main content.

### 10.6 Tools mentioned without hashes
- Add the tool name to `indicators_of_compromise.tools`; leave `file_hashes` empty if no hash is provided.

### 10.7 Malware families without hashes or rules
- Include in `tools` (e.g. `"AveMaria"`, `"Raccoon Stealer"`); if detection rules are provided later, add them to `detection_signatures`.

---

## 11. Quality checklist before finalising each annotation

- [ ] `report_metadata.report_type` is set and correct.
- [ ] `threat_actor` is present **only if explicitly attributed**; omitted otherwise.
- [ ] All IoC entries have `value` and `evidence_snippet`; optional flags (`masked`, `phishing_template`, `source_context`) are used where applicable.
- [ ] TTPs include `tactic`, `technique` and `evidence_snippet`; `mitre_id` is included if stated or confidently inferred.
- [ ] For strategic reports, TTPs are limited to the most emphasised (not exhaustive).
- [ ] Timeline entries use ISO dates with correct `date_precision`.
- [ ] Summary captures the main narrative (actor, techniques, IoCs, impact).
- [ ] All arrays (`target_sectors`, `indicators_of_compromise.*`, `ttps`, `timeline`) may be empty if the document does not provide that information.
- [ ] No placeholders or "TODO" entries remain in the JSON.

---

**End of guidelines. For questions or ambiguities not covered here, consult the schema design notes or escalate to the dissertation supervisor.**
