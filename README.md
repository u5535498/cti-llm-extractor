# CTI LLM Extractor

A pipeline for evaluating the reliability of LLM-based Cyber Threat Intelligence (CTI) extraction via schema-guided prompting, document-provenance grounding, and Chain-of-Verification (CoVe). Built as part of a final year BSc Cyber Security project at the University of Warwick (2025–2026).

***

## Overview

This project investigates whether structured prompting and verification mechanisms can reduce hallucination in open-source LLMs when extracting Indicators of Compromise (IoCs) and Tactics, Techniques, and Procedures (TTPs) from publicly available threat reports.

Four pipeline variants are implemented and evaluated against a manually annotated test set of 28 CTI documents:

| Variant | Description |
|---------|-------------|
| **V0** | Free-text baseline — unstructured prompt, no schema |
| **V1** | Schema-guided extraction using a structured JSON output schema |
| **V2** | Evidence-grounded extraction — each claim must cite a supporting document snippet |
| **V3** | Chain-of-Verification — V1 extraction followed by a self-verification pass |

The underlying model is **Llama 3.1 8B (GGUF, Q4_K_M quantisation)** running locally via `llama-cpp-python` with CUDA acceleration.

***

## Repository Structure

```
cti-llm-extractor/
├── src/
│   ├── run_pipeline.py          # Main entry point — runs all variants
│   ├── v0_extractor.py          # V0: free-text baseline
│   ├── v1_baseline_extractor.py # V1: schema-guided extraction
│   ├── v2_evidence_grounder.py  # V2: document-provenance grounding
│   ├── v3_cove_verifier.py      # V3: Chain-of-Verification
│   ├── prompting.py             # Prompt templates for all variants
│   ├── schema_models.py         # Pydantic output schema definitions
│   ├── document_loader.py       # PDF and HTML report ingestion
│   ├── evaluation.py            # F1, precision, recall, Wilcoxon, bootstrap CIs
│   ├── intra_annotator.py       # Krippendorff alpha inter-annotator reliability
│   └── utils_logging.py         # Logging configuration
├── data/                        # Annotated ground truth (not included in repo)
├── raw_reports/                 # Source CTI PDF/HTML reports (not included)
├── outputs/                     # Pipeline outputs (JSON) — generated at runtime
├── models/                      # Local GGUF model files (not included)
├── schema_v0.2.json             # JSON schema for structured extraction
├── annotation_guidelines_v0.2.md
├── EVALUATION_PROTOCOL.md
├── requirements.txt
└── README.md
```

***

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended; CPU inference is possible but slow)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with CUDA support
- Llama 3.1 8B GGUF model (`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`)

***

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/isobelrclarke/cti-llm-extractor.git
cd cti-llm-extractor
```

### 2. Create and activate a virtual environment

```bash
python -m venv cti_env
source cti_env/bin/activate        # Linux/macOS
cti_env\Scripts\activate           # Windows
```

### 3. Install llama-cpp-python with CUDA support

> Skip the CUDA flag if running on CPU only.

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the model

```bash
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models
```

***

## Usage

### Run all pipeline variants

```bash
python src/run_pipeline.py --input raw_reports/ --output outputs/ --variant all
```

### Run a specific variant

```bash
python src/run_pipeline.py --input raw_reports/ --output outputs/ --variant v3
```

Available values for `--variant`: `v0`, `v1`, `v2`, `v3`, `all`.

### Run evaluation

```bash
python src/evaluation.py --outputs outputs/ --ground-truth data/ground_truth.json
```

This produces `evaluation_results.json` in the `outputs/` directory, containing per-document and macro-averaged precision, recall, F₁, Wilcoxon test statistics, and bootstrap 95% confidence intervals for each variant.

***

## Data

Ground truth annotations and raw CTI reports are included in this repository. The annotation schema and guidelines are documented in:

- [`schema_v0.2.json`](schema_v0.2.json) — JSON output schema for IoC and TTP extraction
- [`annotation_guidelines_v0.2.md`](annotation_guidelines_v0.2.md) — annotation decision rules and examples
- [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) — full evaluation methodology

***

## Dependencies

Key libraries used:

| Library | Purpose |
|---------|---------|
| `llama-cpp-python` | Local LLM inference (Llama 3.1 8B GGUF) |
| `pydantic` | Structured output schema validation |
| `PyMuPDF` | PDF text extraction |
| `scikit-learn` | Precision, recall, F₁ computation |
| `scipy` | Wilcoxon signed-rank test |
| `krippendorff` | Inter-annotator reliability (Krippendorff's alpha) |
| `numpy`, `pandas` | Data processing |

See [`requirements.txt`](requirements.txt) for full version-pinned dependencies.

***

## Notes

- The `cti_env/` virtual environment directory is included in `.gitignore` and should not be committed.
- Model files (`.gguf`) are large and excluded from version control. Download them separately as described above.
- All code in `src/` was written by the project author. Third-party libraries are listed in `requirements.txt`.

***

## Licence

This project was produced for academic assessment purposes at the University of Warwick. It is not licensed for commercial use.
