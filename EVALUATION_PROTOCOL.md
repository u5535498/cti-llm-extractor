# Evaluation Protocol — Pre-Registered Decisions

> **Status:** Locked before test-set evaluation run (2026-04-08).
> These decisions were recorded prior to any test-set results being observed.
> They may not be changed retroactively without explicit acknowledgement that
> the protocol was amended post-hoc.

---

## 1. Context-Overflow Handling Policy

### Decision

Documents whose plain-text content exceeds the **4,096-token context window** of `Llama-3.1-8B-Instruct` after PDF extraction and whitespace normalisation are
**excluded from evaluation** using a sentinel-flag approach.

### Rationale

Two approaches were considered prior to the test-set run:

| Option | Description | Verdict |
|---|---|---|
| **A — Sentinel exclusion (adopted)** | Overflow detected at runtime; document written to disk as an error sentinel (`"context_overflow": true`); excluded from all metric calculations; counted in the reported `Overflow` column. | **Adopted** |
| B — Section-boundary chunking with merge | Document split at structural boundaries; extraction run per chunk; IoCs deduplicated across chunks; results merged before scoring. | Deferred to future work |

Option A was adopted because:

1. It is already implemented, tested, and consistent across all four pipeline variants (V0–V3) on the development set. Option B would require writing,
   testing, and validating new chunking and merge logic under time pressure, immediately before the final evaluation run.

2. The exclusion is applied **identically across all variants**: a document that overflows V0 also overflows V1, V2, and V3 (token counting is pre-model).
   This preserves the within-document paired structure required by the Wilcoxon signed-rank tests.

3. The token-limit boundary is itself an operationally relevant finding. It represents a hard deployment constraint of 8B-parameter open-source models
   for CTI extraction from long-form threat intelligence reports.
   
---

## 2. Sensitivity Analysis Thresholds (V2 Evidence-Grounding)

The evidence-grounding verifier (V2) was evaluated at three fuzzy-match similarity thresholds:

| Threshold | Output file |
|---|---|
| 0.55 | `outputs/sensitivity_t055.json` |
| **0.65 (primary/default)** | `outputs/sensitivity_t065.json` |
| 0.75 | `outputs/sensitivity_t075.json` |

The threshold of **0.65** is designated the primary threshold for all headline results. The 0.55 and 0.75 runs constitute a pre-specified sensitivity analysis,
reported in a dedicated subsection. This designation was made before the test-set run and may not be changed in response to test-set outcomes.

---

## 3. Statistical Testing

- Test: Wilcoxon signed-rank (two-tailed), α = 0.05.
- Effect size: *r* = Z / √N.
- Comparisons: V0 vs V1, V1 vs V2, V2 vs V3, V0 vs V3 (four pre-specified comparisons, no post-hoc additions).
- Both IoC-F1 and TTP-F1 are reported; no correction for multiple comparisons is applied (the analysis is exploratory-comparative, not confirmatory hypothesis testing). This is acknowledged as a limitation.

---

## 5. Test-Set Evaluation Run

- The test set will be evaluated **once**, using the pipeline as it stands at the time of this commit.
- No re-runs to improve scores are permitted after test-set results are observed.
- Any post-hoc changes to the evaluation code must be declared explicitly in the dissertation as amendments made after observing test-set results.
