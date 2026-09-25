# Research Methodology, Scope Boundary & Rigor Addendum

## 1. Executive Summary & Research Contribution Statement (Part AY)

This platform advances thoracic oncology artificial intelligence by synthesizing 3D full-volume volumetric CT representation learning, Contrastive Explainability Learning (CEL), and a clinical decision-support architecture grounded in epistemic humility.

### Formal Research Contribution Statement (Part AY)
> "This work implements and experimentally validates the prediction, explanation-reliability, and abstention components of this framework on LIDC-IDRI and LUNA16 cohorts. The closed-loop learning and drift-monitoring components are architected and instrumented but validated only in a simulated/offline setting due to deployment and institutional data constraints — full online validation is identified as future work."

---

## 2. PART AY-1: Ground Truth Protocol

Every risk, calibration, and disagreement metric in this system is interpreted strictly through its designated ground truth tier:

| Source | Ground Truth Basis | Confidence Level | Tier Classification |
|---|---|---|---|
| **LIDC-IDRI** | Multi-radiologist annotation consensus (up to 4 thoracic readers) — **not** pathology-confirmed for >90% of nodules | Annotation-level, not diagnostic-level | `ANNOTATION_CONSENSUS` |
| **LUNA16** | Derived from LIDC-IDRI candidate annotations ($\ge 3$ of 4 radiologist concordance) | Annotation-level, not diagnostic-level | `DERIVED_ANNOTATION` |
| **IQ-OTH/NCCD** | Per dataset documentation — institutional verification required before diagnostic claims | Verify before use | `UNVERIFIED` |
| **NLST / Future Real-Patient Data** | Tissue biopsy / surgical resection histopathology or $\ge 2$-year serial CT volumetric stability | Diagnostic-level | `PATHOLOGY_CONFIRMED` |

### Inviolable Ground Truth Rule
Every reported table and figure MUST declare which ground-truth tier it is evaluated against. **Never present annotation-consensus-level accuracy and pathology-confirmed accuracy as the same kind of metric in the same table.**

---

## 3. PART AY-2: Statistical Rigor Specification

All experimental tables, hypothesis tests, and ablation matrices adhere to the following standards:

1. **Confidence Intervals (Patient-Level Bootstrap)**:
   - Point estimates are prohibited as standalone claims. Every AUC, sensitivity, specificity, ECE, and abstention precision metric includes a 95% bootstrap confidence interval.
   - Resampling is executed strictly at the **patient level** (clustering all nodules and axial slices belonging to a given patient ID). Slice-level or crop-level resampling is banned because it violates sample independence and artificially deflates confidence interval width.

2. **Significance Testing for Comparisons**:
   - Model comparisons in ablation matrices utilize formal hypothesis testing:
     - **DeLong's Test**: Evaluates the statistical significance of correlated ROC-AUC deltas between model variants on the identical test cohort.
     - **McNemar's Test**: Evaluates paired classification discordance with Edwards continuity correction.

3. **Multiple-Comparison Error Control**:
   - **Confirmatory Hypotheses (H1–H7)**: Pre-declared primary hypotheses are controlled for family-wise error rate or false discovery rate using Bonferroni or Benjamini-Hochberg (FDR) adjustments.
   - **Exploratory Analyses**: Post-hoc phenotype drill-downs, disagreement cluster mining, and drift surveillance are explicitly designated as **hypothesis-generating only** and never reported as confirmed findings.

4. **Inter-Rater Reliability Benchmarking**:
   - When evaluating multi-reader discordance (§AH), Cohen's kappa (for 2 raters) or Fleiss' kappa (for $\ge 3$ raters) is reported to establish baseline inter-radiologist agreement *prior* to comparing clinician decisions to algorithmic predictions.

5. **Non-Circular Reliability Score Weight Fitting**:
   - The Explanation Reliability Score weights $(w_1, w_2, w_3)$ for multi-method overlap, perturbation stability, and CEL reference consistency are fit via logistic regression against radiologist ambiguity (§M) **exclusively on the validation split**.
   - These fitted weights are frozen before touching any test cohort data.

---

## 4. PART AY-3: Scope Boundary (MVP vs. Full Vision)

To maintain defensibility in academic peer review and thesis defense contexts, the project boundary is explicitly delineated:

### In Scope for v1 (Defensible Contribution)
- **Phase 1–17 Core Pipeline**: Isotropic 3D preprocessing, 3D volumetric UNet lung segmentation, nodule segmentation, 3D morphological feature profiling, and multi-modal risk scoring.
- **Phases 11A & 11B (Explanation Reliability Engine)**: Multi-method attribution agreement (3D Grad-CAM, Grad-CAM++, Integrated Gradients), perturbation invariance under clinically plausible noise, and composite reliability scoring.
- **Phases 16 & 16A (Abstention Framework)**: Selective classification routing low-reliability or high-epistemic-uncertainty scans to mandatory `REVIEW REQUIRED`. Reporting Abstention Rate, Abstention Precision, and Conditional vs. Unconditional AUC.
- **Phase 17A (Disagreement Structuring & Data Collection)**: Capturing radiologist overrides, diagnostic rationale, and phenotypic divergence tags.
- **Clinical Radiologist Dashboard & Research Portal**: Full interactive workstation enabling clinical decision-support with embedded governance disclaimers (§55).

### Explicitly Deferred to Future Work
- **Phases 17B & 17C (Automated Online Active Learning & Retraining)**: Continuous weight updates in production environments require longitudinal clinical monitoring and institutional IRB approval.
- **Phase 20 (Live Production Drift Dashboard)**: Long-term drift surveillance requires multi-quarter prospective clinical deployment data. The platform provides architectural instrumentation and offline historical simulation.
- **Phase 24 (External Multi-Center Prospective Validation)**: Multi-institutional clinical validation requiring cross-hospital data use agreements.
