# Limitations & Scientific Scope Boundary

In accordance with academic rigor standards and Part AY-3 of the project specification, the following limitations are formally recognized and stated:

## 1. Ground Truth Quality and Annotation Discrepancies
- **Absence of Uniform Histopathological Confirmation**: The primary evaluation cohorts (LIDC-IDRI and LUNA16) utilize multi-radiologist consensus annotations (up to 4 thoracic readers). Malignancy ratings on these cohorts represent radiologist subjective perceptual suspicion rather than tissue biopsy or surgical pathology confirmation for the majority of nodules.
- **Conflation Guard**: Performance metrics evaluated on LIDC-IDRI/LUNA16 reflect *concordance with expert radiologist consensus* (`ANNOTATION_CONSENSUS` tier) and must not be interpreted as definitive biological truth (`PATHOLOGY_CONFIRMED` tier).

## 2. Validation Scope of Closed-Loop Learning & Drift Surveillance
- **Architectural Instrumentation vs. Prospective Deployment**: The closed-loop disagreement learning, active learning triage, and scanner drift monitoring modules are architecturally implemented, database-backed, and validated in an **offline simulated setting**.
- **Lack of Longitudinal Deployment Stream**: Evaluating continuous drift adaptation in an active clinical radiology reading room requires a multi-month prospective clinical deployment and institutional IRB approval, which is outside the scope of this v1 study.

## 3. Single-Source Dataset Bias & Retrospective Cohorts
- **Demographic and Scanner Distribution**: Models were trained and validated primarily on public retrospective cohorts. While isotropic resampling, HU windowing, and an Out-of-Distribution (OOD) autoencoder are implemented to detect scanner shifts, comprehensive multi-center prospective validation across diverse scanner manufacturers (GE, Siemens, Philips, Canon) remains an essential next step.

## 4. Radiologist Inter-Observer Variability
- Radiologists frequently exhibit inter-observer discordance on subtle sub-solid and ground-glass lesions. Inter-rater reliability (Cohen's and Fleiss' kappa) demonstrates moderate baseline agreement, highlighting that algorithmic "disagreement" with a single clinician often mirrors the discordance between human peers.

## 5. Regulatory & Clinical Decision Support Status
- This software is a research investigational decision-support tool. It is **not** an FDA-cleared or CE-marked medical device. All outputs, including risk probabilities, Lung-RADS categories, and explanation heatmaps, require independent review and verification by a licensed thoracic radiologist prior to clinical action (§55).
