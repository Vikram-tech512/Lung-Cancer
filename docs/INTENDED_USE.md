# Intended Use and Clinical Workflow Specification

> **REGULATORY NOTICE & RESEARCH PROTOTYPE DISCLAIMER**  
> This software is an investigational research prototype designed exclusively for scientific evaluation and clinical decision-support research. It is **NOT** cleared or approved by the U.S. Food and Drug Administration (FDA), CE-marked as a medical device under the EU Medical Device Regulation (MDR 2017/745), or certified by any regulatory agency for diagnostic use. It must **NEVER** be used as a standalone diagnostic tool, and it must never replace independent clinical assessment by a licensed physician.

---

## 1. Intended Users & Clinical Decisions

The system is intended to provide assistive, evidence-grounded decision support to licensed healthcare professionals. Each user group operates within a defined clinical scope:

| Intended User | Clinical Setting | Exact Clinical Decision Supported | Prohibited Actions |
|---|---|---|---|
| **Board-Certified Radiologist** | PACS diagnostic reading room | - Nodule detection confirmation (second-reader paradigm)<br>- Accurate orthogonal measurement and volumetry ($\text{mm}^3$)<br>- Lung-RADS risk category categorization (investigational guidance)<br>- Recommendation for follow-up LDCT interval vs. PET/CT vs. tissue biopsy | Overriding radiologist visual impression; automated sign-off of radiology reports without review. |
| **Pulmonologist** | Pulmonary clinic / Interventional suite | - Assessment of pulmonary nodule referral cases<br>- Planning bronchoscopic vs. transthoracic biopsy access trajectory<br>- Volumetric doubling time estimation across historical scans | Autonomous staging or invasive procedure booking without multidisciplinary review. |
| **Multidisciplinary Tumour Board Member** | Weekly thoracic oncology tumor board | - Rapid case presentation review with standardized 3D multi-planar reconstructions<br>- Quantitative volumetric change analysis between baseline and post-treatment scans<br>- Discrepancy resolution between clinical impressions | Sole-source justification for thoracic surgery or stereotactic body radiotherapy (SBRT). |
| **Non-Specialist Physician** | Primary care / Low-resource setting without on-site radiologist | - Triage of high-risk or indeterminate scans for urgent tele-radiology consultation<br>- Identification of high-uncertainty scans requiring immediate specialist referral | Direct patient communication of cancer diagnosis; definitive benign discharge without specialist sign-off. |

---

## 2. Clinical Workflow Integration

The platform is designed to integrate into clinical radiology workflows without disrupting existing Picture Archiving and Communication Systems (PACS) or Radiology Information Systems (RIS):

```
                        ┌────────────────────────────────────────────────────────┐
                        │             DICOM Ingestion & Quality Audit            │
                        │ (Slice thickness, spacing, reconstruction kernel check)│
                        └──────────────────────────┬─────────────────────────────┘
                                                   │
                         Pass Quality Thresholds   ▼   Fail OOD / Quality Check
                        ┌──────────────────────────┴─────────────────────────────┐
                        │                                                        │
                        ▼                                                        ▼
           ┌─────────────────────────┐                             ┌─────────────────────────┐
           │   Workflow Placement A  │                             │  Hard Abstain Warning   │
           │      Worklist Triage    │                             │ "Scan out-of-distribution│
           │  (Risk/Uncertainty Rank)│                             │  Manual review required"│
           └────────────┬────────────┘                             └─────────────────────────┘
                        │
                        ▼
           ┌─────────────────────────────────────────────────────────────────────┐
           │                        Workflow Placement B                         │
           │         Blind-First Reading & Second-Reader Interactive Review      │
           │                                                                     │
           │  1. Clinician reviews axial/coronal/sagittal slices unassisted     │
           │  2. Clinician records initial impression (reduces automation bias)  │
           │  3. Model reveals candidate locations, volume, score, uncertainty   │
           │  4. Clinician executes: [Accept] / [Edit] / [Reject] per finding   │
           └───────────────────────────────────┬─────────────────────────────────┘
                                               │
                                               ▼
           ┌─────────────────────────────────────────────────────────────────────┐
           │                        Workflow Placement C                         │
           │       Longitudinal Comparison & Structured Reporting (Research)     │
           │                                                                     │
           │  - Rigid centroid alignment to prior scan for growth assessment     │
           │  - Calibrated malignancy probability with MC-Dropout uncertainty    │
           │  - Editable draft report generated (FHIR DiagnosticReport / DICOM) │
           │  - Human-in-the-loop final review and cryptographic audit signature │
           └─────────────────────────────────────────────────────────────────────┘
```

### What the System Must NEVER Do
1. **Never output an autonomous diagnosis:** The software must never state "Patient has lung adenocarcinoma" or "Benign nodule". It outputs candidate coordinates, orthogonal dimensions, volumetric estimations, calibrated risk percentiles, and uncertainty intervals.
2. **Never auto-sign or finalize a report:** Every report draft requires human-in-the-loop verification, clinician identity attribution, and explicit confirmation.
3. **Never auto-dismiss low-suspicion studies:** Low-risk studies are flagged with lower triage priority on the worklist, but are never marked as "cleared", "normal", or removed from the human reading queue.
4. **Never obscure disagreement:** When a clinician rejects an AI finding or alters a contour, the discrepancy is preserved in the audit log as calibration data and never silenced.

---

## 3. Out-of-Scope Uses & Contraindications

Use of this system under any of the following conditions is **strictly contraindicated**:

1. **Paediatric Patients (< 18 years):**
   - *Rationale:* Chest anatomy, nodule etiologies (e.g., infectious vs. neoplastic), and radiation parameters differ fundamentally from adult screening cohorts. The model has zero training exposure to paediatric anatomy.
2. **Non-Chest or Non-Axial CT Scans:**
   - *Rationale:* Abdominal, pelvic, head/neck, or extremity CT scans, as well as direct non-axial reconstructions (coronal/sagittal native acquisitions), are outside the distribution.
3. **Thick-Slice CT (> 2.5 mm slice thickness):**
   - *Rationale:* Thick-slice CT introduces severe partial volume averaging, obscuring small nodule margins and distorting volumetric measurements. LUNA16 inclusion criteria mandate $\le 2.5\text{ mm}$.
4. **Non-Standard Contrast Phases:**
   - *Rationale:* The model was trained on non-contrast low-dose CT (LDCT). High-concentration intravenous iodinated contrast or delayed systemic contrast phases alter Hounsfield Unit (HU) distributions and confound nodule density classification.
5. **Severe Artifact Scans:**
   - *Rationale:* Scans with prominent respiratory motion artifacts, cardiac pulsation blur, or metallic streak artifacts (e.g., pacemakers, spinal fixation hardware) in the thoracic volume must be rejected.
6. **Primary Diagnosis of Primary Cancer vs. Secondary Metastases:**
   - *Rationale:* The system cannot differentiate between primary bronchogenic carcinoma and solitary pulmonary metastases from extrathoracic primaries.

---

## 4. Model Card & Cohort Datasheet

### 4.1 Model Details
- **Architecture:** 3D Deep Residual Network (ResNet3D-18 / ResNet3D-50) with 3D batch normalization and adaptive volumetric pooling.
- **Input Dimension:** $32 \times 32 \times 32$ or $64 \times 64 \times 64$ voxel isotropic volume (resampled to $1.0\times 1.0\times 1.0\text{ mm}^3$ spacing via B-spline interpolation, windowed to $[-1000, 400]\text{ HU}$).
- **Post-Processing Modules:**
  - Post-hoc probability calibration via **Temperature Scaling** (single scalar parameter $T$ fit exclusively on validation NLL).
  - Epistemic uncertainty estimation via **Monte Carlo Dropout** (10 stochastic forward passes with active dropout).
  - Gradient-weighted visual attribution via **3D Grad-CAM** and **Integrated Gradients**.

### 4.2 Benchmark Cohort Specification
- **Primary Source:** LUNA16 (LUng Nodule Analysis 2016), hosted on Zenodo (Record 3723295), curated from TCIA LIDC-IDRI.
- **Scan Count:** 888 thoracic low-dose computed tomography (LDCT) series.
- **Annotation Standard:** Consensus of 4 experienced thoracic radiologists. Solid, part-solid, and non-solid nodules $\ge 3\text{ mm}$ confirmed by $\ge 3$ radiologists form the positive reference set.
- **Partitioning & Split Integrity:**
  - Reduced-study protocol: Subsets 0–1 (Training), Subset 2 (Validation), Subset 3 (Held-Out Test).
  - Patient-level cluster enforcement: Zero patient or series UID overlap across partitions (verified by SHA-256 split hash).
  - Negative candidate handling: Candidate negatives subsampled for training only ($1:4$ ratio); validation and test partitions retain natural clinical candidate prevalence.

### 4.3 Known Biases & Technical Limitations
1. **Geographic and Demographic Sampling Bias:** LIDC-IDRI scans originate from seven academic medical centers in the United States collected between 2000 and 2010. Smoking history, race, and socioeconomic factors are incompletely documented in the public metadata.
2. **Scanner Vendor Distribution:** The cohort reflects historical GE Medical Systems, Siemens, and Philips CT scanners from the 2000s; modern ultra-low-dose iterative reconstruction algorithms and photon-counting CT detectors are not represented.
3. **Sub-centimeter Bias:** False positive rates rise sharply on sub-solid and ground-glass nodules $< 6\text{ mm}$ due to partial volume effects and apical vascular branch mimics.
4. **Epistemic Uncertainty Awareness:** When encountering atypical anatomical variants or out-of-distribution HU ranges, the model must trigger a human-review deferral rather than outputting confident classifications.
