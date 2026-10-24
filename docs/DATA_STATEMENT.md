# Data Statement: LUNA16 & LIDC-IDRI Benchmark Cohort

## 1. Dataset Overview & Provenance
This research platform evaluates 3D deep learning architectures and radiomic representations for pulmonary nodule detection and false-positive reduction using the **LUng Nodule Analysis 2016 (LUNA16)** dataset, hosted publicly on Zenodo (Record 3723295).

LUNA16 is an official curated subset of the **Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI)**, administered by the National Cancer Institute (NCI) and accessed via **The Cancer Imaging Archive (TCIA)**.

- **Primary Source Record:** Zenodo DOI [10.5281/zenodo.3723295](https://doi.org/10.5281/zenodo.3723295)
- **Total LUNA16 Cohort:** 888 thoracic low-dose computed tomography (LDCT) scans partitioned into 10 subsets (subset0 through subset9).
- **Inclusion Criteria:** Scans with slice thickness $\le 2.5\text{ mm}$ and valid reconstruction geometries.
- **Candidate Annotations:** Consensus annotations established by four experienced thoracic radiologists. All nodules $\ge 3\text{ mm}$ identified by at least 3 out of 4 radiologists are indexed as true positive nodules. Non-nodule structures identified by radiologists or automated candidate generators are indexed in `candidates_V2.csv`.

---

## 2. Dataset Licensing
The dataset licenses as stated directly in the source repository records are quoted below:

1. **Zenodo Record 3723295 Metadata License:**  
   The Zenodo metadata specifies:  
   > `"license": {"id": "cc-by-4.0"}`  
   > **Creative Commons Attribution 4.0 International (CC BY 4.0)**

2. **Zenodo Record Description:**  
   The record description states:  
   > *"This dataset includes the images from the LIDC/IDRI dataset in a different format, together with additional annotations. The LIDC/IDRI dataset is available at https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI under a Creative Commons Attribution 3.0 Unported License."*

3. **The Cancer Imaging Archive (TCIA) LIDC-IDRI Page:**  
   The TCIA collection page documents the license as:  
   > **Creative Commons Attribution 3.0 Unported License (CC BY 3.0)**  
   > *"You are free to: Share — copy and redistribute the material in any medium or format; Adapt — remix, transform, and build upon the material for any purpose, even commercially, under the terms of attribution."*

---

## 3. Mandatory Academic Citations
Any publication, technical report, or software benchmark derived from this cohort must cite the primary LUNA16 challenge paper and the foundation LIDC-IDRI TCIA references:

1. **LUNA16 Challenge:**  
   Setio, A. A. A., Traverso, A., de Bel, T., Berens, M. S., van den Bogaard, C., Cerello, P., ... & Jacobs, C. (2017).  
   *Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge.*  
   **Medical Image Analysis**, 42, 1–13. DOI: [10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015).

2. **LIDC-IDRI Overview:**  
   Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., ... & Clarke, L. P. (2011).  
   *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of computed tomography scans of lung nodules.*  
   **Medical Physics**, 38(2), 915–931. DOI: [10.1118/1.3528204](https://doi.org/10.1118/1.3528204).

3. **The Cancer Imaging Archive (TCIA):**  
   Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., ... & Prior, F. (2013).  
   *The Cancer Imaging Archive (TCIA): Maintaining and operating a public information repository.*  
   **Journal of Digital Imaging**, 26(6), 1045–1057. DOI: [10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7).

---

## 4. Patient De-Identification (TCIA Curation)
Per official TCIA documentation, all CT series in LIDC-IDRI / LUNA16 were de-identified by TCIA staff prior to public dissemination:
- TCIA staff curate and de-identify incoming collections, removing direct Protected Health Information (PHI) identifiers from DICOM headers and scrubbing private vendor tags.
- Series and Study UIDs are replaced with non-reversible research identifiers.
- Facial and cranial reconstruction regions were cropped or filtered out where applicable to prevent surface re-identification.
- No identifiable private personal health information is stored, processed, or transmitted by this software platform.

---

## 5. Clinical Safety & Scope Limitations (Research Use Only)
> **MANDATORY NOTICE: RESEARCH USE ONLY — NOT A MEDICAL DEVICE.**  
> This dataset, preprocessing pipeline, and associated machine learning models are developed strictly for academic evaluation, algorithmic benchmarking, and methodology development.  
> They have **NOT** been cleared, certified, or approved by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA/CE-mark), or any health regulatory authority for clinical diagnosis, patient triage, screening, or medical treatment planning.

### Clinical Boundary Disclosures:
1. **Nodule vs. Non-Nodule Only:** LUNA16 annotations index candidate false-positive reduction (distinguishing anatomical/inflammatory non-nodules from consensus pulmonary nodules). The dataset **does not contain verified pathological malignancy outcomes, tissue biopsies, or histopathology grades**.
2. **No Malignancy Diagnosis:** High probability outputs indicate candidate nodular morphometry, **never** a clinical cancer diagnosis.
3. **Cohort Exclusions:** The dataset does not include pediatric patients (< 18 years), post-surgical resections, acute interstitial pneumonia, severe thoracic trauma, or high-contrast arterial phase scans.
4. **Flagged for Manual Review & Unvalidated Uncertainty Defaults:** Uncertainty standard deviation ($\sigma$) tiers are unvalidated default heuristics intended to be tuned on the validation subset. Cases where uncertainty exceeds default thresholds or imaging parameters fail QA checks are flagged for manual review by a qualified radiologist.


