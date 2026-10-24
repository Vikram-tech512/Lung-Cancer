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
The LUNA16 and LIDC-IDRI datasets are distributed under the **Creative Commons Attribution 3.0 Unported License (CC BY 3.0)**.

- **Permissions:** Commercial and non-commercial use, modification, distribution, and reproduction are permitted, provided appropriate credit is attributed to the original creators and source repositories.
- **Full License Text:** [https://creativecommons.org/licenses/by/3.0/](https://creativecommons.org/licenses/by/3.0/)

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

## 4. Patient De-Identification & HIPAA Compliance
All imaging series in LUNA16/LIDC-IDRI underwent rigorous, multi-tiered institutional de-identification in compliance with the **HIPAA Safe Harbor standard** (45 CFR § 164.514(b)(2)) before public release:
- All 18 direct protected health information (PHI) identifiers have been permanently removed from DICOM headers (including patient name, date of birth, medical record numbers, social security numbers, address, and facility identifiers).
- Series Instance UIDs and Study Instance UIDs are hashed with cryptographically non-reversible identifiers.
- Facial and cranial reconstruction regions were cropped or filtered out where applicable to prevent surface re-identification.
- No protected personal health data is processed, stored, or transmitted by this software platform.

---

## 5. Clinical Safety & Scope Limitations (Research Use Only)
> **MANDATORY NOTICE: RESEARCH USE ONLY — NOT A MEDICAL DEVICE.**  
> This dataset, preprocessing pipeline, and associated machine learning models are developed strictly for academic evaluation, algorithmic benchmarking, and methodology development.  
> They have **NOT** been cleared, certified, or approved by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA/CE-mark), or any health regulatory authority for clinical diagnosis, patient triage, screening, or medical treatment planning.

### Clinical Boundary Disclosures:
1. **Nodule vs. Non-Nodule Only:** LUNA16 annotations index candidate false-positive reduction (distinguishing anatomical/inflammatory non-nodules from consensus pulmonary nodules). The dataset **does not contain verified pathological malignancy outcomes, tissue biopsies, or histopathology grades**.
2. **No Malignancy Diagnosis:** High probability outputs indicate candidate nodular morphometry, **never** a clinical cancer diagnosis.
3. **Cohort Exclusions:** The dataset does not include pediatric patients (< 18 years), post-surgical resections, acute interstitial pneumonia, severe thoracic trauma, or high-contrast arterial phase scans.
