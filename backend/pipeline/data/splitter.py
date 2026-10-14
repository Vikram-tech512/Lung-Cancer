"""
backend/pipeline/data/splitter.py
====================================
Patient-stratified train / validation / test splitter.

Guarantees
----------
* All scans for a given patient ID land in exactly ONE split
  (no data leakage across patients)
* Stratified by label (malignant / benign) and optionally by scanner
* Reproducible via fixed random seed
* Supports k-fold cross-validation

Usage
-----
>>> splitter = CTSplitter(val_frac=0.15, test_frac=0.15, seed=42)
>>> train_ids, val_ids, test_ids = splitter.split(records)
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.monitoring import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Record = Dict[str, Any]   # {"patient_id": str, "label": int, "scan_path": str, ...}


class CTSplitter:
    """
    Patient-stratified data splitter for CT datasets.

    Parameters
    ----------
    val_frac   : fraction of patients assigned to validation
    test_frac  : fraction of patients assigned to test
    seed       : random seed (integer or None)
    stratify_by: list of record keys to stratify on (e.g. ["label", "scanner"])
    """

    def __init__(
        self,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        stratify_by: Optional[List[str]] = None,
    ) -> None:
        if val_frac + test_frac >= 1.0:
            raise ValueError("val_frac + test_frac must be < 1.0")
        self.val_frac   = val_frac
        self.test_frac  = test_frac
        self.seed       = seed
        self.stratify_by = stratify_by or ["label"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split(
        self, records: List[Record]
    ) -> Tuple[List[Record], List[Record], List[Record]]:
        """
        Split records into train / val / test.

        Returns
        -------
        (train_records, val_records, test_records)
        """
        # Group records by patient_id
        patient_records: Dict[str, List[Record]] = defaultdict(list)
        for rec in records:
            pid = rec.get("patient_id", rec.get("case_id", self._auto_pid(rec)))
            patient_records[pid].append(rec)

        # Collect unique patients with their strata
        patients = list(patient_records.keys())
        strata_keys = [self._strata_key(patient_records[p][0]) for p in patients]

        train_pids, val_pids, test_pids = self._stratified_split(patients, strata_keys)

        train_records = [r for p in train_pids for r in patient_records[p]]
        val_records   = [r for p in val_pids   for r in patient_records[p]]
        test_records  = [r for p in test_pids  for r in patient_records[p]]

        log.info(
            "data_split_complete",
            total_records=len(records),
            total_patients=len(patients),
            train=len(train_records),
            val=len(val_records),
            test=len(test_records),
        )
        return train_records, val_records, test_records

    def kfold_split(
        self, records: List[Record], n_folds: int = 5
    ) -> List[Tuple[List[Record], List[Record]]]:
        """
        Patient-stratified k-fold cross-validation splits.

        Returns
        -------
        List of (train_records, val_records) tuples, length == n_folds
        """
        patient_records: Dict[str, List[Record]] = defaultdict(list)
        for rec in records:
            pid = rec.get("patient_id", self._auto_pid(rec))
            patient_records[pid].append(rec)

        patients = list(patient_records.keys())
        strata_keys = [self._strata_key(patient_records[p][0]) for p in patients]

        # Sort patients per stratum for deterministic fold assignment
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(patients))
        patients = [patients[i] for i in indices]
        strata_keys = [strata_keys[i] for i in indices]

        # Assign fold index per patient via stratum-aware round-robin
        stratum_buckets: Dict[str, List[str]] = defaultdict(list)
        for pid, sk in zip(patients, strata_keys):
            stratum_buckets[sk].append(pid)

        fold_assignments: Dict[str, int] = {}
        for sk, bucket in stratum_buckets.items():
            for i, pid in enumerate(bucket):
                fold_assignments[pid] = i % n_folds

        folds = []
        for fold_idx in range(n_folds):
            val_pids   = [p for p, f in fold_assignments.items() if f == fold_idx]
            train_pids = [p for p, f in fold_assignments.items() if f != fold_idx]
            folds.append((
                [r for p in train_pids for r in patient_records[p]],
                [r for p in val_pids   for r in patient_records[p]],
            ))

        log.info("kfold_split_complete", n_folds=n_folds, total_patients=len(patients))
        return folds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stratified_split(
        self,
        patients: List[str],
        strata_keys: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Within each stratum, split patients into train/val/test."""
        rng = random.Random(self.seed)

        stratum_patients: Dict[str, List[str]] = defaultdict(list)
        for pid, sk in zip(patients, strata_keys):
            stratum_patients[sk].append(pid)

        train_pids, val_pids, test_pids = [], [], []

        for sk, pids in stratum_patients.items():
            rng.shuffle(pids)
            n = len(pids)
            n_test = max(1, int(n * self.test_frac))
            n_val  = max(1, int(n * self.val_frac))
            n_train = n - n_test - n_val
            if n_train < 1:
                # Not enough patients — all go to train
                train_pids += pids
                continue
            test_pids  += pids[:n_test]
            val_pids   += pids[n_test: n_test + n_val]
            train_pids += pids[n_test + n_val:]

        return train_pids, val_pids, test_pids

    def _strata_key(self, record: Record) -> str:
        parts = [str(record.get(k, "unknown")) for k in self.stratify_by]
        return "|".join(parts)

    @staticmethod
    def _auto_pid(record: Record) -> str:
        """Generate a pseudo patient-ID from the scan path hash."""
        path = str(record.get("scan_path", record.get("dicom_dir", id(record))))
        return hashlib.md5(path.encode()).hexdigest()[:8]
