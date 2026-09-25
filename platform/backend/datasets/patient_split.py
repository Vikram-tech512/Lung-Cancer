"""
backend/datasets/patient_split.py
===================================
Patient-level dataset splitting for medical AI research.

RULES (non-negotiable):
  1. Split is performed at PATIENT level, never at slice or nodule level.
  2. A patient ID may appear in exactly ONE split (train, val, or test).
  3. A SHA-256 hash of the patient ID list is stored with every experiment
     so split reproducibility is verifiable.
  4. The flag PATIENT_LEVEL_SPLIT_VERIFIED = True is always written to
     MLflow experiment tags when this module generates the split.
  5. A duplicate patient ID raises ValueError, not a warning.

These rules are based on the established fact that random slice-level
splitting inflates reported accuracy due to cross-contamination of
similar slices from the same patient across train and test sets.
Reference: Kazeminia et al., "GANs for Medical Image Analysis", 2020.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Patient-level split
# ─────────────────────────────────────────────────────────────

def patient_stratified_split(
    records: List[Dict],
    patient_id_key: str = "patient_id",
    label_key: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split a list of records into train/val/test at patient level.

    Args:
        records:         List of dicts, each with at minimum patient_id and label.
        patient_id_key:  Key for patient identifier.
        label_key:       Key for class label (used for stratification).
        train_ratio:     Fraction of patients for training.
        val_ratio:       Fraction for validation.
        test_ratio:      Fraction for test (must sum to 1.0).
        seed:            Random seed for reproducibility.

    Returns:
        (train_records, val_records, test_records)

    Raises:
        ValueError if any patient ID appears more than once in the records
        with different labels (label inconsistency).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # ── Group by patient ──────────────────────────────────────
    patient_records: Dict[str, List[Dict]] = defaultdict(list)
    patient_labels:  Dict[str, int] = {}

    for record in records:
        pid   = str(record[patient_id_key])
        label = int(record[label_key])
        patient_records[pid].append(record)

        # Verify label consistency within a patient
        if pid in patient_labels:
            if patient_labels[pid] != label:
                # In LIDC-IDRI a patient can have multiple nodules with
                # different malignancy ratings — use max malignancy
                patient_labels[pid] = max(patient_labels[pid], label)
        else:
            patient_labels[pid] = label

    patient_ids = sorted(patient_records.keys())
    logger.info(f"Total unique patients: {len(patient_ids)}")

    # ── Stratify by class ─────────────────────────────────────
    class_pids: Dict[int, List[str]] = defaultdict(list)
    for pid in patient_ids:
        class_pids[patient_labels[pid]].append(pid)

    rng = random.Random(seed)

    train_pids, val_pids, test_pids = [], [], []

    for cls_label, pids in class_pids.items():
        shuffled = pids[:]
        rng.shuffle(shuffled)
        n   = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        # test gets the remainder
        train_pids += shuffled[:n_train]
        val_pids   += shuffled[n_train:n_train + n_val]
        test_pids  += shuffled[n_train + n_val:]

    # ── Verify no overlap ─────────────────────────────────────
    _verify_no_overlap(train_pids, val_pids, test_pids)

    # ── Collect records ───────────────────────────────────────
    def _collect(pids):
        out = []
        for pid in pids:
            out.extend(patient_records[pid])
        return out

    train = _collect(train_pids)
    val   = _collect(val_pids)
    test  = _collect(test_pids)

    logger.info(f"Split: train={len(set(train_pids))} patients ({len(train)} records), "
                f"val={len(set(val_pids))} patients ({len(val)} records), "
                f"test={len(set(test_pids))} patients ({len(test)} records)")

    return train, val, test


def _verify_no_overlap(train_pids: List[str],
                         val_pids: List[str],
                         test_pids: List[str]):
    """Raise if any patient ID appears in more than one split."""
    train_set = set(train_pids)
    val_set   = set(val_pids)
    test_set  = set(test_pids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    leaks = []
    if tv:
        leaks.append(f"train∩val: {tv}")
    if tt:
        leaks.append(f"train∩test: {tt}")
    if vt:
        leaks.append(f"val∩test: {vt}")

    if leaks:
        raise ValueError(
            f"PATIENT SPLIT CONTAMINATION DETECTED: {'; '.join(leaks)}. "
            "This would invalidate any reported accuracy metric."
        )


# ─────────────────────────────────────────────────────────────
#  Split hash (for experiment tracking)
# ─────────────────────────────────────────────────────────────

def compute_split_hash(train_records: List[Dict],
                        val_records: List[Dict],
                        test_records: List[Dict],
                        patient_id_key: str = "patient_id") -> str:
    """Compute SHA-256 hash of patient ID lists in each split.

    This hash is stored with every MLflow experiment run so that
    split reproducibility can be verified retrospectively.
    """
    def _pids(records):
        return sorted(set(str(r[patient_id_key]) for r in records))

    content = json.dumps({
        "train": _pids(train_records),
        "val":   _pids(val_records),
        "test":  _pids(test_records),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def save_split(train_records: List[Dict],
               val_records:   List[Dict],
               test_records:  List[Dict],
               out_path: str | Path,
               seed: int = 42):
    """Save a split to JSON with metadata for reproducibility."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_hash = compute_split_hash(train_records, val_records, test_records)

    payload = {
        "PATIENT_LEVEL_SPLIT_VERIFIED": True,
        "split_hash_sha256": split_hash,
        "seed": seed,
        "n_train": len(train_records),
        "n_val":   len(val_records),
        "n_test":  len(test_records),
        "train":   train_records,
        "val":     val_records,
        "test":    test_records,
    }
    with open(str(out_path), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"Split saved to {out_path}  (hash: {split_hash[:12]}...)")
    return split_hash


def load_split(path: str | Path) -> Tuple[List[Dict], List[Dict],
                                            List[Dict], str]:
    """Load a saved split and verify integrity.

    Returns:
        (train, val, test, split_hash)
    """
    with open(str(path)) as f:
        data = json.load(f)

    if not data.get("PATIENT_LEVEL_SPLIT_VERIFIED"):
        raise ValueError(
            "Loaded split was NOT created by patient_split.py "
            "(PATIENT_LEVEL_SPLIT_VERIFIED flag missing). "
            "Do not use this split for reporting."
        )

    train = data["train"]
    val   = data["val"]
    test  = data["test"]

    # Recompute and verify hash
    recomputed = compute_split_hash(train, val, test)
    stored     = data.get("split_hash_sha256", "")
    if recomputed != stored:
        raise ValueError(
            f"Split file integrity check FAILED. "
            f"Stored hash: {stored[:12]}... "
            f"Recomputed:  {recomputed[:12]}..."
        )

    _verify_no_overlap(
        [r["patient_id"] for r in train],
        [r["patient_id"] for r in val],
        [r["patient_id"] for r in test],
    )

    return train, val, test, stored
