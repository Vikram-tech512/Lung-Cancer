"""Tests for LUNA16 subset assignment and leakage protection."""

import pytest


def _candidate(series_uid, subset_id):
    from backend.datasets.luna16 import LUNA16Candidate

    return LUNA16Candidate(
        series_uid=series_uid,
        coord_x=0.0,
        coord_y=0.0,
        coord_z=0.0,
        label=1,
        diameter_mm=6.0,
        subset_id=subset_id,
    )


def test_subset_mode_uses_fixed_luna16_subsets(tmp_path):
    from backend.datasets.luna16 import LUNA16Loader, luna16_split_hash

    candidates = [
        _candidate("S0", 0),
        _candidate("S1", 1),
        _candidate("S2", 2),
        _candidate("S3", 3),
        _candidate("S4", 4),
    ]
    train, val, test = LUNA16Loader(tmp_path).patient_level_split(
        candidates, split_mode="subset"
    )

    assert {c.subset_id for c in train} == {0, 1}
    assert {c.subset_id for c in val} == {2}
    assert {c.subset_id for c in test} == {3}
    assert luna16_split_hash(train, val, test) == luna16_split_hash(train, val, test)


def test_luna16_split_asserts_patient_and_series_disjointness():
    from backend.datasets.luna16 import assert_luna16_splits_disjoint

    candidate = _candidate("same-series", 0)
    with pytest.raises(AssertionError, match="appears in"):
        assert_luna16_splits_disjoint([candidate], [], [candidate])


def test_repository_defaults_do_not_include_developer_windows_paths():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    files = [
        repo_root / "backend" / "core" / "config.py",
        repo_root / "backend" / "datasets" / "histopathology.py",
        repo_root / "backend" / "training" / "train_luna16_3d.py",
        repo_root / "deploy" / "aws-deploy.ps1",
        repo_root / "deploy" / "deploy-cloudfront.ps1",
    ]
    forbidden = ("C:\\Users\\", "C:\\LUNA16_DATA", "AWSCLIV2\\aws.exe")
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert not any(value in source for value in forbidden), str(path)
