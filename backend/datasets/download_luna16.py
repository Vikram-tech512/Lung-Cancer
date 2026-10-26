"""
backend/datasets/download_luna16.py
===================================
Automated, resumable, verified downloader and extractor for official LUNA16 dataset
from Zenodo record 3723295.

Strict constraints:
1. Target directory: C:\\LUNA16_DATA (outside OneDrive).
2. Sequential subset processing: check disk -> download -> verify size & MD5 ->
   extract -> verify .mhd/.raw pairs -> delete .zip to conserve space.
3. Split generation: train (subsets 0, 1), val (subset 2), test (subset 3).
   Assert zero scan overlap, compute SHA-256 split hash, and write results/split_manifest.json.
"""

import csv
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Metadata for official Zenodo Record 3723295
ZENODO_RECORD_ID = "3723295"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"

FILES_METADATA = {
    "annotations.csv": {
        "url": f"{ZENODO_BASE_URL}/annotations.csv/content",
        "size_bytes": 136986,
        "md5": "f4404df491aee6445a4485061bfdffdd",
        "is_zip": False,
    },
    "candidates_V2.zip": {
        "url": f"{ZENODO_BASE_URL}/candidates_V2.zip/content",
        "size_bytes": 11439452,
        "md5": "2d9c19aa140e6c6705bb81ee26742022",
        "is_zip": True,
        "extract_to_root": True,
    },
    "subset0.zip": {
        "url": f"{ZENODO_BASE_URL}/subset0.zip/content",
        "size_bytes": 6811924508,
        "md5": "1065b0f42b8c25cf29260fd924a3c3a2",
        "is_zip": True,
        "extract_dir": "subset0",
    },
    "subset1.zip": {
        "url": f"{ZENODO_BASE_URL}/subset1.zip/content",
        "size_bytes": 6334778552,
        "md5": "2eda28bb123543074f5cd849499b87ff",
        "is_zip": True,
        "extract_dir": "subset1",
    },
    "subset2.zip": {
        "url": f"{ZENODO_BASE_URL}/subset2.zip/content",
        "size_bytes": 7257937108,
        "md5": "685b484b085bcdb81c525f3b1bf97c51",
        "is_zip": True,
        "extract_dir": "subset2",
    },
    "subset3.zip": {
        "url": f"{ZENODO_BASE_URL}/subset3.zip/content",
        "size_bytes": 6896620114,
        "md5": "54707e9e8954af29326eed60f38ea321",
        "is_zip": True,
        "extract_dir": "subset3",
    },
}

def get_free_disk_gb(path: str | Path) -> float:
    """Return free disk space in GB for the drive containing path."""
    total, used, free = shutil.disk_usage(str(path))
    return free / (1024 ** 3)

def compute_file_md5(file_path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    """Compute MD5 checksum in 4MB streaming chunks."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest().lower()

def download_file_resumable(
    url: str,
    target_path: Path,
    expected_size: int,
    expected_md5: str,
    chunk_size: int = 1024 * 1024,
) -> bool:
    """
    Download a file with HTTP Range support for resumption and progress monitoring.
    Verifies final file size and MD5 checksum.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if target_path.stat().st_size == expected_size:
            print(f"[VERIFY] {target_path.name} already exists. Checking MD5...")
            if compute_file_md5(target_path) == expected_md5.lower():
                print(f"[OK] {target_path.name} already downloaded and verified.")
                return True
            else:
                print(f"[WARN] Existing {target_path.name} MD5 mismatch. Re-downloading.")
                target_path.unlink()
        else:
            print(f"[WARN] Existing {target_path.name} size mismatch. Removing.")
            target_path.unlink()

    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    max_retries = 25
    retry_count = 0
    chunk_size = 2 * 1024 * 1024  # 2MB chunks for better throughput

    while retry_count < max_retries:
        downloaded = temp_path.stat().st_size if temp_path.exists() else 0
        if downloaded > expected_size:
            print(f"[WARN] Partial file {temp_path.name} exceeds expected size ({downloaded} > {expected_size}). Truncating.")
            temp_path.unlink()
            downloaded = 0
        elif downloaded == expected_size:
            print(f"[INFO] Partial file matches expected size {expected_size}.")
            break

        headers = {"User-Agent": "LungCancerAI-Research/1.0"}
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"[RESUME] Resuming {target_path.name} from byte {downloaded:,} / {expected_size:,} ({downloaded/expected_size*100:.1f}%)")
        else:
            print(f"[START] Downloading {target_path.name} ({expected_size / (1024**2):.2f} MB)...")

        try:
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            last_print = t0
            bytes_since_last = 0

            mode = "ab" if downloaded > 0 else "wb"
            with urllib.request.urlopen(req, timeout=30) as response:
                if downloaded > 0 and response.status != 206:
                    print(f"[WARN] Server returned {response.status} instead of 206. Restarting download from byte 0.")
                    downloaded = 0
                    mode = "wb"
                    
                with open(temp_path, mode) as out_f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        bytes_since_last += len(chunk)

                        now = time.time()
                        if now - last_print >= 5.0 or downloaded >= expected_size:
                            elapsed = now - t0
                            interval_speed = bytes_since_last / (now - last_print) if (now - last_print) > 0 else 0
                            pct = (downloaded / expected_size) * 100
                            remaining_bytes = max(0, expected_size - downloaded)
                            eta_sec = remaining_bytes / interval_speed if interval_speed > 0 else 0
                            print(
                                f"  [{target_path.name}] {downloaded / (1024**2):.1f}/{expected_size / (1024**2):.1f} MB "
                                f"({pct:.1f}%) | Speed: {interval_speed / (1024**2):.2f} MB/s | ETA: {eta_sec:.0f}s"
                            )
                            last_print = now
                            bytes_since_last = 0

            if temp_path.stat().st_size >= expected_size:
                break

        except Exception as e:
            retry_count += 1
            print(f"[RETRY {retry_count}/{max_retries}] Transient network error ({e}). Reconnecting in 5s...")
            time.sleep(5)



    if downloaded != expected_size:
        raise IOError(f"Downloaded size ({downloaded}) does not match expected size ({expected_size}) for {target_path.name}")

    print(f"[VERIFY] Computing MD5 for {target_path.name}...")
    actual_md5 = compute_file_md5(temp_path)
    if actual_md5 != expected_md5.lower():
        raise ValueError(f"MD5 mismatch for {target_path.name}: expected {expected_md5}, got {actual_md5}")

    print(f"[OK] MD5 verified: {actual_md5}")
    if target_path.exists():
        target_path.unlink()
    temp_path.rename(target_path)
    return True

def verify_mhd_raw_pairs(directory: Path):
    """
    Verify that every .mhd file has an existing corresponding .raw file.
    Returns (count_of_valid_pairs, list_of_errors).
    """
    mhd_files = list(directory.glob("*.mhd"))
    valid_count = 0
    errors = []
    for mhd in mhd_files:
        raw = mhd.with_suffix(".raw")
        if not raw.exists():
            has_raw = False
            try:
                with open(mhd, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("ElementDataFile"):
                            val = line.split("=")[-1].strip()
                            if (mhd.parent / val).exists():
                                has_raw = True
                            break
            except Exception:
                pass
            if not has_raw:
                errors.append(f"Missing .raw for {mhd.name}")
                continue
        valid_count += 1
    return valid_count, errors


def extract_zip(zip_path: Path, extract_target_dir: Path, expected_subfolder: Optional[str] = None) -> None:
    """Extract a zip archive safely and display progress, handling root vs subfolder layout."""
    print(f"[EXTRACT] Extracting {zip_path.name}...")
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if not members:
            raise ValueError(f"Zip file {zip_path.name} is empty!")

        # Check if the zip already packages files inside expected_subfolder/
        has_subfolder = any(m.startswith(f"{expected_subfolder}/") or m.startswith(f"{expected_subfolder}\\") for m in members if expected_subfolder)
        
        if expected_subfolder and not has_subfolder:
            # Files are flat inside the zip; extract them into extract_target_dir / expected_subfolder
            dest = extract_target_dir / expected_subfolder
            dest.mkdir(parents=True, exist_ok=True)
            print(f"  Extracting flat zip contents into {dest}...")
            zf.extractall(dest)
        else:
            # Files already have subset folder or extracting to root (like candidates_V2)
            print(f"  Extracting into {extract_target_dir}...")
            zf.extractall(extract_target_dir)
            
    elapsed = time.time() - t0
    print(f"[EXTRACT DONE] Extracted {zip_path.name} in {elapsed:.1f}s.")



def generate_split_manifest(
    luna_root: Path,
    output_path: Path,
) -> dict:
    """
    Analyze extracted subsets (0-3), verify zero scan overlap,
    compute candidate counts and SHA-256 split hash, and write results/split_manifest.json.
    """
    print("\n[SPLIT] Generating scan-level split manifest...")
    cand_csv = luna_root / "candidates_V2.csv"
    if not cand_csv.exists():
        # Fallback to candidates.csv if present
        cand_csv = luna_root / "candidates.csv"
    if not cand_csv.exists():
        raise FileNotFoundError(f"Neither candidates_V2.csv nor candidates.csv found in {luna_root}")

    ann_csv = luna_root / "annotations.csv"

    # Scan directory listing for subsets 0, 1, 2, 3
    subset_scans = {}
    for sid in range(4):
        sdir = luna_root / f"subset{sid}"
        if not sdir.exists():
            raise FileNotFoundError(f"Missing subset directory: {sdir}")
        scans = sorted([f.stem for f in sdir.glob("*.mhd")])
        subset_scans[sid] = scans
        print(f"  Subset {sid}: {len(scans)} CT scans found.")

    # Split assignment
    train_scans = sorted(list(set(subset_scans[0] + subset_scans[1])))
    val_scans = sorted(list(set(subset_scans[2])))
    test_scans = sorted(list(set(subset_scans[3])))

    # Assert zero scan overlap
    train_set = set(train_scans)
    val_set = set(val_scans)
    test_set = set(test_scans)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    assert len(overlap_train_val) == 0, f"Overlap between train and val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Overlap between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Overlap between val and test: {overlap_val_test}"
    print("[ASSERT] Zero scan overlap verified across Train (0,1), Val (2), Test (3)!")

    # Parse candidate CSV to compute exact candidate counts per subset
    scan_to_subset = {}
    for sid, scans in subset_scans.items():
        for s in scans:
            scan_to_subset[s] = sid

    subset_counts = {
        sid: {"scans": len(subset_scans[sid]), "total_candidates": 0, "positives": 0, "negatives": 0}
        for sid in range(4)
    }

    print(f"[CANDIDATES] Parsing {cand_csv.name}...")
    with open(cand_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suid = row["seriesuid"]
            if suid in scan_to_subset:
                sid = scan_to_subset[suid]
                is_pos = int(row["class"]) == 1
                subset_counts[sid]["total_candidates"] += 1
                if is_pos:
                    subset_counts[sid]["positives"] += 1
                else:
                    subset_counts[sid]["negatives"] += 1

    # Annotations count
    ann_positives = 0
    if ann_csv.exists():
        with open(ann_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["seriesuid"] in scan_to_subset:
                    ann_positives += 1

    # Compute SHA-256 split hash from canonically sorted scan IDs
    split_fingerprint = {
        "train_scans": train_scans,
        "val_scans": val_scans,
        "test_scans": test_scans,
    }
    split_hash = hashlib.sha256(
        json.dumps(split_fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()

    manifest = {
        "dataset": "LUNA16 (subsets 0-3)",
        "zenodo_record": ZENODO_RECORD_ID,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "split_hash_sha256": split_hash,
        "zero_scan_overlap_asserted": True,
        "split_definitions": {
            "train": {"subsets": [0, 1], "scan_count": len(train_scans)},
            "val": {"subsets": [2], "scan_count": len(val_scans)},
            "test": {"subsets": [3], "scan_count": len(test_scans)},
        },
        "subset_breakdown": subset_counts,
        "summary": {
            "total_scans": sum(s["scans"] for s in subset_counts.values()),
            "total_candidates": sum(s["total_candidates"] for s in subset_counts.values()),
            "total_positives": sum(s["positives"] for s in subset_counts.values()),
            "total_negatives": sum(s["negatives"] for s in subset_counts.values()),
            "annotations_in_subsets_0_to_3": ann_positives,
        },
        "scans": split_fingerprint,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Split manifest written to {output_path}")
    print(f"     Split SHA-256: {split_hash}")
    return manifest


def main():
    default_luna = Path.home() / "LUNA16_DATA" if os.name == "nt" else Path("data/luna16")
    luna_root = Path(os.environ.get("LUNA16_ROOT", str(default_luna)))
    print(f"===========================================================")
    print(f" LUNA16 Resumable Pipeline Downloader & Split Generator")
    print(f" Target Root: {luna_root}")
    print(f"===========================================================")

    # Confirm outside OneDrive
    onedrive_env = os.environ.get("OneDrive", "").lower()
    if onedrive_env and str(luna_root).lower().startswith(onedrive_env):
        raise ValueError(f"CRITICAL: LUNA16_ROOT ({luna_root}) is located inside OneDrive! Must be outside.")

    luna_root.mkdir(parents=True, exist_ok=True)
    free_gb = get_free_disk_gb(luna_root)
    print(f"[DISK] Available disk space on {luna_root.drive}: {free_gb:.2f} GB")
    if free_gb < 20.0:
        raise RuntimeError(f"Insufficient disk space ({free_gb:.2f} GB). Minimum 20 GB required before download.")

    timings = {}

    # 1. Download metadata files (annotations.csv, candidates_V2.zip)
    print("\n--- STEP 1: Metadata Files ---")
    for fname in ["annotations.csv", "candidates_V2.zip"]:
        meta = FILES_METADATA[fname]
        target_file = luna_root / fname
        t0 = time.time()
        # If candidate csv already extracted, skip zip download
        if fname == "candidates_V2.zip" and (luna_root / "candidates_V2.csv").exists():
            print(f"[SKIP] candidates_V2.csv already extracted.")
            continue
        if fname == "annotations.csv" and target_file.exists() and target_file.stat().st_size == meta["size_bytes"]:
            print(f"[SKIP] annotations.csv already downloaded and size matches.")
            continue

        download_file_resumable(meta["url"], target_file, meta["size_bytes"], meta["md5"])
        timings[fname] = time.time() - t0

        if meta.get("is_zip"):
            extract_zip(target_file, luna_root)
            # Delete zip
            print(f"[CLEANUP] Removing {target_file.name} to save disk space.")
            target_file.unlink()

    # 2. Sequential Subsets 0-3
    print("\n--- STEP 2: Subsets 0 to 3 (Sequential Download & Extraction) ---")
    for sid in range(4):
        subset_zip_name = f"subset{sid}.zip"
        meta = FILES_METADATA[subset_zip_name]
        subset_dir = luna_root / f"subset{sid}"

        # Check if already extracted and valid
        if subset_dir.exists():
            n_valid, errs = verify_mhd_raw_pairs(subset_dir)
            if n_valid > 80 and len(errs) == 0:
                print(f"[SKIP] Subset {sid} already extracted ({n_valid} valid .mhd/.raw pairs). Skipping download.")
                continue

        # Check free disk before starting download
        free_gb = get_free_disk_gb(luna_root)
        print(f"\n[DISK CHECK] Subset {sid}: {free_gb:.2f} GB free.")
        needed_gb = (meta["size_bytes"] / (1024**3)) * 2.1  # zip + extracted
        if free_gb < needed_gb:
            raise RuntimeError(f"Insufficient disk space for subset {sid}: {free_gb:.2f} GB free, need ~{needed_gb:.2f} GB")

        target_zip = luna_root / subset_zip_name
        t0 = time.time()
        download_file_resumable(meta["url"], target_zip, meta["size_bytes"], meta["md5"])
        t_download = time.time() - t0
        timings[f"subset{sid}_download"] = t_download

        # Extract
        t0 = time.time()
        # Extract into subset_dir: Zenodo subset zip usually contains files directly or within subsetN/
        extract_zip(target_zip, luna_root, expected_subfolder=f"subset{sid}")
        t_extract = time.time() - t0
        timings[f"subset{sid}_extract"] = t_extract

        # Verify .mhd/.raw pairs
        n_valid, errs = verify_mhd_raw_pairs(subset_dir)
        print(f"[VERIFY] Subset {sid}: {n_valid} .mhd/.raw scan pairs verified.")
        if errs:
            raise ValueError(f"Subset {sid} verification failed with errors: {errs[:5]}")

        # Delete zip
        print(f"[CLEANUP] Deleting {target_zip.name} to reclaim {meta['size_bytes'] / (1024**3):.2f} GB...")
        target_zip.unlink()

    # 3. Generate Split Manifest
    manifest_path = Path("results/split_manifest.json")
    manifest = generate_split_manifest(luna_root, manifest_path)

    print("\n===========================================================")
    print(" LUNA16 Step 1 Download & Split Complete!")
    print("===========================================================")
    print("Timings Summary:")
    for k, v in timings.items():
        print(f"  {k}: {v:.1f}s")
    print(f"Total Scans in Subsets 0-3: {manifest['summary']['total_scans']}")
    print(f"Total Candidates: {manifest['summary']['total_candidates']} (Positives: {manifest['summary']['total_positives']}, Negatives: {manifest['summary']['total_negatives']})")
    print(f"Split Hash: {manifest['split_hash_sha256']}")
    print("===========================================================")


if __name__ == "__main__":
    main()

