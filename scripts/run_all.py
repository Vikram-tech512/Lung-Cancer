# scripts/run_all.py -- Master Lung Cancer Execution Script
from __future__ import annotations

import io
import os
import pathlib
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
log_fp = open(LOGS / "run_all.log", "w", encoding="utf-8")

PASS_: list[str] = []
FAIL_: list[str] = []
SKIP_: list[str] = []


def hdr(msg: str) -> None:
    banner = "=" * 80 + "\n  " + msg + "\n" + "-" * 80
    print("\n" + banner)
    log_fp.write("\n" + banner + "\n")
    log_fp.flush()


def info(msg: str) -> None:
    line = "  " + msg
    print(line)
    log_fp.write(line + "\n")
    log_fp.flush()


def run(cmd: list[str], cwd: pathlib.Path, label: str, check: bool = False) -> bool:
    info("$ " + " ".join(str(x) for x in cmd))
    t0 = time.time()
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.time() - t0
    status = "PASS" if proc.returncode == 0 else "FAIL"
    info(f"[{status}] {label} ({dt:.1f}s)")
    if proc.stdout:
        for line in proc.stdout.strip().splitlines()[-15:]:
            info("    " + line)
    if proc.returncode == 0:
        PASS_.append(label)
    else:
        FAIL_.append(label)
        if check:
            raise RuntimeError(f"Step {label} failed")
    return proc.returncode == 0


def main() -> int:
    hdr("LUNG CANCER AI RESEARCH PLATFORM -- MASTER EXECUTION SUITE")

    # Step 1: Model Smoke Test
    hdr("STEP 1: 3D Deep Learning Model Architecture Smoke Tests")
    run([PY, "-m", "pytest", "backend/tests/test_models.py", "-v"], ROOT, "Model forward pass smoke tests")

    # Step 2: Full Test Suite
    hdr("STEP 2: Automated Backend Test Suite (DICOM, Pipeline, Interop, XAI, Continuous Reporting)")
    run([PY, "-m", "pytest", "backend/tests", "-v"], ROOT, "Full backend test suite")

    # Step 2b: Continuous Reporting & CPU Benchmark Verification
    hdr("STEP 2B: Continuous Reporting & CPU-Level Inference Benchmark")
    run([PY, "experiments/benchmark_continuous_reporting.py"], ROOT, "CPU continuous reporting benchmark")

    # Step 3: Frontend Build Verification
    hdr("STEP 3: Frontend Client Build Verification")
    dist_index = ROOT / "frontend" / "dist" / "index.html"
    if dist_index.exists():
        info(f"[PASS] Frontend production bundle exists: {dist_index}")
        PASS_.append("Frontend production bundle")
    else:
        info("[INFO] Building frontend production bundle...")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        run([npm_cmd, "run", "build"], ROOT / "frontend", "Frontend production build")

    # Summary
    hdr("MASTER RESULTS SUMMARY")
    info(f"Steps PASSED : {len(PASS_)}")
    info(f"Steps FAILED : {len(FAIL_)}  -->  {FAIL_}")
    info(f"Steps SKIPPED: {len(SKIP_)}  -->  {SKIP_}")
    info(f"Log file     : {LOGS / 'run_all.log'}")

    log_fp.close()
    return 0 if not FAIL_ else 1


if __name__ == "__main__":
    sys.exit(main())
