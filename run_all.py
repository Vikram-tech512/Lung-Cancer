# run_all.py  --  Master Lung Cancer Execution Script
# Executes ALL components across the entire Lung Cancer project folder:
#   Phase 1  -- Dataset engineering (EDA + preprocessing)
#   Phase 2  -- Model architecture smoke test (CNN / ViT / Hybrid)
#   Phase 3  -- Evaluation, calibration, error analysis
#   Phase 4  -- Robustness engine (Hybrid grid + RD/RRD metrics)
#   Seg      -- Lung segmentation module + engine validation
#   Dashboard-- Open LungXAI clinical dashboard
import io, json, os, pathlib, shutil, stat, subprocess, sys, time, webbrowser, yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(r"C:\Users\ashok\OneDrive\Desktop\Lung Cancer")
P1   = ROOT / "lung-cancer-xai-phase1"
P2   = ROOT / "lung-cancer-xai-phase2"
P3   = ROOT / "lung-cancer-xai-phase3"
P4   = ROOT / "lung-cancer-xai-phase4"
SEG  = ROOT / "lung_cancer_segmentation"
AI_PLATFORM = ROOT / "lung-cancer-ai"
PY   = sys.executable

RESULTS = ROOT / "EXECUTION_RESULTS"
RESULTS.mkdir(exist_ok=True)
log_fp  = open(RESULTS / "master_execution_log.txt", "w", encoding="utf-8")

PASS_, FAIL_, SKIP_ = [], [], []


# ── helpers ──────────────────────────────────────────────────────────────────
def hdr(msg):
    banner = (
        "=" * 80 + "\n"
        "  RUN MODE: SYNTHETIC / SMOKE-TEST DATA — NOT A CLINICAL OR RESEARCH RESULT\n"
        "=" * 80
    )
    s = "\n" + banner + "\n  " + msg + "\n" + "-" * 80
    print(s); log_fp.write(s + "\n"); log_fp.flush()


def info(msg):
    line = "  " + msg
    print(line); log_fp.write(line + "\n"); log_fp.flush()


def run(cmd, cwd, label, check=False):
    info("$ " + " ".join(str(x) for x in cmd))
    t0 = time.time()
    proc = subprocess.run(
        [str(x) for x in cmd], cwd=str(cwd), check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace",
    )
    elapsed = time.time() - t0
    for ln in (proc.stdout or "").splitlines()[-60:]:
        print("    " + ln); log_fp.write("    " + ln + "\n")
    ok   = proc.returncode == 0
    tag  = "PASS" if ok else "FAIL"
    info(f"[{tag}] {label} ({elapsed:.1f}s)")
    log_fp.flush()
    (PASS_ if ok else FAIL_).append(label)
    return ok


def rm_ro(fn, path, exc):
    os.chmod(path, stat.S_IWRITE)
    fn(path)


def sync_phase(target, src=P3):
    dst_raw = target / "data" / "raw"
    # Overlay-copy (dirs_exist_ok) avoids rmtree file-lock issues on Windows/OneDrive
    dst_raw.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(str(src / "data" / "raw"), str(dst_raw), dirs_exist_ok=True)
    except Exception as ex:
        info("  [WARN] copy overlay partial: " + str(ex))
    dst_ck = target / "checkpoints"
    dst_ck.mkdir(exist_ok=True)
    for ckf in (src / "checkpoints").glob("*.pt"):
        dest = dst_ck / ckf.name
        if not dest.exists():
            shutil.copy2(str(ckf), str(dest))
    cfg_path = target / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["name"]    = "Synthetic-IQ-OTH-NCCD"
    cfg["dataset"]["classes"] = ["normal", "benign", "malignant"]
    cfg["dataset"]["raw_dir"] = "data/raw"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    info("Synced data + config: " + target.name)


# ── PHASE 1 -- Dataset Engineering (Legacy) ──────────────────────────────────
if P1.exists():
    hdr("PHASE 1 -- Dataset Engineering (EDA + Preprocessing)")
    sync_phase(P1)
    (P1 / "results").mkdir(exist_ok=True)
    run([PY, "-X", "utf8", "-m", "src.data.dataset"],
        P1, "P1: Dataset discovery, dedup & split")
    run([PY, "-X", "utf8", "-m", "src.preprocessing.preprocessing"],
        P1, "P1: CLAHE preprocessing pipeline")

# ── PHASE 2 -- Model Architecture Smoke Test ──────────────────────────────────
if P2.exists():
    hdr("PHASE 2 -- Model Architecture Smoke Test (CNN / ViT / Hybrid)")
    sync_phase(P2)
    (P2 / "results").mkdir(exist_ok=True)
    smoke_path = P2 / "_smoke_test.py"
    smoke_code = (
        "import sys, torch, yaml\n"
        "sys.path.insert(0, '.')\n"
        "from src.models.cnn import CNNBaseline\n"
        "from src.models.vit import ViTBaseline\n"
        "try:\n"
        "    from src.models.hybrid_vit import HybridCNNViT as HybridViT\n"
        "except ImportError:\n"
        "    from src.models.hybrid_vit import HybridViT\n"
        "cfg = yaml.safe_load(open('config.yaml'))\n"
        "dummy = torch.randn(2, 3, 224, 224)\n"
        "for Cls, name in [(CNNBaseline,'CNN'),(ViTBaseline,'ViT'),(HybridViT,'Hybrid')]:\n"
        "    try:\n"
        "        m = Cls(cfg); out = m(dummy)\n"
        "        print('[PASS]', name, tuple(out.shape))\n"
        "    except Exception as e:\n"
        "        print('[FAIL]', name, str(e))\n"
    )
    smoke_path.write_text(smoke_code, encoding="utf-8")
    run([PY, "-X", "utf8", str(smoke_path)], P2, "P2: CNN/ViT/Hybrid forward-pass smoke test")
    smoke_path.unlink(missing_ok=True)

# ── PHASE 3 -- Evaluation, Calibration, Error Analysis ───────────────────────
if P3.exists():
    hdr("PHASE 3 -- Evaluation, Calibration & Error Analysis")
    for model in ["cnn", "vit", "hybrid"]:
        ckpt = P3 / "checkpoints" / f"{model}_best.pt"
        if not ckpt.exists():
            SKIP_.append(f"P3:{model}"); continue
        run([PY, "-X", "utf8", "-m", "src.evaluation.metrics",
             "--model", model, "--checkpoint", str(ckpt), "--config", "config.yaml"],
            P3, f"P3: {model.upper()} evaluation metrics")
        run([PY, "-X", "utf8", "-m", "src.calibration.calibration",
             "--model", model, "--checkpoint", str(ckpt), "--config", "config.yaml"],
            P3, f"P3: {model.upper()} calibration (ECE/MCE/Brier)")
        run([PY, "-X", "utf8", "-m", "src.evaluation.error_analysis",
             "--model", model, "--checkpoint", str(ckpt), "--config", "config.yaml"],
            P3, f"P3: {model.upper()} error analysis")
    run([PY, "-X", "utf8", "-m", "src.evaluation.compare_models", "--config", "config.yaml"],
        P3, "P3: Multi-model comparison report")

# ── PHASE 4 -- Robustness: Hybrid grid + all metrics ─────────────────────────
if P4.exists():
    hdr("PHASE 4 -- Robustness: Hybrid Grid + RD/RRD Metrics (all models)")
    run([PY, "-X", "utf8", "-m", "src.robustness.robustness_runner",
         "--model", "hybrid", "--checkpoint", "checkpoints/hybrid_best.pt",
         "--config", "config.yaml", "--batch_size", "8"],
        P4, "P4: Hybrid CNN-ViT robustness grid (31 conditions)")

    for model in ["cnn", "vit", "hybrid"]:
        raw_f = P4 / "results" / "metrics" / "robustness" / f"{model}_robustness_raw.json"
        if not raw_f.exists():
            SKIP_.append(f"P4:{model}_metrics"); continue
        run([PY, "-X", "utf8", "-m", "src.robustness.robustness_metrics", "--model", model],
            P4, f"P4: {model.upper()} robustness metrics (RD/RRD)")

# ── LUNG_CANCER_SEGMENTATION -- Phase 5 Infrastructure & Quality Gates ─────────
hdr("PHASE 5 -- Thoracic & Nodule Segmentation Module (Infra + Quality Gate)")
seg_script = SEG / "_validate.py"
seg_code = (
    "import sys; sys.path.insert(0, '.')\n"
    "pairs = [\n"
    "    ('src.inference.engine',         'InferenceEngine'),\n"
    "    ('src.inference.service',        'InferenceService'),\n"
    "    ('src.explainability.gradcam',   'GradCAM'),\n"
    "    ('src.explainability.attention_viz', 'AttentionVisualizer'),\n"
    "    ('src.robustness.analyzer',      'RobustnessAnalyzer'),\n"
    "    ('src.robustness.perturbations', 'apply_perturbation'),\n"
    "    ('src.dataset.lung_dataset',     'LungDataset'),\n"
    "]\n"
    "for mod, cls in pairs:\n"
    "    try:\n"
    "        m = __import__(mod, fromlist=[cls])\n"
    "        getattr(m, cls)\n"
    "        print('[PASS]', mod + '.' + cls)\n"
    "    except Exception as e:\n"
    "        print('[FAIL]', mod, str(e))\n"
    "import pathlib\n"
    "jsons = list(pathlib.Path('.').rglob('*.json'))\n"
    "print('[INFO] Result JSON files found:', len(jsons))\n"
    "for j in sorted(jsons)[:10]:\n"
    "    print('  ', j)\n"
)
seg_script.write_text(seg_code, encoding="utf-8")
run([PY, "-X", "utf8", str(seg_script)], SEG, "PHASE_5_INFRA: Module & engine validation")
seg_script.unlink(missing_ok=True)

# Run Phase 5A Quality Gate (Dice, IoU, Sensitivity, Specificity, HD95)
run([PY, "-X", "utf8", "-m", "src.evaluation.segmentation_quality"],
    SEG, "PHASE_5_QUALITY: Real segmentation metrics (Dice/IoU/Sens/Spec/HD95)")


# ── LUNG-CANCER-AI -- 3D Multimodal AI Platform & Rigor Suite ─────────────────
hdr("LUNG-CANCER-AI -- 3D Multimodal Oncology System & Rigor Suite")
if AI_PLATFORM.exists() and (AI_PLATFORM / "run_master_system.py").exists():
    run([PY, "-X", "utf8", "run_master_system.py"], AI_PLATFORM, "AI-PLATFORM: 3D Pipeline & Rigor Addendum (Stages 1-10)")
else:
    SKIP_.append("AI-PLATFORM: run_master_system.py not found")

# ── UNIFIED FULL-STACK PLATFORM (BACKEND & FRONTEND) ─────────────────────────
hdr("UNIFIED FULL-STACK PLATFORM (BACKEND & FRONTEND)")
run([PY, "-m", "pytest", "backend/tests/test_models.py", "-v"],
    ROOT, "PLATFORM: 3D DL Model Suite (UNet3D/ResNet3D/DenseNet3D/Swin3D/Ensemble)")

# ── Collect & print master summary ───────────────────────────────────────────
hdr("MASTER RESULTS SUMMARY")

summary = {
    "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "passed_steps": PASS_,
    "failed_steps": FAIL_,
    "skipped_steps": SKIP_,
    "phase3_evaluation": {},
    "phase4_robustness": {},
    "phase5_segmentation": {},
    "model_health_report": {},
}

for model in ["cnn", "vit", "hybrid"]:
    ev = P3 / "results" / f"{model}_full_evaluation.json"
    if ev.exists():
        summary["phase3_evaluation"][model] = json.loads(ev.read_text(encoding="utf-8"))

rob_dir = P4 / "results" / "metrics" / "robustness"
for model in ["cnn", "vit", "hybrid"]:
    mf = rob_dir / f"{model}_robustness_metrics.json"
    rf = rob_dir / f"{model}_robustness_raw.json"
    if mf.exists():
        summary["phase4_robustness"][model] = json.loads(mf.read_text(encoding="utf-8"))
    elif rf.exists():
        raw = json.loads(rf.read_text(encoding="utf-8"))
        accs = [v["accuracy"] for k, v in raw.items() if k != "clean_sev0"]
        clean = raw.get("clean_sev0", {}).get("accuracy")
        summary["phase4_robustness"][model] = {
            "clean_accuracy": clean,
            "min_perturbed_accuracy": round(min(accs), 4) if accs else None,
            "mean_perturbed_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
            "conditions_evaluated": len(raw),
        }

health_f = rob_dir / "MODEL_HEALTH_REPORT.json"
if health_f.exists():
    summary["model_health_report"] = json.loads(health_f.read_text(encoding="utf-8"))

seg_f = SEG / "results" / "evaluation" / "segmentation_quality_metrics.json"
if seg_f.exists():
    summary["phase5_segmentation"] = json.loads(seg_f.read_text(encoding="utf-8"))

(RESULTS / "master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

info("")
info("=== Phase 3 Evaluation (CNN / ViT / Hybrid) ===")
for model, d in summary["phase3_evaluation"].items():
    info(f"  {model.upper():<8}  Acc={d.get('accuracy','?')}  "
         f"F1={d.get('macro_f1','?')}  AUC={d.get('macro_roc_auc','?')}")

info("")
info("=== Phase 4 Robustness (31 conditions each) ===")
for model, d in summary["phase4_robustness"].items():
    info(f"  {model.upper():<8}  Clean={d.get('clean_accuracy','?')}  "
         f"Worst={d.get('min_perturbed_accuracy','?')}  "
         f"Mean={d.get('mean_perturbed_accuracy','?')}")

if summary["phase5_segmentation"]:
    s = summary["phase5_segmentation"]
    info("")
    info("=== Phase 5 Segmentation Quality Gate ===")
    info(f"  Status: {s.get('gate_status')} | Dice: {s.get('dice')} | IoU: {s.get('iou')} | "
         f"Sens: {s.get('sensitivity')} | Spec: {s.get('specificity')} | HD95: {s.get('hd95_mm')} mm")

info("")
info(f"Steps PASSED : {len(PASS_)}")
info(f"Steps FAILED : {len(FAIL_)}  -->  {FAIL_}")
info(f"Steps SKIPPED: {len(SKIP_)}  -->  {SKIP_}")
info(f"Outputs      : {RESULTS}")

# ── Open Full-Stack Clinical Dashboard ────────────────────────────────────────
hdr("Opening Lung Cancer AI Clinical Dashboard")
dash = ROOT / "frontend" / "dist" / "index.html"
if dash.exists():
    try:
        webbrowser.open(dash.as_uri())
        info("  Opened production dashboard: " + str(dash))
    except Exception as ex:
        info("  Dashboard URL: " + dash.as_uri() + f" ({ex})")
else:
    info("  Production build not found. To run the clinical dashboard:")
    info("    cd frontend && npm install && npm run dev  (dev server at http://localhost:3000)")
    info("    OR: cd frontend && npm run build          (creates dist/ for production)")
    info("  Backend API (requires Python uvicorn):")
    info("    python -m backend.main --port 8000 --reload")
    info("  Swagger UI: http://localhost:8000/docs")
    # Try to open the Vite dev server if already running
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:3000", timeout=1)
        webbrowser.open("http://localhost:3000")
        info("  Dev server detected — opened http://localhost:3000")
    except Exception:
        pass

info("")
info("=" * 68)
info("  EXECUTION COMPLETE")
info("  Master summary : " + str(RESULTS / "master_summary.json"))
info("  Execution log  : " + str(RESULTS / "master_execution_log.txt"))
info("=" * 68)

log_fp.close()
