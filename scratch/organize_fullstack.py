import os
import shutil
import pathlib
import re

ROOT = pathlib.Path(r"C:\Users\ashok\OneDrive\Desktop\Lung Cancer")
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

def copy_tree_clean(src: pathlib.Path, dst: pathlib.Path, ignore_patterns=None):
    if not src.exists():
        print(f"Source {src} does not exist, skipping.")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = pathlib.Path(root).relative_to(src)
        # Filter directories
        dirs[:] = [d for d in dirs if d not in ["__pycache__", "node_modules", "dist", ".pytest_cache", ".git"]]
        dest_dir = dst / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            src_file = pathlib.Path(root) / f
            dst_file = dest_dir / f
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"Error copying {src_file} -> {dst_file}: {e}")

print("--- 1. Setting up Backend ---")
# Copy platform/backend
copy_tree_clean(ROOT / "platform" / "backend", BACKEND)

# Copy complementary modules from lung-cancer-ai/backend
ai_backend = ROOT / "lung-cancer-ai" / "backend"
if ai_backend.exists():
    for sub in ["auth", "database", "reports", "schemas", "storage", "websocket", "core"]:
        src_sub = ai_backend / sub
        if src_sub.exists():
            copy_tree_clean(src_sub, BACKEND / sub)
    
    # Deidentifier
    deid = ai_backend / "dicom" / "deidentifier.py"
    if deid.exists():
        (BACKEND / "dicom").mkdir(parents=True, exist_ok=True)
        shutil.copy2(deid, BACKEND / "dicom" / "deidentifier.py")

# Copy segmentation modules from lung_cancer_segmentation/src
seg_src = ROOT / "lung_cancer_segmentation" / "src"
if seg_src.exists():
    copy_tree_clean(seg_src / "inference", BACKEND / "segmentation")
    copy_tree_clean(seg_src / "evaluation", BACKEND / "evaluation")
    if (seg_src / "models_db.py").exists():
        (BACKEND / "database").mkdir(parents=True, exist_ok=True)
        shutil.copy2(seg_src / "models_db.py", BACKEND / "database" / "models_db.py")

# Create __init__.py files in all python directories
for dirpath, dirs, files in os.walk(BACKEND):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    init_file = pathlib.Path(dirpath) / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# init\n", encoding="utf-8")

print("--- 2. Fixing Import References in Backend ---")
# Replace 'from platform.backend.' or 'platform.backend.' with 'backend.' or relative imports
for py_file in BACKEND.rglob("*.py"):
    try:
        content = py_file.read_text(encoding="utf-8")
        new_content = content
        new_content = re.sub(r'\bfrom platform\.backend\.', 'from backend.', new_content)
        new_content = re.sub(r'\bimport platform\.backend\.', 'import backend.', new_content)
        new_content = re.sub(r'\bplatform\.backend\.', 'backend.', new_content)
        if new_content != content:
            py_file.write_text(new_content, encoding="utf-8")
            print(f"Fixed imports in: {py_file.relative_to(ROOT)}")
    except Exception as e:
        print(f"Error reading/writing {py_file}: {e}")

print("--- 3. Setting up Frontend ---")
# Copy platform/frontend
copy_tree_clean(ROOT / "platform" / "frontend", FRONTEND)

print("--- Complete! ---")
