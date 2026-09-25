import os
import shutil
import pathlib

ROOT = pathlib.Path(r"C:\Users\ashok\OneDrive\Desktop\Lung Cancer")
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# 1. Preserve trained checkpoints if any
for p in [ROOT / "lung-cancer-xai-phase3", ROOT / "lung-cancer-xai-phase4", ROOT / "lung-cancer-xai-phase1"]:
    ck_dir = p / "checkpoints"
    if ck_dir.exists():
        for ckf in ck_dir.glob("*.pt"):
            dest = CHECKPOINTS / ckf.name
            if not dest.exists():
                shutil.copy2(str(ckf), str(dest))
                print(f"Preserved checkpoint: {ckf.name} -> checkpoints/")

# 2. Remove read-only attributes on Windows before deletion to prevent permission errors
def remove_readonly(func, path, exc_info):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

# 3. Remove the 4 phase directories
phases = [
    ROOT / "lung-cancer-xai-phase1",
    ROOT / "lung-cancer-xai-phase2",
    ROOT / "lung-cancer-xai-phase3",
    ROOT / "lung-cancer-xai-phase4",
]

for p in phases:
    if p.exists():
        print(f"Removing {p.name}...")
        try:
            shutil.rmtree(p, onerror=remove_readonly)
            print(f"Successfully removed {p.name}")
        except Exception as e:
            print(f"Error removing {p.name}: {e}")

print("Done removing phase directories.")
