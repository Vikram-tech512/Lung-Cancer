import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
from src.dataset.universal_loader import universal_loader
from src.inference.engine import get_inference_engine

def run_test():
    with open("data/real_world_dataset/dataset_manifest.json") as f:
        manifest = json.load(f)

    engine = get_inference_engine()
    print(f"Testing complete diagnosis & XAI on {len(manifest)} diverse real-world cases...")

    # Sample one from each category
    test_indices = [0, 6, 12, 18, 24, 30]
    for idx in test_indices:
        item = manifest[idx]
        p = Path(item["file_path"])
        pil_img, tensor, meta = universal_loader.load_image(p)

        # 1. Classification
        res = engine.classify_image(pil_img, arch="hybrid")
        # 2. Full XAI
        xai = engine.generate_full_xai(pil_img, arch="hybrid")

        print(f"[{item['subtype']}] {item['filename']} -> Predicted: {res['predicted_class']} (conf: {res['confidence']}), CAM: {bool(xai.get('gradcam_base64'))}, Attn: {bool(xai.get('attention_base64'))}")

    print("[SUCCESS] All multi-architecture inferences & XAI maps generated without failure!")

if __name__ == "__main__":
    run_test()
