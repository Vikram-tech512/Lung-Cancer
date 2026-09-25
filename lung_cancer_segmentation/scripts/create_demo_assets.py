# scripts/create_demo_assets.py
"""Generate high-fidelity synthetic CT scan lung slices for demonstration and local testing.
Creates:
  - sample_malignant.png: CT slice showing a spiculated, hyperdense suspicious pulmonary nodule.
  - sample_benign.png: CT slice showing a well-circumscribed, smooth benign calcified granuloma.
  - sample_normal.png: Clear lung parenchyma with vascular branching and no focal lesions.
Also creates sample class directories in `data/demo_samples/`.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_base_chest_ct(size=512):
    """Draw a realistic axial chest CT slice structure."""
    arr = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    cy, cx = size / 2, size / 2

    # Thoracic body contour (oval)
    body_mask = ((x - cx)**2 / (cx * 0.88)**2 + (y - cy)**2 / (cy * 0.78)**2) <= 1.0
    arr[body_mask] = 40.0  # soft tissue baseline HU equivalent

    # Subcutaneous fat layer (slightly darker)
    fat_mask = ((x - cx)**2 / (cx * 0.85)**2 + (y - cy)**2 / (cy * 0.75)**2) <= 1.0
    arr[fat_mask] = 55.0

    # Thoracic spine / vertebral body
    spine_mask = ((x - cx)**2 / 25**2 + (y - (cy + 130))**2 / 25**2) <= 1.0
    arr[spine_mask] = 220.0  # bone (hyperdense)

    # Ribs around the perimeter
    for angle in np.linspace(0, 2 * np.pi, 16):
        rx = cx + cx * 0.72 * np.cos(angle)
        ry = cy + cy * 0.62 * np.sin(angle)
        rib_mask = ((x - rx)**2 + (y - ry)**2) <= 14**2
        arr[rib_mask] = 210.0

    # Sternum (anterior)
    sternum_mask = ((x - cx)**2 / 30**2 + (y - (cy - 160))**2 / 12**2) <= 1.0
    arr[sternum_mask] = 200.0

    # Mediastinum (central heart and great vessels)
    med_mask = ((x - (cx - 15))**2 / 55**2 + (y - (cy + 10))**2 / 75**2) <= 1.0
    arr[med_mask] = 75.0

    # Left Lung cavity (dark, air-filled)
    left_lung = ((x - (cx - 100))**2 / 60**2 + (y - cy)**2 / 100**2) <= 1.0
    arr[left_lung] = 15.0

    # Right Lung cavity (dark, air-filled)
    right_lung = ((x - (cx + 100))**2 / 65**2 + (y - cy)**2 / 95**2) <= 1.0
    arr[right_lung] = 15.0

    # Bronchovascular tree (subtle branching lines in lungs)
    for _ in range(35):
        # random lung points
        lx = np.random.choice([cx - 100 + np.random.normal(0, 25), cx + 100 + np.random.normal(0, 25)])
        ly = cy + np.random.normal(0, 45)
        vessel = ((x - lx)**2 + (y - ly)**2) <= np.random.uniform(2, 6)**2
        arr[vessel] = np.random.uniform(50, 95)

    # Add realistic CT quantum mottle noise
    noise = np.random.normal(0, 3.5, (size, size))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def create_malignant_sample(output_path: str):
    """Malignant: Irregular, spiculated hyperdense lesion in right upper lobe."""
    img = create_base_chest_ct(512)
    draw = ImageDraw.Draw(img)
    # Right upper lobe nodule position: cx + 105, cy - 35
    cx, cy = 512 / 2 + 105, 512 / 2 - 35
    # Irregular spiculations
    spicules = []
    num_points = 24
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        radius = np.random.uniform(20, 36)
        # Spiculated projections
        if i % 3 == 0:
            radius += np.random.uniform(12, 22)
        px = cx + radius * np.cos(angle)
        py = cy + radius * np.sin(angle)
        spicules.append((px, py))

    draw.polygon(spicules, fill=(185, 185, 185), outline=(215, 215, 215))
    # Core density variation (necrosis/heterogeneity)
    draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(130, 130, 130))
    # Subtle blur
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(output_path)
    return output_path


def create_benign_sample(output_path: str):
    """Benign: Smooth, well-demarcated nodule (e.g. hamartoma / granuloma) with central calcification."""
    img = create_base_chest_ct(512)
    draw = ImageDraw.Draw(img)
    # Left lower lobe nodule: cx - 95, cy + 30
    cx, cy = 512 / 2 - 95, 512 / 2 + 30
    r = 16
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(140, 140, 140), outline=(160, 160, 160))
    # Central pop-corn calcification
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(230, 230, 230))
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    img.save(output_path)
    return output_path


def create_normal_sample(output_path: str):
    """Normal: Clean pulmonary parenchyma, bilateral aeration, no focal nodular lesions."""
    img = create_base_chest_ct(512)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(output_path)
    return output_path


def generate_all_demo_samples():
    demo_dir = "data/demo_samples"
    os.makedirs(f"{demo_dir}/malignant", exist_ok=True)
    os.makedirs(f"{demo_dir}/benign", exist_ok=True)
    os.makedirs(f"{demo_dir}/normal", exist_ok=True)
    os.makedirs("web/public/samples", exist_ok=True)

    print("Generating demo medical imaging slices...")
    create_malignant_sample(f"{demo_dir}/malignant/sample_malignant.png")
    create_malignant_sample("web/public/samples/sample_malignant.png")

    create_benign_sample(f"{demo_dir}/benign/sample_benign.png")
    create_benign_sample("web/public/samples/sample_benign.png")

    create_normal_sample(f"{demo_dir}/normal/sample_normal.png")
    create_normal_sample("web/public/samples/sample_normal.png")

    # Generate a few more for the training dataset folder if desired
    for i in range(1, 6):
        create_malignant_sample(f"{demo_dir}/malignant/case_{i:03d}.png")
        create_benign_sample(f"{demo_dir}/benign/case_{i:03d}.png")
        create_normal_sample(f"{demo_dir}/normal/case_{i:03d}.png")

    print("Demo CT assets created successfully.")


if __name__ == "__main__":
    generate_all_demo_samples()
