# scripts/generate_dummy_image.py
"""Generate a random dummy CT slice (PNG) for quick demo.
The image will be saved to `data/images/dummy.png`.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image

out_dir = Path('data/images')
out_dir.mkdir(parents=True, exist_ok=True)

# Create a 224x224 grayscale image with random values, then repeat channels to RGB
arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
img = Image.fromarray(arr, mode='L').convert('RGB')
out_path = out_dir / 'dummy.png'
img.save(out_path)
print(f'Dummy image saved to {out_path}')
