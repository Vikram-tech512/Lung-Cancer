# scripts/create_dummy_checkpoint.py
"""Create a dummy checkpoint for the ViTSeg model.
This is useful for a quick end‑to‑end demo when no trained weights are available.
The script instantiates the model, initializes random weights, and saves them to
`checkpoints/vit_lung_seg.pt`.
"""
import os
from pathlib import Path
import torch

# Ensure the checkpoint directory exists
ckpt_dir = Path('checkpoints')
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Import the model (relative import works when run from repo root)
from src.model.vit_seg import ViTSeg

model = ViTSeg()
# Optionally, you could load a pretrained backbone from timm here.
# For now we keep random initialization.
ckpt_path = ckpt_dir / 'vit_lung_seg.pt'
torch.save(model.state_dict(), ckpt_path)
print(f'Dummy checkpoint saved to {ckpt_path}')
