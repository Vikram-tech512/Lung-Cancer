import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_image(input_path: str, output_path: str):
    """Resize, normalize and save as PNG for inference.
    This simple preprocessing mirrors the one used in the inference script.
    """
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(input_path).convert('RGB')
    tensor = tf(img)
    # Convert back to PIL for saving (denormalize)
    inv_tf = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                             std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
    ])
    pil = inv_tf(tensor)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw image (PNG/JPEG)")
    parser.add_argument("--output", required=True, help="Path to saved preprocessed PNG")
    args = parser.parse_args()
    preprocess_image(args.input, args.output)
