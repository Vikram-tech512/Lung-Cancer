# src/explainability/shap_explainer.py
"""SHAP (SHapley Additive exPlanations) wrapper for lung cancer classification.
Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
"""
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SHAPExplainer:
    """SHAP DeepExplainer / GradientExplainer for image models."""

    def __init__(self, model, background_images: torch.Tensor = None, device="cpu"):
        """
        Args:
            model: PyTorch classification model.
            background_images: (N, C, H, W) tensor of background samples.
            device: "cpu" or "cuda".
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap is not installed. Run: pip install shap")
        self.model = model.eval().to(device)
        self.device = device
        if background_images is None:
            background_images = torch.randn(10, 3, 224, 224).to(device)
        self.explainer = shap.GradientExplainer(model, background_images)

    def explain(self, img: Image.Image) -> np.ndarray:
        """Generate SHAP values for a single image.

        Returns:
            shap_values: list of arrays, one per class. Each array: (1, C, H, W).
        """
        tensor = _preprocess(img.resize((224, 224))).unsqueeze(0).to(self.device)
        shap_values = self.explainer.shap_values(tensor)
        return shap_values

    def save_explanation(self, img: Image.Image, shap_values, class_idx: int, path: str):
        """Save SHAP summary figure."""
        # shap_values[class_idx]: (1, C, H, W)
        sv = shap_values[class_idx][0]  # (C, H, W)
        # Average across channels
        sv_mean = np.mean(np.abs(sv), axis=0)  # (H, W)
        if sv_mean.max() > 0:
            sv_mean = sv_mean / sv_mean.max()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img.resize((224, 224)))
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(sv_mean, cmap="hot")
        axes[1].set_title(f"SHAP (class {class_idx})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
