# src/explainability/lime_explainer.py
"""LIME (Local Interpretable Model-Agnostic Explanations) wrapper for lung cancer classification.
Reference: Ribeiro et al., "Why Should I Trust You?", KDD 2016.
"""
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LIMEExplainer:
    """LIME explainer for image classification models."""

    def __init__(self, model, device="cpu"):
        if not LIME_AVAILABLE:
            raise ImportError("lime is not installed. Run: pip install lime")
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """Batch prediction function expected by LIME.
        images: (N, H, W, C) numpy array of uint8.
        Returns: (N, num_classes) probability array.
        """
        self.model.eval()
        batch = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            tensor = _preprocess(pil)
            batch.append(tensor)
        batch = torch.stack(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def explain(self, img: Image.Image, num_samples: int = 500, top_labels: int = 3):
        """Generate LIME explanation for a single image.

        Args:
            img: PIL Image (RGB).
            num_samples: Number of perturbed samples.
            top_labels: Number of top labels to explain.

        Returns:
            lime explanation object.
        """
        img_arr = np.array(img.resize((224, 224)))
        explanation = self.explainer.explain_instance(
            img_arr,
            self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
        )
        return explanation

    def save_explanation(self, explanation, label: int, path: str, positive_only: bool = True):
        """Save LIME explanation as an image."""
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=positive_only,
            num_features=10,
            hide_rest=False,
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(temp)
        axes[0].set_title("LIME Explanation")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Feature Mask")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
