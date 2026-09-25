# src/explainability/gradcam.py
"""Grad-CAM and Grad-CAM++ implementations for CNN and Hybrid models.
Reference:
  - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017.
  - Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual Explanations," WACV 2018.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Standard Grad-CAM for any model with a target convolutional layer."""

    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network.
            target_layer: The nn.Module whose output feature maps to use.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input image tensor.
            target_class: Class index. If None, uses the predicted class.

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


class GradCAMPP(GradCAM):
    """Grad-CAM++ – improved localization with second-order gradients."""

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        grads = self.gradients          # (1, C, h, w)
        acts = self.activations         # (1, C, h, w)

        # Grad-CAM++ weights: alpha = grad^2 / (2*grad^2 + sum(act * grad^3) + eps)
        grads_sq = grads ** 2
        grads_cb = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        alpha = grads_sq / (2 * grads_sq + sum_acts * grads_cb + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4,
                    colormap: str = "jet") -> Image.Image:
    """Overlay a heatmap on the original image.

    Args:
        img: Original PIL Image (RGB).
        heatmap: 2D numpy array in [0, 1].
        alpha: Overlay transparency.
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image with heatmap overlay.
    """
    # Resize heatmap to image size
    h, w = img.size[1], img.size[0]
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h)))
    heatmap_resized = heatmap_resized.astype(np.float32) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_resized)[:, :, :3]  # drop alpha channel
    colored = (colored * 255).astype(np.uint8)

    # Blend
    img_arr = np.array(img).astype(np.float32)
    overlay = img_arr * (1 - alpha) + colored.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_gradcam_figure(img: Image.Image, heatmap: np.ndarray, title: str, path: str):
    """Save a side-by-side figure of original + heatmap overlay."""
    overlay = overlay_heatmap(img, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
