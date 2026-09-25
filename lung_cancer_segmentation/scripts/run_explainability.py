# scripts/run_explainability.py
"""Generate explainability visualizations for a trained model.
Produces Grad-CAM, Grad-CAM++, and Attention Map outputs for sample images,
both on original and perturbed versions (for explanation consistency analysis).
"""
import argparse
import os
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from scripts.evaluate import get_model
from src.explainability.gradcam import GradCAM, GradCAMPP, save_gradcam_figure
from src.explainability.attention_viz import extract_attention_map, attention_rollout, save_attention_figure
from src.robustness.perturbations import apply_perturbation


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_target_layer(model, arch):
    """Get the appropriate target layer for Grad-CAM based on architecture."""
    if arch == "cnn":
        # Last conv block in the features sequential
        return model.features[-1]
    elif arch == "hybrid":
        # CNN backbone's last conv layer
        return model.cnn_backbone.conv_layers[-1]
    else:
        return None  # ViT uses attention maps instead


def main():
    parser = argparse.ArgumentParser(description="Generate XAI explanations")
    parser.add_argument("--image", required=True, help="Path to a test image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", choices=["cnn", "vit", "hybrid"], default="hybrid")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--output-dir", default="results/explanations")
    parser.add_argument("--perturbations", nargs="*", default=["gaussian_noise_med", "blur_med", "contrast_low"])
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = get_model(args.arch, args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    img_resized = img.resize((224, 224))

    # ── Grad-CAM / Grad-CAM++ (for CNN and Hybrid) ──
    target_layer = get_target_layer(model, args.arch)
    if target_layer is not None:
        # Original image
        tensor = _preprocess(img_resized).unsqueeze(0).to(device)

        gcam = GradCAM(model, target_layer)
        heatmap = gcam.generate(tensor)
        save_gradcam_figure(img_resized, heatmap, f"Grad-CAM ({args.arch}) – Original",
                            os.path.join(args.output_dir, "gradcam_original.png"))

        # Re-instantiate for Grad-CAM++ (hooks need fresh registration)
        model2 = get_model(args.arch, args.num_classes).to(device)
        model2.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model2.eval()
        target_layer2 = get_target_layer(model2, args.arch)
        gcampp = GradCAMPP(model2, target_layer2)
        heatmap_pp = gcampp.generate(tensor)
        save_gradcam_figure(img_resized, heatmap_pp, f"Grad-CAM++ ({args.arch}) – Original",
                            os.path.join(args.output_dir, "gradcampp_original.png"))

        # Perturbed versions
        for pname in args.perturbations:
            perturbed = apply_perturbation(img_resized, pname)
            tensor_p = _preprocess(perturbed).unsqueeze(0).to(device)

            model3 = get_model(args.arch, args.num_classes).to(device)
            model3.load_state_dict(torch.load(args.checkpoint, map_location=device))
            model3.eval()
            tl3 = get_target_layer(model3, args.arch)
            gcam_p = GradCAM(model3, tl3)
            hm_p = gcam_p.generate(tensor_p)
            save_gradcam_figure(perturbed, hm_p, f"Grad-CAM ({args.arch}) – {pname}",
                                os.path.join(args.output_dir, f"gradcam_{pname}.png"))
        print(f"Grad-CAM outputs saved to {args.output_dir}/")

    # ── Attention Maps (for ViT and Hybrid) ──
    if args.arch in ("vit", "hybrid"):
        tensor = _preprocess(img_resized).unsqueeze(0).to(device)

        # Last layer attention
        attn_map = extract_attention_map(model, tensor, layer_idx=-1)
        save_attention_figure(img_resized, attn_map, f"Attention Map ({args.arch}) – Original",
                              os.path.join(args.output_dir, "attention_original.png"))

        # Attention rollout
        rollout = attention_rollout(model, tensor)
        save_attention_figure(img_resized, rollout, f"Attention Rollout ({args.arch}) – Original",
                              os.path.join(args.output_dir, "attention_rollout_original.png"))

        # Perturbed
        for pname in args.perturbations:
            perturbed = apply_perturbation(img_resized, pname)
            tensor_p = _preprocess(perturbed).unsqueeze(0).to(device)
            attn_p = extract_attention_map(model, tensor_p, layer_idx=-1)
            save_attention_figure(perturbed, attn_p, f"Attention ({args.arch}) – {pname}",
                                  os.path.join(args.output_dir, f"attention_{pname}.png"))
        print(f"Attention map outputs saved to {args.output_dir}/")

    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()
