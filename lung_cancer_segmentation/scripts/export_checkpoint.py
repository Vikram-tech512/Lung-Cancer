# scripts/export_checkpoint.py
"""Export a trained ViTSeg checkpoint to ONNX or TorchScript for deployment.
Usage:
    python scripts/export_checkpoint.py --ckpt checkpoints/vit_lung_seg.pt --format onnx
    python scripts/export_checkpoint.py --ckpt checkpoints/vit_lung_seg.pt --format torchscript
"""
import argparse
import torch
from src.model.vit_seg import ViTSeg


def export_onnx(model, out_path: str):
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["mask"],
        opset_version=16,
        dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
    )
    print(f"ONNX model saved to {out_path}")


def export_torchscript(model, out_path: str):
    dummy = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, dummy)
    traced.save(out_path)
    print(f"TorchScript model saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export ViTSeg checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--out", default=None, help="Output file path")
    args = parser.parse_args()

    model = ViTSeg()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    ext = "onnx" if args.format == "onnx" else "pt"
    out_path = args.out or f"model_exported.{ext}"

    if args.format == "onnx":
        export_onnx(model, out_path)
    else:
        export_torchscript(model, out_path)


if __name__ == "__main__":
    main()
