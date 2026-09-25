# scripts/run_robustness.py
"""Run the full robustness analysis across all perturbations for a trained model.
Outputs a JSON report and a bar chart comparing accuracy under each perturbation.
"""
import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.evaluate import get_model
from src.robustness.analyzer import robustness_analysis
from src.robustness.perturbations import get_all_perturbation_names


def main():
    parser = argparse.ArgumentParser(description="Robustness analysis")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", choices=["cnn", "vit", "hybrid"], default="hybrid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="results/robustness")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.dataset.lung_dataset import LungCancerDataset
    ds_temp = LungCancerDataset(args.data_dir)
    num_classes = len(set(label for _, label in ds_temp.samples))

    model = get_model(args.arch, num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{args.arch}_robustness.json")

    results = robustness_analysis(
        model,
        img_dir=args.data_dir,
        label_file=None,
        batch_size=args.batch_size,
        device=str(device),
        output_path=report_path,
    )

    # Bar chart
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    colors = ["#34d399" if n == "original" else ("#fb7185" if results[n]["degradation"] > 0.1 else "#6366f1") for n in names]

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(range(len(names)), accs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Robustness Analysis – {args.arch.upper()}")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    chart_path = os.path.join(args.output_dir, f"{args.arch}_robustness_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nRobustness report saved to: {report_path}")
    print(f"Bar chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
