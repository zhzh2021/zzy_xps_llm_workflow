"""
Stage 06 - Model performance evaluation.
Evaluates predictions and generates plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xas_deep_learning.dl_training import (
    evaluate_classification,
    evaluate_regression,
    plot_confusion_matrix,
    plot_regression_results,
    plot_residuals,
    plot_error_hist,
)
from xas_deep_learning.dl_utils import get_output_dirs, save_json, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--task", type=str, choices=["regression", "classification"], required=True)
    parser.add_argument("--y-true", type=str, required=True, help="CSV of true labels")
    parser.add_argument("--y-pred", type=str, required=True, help="CSV of predictions")
    parser.add_argument("--output-dir", type=str, default=None, help="Override base output dir")
    args = parser.parse_args()

    output_dirs = get_output_dirs(Path(args.output_dir) if args.output_dir else None)
    setup_logging(output_dirs["logs"], level="INFO")

    y_true = np.loadtxt(args.y_true, delimiter=",")
    y_pred = np.loadtxt(args.y_pred, delimiter=",")

    if args.task == "regression":
        metrics = evaluate_regression(y_true, y_pred)
        plot_regression_results(y_true, y_pred, output_dirs["plots"] / "evaluation_scatter.png", "Model Evaluation")
        plot_residuals(y_true, y_pred, output_dirs["plots"] / "evaluation_residuals.png", "Residuals")
        plot_error_hist(y_true, y_pred, output_dirs["plots"] / "evaluation_residual_hist.png", "Residual Histogram")
    else:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        metrics = evaluate_classification(y_true, y_pred)
        plot_confusion_matrix(y_true, y_pred, output_dirs["plots"] / "evaluation_confusion_matrix.png", "Model Evaluation")

    save_json(metrics, output_dirs["reports"] / "evaluation_metrics.json")
    print(f"Saved metrics: {output_dirs['reports'] / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
