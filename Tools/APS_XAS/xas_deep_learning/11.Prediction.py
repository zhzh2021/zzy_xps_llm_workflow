"""
Stage 06 - Prediction.
Loads a trained model and runs predictions on the test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xas_deep_learning.dl_data import (
    find_latest_splits,
    get_default_features_dir,
    get_default_spectra_dir,
    load_features_only,
    load_spectra_only,
    load_splits_npz,
)
from xas_deep_learning.dl_utils import get_output_dirs, load_deep_learning_config, setup_logging


def _load_model(model_path: Path):
    if model_path.suffix.lower() in {".h5", ".keras"}:
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    if model_path.suffix.lower() in {".pkl", ".joblib"}:
        return joblib.load(model_path)
    raise ValueError(f"Unsupported model type: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.h5 or .pkl)")
    parser.add_argument("--splits", type=str, default=None, help="Path to splits_*.npz (optional)")
    parser.add_argument("--features-dir", type=str, default=None, help="Feature JSONs directory (optional)")
    parser.add_argument("--spectra-dir", type=str, default=None, help="Normalized spectra CSVs directory (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override base output dir")
    parser.add_argument("--output", type=str, default=None, help="Output predictions CSV (optional)")
    parser.add_argument("--config", type=str, default=None, help="Path to xas_ml_settings.yaml (optional)")
    args = parser.parse_args()

    output_dirs = get_output_dirs(Path(args.output_dir) if args.output_dir else None)
    setup_logging(output_dirs["logs"], level="INFO")

    cfg = load_deep_learning_config(args.config)
    data_cfg = cfg.get("dataset", {})
    if args.features_dir or args.spectra_dir:
        if args.features_dir:
            features_dir = Path(args.features_dir)
        else:
            features_dir = None
        if args.spectra_dir:
            spectra_dir = Path(args.spectra_dir)
        else:
            spectra_dir = None
        if features_dir:
            X_test, names, _ = load_features_only(features_dir)
        else:
            grid = data_cfg.get("spectra_grid", {})
            spectra_dir = spectra_dir if spectra_dir else get_default_spectra_dir()
            X_test, names, _ = load_spectra_only(
                spectra_dir,
                grid_min=float(grid.get("min_energy", 6900.0)),
                grid_max=float(grid.get("max_energy", 7400.0)),
                n_points=int(grid.get("n_points", 400)),
            )
        y_test = None
        # Apply saved scaler/PCA if available
        scaler_path = output_dirs["datasets"] / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(X_test)
        pca_path = output_dirs["datasets"] / "pca.pkl"
        if pca_path.exists():
            pca = joblib.load(pca_path)
            X_test = pca.transform(X_test)
    else:
        splits_path = Path(args.splits) if args.splits else find_latest_splits(output_dirs["datasets"])
        splits = load_splits_npz(splits_path)
        X_test = splits["X_test"]
        y_test = splits["y_test"]
        names = splits.get("names_test")

    model = _load_model(Path(args.model))
    if hasattr(model, "predict"):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Loaded model does not support predict()")

    y_pred = np.array(y_pred).reshape(-1)
    output_path = Path(args.output) if args.output else output_dirs["reports"] / "predictions.csv"

    np.savetxt(output_path, y_pred, delimiter=",")
    if y_test is not None:
        np.savetxt(output_dirs["reports"] / "y_test.csv", y_test, delimiter=",")
    if names:
        with open(output_dirs["reports"] / "prediction_names.txt", "w", encoding="utf-8") as f:
            f.write("\n".join([str(n) for n in names]))

    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
