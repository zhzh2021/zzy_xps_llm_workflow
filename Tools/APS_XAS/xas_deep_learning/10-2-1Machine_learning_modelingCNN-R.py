"""
Stage 06 - CNN Regression.
Trains a 1D CNN on spectra datasets.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xas_deep_learning.dl_data import find_latest_splits, load_splits_npz
from xas_deep_learning.dl_models import build_cnn
from xas_deep_learning.dl_training import (
    compile_model,
    evaluate_regression,
    plot_error_hist,
    plot_regression_results,
    plot_residuals,
    plot_training_history,
    train_keras_model,
)
from xas_deep_learning.dl_utils import (
    get_output_dirs,
    load_deep_learning_config,
    save_config_snapshot,
    save_json,
    set_random_seed,
    setup_logging,
)


def _add_channel_dim(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return X[:, :, None]
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN regression model.")
    parser.add_argument("--config", type=str, default=None, help="Path to xas_ml_settings.yaml (optional)")
    parser.add_argument("--splits", type=str, default=None, help="Path to splits_*.npz (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override base output dir")
    args = parser.parse_args()

    cfg = load_deep_learning_config(args.config)
    model_cfg = cfg.get("models", {}).get("cnn_regression", {})
    output_dirs = get_output_dirs(Path(args.output_dir) if args.output_dir else None)
    setup_logging(output_dirs["logs"], level="INFO")
    set_random_seed(int(cfg.get("dataset", {}).get("random_seed", 42)))

    splits_path = Path(args.splits) if args.splits else find_latest_splits(output_dirs["datasets"])
    splits = load_splits_npz(splits_path)
    if "spectra" not in splits_path.name:
        print(f"[WARN] CNN typically expects spectra-based splits. Using: {splits_path.name}")
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    X_train = _add_channel_dim(X_train)
    X_val = _add_channel_dim(X_val)
    X_test = _add_channel_dim(X_test)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"cnn_regression_{run_id}"
    model_dir = output_dirs["models"] / run_name
    plot_dir = output_dirs["plots"] / run_name
    report_dir = output_dirs["reports"] / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    filters = model_cfg.get("filters", [32, 64])
    kernel_size = int(model_cfg.get("kernel_size", 5))
    pool_size = int(model_cfg.get("pool_size", 2))
    dense_units = model_cfg.get("dense_units", [128, 64])
    dropout = float(model_cfg.get("dropout", 0.0))
    use_gap = bool(model_cfg.get("use_global_avg_pool", False))
    dilations = model_cfg.get("dilations", None)
    learning_rate = float(model_cfg.get("learning_rate", 0.001))
    batch_size = int(model_cfg.get("batch_size", 32))
    epochs = int(model_cfg.get("epochs", 100))
    patience = int(model_cfg.get("patience", 12))

    model = build_cnn(
        input_length=X_train.shape[1],
        task_type="regression",
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dense_units=dense_units,
        dropout=dropout,
        use_global_avg_pool=use_gap,
        dilations=dilations,
    )
    target_scale = bool(cfg.get("dataset", {}).get("target_standardize", False))
    if target_scale:
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
        import joblib
        joblib.dump(y_scaler, model_dir / "target_scaler.pkl")
    else:
        y_scaler = None

    model = compile_model(model, "regression", learning_rate)
    history, best_model_path = train_keras_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        model_dir,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    y_pred = model.predict(X_test).reshape(-1)
    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
        y_test_eval = splits["y_test"]
    else:
        y_test_eval = y_test
    metrics = evaluate_regression(y_test_eval, y_pred)

    plot_training_history(history, plot_dir / "training_loss.png", "CNN Regression Loss")
    plot_regression_results(y_test_eval, y_pred, plot_dir / "prediction_scatter.png", "CNN Regression")
    plot_residuals(y_test_eval, y_pred, plot_dir / "residuals.png", "CNN Regression Residuals")
    plot_error_hist(y_test_eval, y_pred, plot_dir / "residual_hist.png", "CNN Regression Residuals")

    np.savetxt(report_dir / "y_test.csv", y_test_eval, delimiter=",")
    np.savetxt(report_dir / "y_pred.csv", y_pred, delimiter=",")
    save_json(metrics, report_dir / "metrics.json")
    save_config_snapshot(cfg, report_dir / "config_snapshot.json")
    save_json(
        {
            "splits_path": str(splits_path),
            "best_model": str(best_model_path),
            "model_dir": str(model_dir),
            "plot_dir": str(plot_dir),
            "report_dir": str(report_dir),
        },
        report_dir / "run_info.json",
    )

    print(f"Saved model: {best_model_path}")
    print(f"Metrics: {report_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
