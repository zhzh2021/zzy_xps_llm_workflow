"""
Stage 06 - CNN Classification.
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
from sklearn.preprocessing import LabelEncoder

from xas_deep_learning.dl_data import find_latest_splits, load_splits_npz
from xas_deep_learning.dl_models import build_cnn
from xas_deep_learning.dl_training import (
    compile_model,
    evaluate_classification,
    plot_confusion_matrix,
    plot_training_history,
    get_focal_loss,
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
    parser = argparse.ArgumentParser(description="Train CNN classification model.")
    parser.add_argument("--config", type=str, default=None, help="Path to xas_ml_settings.yaml (optional)")
    parser.add_argument("--splits", type=str, default=None, help="Path to splits_*.npz (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override base output dir")
    args = parser.parse_args()

    cfg = load_deep_learning_config(args.config)
    model_cfg = cfg.get("models", {}).get("cnn_classification", {})
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

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)
    n_classes = int(len(encoder.classes_))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"cnn_classification_{run_id}"
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
        task_type="classification",
        n_classes=n_classes,
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dense_units=dense_units,
        dropout=dropout,
        use_global_avg_pool=use_gap,
        dilations=dilations,
    )
    use_focal = bool(model_cfg.get("use_focal_loss", False))
    if use_focal:
        loss_fn = get_focal_loss(
            gamma=float(model_cfg.get("focal_gamma", 2.0)),
            alpha=float(model_cfg.get("focal_alpha", 0.25)),
        )
    else:
        loss_fn = None
    model = compile_model(model, "classification", learning_rate, loss_override=loss_fn)
    class_weight = None
    if bool(model_cfg.get("use_class_weights", True)):
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_enc)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_enc)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    history, best_model_path = train_keras_model(
        model,
        X_train,
        y_train_enc,
        X_val,
        y_val_enc,
        model_dir,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        class_weight=class_weight,
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    metrics = evaluate_classification(y_test_enc, y_pred)

    plot_training_history(history, plot_dir / "training_loss.png", "CNN Classification Loss")
    plot_confusion_matrix(y_test_enc, y_pred, plot_dir / "confusion_matrix.png", "CNN Classification")

    np.savetxt(report_dir / "y_test.csv", y_test_enc, delimiter=",")
    np.savetxt(report_dir / "y_pred.csv", y_pred, delimiter=",")
    save_json(metrics, report_dir / "metrics.json")
    save_config_snapshot(cfg, report_dir / "config_snapshot.json")
    save_json(
        {
            "splits_path": str(splits_path),
            "best_model": str(best_model_path),
            "label_classes": encoder.classes_.tolist(),
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
