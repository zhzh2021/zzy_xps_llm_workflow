"""
Stage 06 - Random Forest Regression.
Trains a scikit-learn RF model on feature datasets.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xas_deep_learning.dl_data import find_latest_splits, load_splits_npz
from xas_deep_learning.dl_training import (
    evaluate_regression,
    plot_error_hist,
    plot_regression_results,
    plot_residuals,
)
from xas_deep_learning.dl_utils import (
    get_output_dirs,
    load_deep_learning_config,
    apply_plot_style,
    save_config_snapshot,
    save_json,
    set_random_seed,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF regression model.")
    parser.add_argument("--config", type=str, default=None, help="Path to xas_ml_settings.yaml (optional)")
    parser.add_argument("--splits", type=str, default=None, help="Path to splits_*.npz (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override base output dir")
    args = parser.parse_args()

    cfg = load_deep_learning_config(args.config)
    model_cfg = cfg.get("models", {}).get("rf_regression", {})
    output_dirs = get_output_dirs(Path(args.output_dir) if args.output_dir else None)
    setup_logging(output_dirs["logs"], level="INFO")
    set_random_seed(int(cfg.get("dataset", {}).get("random_seed", 42)))

    splits_path = Path(args.splits) if args.splits else find_latest_splits(output_dirs["datasets"])
    splits = load_splits_npz(splits_path)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rf_regression_{run_id}"
    model_dir = output_dirs["models"] / run_name
    plot_dir = output_dirs["plots"] / run_name
    report_dir = output_dirs["reports"] / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    base_model = RandomForestRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 200)),
        max_depth=model_cfg.get("max_depth", None),
        min_samples_split=int(model_cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
        max_features=model_cfg.get("max_features", "sqrt"),
        random_state=int(model_cfg.get("random_state", 42)),
        n_jobs=-1,
    )

    if bool(model_cfg.get("grid_search", False)):
        grid_params = model_cfg.get("grid_params", {})
        search = GridSearchCV(
            base_model,
            grid_params,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = base_model.fit(X_train, y_train)
        best_params = None

    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)

    plot_regression_results(y_test, y_pred, plot_dir / "prediction_scatter.png", "RF Regression")
    plot_residuals(y_test, y_pred, plot_dir / "residuals.png", "RF Regression Residuals")
    plot_error_hist(y_test, y_pred, plot_dir / "residual_hist.png", "RF Regression Residuals")

    np.savetxt(report_dir / "y_test.csv", y_test, delimiter=",")
    np.savetxt(report_dir / "y_pred.csv", y_pred, delimiter=",")
    save_json(metrics, report_dir / "metrics.json")
    save_config_snapshot(cfg, report_dir / "config_snapshot.json")
    if best_params:
        save_json(best_params, report_dir / "best_params.json")
    model_path = model_dir / "rf_model.pkl"
    joblib.dump(model, model_path)

    # Feature importance
    dataset_tag = splits_path.stem.replace("splits_", "")
    summary_path = output_dirs["datasets"] / f"dataset_summary_{dataset_tag}.json"
    feature_names = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            feature_names = summary.get("feature_names")
        except Exception:
            feature_names = None
    if feature_names:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        apply_plot_style()
        importances = model.feature_importances_
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
        df_imp.sort_values("importance", ascending=False, inplace=True)
        df_imp.to_csv(report_dir / "feature_importance.csv", index=False)

        top = df_imp.head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title("Top Feature Importances (RF)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(plot_dir / "feature_importance.png", bbox_inches="tight")
        plt.close()

    save_json(
        {
            "splits_path": str(splits_path),
            "model_path": str(model_path),
            "best_params": best_params,
            "model_dir": str(model_dir),
            "plot_dir": str(plot_dir),
            "report_dir": str(report_dir),
        },
        report_dir / "run_info.json",
    )

    print(f"Saved model: {model_path}")
    print(f"Metrics: {report_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
