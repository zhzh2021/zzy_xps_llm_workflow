"""
Stage 06 - Dataset standardization and split.

Builds datasets from:
  - Feature JSONs (stage 03)
  - Normalized spectra CSVs (stage 02)

Outputs standardized train/val/test splits to:
  project_root/xas_results/06_deep_learning/datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xas_deep_learning.dl_data import (
    build_dataset_from_features,
    build_dataset_from_spectra,
    get_default_features_dir,
    get_default_spectra_dir,
    save_splits_npz,
    split_dataset,
    standardize_and_pca,
)
from xas_deep_learning.dl_utils import (
    get_output_dirs,
    load_deep_learning_config,
    save_json,
    save_config_snapshot,
    set_random_seed,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and standardize deep learning datasets.")
    parser.add_argument("--config", type=str, default=None, help="Path to xas_ml_settings.yaml (optional)")
    parser.add_argument("--source", type=str, choices=["features", "spectra"], default=None)
    parser.add_argument("--label-source", type=str, choices=["metadata", "feature", "file"], default=None)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--label-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Override datasets output dir")
    parser.add_argument("--no-standardize", action="store_true", help="Disable standardization")
    parser.add_argument("--apply-pca", action="store_true", help="Enable PCA (overrides config)")
    args = parser.parse_args()

    cfg = load_deep_learning_config(args.config)
    data_cfg = cfg.get("dataset", {})

    source = args.source or data_cfg.get("source", "features")
    label_source = args.label_source or data_cfg.get("label_source", "metadata")
    label_column = args.label_column or data_cfg.get("label_column", "ligand_type")
    label_file = Path(args.label_file) if args.label_file else None

    random_seed = int(data_cfg.get("random_seed", 42))
    test_size = float(data_cfg.get("test_size", 0.2))
    val_size = float(data_cfg.get("val_size", 0.1))
    stratify = bool(data_cfg.get("stratify", True))
    standardize = not args.no_standardize and bool(data_cfg.get("standardize", True))
    scaler_type = data_cfg.get("scaler_type", "standard").lower()
    apply_pca = bool(args.apply_pca or data_cfg.get("apply_pca", False))
    pca_variance = data_cfg.get("pca_variance", 0.99)
    pca_components = data_cfg.get("pca_components", None)

    output_dirs = get_output_dirs(Path(args.output_dir) if args.output_dir else None)
    setup_logging(output_dirs["logs"], level="INFO")
    set_random_seed(random_seed)

    if source == "features":
        features_dir = data_cfg.get("features_dir")
        features_dir = Path(features_dir) if features_dir else get_default_features_dir()
        X, y, sample_names, feature_names, meta = build_dataset_from_features(
            features_dir, label_source, label_column, label_file
        )
        extra_meta = {"feature_names": feature_names}
    else:
        spectra_dir = data_cfg.get("spectra_dir")
        spectra_dir = Path(spectra_dir) if spectra_dir else get_default_spectra_dir()
        grid = data_cfg.get("spectra_grid", {})
        X, y, sample_names, energy_grid, meta = build_dataset_from_spectra(
            spectra_dir,
            label_source,
            label_column,
            label_file,
            grid_min=float(grid.get("min_energy", 6900.0)),
            grid_max=float(grid.get("max_energy", 7400.0)),
            n_points=int(grid.get("n_points", 400)),
        )
        extra_meta = {"energy_grid": energy_grid.tolist()}

    splits = split_dataset(
        X,
        y,
        sample_names,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
        stratify=stratify,
    )
    splits = standardize_and_pca(
        splits,
        output_dirs["datasets"],
        standardize=standardize,
        scaler_type=scaler_type,
        apply_pca=apply_pca,
        pca_components=pca_components,
        pca_variance=pca_variance,
    )

    dataset_tag = f"{source}_{label_column}"
    splits_path = output_dirs["datasets"] / f"splits_{dataset_tag}.npz"
    save_splits_npz(splits, splits_path)

    summary = {
        "source": source,
        "label_source": label_source,
        "label_column": label_column,
        "n_samples": int(X.shape[0]),
        "n_features": int(splits["X_train"].shape[1]),
        "train_size": int(splits["X_train"].shape[0]),
        "val_size": int(splits["X_val"].shape[0]),
        "test_size": int(splits["X_test"].shape[0]),
        "standardize": standardize,
        "apply_pca": apply_pca,
        "splits_path": str(splits_path),
    }
    summary.update(extra_meta)
    save_json(summary, output_dirs["datasets"] / f"dataset_summary_{dataset_tag}.json")
    save_config_snapshot(cfg, output_dirs["datasets"] / f"config_snapshot_{dataset_tag}.json")

    # Save metadata and label distribution
    meta_df = pd.DataFrame(meta, index=sample_names)
    meta_df.to_csv(output_dirs["datasets"] / f"metadata_{dataset_tag}.csv")
    if label_source == "metadata" and label_column in meta_df.columns:
        counts = meta_df[label_column].value_counts(dropna=False).to_dict()
        save_json({"label_counts": counts}, output_dirs["datasets"] / f"label_distribution_{dataset_tag}.json")
    else:
        # Use y distribution for feature/file labels
        try:
            ser = pd.Series(y)
            counts = ser.value_counts(dropna=False).to_dict()
            save_json({"label_counts": counts}, output_dirs["datasets"] / f"label_distribution_{dataset_tag}.json")
        except Exception:
            pass

    print(f"Saved splits: {splits_path}")
    print(f"Summary: {output_dirs['datasets'] / f'dataset_summary_{dataset_tag}.json'}")


if __name__ == "__main__":
    main()
