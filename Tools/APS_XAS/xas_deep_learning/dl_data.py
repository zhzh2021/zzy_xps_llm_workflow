"""
Dataset utilities for XAS deep learning.
Builds datasets from feature JSONs or normalized spectra CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .dl_utils import get_project_root


def parse_metadata_from_name(sample_name: str) -> Dict[str, Any]:
    """Parse metadata from sample name. Mirrors ML stage parsing."""
    metadata: Dict[str, Any] = {}

    # Iron source (anion)
    if "FeCl2" in sample_name:
        metadata["iron_source"] = "FeCl2"
        metadata["anion_type"] = 0
    elif "FeSO4" in sample_name:
        metadata["iron_source"] = "FeSO4"
        metadata["anion_type"] = 1
    elif "FeAcetate" in sample_name:
        metadata["iron_source"] = "FeAcetate"
        metadata["anion_type"] = 2
    elif "FeTFSI" in sample_name:
        metadata["iron_source"] = "FeTFSI"
        metadata["anion_type"] = 3
    else:
        metadata["iron_source"] = "Unknown"
        metadata["anion_type"] = -1

    # Ligand
    if "Malic_acid" in sample_name:
        metadata["ligand"] = "Malic_acid"
        metadata["ligand_type"] = 0
    elif "Tartaric_acid" in sample_name:
        metadata["ligand"] = "Tartaric_acid"
        metadata["ligand_type"] = 1
    elif "H2O" in sample_name:
        metadata["ligand"] = "H2O"
        metadata["ligand_type"] = 2
    else:
        metadata["ligand"] = "Unknown"
        metadata["ligand_type"] = -1

    # pH
    ph_match = None
    for part in sample_name.split("_"):
        if part.startswith("pH"):
            ph_match = part
            break
    if ph_match:
        try:
            metadata["pH"] = float(ph_match.replace("pH", ""))
        except ValueError:
            metadata["pH"] = None
    else:
        metadata["pH"] = None

    # State (gel/solution)
    if "gel" in sample_name.lower():
        metadata["state"] = "gel"
        metadata["state_type"] = 1
    else:
        metadata["state"] = "solution"
        metadata["state_type"] = 0

    # Replicate
    replicate = 1
    if "_R" in sample_name:
        try:
            replicate = int(sample_name.split("_R")[-1].split("_")[0])
        except Exception:
            replicate = 1
    metadata["replicate"] = replicate

    # Concentrations inside parentheses (0_1-0_05)
    metadata["anion_conc"] = None
    metadata["ligand_conc"] = None
    metadata["conc_ratio"] = None
    if "(" in sample_name and ")" in sample_name:
        try:
            inside = sample_name.split("(")[-1].split(")")[0]
            if "-" in inside:
                a_str, l_str = inside.split("-", 1)
                a_str = a_str.replace("_", ".")
                l_str = l_str.replace("_", ".")
                metadata["anion_conc"] = float(a_str.replace("M", ""))
                metadata["ligand_conc"] = float(l_str.replace("M", ""))
                if metadata["ligand_conc"] and metadata["ligand_conc"] > 0:
                    metadata["conc_ratio"] = metadata["anion_conc"] / metadata["ligand_conc"]
        except Exception:
            pass

    return metadata


def _load_label_file(label_file: Path) -> Dict[str, Any]:
    df = pd.read_csv(label_file)
    if "sample_name" not in df.columns:
        raise ValueError("Label file must include 'sample_name' column")
    label_col = "label"
    for col in df.columns:
        if col != "sample_name":
            label_col = col
            break
    return dict(zip(df["sample_name"].astype(str), df[label_col]))


def _build_metadata_dict(sample_names: List[str]) -> Dict[str, List[Any]]:
    meta: Dict[str, List[Any]] = {}
    for name in sample_names:
        row = parse_metadata_from_name(name)
        for key, value in row.items():
            meta.setdefault(key, []).append(value)
    return meta


def load_feature_jsons(features_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    sample_names: List[str] = []
    for path in sorted(features_dir.glob("*_features.json")):
        if path.name == "batch_features_summary.json":
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sample_name = data.get("sample_name", path.stem.replace("_features", ""))
        sample_names.append(sample_name)
        rows.append(data)
    if not rows:
        raise FileNotFoundError(f"No feature JSONs found in: {features_dir}")
    df = pd.DataFrame(rows)
    return df, sample_names


def build_dataset_from_features(
    features_dir: Path,
    label_source: str,
    label_column: str,
    label_file: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, List[Any]]]:
    df, sample_names = load_feature_jsons(features_dir)
    meta = _build_metadata_dict(sample_names)

    # Remove non-feature columns
    drop_cols = {"sample_name"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].to_numpy(dtype=float)

    if label_source == "feature":
        if label_column not in df.columns:
            raise ValueError(f"Label column not found in features: {label_column}")
        y = df[label_column].to_numpy()
    elif label_source == "metadata":
        if label_column not in meta:
            raise ValueError(f"Label column not found in metadata: {label_column}")
        y = np.array(meta[label_column])
    elif label_source == "file":
        if label_file is None:
            raise ValueError("label_file is required when label_source='file'")
        mapping = _load_label_file(label_file)
        y = np.array([mapping.get(n) for n in sample_names])
    else:
        raise ValueError(f"Unknown label_source: {label_source}")

    # Drop rows with missing labels
    valid = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y])
    X = X[valid]
    y = y[valid]
    sample_names = [n for n, ok in zip(sample_names, valid) if ok]

    return X, y, sample_names, feature_cols, meta


def load_features_only(features_dir: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    df, sample_names = load_feature_jsons(features_dir)
    drop_cols = {"sample_name"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].to_numpy(dtype=float)
    return X, sample_names, feature_cols


def load_spectra_only(
    spectra_dir: Path, grid_min: float, grid_max: float, n_points: int
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    samples = []
    names: List[str] = []
    energy_grid = np.linspace(grid_min, grid_max, n_points)
    for path in sorted(spectra_dir.glob("*_analyzed.csv")):
        df = pd.read_csv(path)
        if "energy" not in df.columns or "mu_normalized" not in df.columns:
            continue
        energy = df["energy"].to_numpy(dtype=float)
        mu = df["mu_normalized"].to_numpy(dtype=float)
        mu_interp = np.interp(energy_grid, energy, mu)
        samples.append(mu_interp)
        name = path.stem.replace("_analyzed", "")
        names.append(name)
    if not samples:
        raise FileNotFoundError(f"No analyzed spectra found in: {spectra_dir}")
    X = np.vstack(samples)
    return X, names, energy_grid


def build_dataset_from_spectra(
    spectra_dir: Path,
    label_source: str,
    label_column: str,
    label_file: Optional[Path],
    grid_min: float,
    grid_max: float,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Dict[str, List[Any]]]:
    samples = []
    names: List[str] = []

    energy_grid = np.linspace(grid_min, grid_max, n_points)
    for path in sorted(spectra_dir.glob("*_analyzed.csv")):
        df = pd.read_csv(path)
        if "energy" not in df.columns or "mu_normalized" not in df.columns:
            continue
        energy = df["energy"].to_numpy(dtype=float)
        mu = df["mu_normalized"].to_numpy(dtype=float)
        mu_interp = np.interp(energy_grid, energy, mu)
        samples.append(mu_interp)
        name = path.stem.replace("_analyzed", "")
        names.append(name)

    if not samples:
        raise FileNotFoundError(f"No analyzed spectra found in: {spectra_dir}")

    X = np.vstack(samples)
    meta = _build_metadata_dict(names)

    if label_source == "metadata":
        if label_column not in meta:
            raise ValueError(f"Label column not found in metadata: {label_column}")
        y = np.array(meta[label_column])
    elif label_source == "file":
        if label_file is None:
            raise ValueError("label_file is required when label_source='file'")
        mapping = _load_label_file(label_file)
        y = np.array([mapping.get(n) for n in names])
    elif label_source == "feature":
        raise ValueError("label_source='feature' is not supported for spectra datasets")
    else:
        raise ValueError(f"Unknown label_source: {label_source}")

    valid = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y])
    X = X[valid]
    y = y[valid]
    names = [n for n, ok in zip(names, valid) if ok]

    return X, y, names, energy_grid, meta


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    sample_names: List[str],
    test_size: float,
    val_size: float,
    random_seed: int,
    stratify: bool,
) -> Dict[str, Any]:
    stratify_labels = None
    if stratify:
        try:
            unique_vals = np.unique(y)
            if len(unique_vals) <= max(2, min(20, len(y) // 2)):
                stratify_labels = y
        except Exception:
            stratify_labels = None
    X_train, X_tmp, y_train, y_tmp, names_train, names_tmp = train_test_split(
        X, y, sample_names, test_size=test_size, random_state=random_seed, stratify=stratify_labels
    )
    val_ratio = val_size / max(1e-6, (1.0 - test_size))
    stratify_tmp = y_tmp if stratify else None
    X_val, X_test, y_val, y_test, names_val, names_test = train_test_split(
        X_tmp, y_tmp, names_tmp, test_size=val_ratio, random_state=random_seed, stratify=stratify_tmp
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "names_train": names_train,
        "names_val": names_val,
        "names_test": names_test,
    }


def standardize_and_pca(
    splits: Dict[str, Any],
    output_dir: Path,
    standardize: bool,
    scaler_type: str,
    apply_pca: bool,
    pca_components: Optional[int],
    pca_variance: Optional[float],
) -> Dict[str, Any]:
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]

    scaler = None
    if standardize:
        if scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, output_dir / "scaler.pkl")

    if apply_pca:
        if pca_components is None and pca_variance is None:
            pca_components = min(20, X_train.shape[1])
        if pca_variance is not None:
            pca = PCA(n_components=pca_variance)
        else:
            pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        joblib.dump(pca, output_dir / "pca.pkl")
    else:
        pca = None

    splits = dict(splits)
    splits.update({"X_train": X_train, "X_val": X_val, "X_test": X_test})
    return splits


def save_splits_npz(splits: Dict[str, Any], output_path: Path) -> None:
    np.savez_compressed(
        output_path,
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        names_train=np.array(splits["names_train"]),
        names_val=np.array(splits["names_val"]),
        names_test=np.array(splits["names_test"]),
    )


def load_splits_npz(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "names_train": list(data["names_train"]),
        "names_val": list(data["names_val"]),
        "names_test": list(data["names_test"]),
    }


def get_default_features_dir() -> Path:
    return get_project_root() / "xas_results" / "03_feature_extraction" / "extracted_features"


def get_default_spectra_dir() -> Path:
    return get_project_root() / "xas_results" / "02_analyzed_data" / "normalized_data"


def find_latest_splits(datasets_dir: Path, pattern: str = "splits_*.npz") -> Path:
    candidates = list(datasets_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No split files found in: {datasets_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
