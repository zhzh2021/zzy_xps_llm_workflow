"""
ML Correlator: Link XPS-derived SEI composition to electrochemical performance
Author: Argo (Argonne National Laboratory)

Enhancements:
- Robust path handling: directories and glob patterns for XPS files
- Excel support for electrochemical data (.xlsx/.xls)
- Region-aware feature names (e.g., F1s_LiF_perc) to avoid collisions
- Auto-fallback between Area_percent and Component_atomic_percent
"""

import os
import json
import glob
import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from joblib import dump


# ========== XPS WORKFLOW INTEGRATION ==========
try:
    import sys
    from pathlib import Path
    
    # Add Tools directory to path for import
    tools_dir = Path(__file__).resolve().parents[1]
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    from xps_workflow_manager import get_workflow_config, update_module_paths
    WORKFLOW_MANAGER_AVAILABLE = True
except ImportError:
    WORKFLOW_MANAGER_AVAILABLE = False
    print("⚠️  Workflow manager not available, using legacy paths")

def resolve_correlation_paths(config: XPSConfig):
    """
    Resolve correlation paths using workflow manager for unified structure.
    """
    if WORKFLOW_MANAGER_AVAILABLE:
        try:
            workflow_config = get_workflow_config(str(config.project_root))
            correlator_config = update_module_paths('correlator', workflow_config)
            
            return {
                "quant_dir": correlator_config['input_dir'], 
                "output_dir": correlator_config['output_dir']
            }
        except Exception as e:
            print(f"⚠️  Workflow manager error: {e}, falling back to legacy paths")
    
    # Legacy fallback
    root = Path(config.project_root)
    quant_dir = root / "quant"
    output_dir = root / "correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {"quant_dir": quant_dir, "output_dir": output_dir}


CONFIG = {
    "xps_csv_paths": [r"N:\zhenzhen\Python\ZZY_XPS\F1s\fits_multilayer\AC_quantification\all_components_with_atomic_percent.csv"],
    "echem_csv_path": r"N:\zhenzhen\Python\ZZY_XPS\EC_data.xlsx",
    "echem_sheet_name": 0,  # or "Sheet1"

    "sample_id_col": "Sample",
    "xps_region_filter": None,
    # or "Area_percent" if atomic % not available
    "xps_value_col": "Component_atomic_percent",
    "xps_component_col": "Component",
    "xps_layer_aggregation": "sum",
    "xps_normalize_rows": False,  # False for atomic percent; True if using raw areas
    "include_region_in_feature_name": True,

    "echem_cycle_of_interest": None,
    "target_column": "Capacity_Fade_perc",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 500,
    "max_depth": None,
    "n_jobs": -1,

    "output_dir": "outputs",
    "processed_dataset_csv": "outputs/processed_dataset.csv",
    "corr_heatmap_path": "outputs/corr_heatmap.png",
    "fi_plot_path": "outputs/feature_importances.png",
    "model_path": "outputs/random_forest_model.joblib",
    "metrics_path": "outputs/metrics.json",
}
# -------------------- Utilities --------------------


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_xps_paths(paths_or_patterns: List[str]) -> List[str]:
    """
    Expand a list of file paths, directories, or glob patterns into concrete CSV file paths.
    """
    resolved = []
    for p in paths_or_patterns:
        if not isinstance(p, str):
            continue
        p = p.strip()
        if not p:
            continue
        # If it's a directory, include all .csv files in it
        if os.path.isdir(p):
            resolved.extend(glob.glob(os.path.join(p, "*.csv")))
        else:
            # If it looks like a glob pattern, expand it
            if any(ch in p for ch in ["*", "?", "[", "]"]):
                resolved.extend(glob.glob(p))
            else:
                # If it's an explicit file path
                if os.path.isfile(p):
                    resolved.append(p)
                else:
                    warnings.warn(f"XPS path not found: {p}")
    # Deduplicate
    resolved = sorted(list(set(resolved)))
    if not resolved:
        raise ValueError(
            "No XPS CSV files found. Check 'xps_csv_paths' paths/patterns.")
    return resolved


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "sample": "Sample",
        "Sample": "Sample",
        "File": "File",
        "Layer": "Layer",
        "Region": "Region",
        "Component": "Component",
        "Area_percent": "Area_percent",
        "Area_percent(%)": "Area_percent",
        "Area_counts": "Area_counts",
        "Center_eV": "Center_eV",
        "FWHM_eV": "FWHM_eV",
        "Eta_mix": "Eta_mix",
        "R_squared": "R_squared",
        "Template": "Template",
        "Capacity retention": "Capacity_Retention_perc",
        "Capacity Retention": "Capacity_Retention_perc",
        "capacity_retention": "Capacity_Retention_perc",
        "CE": "CE",
        "Cycle": "Cycle",
        "R-(electrolyte)": "R_electrolyte",
        "R-(interface)": "R_interface",
        "R-(SEI)": "R_SEI",
        "R-(ct)": "R_ct",
        "Element_atomic_percent": "Element_atomic_percent",
        "Component_atomic_percent": "Component_atomic_percent",
    }
    df.rename(columns=rename_map, inplace=True)
    df.columns = [c.replace("-", "_").replace("(", "").replace(")",
                                                               "").replace(" ", "_") for c in df.columns]
    return df


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_xps(csv_paths: List[str], region_filter: Optional[str] = None, value_col_pref: Optional[str] = None) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        dfi = pd.read_csv(p)
        dfi = clean_column_names(dfi)
        if region_filter is not None and "Region" in dfi.columns:
            dfi = dfi[dfi["Region"] == region_filter]
        # Required columns
        required_cols = {"Sample", "Component"}
        missing = required_cols - set(dfi.columns)
        if missing:
            raise ValueError(f"XPS file {p} missing columns: {missing}")

        # Numeric conversions
        numeric_cols = [
            "Area_percent", "Area_counts", "Center_e_V", "Center_eV", "FWHM_eV",
            "Eta_mix", "R_squared", "Element_atomic_percent", "Component_atomic_percent", "Layer"
        ]
        for col in numeric_cols:
            if col in dfi.columns:
                dfi[col] = to_numeric_safe(dfi[col])

        dfs.append(dfi)

    xps_df = pd.concat(dfs, ignore_index=True)
    xps_df["Sample"] = xps_df["Sample"].astype(str).str.strip()

    # Determine value column to use
    if value_col_pref and value_col_pref in xps_df.columns:
        xps_df["_pivot_value_"] = xps_df[value_col_pref]
    elif "Component_atomic_percent" in xps_df.columns:
        xps_df["_pivot_value_"] = xps_df["Component_atomic_percent"]
    elif "Area_percent" in xps_df.columns:
        xps_df["_pivot_value_"] = xps_df["Area_percent"]
    else:
        raise ValueError(
            "Neither 'Component_atomic_percent' nor 'Area_percent' present in XPS data.")

    return xps_df


def pivot_xps_components(
    xps_df: pd.DataFrame,
    id_col: str = "Sample",
    component_col: str = "Component",
    value_col: str = "_pivot_value_",
    layer_agg: str = "sum",
    normalize_rows: bool = False,
    include_region_in_feature_name: bool = True
) -> pd.DataFrame:
    # Aggregate across layers per sample per region per component (if Region exists)
    agg_func = {"sum": "sum", "mean": "mean",
                "max": "max"}.get(layer_agg, "sum")
    group_cols = [id_col]
    if include_region_in_feature_name and "Region" in xps_df.columns:
        group_cols += ["Region"]
    group_cols += [component_col]

    grouped = xps_df.groupby(group_cols, dropna=False)[
        value_col].agg(agg_func).reset_index()

    # Pivot to wide format
    if include_region_in_feature_name and "Region" in xps_df.columns:
        wide = grouped.pivot(index=id_col, columns=[
                             "Region", component_col], values=value_col).fillna(0.0)
        # Flatten MultiIndex columns: Region_Component_perc
        wide.columns = [f"{str(r)}_{str(c)}_perc" for r, c in wide.columns]
    else:
        wide = grouped.pivot(
            index=id_col, columns=component_col, values=value_col).fillna(0.0)
        wide.columns = [f"{str(c)}_perc" for c in wide.columns]

    # Normalize per-sample rows if requested (useful for raw areas, not atomic %)
    if normalize_rows:
        row_sums = wide.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        wide = wide.div(row_sums, axis=0) * 100.0

    wide.reset_index(inplace=True)
    return wide


def load_echem(echem_path: str, sample_id_col: str = "Sample", cycle_of_interest: Optional[int] = None, sheet_name: Optional[str] = None) -> pd.DataFrame:
    # Read CSV vs Excel
    ext = os.path.splitext(echem_path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        echem = pd.read_excel(echem_path, sheet_name=sheet_name)
    else:
        echem = pd.read_csv(echem_path)

    echem = clean_column_names(echem)

    if sample_id_col not in echem.columns:
        if "sample" in echem.columns:
            echem.rename(columns={"sample": sample_id_col}, inplace=True)
        else:
            raise ValueError(
                f"Electrochemical file missing sample ID column '{sample_id_col}'.")

    echem[sample_id_col] = echem[sample_id_col].astype(str).str.strip()

    for col in ["Capacity_Retention_perc", "CE", "Cycle", "R_electrolyte", "R_interface", "R_SEI", "R_ct"]:
        if col in echem.columns:
            echem[col] = to_numeric_safe(echem[col])

    # Target: Capacity_Fade_perc = 100 - Capacity_Retention_perc if available
    if "Capacity_Retention_perc" in echem.columns:
        echem["Capacity_Fade_perc"] = 100.0 - echem["Capacity_Retention_perc"]
    else:
        warnings.warn(
            "Capacity_Retention_perc not found; target Capacity_Fade_perc will be missing unless provided.")

    # Choose cycle (robust to NaNs and missing values)
    if "Cycle" in echem.columns:
        if cycle_of_interest is not None:
            # If Cycle might be non-numeric strings, ensure numeric comparison
            # (you already called to_numeric with coerce above)
            echem = echem[echem["Cycle"] == cycle_of_interest].copy()
        else:
            # Sort stably so fallback to "last row" is deterministic
            echem = echem.sort_values(
                [sample_id_col, "Cycle"], kind="mergesort")

            selected_idx = []
            for _, g in echem.groupby(sample_id_col, sort=False):
                valid_cycles = g["Cycle"].dropna()
                if not valid_cycles.empty:
                    # Take the row index of the maximum valid cycle
                    selected_idx.append(valid_cycles.idxmax())
                else:
                    # Fallback: no valid cycle; take the last row in this sample group
                    selected_idx.append(g.index[-1])

            echem = echem.loc[selected_idx].copy()

    return echem


def merge_datasets(xps_wide: pd.DataFrame, echem_df: pd.DataFrame, sample_id_col: str = "Sample") -> pd.DataFrame:
    return pd.merge(echem_df, xps_wide, on=sample_id_col, how="inner")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str, figsize=(12, 10)):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2 or num.isna().all().all():
        print("Skip heatmap: insufficient numeric data or all NaN.")
        return
    corr = num.corr(method="pearson")
    if corr.isna().all().all():
        print("Skip heatmap: correlation matrix is all NaN.")
        return
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# In main, before training:
xps_feature_cols = [c for c in merged.columns if c.endswith("_perc")]
if len(xps_feature_cols) == 0:
    raise ValueError(
        "No XPS composition feature columns found. Check pivot settings and value column.")

merged = merged.dropna(subset=[cfg["target_column"]])
X = merged[xps_feature_cols].copy()
y = merged[cfg["target_column"]].copy()

if X.shape[0] == 0:
    # Helpful debug summary
    print("No rows available for training after merge and NaN filtering.")
    print(
        f"xps_wide samples: {xps_wide['Sample'].nunique()} | echem samples: {echem_df['Sample'].nunique()}")
    print("Example XPS Sample IDs:", xps_wide["Sample"].head().tolist())
    print("Example EC Sample IDs:", echem_df["Sample"].head().tolist())
    print("Intersection size:", len(
        set(xps_wide["Sample"]) & set(echem_df["Sample"])))
    raise ValueError(
        "Empty training set. Fix Sample ID alignment, value columns, or cycle selection.")


def train_random_forest(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = 0.2, random_state: int = 42,
    n_estimators: int = 500, max_depth: Optional[int] = None,
    n_jobs: int = -1
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    return rf, (X_train, X_test, y_train, y_test), metrics


def plot_feature_importances(model: RandomForestRegressor, feature_names: List[str], save_path: str, top_n: Optional[int] = None):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    if top_n is not None:
        order = order[:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[order][::-1],
             importances[order][::-1], color="#2c7fb8")
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main(cfg: Dict):
    ensure_output_dir(cfg["output_dir"])

    # Resolve XPS paths (files, dirs, or globs)
    xps_paths = resolve_xps_paths(cfg["xps_csv_paths"])

    # Load XPS
    xps_df = load_xps(
        xps_paths, region_filter=cfg["xps_region_filter"], value_col_pref=cfg["xps_value_col"])

    # Pivot XPS components to wide
    xps_wide = pivot_xps_components(
        xps_df,
        id_col=cfg["sample_id_col"],
        component_col=cfg["xps_component_col"],
        value_col="_pivot_value_",
        layer_agg=cfg["xps_layer_aggregation"],
        normalize_rows=cfg["xps_normalize_rows"],
        include_region_in_feature_name=cfg["include_region_in_feature_name"]
    )

    # Load electrochemical data
    echem_df = load_echem(
        cfg["echem_csv_path"],
        sample_id_col=cfg["sample_id_col"],
        cycle_of_interest=cfg["echem_cycle_of_interest"],
        sheet_name=cfg.get("echem_sheet_name"),
    )

    # Merge datasets
    merged = merge_datasets(
        xps_wide, echem_df, sample_id_col=cfg["sample_id_col"])
    merged.to_csv(cfg["processed_dataset_csv"], index=False)

    # EDA: correlation heatmap
    plot_correlation_heatmap(merged, cfg["corr_heatmap_path"])

    # Features: XPS composition columns (suffix '_perc')
    xps_feature_cols = [c for c in merged.columns if c.endswith("_perc")]
    if len(xps_feature_cols) == 0:
        raise ValueError(
            "No XPS composition feature columns found (suffix '_perc'). Check pivot settings.")

    # Target
    target_col = cfg["target_column"]
    if target_col not in merged.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Ensure electrochem data includes Capacity_Retention_perc or '{target_col}'.")

    merged = merged.dropna(subset=[target_col])
    X = merged[xps_feature_cols].copy()
    y = merged[target_col].copy()

    # Train model
    rf_model, splits, metrics = train_random_forest(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        n_jobs=cfg["n_jobs"]
    )

    # Save model and metrics
    dump(rf_model, cfg["model_path"])
    with open(cfg["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importances
    plot_feature_importances(rf_model, xps_feature_cols,
                             cfg["fi_plot_path"], top_n=None)

    # Permutation importances
    X_train, X_test, y_train, y_test = splits
    pi = permutation_importance(rf_model, X_test, y_test, n_repeats=10,
                                random_state=cfg["random_state"], n_jobs=cfg["n_jobs"])
    pi_dict = {
        "features": xps_feature_cols,
        "importances_mean": pi.importances_mean.tolist(),
        "importances_std": pi.importances_std.tolist()
    }
    with open(os.path.join(cfg["output_dir"], "permutation_importances.json"), "w") as f:
        json.dump(pi_dict, f, indent=2)

    print("=== ML Correlator Summary ===")
    print(f"Processed dataset: {cfg['processed_dataset_csv']}")
    print(f"Correlation heatmap: {cfg['corr_heatmap_path']}")
    print(f"Model saved to: {cfg['model_path']}")
    print(f"Feature importances plot: {cfg['fi_plot_path']}")
    print(f"Metrics: {cfg['metrics_path']}")
    print(f"Metrics summary: {metrics}")


if __name__ == "__main__":
    if not CONFIG["xps_csv_paths"] or not CONFIG["echem_csv_path"]:
        print(
            "Please set CONFIG['xps_csv_paths'] and CONFIG['echem_csv_path'] before running.")
    else:
        main(CONFIG)
