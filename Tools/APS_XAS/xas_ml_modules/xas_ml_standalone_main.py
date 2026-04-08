"""
Standalone ML Analysis Main Entry Point - XAS Workflow

This is the MAIN ENTRY POINT for all ML analysis in the XAS workflow.
It provides a unified interface for:
  1. Feature-based ML (PCA, clustering, correlations)
  2. Whole-spectrum PCA analysis
  3. Experiment planning with conditions overlay
  4. Comprehensive result reporting

Usage:
  python xas_ml_standalone_main.py [--mode MODE] [--output-dir DIR]
  
Modes:
  - all: Run both feature-based and spectrum-based analysis (default)
  - features: Feature-based ML only
  - spectrum: Whole-spectrum PCA only
  - planning: Spectrum PCA + experiment suggestions only

"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import importlib.util
import re

# Ensure APS_XAS tools directory is on sys.path for direct script execution
_current_dir = Path(__file__).resolve().parent
_tools_dir = _current_dir.parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from xas_ml_modules.xas_tsne_umap import XASTSNEUMAP
from xas_ml_modules.config_utils import ConfigLoader
from xas_ml_modules.xas_pca_analyzer import XASPCAAnalyzer
from xas_ml_modules.xas_clusterer import XASClusterer
from xas_ml_modules.xas_trend_analyzer import XASTrendAnalyzer
from xas_analyzer.xas_models import XASDataset
from xas_feature_extraction.xas_feature_extractor import XASFeatureExtractor
try:
    from xas_plotter.xas_ml_plotter import (
        plot_pca_scree,
        plot_pca_scores,
        plot_pca_loadings,
        plot_cluster_scatter,
        plot_correlation_heatmap
    )
except Exception:
    plot_pca_scree = None
    plot_pca_scores = None
    plot_pca_loadings = None
    plot_cluster_scatter = None
    plot_correlation_heatmap = None

try:
    from xas_plotter.xas_features_plotter import create_feature_comparison_plots
except Exception:
    create_feature_comparison_plots = None

try:
    from xas_ml_modules.xas_vae import XASVAEEmbedding
except Exception:
    XASVAEEmbedding = None

try:
    from xas_ml_modules.xas_structure_condition_plots import generate_structure_condition_plots
except Exception:
    generate_structure_condition_plots = None

# sklearn imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats

# Matplotlib (non-interactive backend for automated runs)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


def load_features_from_json(json_dir: Path):
    """Load XAS features from individual JSON files."""
    json_files = sorted(json_dir.glob("*_features.json"))
    
    features_list = []
    sample_names = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            features = json.load(f)
            features_list.append(features)
            sample_names.append(features['sample_name'])
    
    return features_list, sample_names


def load_features_from_csv_dir(csv_dir: Path, feature_extractor: XASFeatureExtractor):
    """
    Load analyzed CSV spectra (energy + mu) and extract features on the fly.
    Expected columns: energy and mu_normalized (fallback to mu_cleaned).
    """
    csv_files = sorted(csv_dir.glob("*.csv"))

    features_list = []
    sample_names = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            print(f"  [WARN] Failed to read {csv_file.name}: {exc}")
            continue

        if 'energy' not in df.columns:
            print(f"  [WARN] Missing 'energy' column in {csv_file.name}, skipping")
            continue

        if 'mu_normalized' in df.columns:
            mu = df['mu_normalized'].to_numpy()
        elif 'mu_cleaned' in df.columns:
            mu = df['mu_cleaned'].to_numpy()
        else:
            print(f"  [WARN] Missing 'mu_normalized' or 'mu_cleaned' in {csv_file.name}, skipping")
            continue

        energy = df['energy'].to_numpy()
        sample_name = csv_file.stem

        try:
            features = feature_extractor.extract_features_from_arrays(
                energy=energy,
                mu_normalized=mu,
                sample_name=sample_name
            )
        except Exception as exc:
            print(f"  [WARN] Feature extraction failed for {csv_file.name}: {exc}")
            continue

        features_list.append(features.model_dump())
        sample_names.append(sample_name)

    return features_list, sample_names


def create_feature_matrix(features_list):
    """Convert list of feature dicts to numpy matrix."""
    # Define feature order (exclude sample_name and null features)
    feature_names = [
        'e0', 'edge_step', 'edge_slope', 'pre_edge_area',
        'white_line_intensity', 'white_line_prominence', 'white_line_energy', 'white_line_fwhm',
        'xanes_area', 'xanes_centroid',
        'first_derivative_max', 'second_derivative_zero',
        'spectral_mean', 'spectral_variance', 'spectral_skewness', 'spectral_kurtosis'
    ]
    
    # Build matrix
    matrix = []
    for features in features_list:
        row = []
        for fname in feature_names:
            value = features.get(fname)
            # Handle None values
            if value is None:
                value = float('nan')
            row.append(float(value))
        matrix.append(row)
    
    return np.array(matrix), feature_names


def parse_metadata_from_name(sample_name: str):
    """Extract experimental metadata from sample name."""
    import re
    
    metadata = {}
    
    # Extract iron source (anion)
    if 'FeCl2' in sample_name:
        metadata['iron_source'] = 'FeCl2'
        metadata['anion_type'] = 0
    elif 'FeSO4' in sample_name:
        metadata['iron_source'] = 'FeSO4'
        metadata['anion_type'] = 1
    elif 'FeAcetate' in sample_name:
        metadata['iron_source'] = 'FeAcetate'
        metadata['anion_type'] = 2
    elif 'FeTFSI' in sample_name:
        metadata['iron_source'] = 'FeTFSI'
        metadata['anion_type'] = 3
    else:
        metadata['iron_source'] = 'Unknown'
        metadata['anion_type'] = -1
    
    # Extract ligand
    if 'Malic' in sample_name:
        metadata['ligand'] = 'Malic_acid'
        metadata['ligand_type'] = 0
    elif 'Tartaric' in sample_name:
        metadata['ligand'] = 'Tartaric_acid'
        metadata['ligand_type'] = 1
    elif 'H2O' in sample_name:
        metadata['ligand'] = 'H2O'
        metadata['ligand_type'] = 2
    else:
        metadata['ligand'] = 'Unknown'
        metadata['ligand_type'] = -1
    
    # Extract pH
    ph_match = re.search(r'pH(\d+(?:[_\.]\d+)?)', sample_name)
    if ph_match:
        ph_str = ph_match.group(1).replace('_', '.')
        metadata['pH'] = float(ph_str)
    else:
        metadata['pH'] = None
    
    # Extract gel/solution state
    if 'gel' in sample_name.lower():
        metadata['state'] = 'gel'
        metadata['state_type'] = 1
    else:
        metadata['state'] = 'solution'
        metadata['state_type'] = 0
    
    # Extract replicate number
    replicate_match = re.search(r'_R(\d+)', sample_name)
    if replicate_match:
        metadata['replicate'] = int(replicate_match.group(1))
    else:
        metadata['replicate'] = 1
    
    # Extract concentrations
    conc_match = re.search(r'\((\d+(?:[_\.]\d+)?)-(\d+(?:[_\.]\d+)?)\)', sample_name)
    if conc_match:
        anion_str = conc_match.group(1).replace('_', '.')
        ligand_str = conc_match.group(2).replace('_', '.')
        
        try:
            metadata['anion_conc'] = float(anion_str)
            metadata['ligand_conc'] = float(ligand_str)
            if metadata['ligand_conc'] > 0:
                metadata['conc_ratio'] = metadata['anion_conc'] / metadata['ligand_conc']
            else:
                metadata['conc_ratio'] = None
        except ValueError:
            metadata['anion_conc'] = None
            metadata['ligand_conc'] = None
            metadata['conc_ratio'] = None
    else:
        metadata['anion_conc'] = None
        metadata['ligand_conc'] = None
        metadata['conc_ratio'] = None
    
    return metadata


def create_metadata_dict(sample_names):
    """Create metadata dictionary from sample names."""
    metadata_dict = {
        'iron_source': [],
        'anion_type': [],
        'ligand': [],
        'ligand_type': [],
        'pH': [],
        'state': [],
        'state_type': [],
        'replicate': [],
        'anion_conc': [],
        'ligand_conc': [],
        'conc_ratio': []
    }
    
    for name in sample_names:
        meta = parse_metadata_from_name(name)
        for key in metadata_dict.keys():
            metadata_dict[key].append(meta.get(key))
    
    return metadata_dict


def run_pca(feature_matrix, n_components=5):
    """Run PCA analysis with imputation and scaling."""
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(feature_matrix)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    results = {
        'n_components': n_components,
        'variance_explained': pca.explained_variance_ratio_,
        'variance_cumulative': np.cumsum(pca.explained_variance_ratio_),
        'loadings': pca.components_.T,  # Features x Components
        'scores': scores,  # Samples x Components
        'scaler_mean': scaler.mean_,
        'scaler_std': scaler.scale_,
        'imputer_statistics': imputer.statistics_.tolist()
    }

    return results


def run_clustering(X, n_clusters=4):
    """Run K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Compute validation metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    results = {
        'n_clusters': n_clusters,
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz
    }
    
    return results


def compute_correlations(feature_matrix, metadata_dict, feature_names):
    """Compute feature-metadata correlations with NaN handling."""
    correlations = {}

    for meta_key, meta_values in metadata_dict.items():
        meta_array = []
        feature_arrays = [[] for _ in range(feature_matrix.shape[1])]

        for i, val in enumerate(meta_values):
            if val is None or isinstance(val, str):
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue

            row = feature_matrix[i, :]
            if np.any(~np.isfinite(row)):
                continue

            meta_array.append(v)
            for j in range(feature_matrix.shape[1]):
                feature_arrays[j].append(row[j])

        if len(meta_array) < 3:
            continue

        meta_array = np.array(meta_array)
        correlations[meta_key] = {}

        for j, fname in enumerate(feature_names):
            feat_array = np.array(feature_arrays[j])
            try:
                r, p = stats.pearsonr(feat_array, meta_array)
                correlations[meta_key][fname] = {
                    'correlation': float(r),
                    'p_value': float(p),
                    'significant': p < 0.05
                }
            except:
                pass

    return correlations


def save_results(output_dir, sample_names, feature_matrix, feature_names,
                metadata_dict, pca_results, cluster_results, trend_results):
    """Save all ML analysis results."""
    
    # Create subdirectories
    results_dir = output_dir / 'analysis_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature matrix as CSV
    # Align feature_names with matrix shape if needed
    if feature_matrix.shape[1] != len(feature_names):
        print(f"  [WARN] Feature name count ({len(feature_names)}) does not match matrix columns ({feature_matrix.shape[1]}). Truncating feature names.")
        feature_names = feature_names[:feature_matrix.shape[1]]
    df_features = pd.DataFrame(
        feature_matrix,
        columns=feature_names,
        index=sample_names
    )
    df_features.to_csv(results_dir / 'feature_matrix.csv')
    
    # 2. Metadata as CSV
    if metadata_dict:
        df_metadata = pd.DataFrame(metadata_dict, index=sample_names)
        df_metadata.to_csv(results_dir / 'metadata.csv')
    
    # 3. PCA results
    pca_data = {
        'n_components': int(pca_results.n_components),
        'variance_explained': list(pca_results.explained_variance),
        'variance_cumulative': list(pca_results.cumulative_variance),
        'total_variance_captured': float(pca_results.variance_captured),
        'confidence': float(pca_results.confidence),
        'flags': list(pca_results.flags),
        'feature_importance': pca_results.feature_importance
    }
    
    with open(results_dir / 'pca_summary.json', 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    # PCA scores as CSV
    df_pca = pd.DataFrame(
        pca_results.scores,
        columns=[f'PC{i+1}' for i in range(pca_results.n_components)],
        index=sample_names
    )
    df_pca.to_csv(results_dir / 'pca_scores.csv')
    
    # PCA loadings as CSV
    # Align feature_names with loadings shape if needed
    loadings = pca_results.loadings
    if loadings.shape[0] != len(feature_names):
        print(f"  [WARN] Loading rows ({loadings.shape[0]}) do not match feature names ({len(feature_names)}). Truncating feature names.")
        feature_names = feature_names[:loadings.shape[0]]
    df_loadings = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(pca_results.n_components)],
        index=feature_names
    )
    df_loadings.to_csv(results_dir / 'pca_loadings.csv')
    
    # 4. Clustering results
    cluster_data = {
        'method': cluster_results.method,
        'n_clusters': int(cluster_results.n_clusters),
        'labels': cluster_results.labels.tolist() if cluster_results.labels is not None else [],
        'silhouette_score': float(cluster_results.silhouette_score),
        'davies_bouldin_score': float(cluster_results.davies_bouldin_index) if cluster_results.davies_bouldin_index is not None else None,
        'calinski_harabasz_score': float(cluster_results.calinski_harabasz_score) if cluster_results.calinski_harabasz_score is not None else None,
        'confidence': float(cluster_results.confidence),
        'flags': list(cluster_results.flags)
    }
    
    with open(results_dir / 'cluster_summary.json', 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    # Cluster assignments as CSV
    df_clusters = pd.DataFrame({
        'sample_name': sample_names,
        'cluster': cluster_results.labels if cluster_results.labels is not None else []
    })
    df_clusters.to_csv(results_dir / 'cluster_assignments.csv', index=False)
    
    # 5. Trend analysis summary
    if trend_results is not None:
        trend_data = {
            'correlations': trend_results.correlations,
            'p_values': trend_results.p_values,
            'significant_correlations': trend_results.significant_correlations,
            'cluster_metadata_stats': trend_results.cluster_metadata_stats,
            'outlier_indices': trend_results.outlier_indices,
            'outlier_scores': trend_results.outlier_scores.tolist() if trend_results.outlier_scores is not None else None,
            'outlier_method': trend_results.outlier_method,
            'confidence': float(trend_results.confidence),
            'flags': list(trend_results.flags)
        }

        with open(results_dir / 'trend_analysis_summary.json', 'w') as f:
            json.dump(trend_data, f, indent=2)

        correlations_json = {}
        for feat_name, corr_dict in trend_results.correlations.items():
            correlations_json[feat_name] = {}
            for meta_key, corr_val in corr_dict.items():
                pval = trend_results.p_values.get(feat_name, {}).get(meta_key)
                correlations_json[feat_name][meta_key] = {
                    'correlation': corr_val,
                    'p_value': pval,
                    'significant': int(
                        any(
                            (sc.get('feature') == feat_name and sc.get('metadata') == meta_key)
                            for sc in trend_results.significant_correlations
                        )
                    )
                }

        with open(results_dir / 'feature_metadata_correlations.json', 'w') as f:
            json.dump(correlations_json, f, indent=2)
    
    print(f"\n[OK] Results saved to: {results_dir}")
    print(f"  - feature_matrix.csv ({len(sample_names)} x {len(feature_names)})")
    print(f"  - metadata.csv")
    print(f"  - pca_summary.json")
    print(f"  - pca_scores.csv")
    print(f"  - pca_loadings.csv")
    print(f"  - cluster_summary.json")
    print(f"  - cluster_assignments.csv")
    if trend_results is not None:
        print(f"  - trend_analysis_summary.json")
        print(f"  - feature_metadata_correlations.json")




def _plot_embedding(embedding, labels, title, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))

    # Numeric vs categorical
    try:
        vals = np.array(labels, dtype=float)
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False
        vals = np.array(labels, dtype=object)

    if is_numeric:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=vals, cmap='viridis', s=90, edgecolors='black')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(title, fontsize=18)
    else:
        unique_vals = sorted(set(vals))
        if len(unique_vals) <= 9:
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_vals)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
        for v, c in zip(unique_vals, colors):
            mask = vals == v
            ax.scatter(embedding[mask, 0], embedding[mask, 1], label=str(v), c=[c], s=90, edgecolors='black')
        ax.legend(title=title, fontsize=18)

    ax.set_xlabel('Dim 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dim 2', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def main():
    """Run complete ML analysis workflow."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='XAS ML Analysis Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis (default)
  python xas_ml_standalone_main.py
  
  # Feature-based ML only
  python xas_ml_standalone_main.py --mode features
  
  # Whole-spectrum PCA + planning only
  python xas_ml_standalone_main.py --mode spectrum
  
  # Custom output directory
  python xas_ml_standalone_main.py --output-dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'features', 'spectrum', 'planning'],
        default='all',
        help='Analysis mode (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: project_root/xas_results/04_ml_analysis)'
    )

    parser.add_argument(
        '--features-dir',
        type=str,
        default=None,
        help='Directory of *_features.json (default: project_root/xas_results/03_feature_extraction/extracted_features)'
    )

    parser.add_argument(
        '--filtered-dir',
        type=str,
        default=None,
        help='Directory of analyzed CSV spectra (energy + mu_normalized) to extract features on the fly'
    )
    
    parser.add_argument(
        '--n-components',
        type=int,
        default=5,
        help='Number of PCA components (default: 5)'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=4,
        help='Number of clusters for K-means (default: 4)'
    )

    parser.add_argument(
        '--tsne',
        action='store_true',
        help='Run t-SNE embedding (feature-based)'
    )

    parser.add_argument(
        '--umap',
        action='store_true',
        help='Run UMAP embedding (feature-based)'
    )

    parser.add_argument(
        '--vae',
        action='store_true',
        help='Run VAE embedding on spectra (requires torch)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("XAS ML ANALYSIS - MAIN ENTRY POINT")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    config = ConfigLoader()
    limits = config.get_section('analysis_limits')
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths - find project_root correctly
    current_dir = Path(__file__).parent  # xas_ml_modules/
    tools_dir = current_dir.parent  # Tools/APS_XAS/
    zzy_llm_dir = tools_dir.parent.parent  # zzy_llm/
    project_root = zzy_llm_dir / "project_root"
    
    print(f"Project root: {project_root}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "xas_results" / "04_ml_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run appropriate analysis mode
    if args.mode in ['all', 'features']:
        print("\n" + "=" * 80)
        print("PART 1: FEATURE-BASED ML ANALYSIS")
        print("=" * 80)
        run_feature_based_ml(project_root, output_dir, args)
    
    if args.mode in ['all', 'spectrum', 'planning']:
        print("\n" + "=" * 80)
        print("PART 2: WHOLE-SPECTRUM PCA & EXPERIMENT PLANNING")
        print("=" * 80)
        run_spectrum_pca_and_planning(project_root, output_dir, args)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ALL ML ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated outputs:")
    print("  analysis_results/    - Data files (CSV, JSON)")
    print("  analysis_plots/      - Visualizations (PNG)")
    if args.mode in ['all', 'spectrum', 'planning']:
        print("  spectrum_pca_results/ - Spectrum PCA outputs")
    print("\n" + "=" * 80)


def run_feature_based_ml(project_root, output_dir, args):
    """Run feature-based ML analysis."""
    config = ConfigLoader()
    limits = config.get_section('analysis_limits')
    cluster_cfg = config.get_section('clustering')

    features_dir = project_root / "xas_results" / "03_feature_extraction" / "extracted_features"

    # Step 1: Load features
    print("\n[1/6] Loading extracted features...")
    if args.filtered_dir:
        filtered_dir = Path(args.filtered_dir)
        print(f"  Using filtered data: {filtered_dir}")
        feature_extractor = XASFeatureExtractor()
        features_list, sample_names = load_features_from_csv_dir(filtered_dir, feature_extractor)
    else:
        if args.features_dir:
            features_dir = Path(args.features_dir)
        print(f"  Using feature JSONs: {features_dir}")
        features_list, sample_names = load_features_from_json(features_dir)

    print(f"  Loaded {len(features_list)} samples")
    min_warn = limits.get('min_samples_warning', 10)
    if len(features_list) < min_warn:
        print(f"  [WARN] Small sample size ({len(features_list)} < {min_warn}). Statistical analyses may be unreliable.")
    if len(features_list) < 10:
        print(f"  [WARN] Small sample size ({len(features_list)}). Statistical analyses may be unreliable.")

    if len(features_list) < 2:
        print("\n[SKIP] Need at least 2 samples for ML analysis")
        return

    # Step 2: Create feature matrix
    print("\n[2/6] Creating feature matrix...")
    feature_matrix, feature_names = create_feature_matrix(features_list)
    print(f"  Shape: {feature_matrix.shape}")
    print(f"  Features: {len(feature_names)}")

    # Drop all-NaN feature columns (prevents imputer from silently dropping)
    valid_cols = np.any(np.isfinite(feature_matrix), axis=0)
    if not np.all(valid_cols):
        removed = [feature_names[i] for i, ok in enumerate(valid_cols) if not ok]
        feature_matrix = feature_matrix[:, valid_cols]
        feature_names = [name for name, ok in zip(feature_names, valid_cols) if ok]
        print(f"  [WARN] Dropped all-NaN features: {removed}")
        print(f"  Updated feature matrix shape: {feature_matrix.shape}")

    # Extract metadata
    metadata_dict = create_metadata_dict(sample_names)
    print(f"  Metadata columns: {len(metadata_dict)}")

    dataset = XASDataset(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        sample_names=sample_names,
        metadata_dict=metadata_dict,
        quality_flags={},
        n_samples=len(sample_names),
        n_features=len(feature_names),
        dataset_id=f"xas_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        creation_timestamp=datetime.now()
    )

    # PCA sample-size warning: at least 5x features
    pca_min_samples = int(limits.get('pca_samples_per_feature', 5) * feature_matrix.shape[1])
    if len(features_list) < pca_min_samples:
        print(f"  [WARN] PCA may be unstable: {len(features_list)} samples < 5x features ({pca_min_samples}).")

    # Step 3: PCA Analysis (module)
    print("\n[3/6] Running PCA...")
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(feature_matrix)
    pca_dataset = XASDataset(
        feature_matrix=X_imp,
        feature_names=feature_names,
        sample_names=sample_names,
        metadata_dict=metadata_dict,
        quality_flags={},
        n_samples=len(sample_names),
        n_features=len(feature_names),
        dataset_id=dataset.dataset_id,
        creation_timestamp=dataset.creation_timestamp
    )
    pca_analyzer = XASPCAAnalyzer()
    pca_results = pca_analyzer.analyze(pca_dataset, n_components=args.n_components)
    print(f"  Components: {pca_results.n_components}")
    print(f"  Variance explained by each PC:")
    for i, var in enumerate(pca_results.explained_variance):
        cum_var = pca_results.cumulative_variance[i]
        print(f"    PC{i+1}: {var*100:.1f}% (cumulative: {cum_var*100:.1f}%)")

    # Clustering sample-size warning: at least 2x clusters
    cluster_min_samples = int(limits.get('clustering_samples_per_cluster', 2) * args.n_clusters)
    if len(features_list) < cluster_min_samples:
        print(f"  [WARN] Clustering may be unreliable: {len(features_list)} samples < 2x clusters ({cluster_min_samples}).")

    # Step 4: Clustering (module)
    print("\n[4/6] Running clustering...")
    use_pca_space = cluster_cfg.get('use_pca_space', True)
    pca_scores = None
    if use_pca_space and pca_results.scores is not None:
        n_use = min(3, pca_results.n_components)
        pca_scores = pca_results.scores[:, :n_use]
    clusterer = XASClusterer()
    cluster_results = clusterer.cluster(dataset, n_clusters=args.n_clusters, use_pca_scores=pca_scores)
    print(f"  Clusters: {cluster_results.n_clusters}")
    print(f"  Silhouette score: {cluster_results.silhouette_score:.3f}")
    if cluster_results.davies_bouldin_index is not None:
        print(f"  Davies-Bouldin index: {cluster_results.davies_bouldin_index:.3f} (lower is better)")
    if cluster_results.calinski_harabasz_score is not None:
        print(f"  Calinski-Harabasz score: {cluster_results.calinski_harabasz_score:.1f} (higher is better)")

    if cluster_results.labels is not None:
        unique, counts = np.unique(cluster_results.labels, return_counts=True)
        print("  Cluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"    Cluster {cluster_id}: {count} samples")

    # Step 5: Trend analysis (module)
    print("\n[5/6] Computing feature-metadata trends...")
    trend_analyzer = XASTrendAnalyzer()
    trend_results = trend_analyzer.analyze(dataset, clustering_result=cluster_results)
    print(f"  Significant correlations: {len(trend_results.significant_correlations)}")
    if trend_results.outlier_indices:
        print(f"  Outliers detected: {len(trend_results.outlier_indices)}")

    # Step 6: Save results
    print("\n[6/6] Saving feature-based ML results...")
    save_results(output_dir, sample_names, feature_matrix, feature_names,
                metadata_dict, pca_results, cluster_results, trend_results)

    # Feature-based plots
    plots_dir = output_dir / 'analysis_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    if plot_pca_scree is not None:
        plot_pca_scree(pca_results, plots_dir / 'feature_pca_scree.png')
    if plot_pca_scores is not None:
        color_by = None
        if 'pH' in metadata_dict:
            color_by = metadata_dict['pH']
        plot_pca_scores(pca_results, plots_dir / 'feature_pca_scores.png', color_by=color_by, title="Feature PCA Scores (PC1 vs PC2)")
    if plot_pca_loadings is not None:
        plot_pca_loadings(pca_results, feature_names, plots_dir / 'feature_pca_loadings.png')
    if plot_cluster_scatter is not None and cluster_results.labels is not None:
        plot_cluster_scatter(pca_results, cluster_results.labels, plots_dir / 'feature_clusters_pca.png')
    if plot_correlation_heatmap is not None and trend_results is not None:
        plot_correlation_heatmap(trend_results, plots_dir / 'feature_metadata_correlation_heatmap.png')

    # Structure–Condition plots (from interpretation guide)
    if generate_structure_condition_plots is not None and trend_results is not None:
        try:
            generate_structure_condition_plots(
                feature_matrix=feature_matrix,
                feature_names=feature_names,
                metadata_dict=metadata_dict,
                trend_results=trend_results,
                output_dir=plots_dir
            )
        except Exception as e:
            print(f"  [WARN] Structure–condition plot generation failed: {e}")

    # Optional t-SNE / UMAP embeddings
    if args.tsne or args.umap:
        print("\n[Optional] Running non-linear embeddings...")
        embedder = XASTSNEUMAP()
        try:
            if args.tsne:
                tsne_res = embedder.fit_tsne(dataset)
                results_dir = output_dir / 'analysis_results'
                np.savetxt(results_dir / 'tsne_embedding.csv', tsne_res['embedding'], delimiter=',')
                print("  [OK] t-SNE embedding saved")
                plots_dir = output_dir / 'analysis_plots'
                plots_dir.mkdir(parents=True, exist_ok=True)
                if 'pH' in metadata_dict:
                    _plot_embedding(tsne_res['embedding'], metadata_dict['pH'], 't-SNE colored by pH', plots_dir / 'tsne_pH.png')
                if 'ligand_conc' in metadata_dict:
                    _plot_embedding(tsne_res['embedding'], metadata_dict['ligand_conc'], 't-SNE colored by ligand concentration', plots_dir / 'tsne_ligand_concentration.png')
                if 'ligand' in metadata_dict:
                    _plot_embedding(tsne_res['embedding'], metadata_dict['ligand'], 't-SNE colored by ligand', plots_dir / 'tsne_ligand.png')
            if args.umap:
                umap_res = embedder.fit_umap(dataset)
                results_dir = output_dir / 'analysis_results'
                np.savetxt(results_dir / 'umap_embedding.csv', umap_res['embedding'], delimiter=',')
                print("  [OK] UMAP embedding saved")
                plots_dir = output_dir / 'analysis_plots'
                plots_dir.mkdir(parents=True, exist_ok=True)
                if 'pH' in metadata_dict:
                    _plot_embedding(umap_res['embedding'], metadata_dict['pH'], 'UMAP colored by pH', plots_dir / 'umap_pH.png')
                if 'ligand_conc' in metadata_dict:
                    _plot_embedding(umap_res['embedding'], metadata_dict['ligand_conc'], 'UMAP colored by ligand concentration', plots_dir / 'umap_ligand_concentration.png')
                if 'ligand' in metadata_dict:
                    _plot_embedding(umap_res['embedding'], metadata_dict['ligand'], 'UMAP colored by ligand', plots_dir / 'umap_ligand.png')
        except Exception as e:
            print(f"  [WARN] Embedding failed: {e}")


def run_spectrum_pca_and_planning(project_root, output_dir, args):
    """Run whole-spectrum PCA and experiment planning."""
    config = ConfigLoader()
    limits = config.get_section('analysis_limits')
    
    # Load spectrum PCA module
    tools_dir = Path(__file__).parent
    spec_pca = importlib.util.spec_from_file_location(
        "xas_spectrum_pca",
        tools_dir / "xas_spectrum_pca.py"
    )
    spectrum_pca_module = importlib.util.module_from_spec(spec_pca)
    spec_pca.loader.exec_module(spectrum_pca_module)
    
    # Load experiment planner module
    spec_planner = importlib.util.spec_from_file_location(
        "xas_experiment_planner",
        tools_dir / "xas_experiment_planner.py"
    )
    exp_planner_module = importlib.util.module_from_spec(spec_planner)
    spec_planner.loader.exec_module(exp_planner_module)
    
    XASSpectrumPCA = spectrum_pca_module.XASSpectrumPCA
    XASExperimentPlanner = exp_planner_module.XASExperimentPlanner
    
    # Load normalized spectra
    data_dir = project_root / "xas_results" / "02_analyzed_data" / "normalized_data"
    
    print("\n[1/7] Loading normalized spectra...")
    csv_files = sorted(data_dir.glob("*_analyzed.csv"))
    
    energies = []
    spectra = []
    sample_names = []
    
    for csv_file in csv_files:
        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            energy = data[:, 0]
            mu_norm = data[:, 2]  # mu_normalized column
            
            energies.append(energy)
            spectra.append(mu_norm)
            sample_names.append(csv_file.stem.replace('_analyzed', ''))
        except Exception as e:
            print(f"  Warning: Skipped {csv_file.name}: {e}")
    
    print(f"  Loaded {len(sample_names)} spectra")
    min_warn = limits.get('min_samples_warning', 10)
    if len(sample_names) < min_warn:
        print(f"  [WARN] Small sample size ({len(sample_names)} < {min_warn}). PCA and suggestions may be unstable.")
    
    if len(sample_names) < 2:
        print("\n[SKIP] Need at least 2 spectra for PCA")
        return
    
    # Run PCA
    print("\n[2/7] Running whole-spectrum PCA...")
    analyzer = XASSpectrumPCA(
        n_components=args.n_components,
        normalization='standard',
        energy_range=(7100, 7200),
        n_grid_points=500
    )
    
    pca_result = analyzer.analyze_spectra(energies, spectra, sample_names)
    
    print(f"  Components: {pca_result.n_components}")
    print(f"  Variance explained:")
    for i in range(pca_result.n_components):
        print(f"    PC{i+1}: {pca_result.variance_ratio[i]*100:.1f}% "
              f"(cumulative: {pca_result.cumulative_variance[i]*100:.1f}%)")
    
    # Interpret components
    print("\n[3/7] Interpreting principal components...")
    planner = XASExperimentPlanner(edge_energy=7120.0)
    interpretations = planner.interpret_components(pca_result, peak_threshold=0.15)
    
    for interp in interpretations[:3]:  # Show first 3
        print(f"  PC{interp.pc_number}: {interp.interpretation}")
    
    # Extract experimental parameters
    print("\n[4/7] Extracting experimental conditions...")
    experimental_params = extract_experimental_params_from_names(sample_names)
    
    unique_pH = len(set(p for p in experimental_params['pH'] if p is not None))
    unique_sources = len(set(experimental_params['iron_source']))
    unique_ligands = len(set(experimental_params['ligand']))
    
    print(f"  pH values: {unique_pH} unique")
    print(f"  Iron sources: {unique_sources} unique")
    print(f"  Ligands: {unique_ligands} unique")
    
    # Generate experiment suggestions
    print("\n[5/7] Generating experiment suggestions...")
    
    all_suggestions = []
    strategies = ['maxdist', 'boundary', 'hull']
    
    for strategy in strategies:
        try:
            suggestions = planner.suggest_experiments(
                pca_result,
                experimental_params=experimental_params,
                strategy=strategy,
                n_suggestions=2,
                pc_x=1,
                pc_y=2
            )
            all_suggestions.extend(suggestions)
            print(f"  Generated {len(suggestions)} suggestions using '{strategy}' strategy")
        except Exception as e:
            print(f"  Warning: '{strategy}' strategy failed: {e}")
    
    all_suggestions.sort(key=lambda x: x.priority, reverse=True)
    top_suggestions = all_suggestions[:5]
    
    print(f"\n  Top 3 suggestions:")
    for i, sug in enumerate(top_suggestions[:3]):
        print(f"    {i+1}. {sug.strategy}: PC1={sug.predicted_scores[0]:.2f}, "
              f"PC2={sug.predicted_scores[1]:.2f}, priority={sug.priority:.2f}")
    
    # Save spectrum PCA results
    print("\n[6/7] Saving spectrum PCA results...")
    save_spectrum_pca_results(output_dir, pca_result, interpretations, 
                              top_suggestions, experimental_params)
    
    # Generate plots
    print("\n[7/7] Generating visualizations...")
    plots_dir = output_dir / 'analysis_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        analyzer.plot_scree(pca_result, save_path=plots_dir / 'spectrum_pca_scree.png')
        print("  [OK] Scree plot")
    except Exception as e:
        print(f"  [WARN] Scree plot failed: {e}")
    
    try:
        pH_values = [p if p is not None else 0 for p in experimental_params['pH']]
        analyzer.plot_scores(pca_result, pc_x=1, pc_y=2, color_by=pH_values,
                           save_path=plots_dir / 'spectrum_pca_scores.png')
        print("  [OK] Scores plot")
    except Exception as e:
        print(f"  [WARN] Scores plot failed: {e}")
    
    try:
        analyzer.plot_loadings(pca_result, components=[1, 2, 3],
                             save_path=plots_dir / 'spectrum_pca_loadings.png')
        print("  [OK] Loadings plot")
    except Exception as e:
        print(f"  [WARN] Loadings plot failed: {e}")
    
    try:
        planner.plot_conditions_overlay(
            pca_result,
            experimental_params=experimental_params,
            pc_x=1, pc_y=2,
            figsize=(16, 10),
            save_path=plots_dir / 'conditions_overlay.png'
        )
        print("  [OK] Conditions overlay plot")
    except Exception as e:
        print(f"  [WARN] Conditions overlay plot failed: {e}")
    
    try:
        planner.plot_experiment_planning(
            pca_result,
            experimental_params=experimental_params,
            suggestions=top_suggestions,
            pc_x=1, pc_y=2,
            color_by='pH',
            save_path=plots_dir / 'experiment_planning.png'
        )
        print("  [OK] Experiment planning plot")
    except Exception as e:
        print(f"  [WARN] Experiment planning plot failed: {e}")

    # Optional VAE embedding on spectra
    if args.vae:
        if XASVAEEmbedding is None:
            print("  [WARN] VAE requested but torch or module unavailable.")
        else:
            try:
                print("\n[Optional] Running VAE embedding on spectra...")
                spectra_matrix = np.array(spectra, dtype=float)
                vae = XASVAEEmbedding()
                vae_res = vae.fit_transform(spectra_matrix)
                results_dir = output_dir / 'analysis_results'
                results_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(results_dir / 'vae_embedding.csv', vae_res.embedding, delimiter=',')

                # Plot latent space (first 2 dims)
                if vae_res.embedding.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(7, 6))
                    color_vals = experimental_params.get('pH') if experimental_params else None
                    if color_vals is not None:
                        try:
                            colors = np.array(color_vals, dtype=float)
                            sc = ax.scatter(vae_res.embedding[:, 0], vae_res.embedding[:, 1],
                                            c=colors, cmap='viridis', s=80, edgecolors='black')
                            plt.colorbar(sc, ax=ax, label='pH')
                        except (ValueError, TypeError):
                            ax.scatter(vae_res.embedding[:, 0], vae_res.embedding[:, 1],
                                       s=80, edgecolors='black')
                    else:
                        ax.scatter(vae_res.embedding[:, 0], vae_res.embedding[:, 1],
                                   s=80, edgecolors='black')
                    ax.set_xlabel("VAE Dim 1")
                    ax.set_ylabel("VAE Dim 2")
                    ax.set_title("VAE Embedding (Spectra)")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(plots_dir / 'vae_embedding.png', dpi=300)
                    plt.close(fig)
                print("  [OK] VAE embedding saved")
            except Exception as e:
                print(f"  [WARN] VAE embedding failed: {e}")


def extract_experimental_params_from_names(sample_names):
    """Extract experimental parameters from sample names."""
    experimental_params = {
        'pH': [],
        'iron_source': [],
        'ligand': [],
        'ligand_concentration': [],
        'state': []
    }
    
    for name in sample_names:
        # Iron source
        if 'FeCl2' in name:
            experimental_params['iron_source'].append('FeCl2')
        elif 'FeSO4' in name:
            experimental_params['iron_source'].append('FeSO4')
        else:
            experimental_params['iron_source'].append('Unknown')
        
        # Ligand
        if 'Malic' in name:
            experimental_params['ligand'].append('Malic_acid')
        elif 'Tartaric' in name:
            experimental_params['ligand'].append('Tartaric_acid')
        else:
            experimental_params['ligand'].append('Unknown')
        
        # pH
        ph_match = re.search(r'pH(\d+(?:[_\.]\d+)?)', name)
        if ph_match:
            ph_str = ph_match.group(1).replace('_', '.')
            try:
                experimental_params['pH'].append(float(ph_str))
            except ValueError:
                experimental_params['pH'].append(None)
        else:
            experimental_params['pH'].append(None)
        
        # Ligand concentration
        conc_match = re.search(r'\((\d+(?:[_\.]\d+)?)-(\d+(?:[_\.]\d+)?)\)', name)
        if conc_match:
            conc_str = conc_match.group(2).replace('_', '.')
            try:
                experimental_params['ligand_concentration'].append(float(conc_str))
            except:
                experimental_params['ligand_concentration'].append(None)
        else:
            experimental_params['ligand_concentration'].append(None)
        
        # State
        if 'gel' in name.lower():
            experimental_params['state'].append('gel')
        else:
            experimental_params['state'].append('solution')
    
    return experimental_params


def save_spectrum_pca_results(output_dir, pca_result, interpretations, 
                               suggestions, experimental_params):
    """Save spectrum PCA results."""
    results_dir = output_dir / 'spectrum_pca_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # PCA scores
    df_scores = pd.DataFrame(
        pca_result.scores,
        columns=[f'PC{i+1}' for i in range(pca_result.n_components)],
        index=pca_result.sample_names
    )
    df_scores.to_csv(results_dir / 'spectrum_pca_scores.csv')
    
    # PCA loadings
    df_loadings = pd.DataFrame(
        pca_result.loadings.T,
        columns=[f'PC{i+1}' for i in range(pca_result.n_components)]
    )
    df_loadings['energy'] = pca_result.energy_grid
    df_loadings.to_csv(results_dir / 'spectrum_pca_loadings.csv', index=False)
    
    # Component interpretations
    interp_data = []
    for interp in interpretations:
        interp_data.append({
            'pc_number': interp.pc_number,
            'variance_explained': interp.variance_explained,
            'interpretation': interp.interpretation,
            'peak_energies': interp.peak_energies,
            'peak_regions': interp.peak_regions
        })
    
    with open(results_dir / 'component_interpretations.json', 'w') as f:
        json.dump(interp_data, f, indent=2)
    
    # Experiment suggestions
    sug_data = []
    for sug in suggestions:
        sug_data.append({
            'strategy': sug.strategy,
            'predicted_scores': sug.predicted_scores.tolist(),
            'distance_to_nearest': sug.distance_to_nearest,
            'priority': sug.priority,
            'reason': sug.reason
        })
    
    with open(results_dir / 'experiment_suggestions.json', 'w') as f:
        json.dump(sug_data, f, indent=2)
    
    # Experimental parameters
    df_params = pd.DataFrame(experimental_params, index=pca_result.sample_names)
    df_params.to_csv(results_dir / 'experimental_parameters.csv')
    
    print(f"  Saved to: {results_dir}")


def main_old():
    """Original main function (kept for backwards compatibility)."""
    print("=" * 80)
    print("XAS ML ANALYSIS - EXTRACTED FEATURES (LEGACY)")
    print("=" * 80)
    project_root = Path(__file__).parent.parent.parent / "project_root"
    output_dir = project_root / "xas_results" / "04_ml_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = argparse.Namespace(
        n_components=5,
        n_clusters=4,
        tsne=False,
        umap=False,
        features_dir=None,
        filtered_dir=None
    )
    run_feature_based_ml(project_root, output_dir, args)


if __name__ == "__main__":
    main()
