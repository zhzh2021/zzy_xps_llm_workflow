"""
Test ML Analysis on Extracted XAS Features

Runs complete ML workflow:
1. Load features from JSON files
2. Assemble batch dataset
3. PCA analysis
4. Clustering
5. Trend analysis

Author: XAS Workflow Team
Date: March 6, 2026
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add paths - need both Tools/APS_XAS and its parent for relative imports
tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))

# Import ML modules directly (avoiding __init__.py)
import xas_ml_modules.xas_pca_analyzer as pca_module
import xas_ml_modules.xas_clusterer as cluster_module
import xas_ml_modules.xas_trend_analyzer as trend_module

XASPCAAnalyzer = pca_module.XASPCAAnalyzer
XASClusterer = cluster_module.XASClusterer
XASTrendAnalyzer = trend_module.XASTrendAnalyzer


def load_features_from_json(json_dir: Path):
    """
    Load XAS features from individual JSON files.
    
    Args:
        json_dir: Directory containing feature JSON files
        
    Returns:
        List of feature dictionaries
    """
    json_files = sorted(json_dir.glob("*_features.json"))
    
    features_list = []
    sample_names = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            features = json.load(f)
            features_list.append(features)
            sample_names.append(features['sample_name'])
    
    return features_list, sample_names


def create_feature_matrix(features_list):
    """
    Convert list of feature dicts to numpy matrix.
    
    Args:
        features_list: List of feature dictionaries
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    # Define feature order (exclude sample_name and null features)
    feature_names = [
        'e0', 'edge_step', 'edge_slope', 'pre_edge_area',
        'white_line_intensity', 'white_line_energy', 'white_line_fwhm',
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
                value = 0.0
            row.append(float(value))
        matrix.append(row)
    
    return np.array(matrix), feature_names


def parse_metadata_from_name(sample_name: str):
    """
    Extract experimental metadata from sample name.
    
    Example: FeCl2-Malic_acid_(0_5-0_5)-pH2_R1
    """
    import re
    
    metadata = {}
    
    # Extract iron source (anion)
    if 'FeCl2' in sample_name:
        metadata['iron_source'] = 'FeCl2'
        metadata['anion_type'] = 0  # Label encoding
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
    
    # Extract ligand (organic acid)
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
    ph_match = re.search(r'pH(\d+)', sample_name)
    if ph_match:
        metadata['pH'] = int(ph_match.group(1))
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
    
    # Extract concentrations (0.1, 0.5, 1)
    conc_match = re.search(r'\((\d+(?:_\d+)?)-(\d+(?:_\d+)?)\)', sample_name)
    if conc_match:
        # Convert 0_1 to 0.1, 0_5 to 0.5, etc.
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
    """
    Create metadata dictionary from sample names.
    
    Args:
        sample_names: List of sample names
        
    Returns:
        Dict with metadata columns
    """
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


def save_results(output_dir: Path, dataset, pca_results, cluster_results, trend_results):
    """Save all ML analysis results."""
    
    # 1. Dataset summary
    dataset_info = {
        'dataset_id': dataset.dataset_id,
        'timestamp': datetime.now().isoformat(),
        'n_samples': dataset.n_samples,
        'n_features': dataset.n_features,
        'sample_names': dataset.sample_names,
        'feature_names': dataset.feature_names
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 2. Feature matrix as CSV
    df_features = pd.DataFrame(
        dataset.feature_matrix,
        columns=dataset.feature_names,
        index=dataset.sample_names
    )
    df_features.to_csv(output_dir / 'feature_matrix.csv')
    
    # 3. Metadata as CSV
    if dataset.metadata_dict:
        df_metadata = pd.DataFrame(dataset.metadata_dict, index=dataset.sample_names)
        df_metadata.to_csv(output_dir / 'metadata.csv')
    
    # 4. PCA results
    pca_data = {
        'n_components': pca_results.n_components,
        'variance_explained': pca_results.variance_explained.tolist(),
        'variance_captured': pca_results.variance_captured,
        'loadings': pca_results.loadings.tolist(),
        'scores': pca_results.scores.tolist(),
        'feature_names': dataset.feature_names
    }
    
    with open(output_dir / 'pca_results.json', 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    # PCA scores as CSV
    df_pca = pd.DataFrame(
        pca_results.scores,
        columns=[f'PC{i+1}' for i in range(pca_results.n_components)],
        index=dataset.sample_names
    )
    df_pca.to_csv(output_dir / 'pca_scores.csv')
    
    # PCA loadings as CSV
    df_loadings = pd.DataFrame(
        pca_results.loadings,
        columns=[f'PC{i+1}' for i in range(pca_results.n_components)],
        index=dataset.feature_names
    )
    df_loadings.to_csv(output_dir / 'pca_loadings.csv')
    
    # 5. Clustering results
    cluster_data = {
        'n_clusters': cluster_results.n_clusters,
        'labels': cluster_results.labels.tolist(),
        'silhouette_score': cluster_results.silhouette_score,
        'davies_bouldin_score': cluster_results.davies_bouldin_score,
        'calinski_harabasz_score': cluster_results.calinski_harabasz_score,
        'method': cluster_results.method
    }
    
    with open(output_dir / 'cluster_results.json', 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    # Cluster assignments as CSV
    df_clusters = pd.DataFrame({
        'sample_name': dataset.sample_names,
        'cluster': cluster_results.labels
    })
    df_clusters.to_csv(output_dir / 'cluster_assignments.csv', index=False)
    
    # 6. Trend analysis results
    if hasattr(trend_results, 'correlations') and trend_results.correlations is not None:
        # Save correlation matrix
        if hasattr(trend_results, 'correlation_matrix'):
            df_corr = pd.DataFrame(
                trend_results.correlation_matrix,
                columns=list(dataset.metadata_dict.keys()) if dataset.metadata_dict else [],
                index=dataset.feature_names
            )
            df_corr.to_csv(output_dir / 'feature_metadata_correlations.csv')
    
    # Save outliers if detected
    if hasattr(trend_results, 'outlier_indices') and trend_results.outlier_indices:
        outlier_names = [dataset.sample_names[i] for i in trend_results.outlier_indices]
        with open(output_dir / 'outliers.json', 'w') as f:
            json.dump({
                'outlier_indices': trend_results.outlier_indices,
                'outlier_names': outlier_names
            }, f, indent=2)
    
    print(f"\n[OK] Results saved to: {output_dir}")
    print(f"  - dataset_info.json")
    print(f"  - feature_matrix.csv ({dataset.n_samples} x {dataset.n_features})")
    print(f"  - pca_results.json ({pca_results.n_components} components)")
    print(f"  - pca_scores.csv")
    print(f"  - pca_loadings.csv")
    print(f"  - cluster_results.json ({cluster_results.n_clusters} clusters)")
    print(f"  - cluster_assignments.csv")


def main():
    """Run complete ML analysis workflow."""
    
    print("=" * 80)
    print("XAS ML ANALYSIS - EXTRACTED FEATURES")
    print("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent / "project_root"
    features_dir = project_root / "xas_results" / "03_feature_extraction"
    output_dir = project_root / "xas_results" / "04_ml_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load features
    print("\n[1/5] Loading extracted features...")
    features_list, sample_names = load_features_from_json(features_dir)
    print(f"  Loaded {len(features_list)} samples")
    
    if len(features_list) < 2:
        print("\n[FAIL] Need at least 2 samples for ML analysis")
        return
    
    # Step 2: Create dataset
    print("\n[2/5] Creating feature matrix...")
    feature_matrix, feature_names = create_feature_matrix(features_list)
    print(f"  Shape: {feature_matrix.shape}")
    print(f"  Features: {len(feature_names)}")
    
    # Extract metadata from sample names
    metadata_dict = create_metadata_dict(sample_names)
    print(f"  Metadata columns: {len(metadata_dict)}")
    
    # Create dataset object (mimicking XASDataset)
    class SimpleDataset:
        def __init__(self, feature_matrix, feature_names, sample_names, metadata_dict):
            self.feature_matrix = feature_matrix
            self.feature_names = feature_names
            self.sample_names = sample_names
            self.metadata_dict = metadata_dict
            self.n_samples = len(sample_names)
            self.n_features = len(feature_names)
            self.dataset_id = f"xas_030525_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.quality_flags = None
            self.creation_timestamp = datetime.now()
    
    dataset = SimpleDataset(feature_matrix, feature_names, sample_names, metadata_dict)
    
    # Step 3: PCA Analysis
    print("\n[3/5] Running PCA...")
    try:
        pca_analyzer = XASPCAAnalyzer()
        pca_results = pca_analyzer.analyze(dataset, n_components=5)
        print(f"  Components: {pca_results.n_components}")
        print(f"  Variance explained by each PC:")
        for i, var in enumerate(pca_results.variance_explained):
            print(f"    PC{i+1}: {var*100:.1f}%")
        print(f"  Total variance captured: {pca_results.variance_captured*100:.1f}%")
    except Exception as e:
        print(f"  [FAIL] PCA error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Clustering
    print("\n[4/5] Running clustering...")
    try:
        clusterer = XASClusterer()
        cluster_results = clusterer.cluster(
            dataset,
            n_clusters=4,  # Can be auto-determined
            use_pca_scores=pca_results.scores
        )
        print(f"  Clusters: {cluster_results.n_clusters}")
        print(f"  Silhouette score: {cluster_results.silhouette_score:.3f}")
        print(f"  Davies-Bouldin index: {cluster_results.davies_bouldin_score:.3f}")
        print(f"  Calinski-Harabasz score: {cluster_results.calinski_harabasz_score:.1f}")
        
        # Show cluster distribution
        unique, counts = np.unique(cluster_results.labels, return_counts=True)
        print(f"  Cluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"    Cluster {cluster_id}: {count} samples")
    except Exception as e:
        print(f"  [FAIL] Clustering error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Trend Analysis
    print("\n[5/5] Running trend analysis...")
    try:
        trend_analyzer = XASTrendAnalyzer()
        trend_results = trend_analyzer.analyze(dataset, cluster_results)
        
        n_outliers = len(trend_results.outlier_indices) if hasattr(trend_results, 'outlier_indices') else 0
        print(f"  Outliers detected: {n_outliers}")
        
        if hasattr(trend_results, 'significant_correlations'):
            print(f"  Significant correlations: {len(trend_results.significant_correlations)}")
        
    except Exception as e:
        print(f"  [WARNING] Trend analysis error: {e}")
        # Continue even if trend analysis fails
        trend_results = None
    
    # Step 6: Save results
    print("\n[6/6] Saving results...")
    save_results(output_dir, dataset, pca_results, cluster_results, trend_results)
    
    # Summary
    print("\n" + "=" * 80)
    print("ML ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nDataset: {dataset.n_samples} samples x {dataset.n_features} features")
    print(f"PCA: {pca_results.n_components} components, {pca_results.variance_captured*100:.1f}% variance")
    print(f"Clustering: {cluster_results.n_clusters} clusters, silhouette={cluster_results.silhouette_score:.3f}")
    print(f"\nOutput: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
