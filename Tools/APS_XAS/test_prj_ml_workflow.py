"""
Test ML workflow with Athena .prj file data.

This demonstrates using pre-normalized data from Athena project files
directly in the ML analysis pipeline.
"""
import sys
import os
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prj_reader import read_prj_file
from xas_analyzer.xas_feature_extractor import XASFeatureExtractor
from xas_ml_modules.xas_batch_assembler import XASBatchAssembler
from xas_ml_modules.xas_pca_analyzer import XASPCAAnalyzer
from xas_ml_modules.xas_clusterer import XASClusterer
from xas_ml_modules.xas_trend_analyzer import XASTrendAnalyzer
from xas_plotter.xas_rawdata_plotter import XASPlotter

# Additional ML/stats imports
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.stats import f_oneway


def parse_sample_metadata(sample_name: str) -> dict:
    """
    Extract experimental metadata from sample name.
    
    Example: FeCl2_Tartaric_Acid_0.1M_0.1M_pH2
    Returns: {anion, ligand, anion_conc, ligand_conc, pH}
    """
    import re
    
    metadata = {
        'sample_id': sample_name,
        'source': 'athena_prj',
        'edge': 'Fe K',
    }
    
    # Extract anion (iron source)
    if 'FeCl2' in sample_name:
        metadata['anion'] = 'FeCl2'
        metadata['iron_source'] = 'FeCl2'
        metadata['chloride'] = True
    elif 'FeSO4' in sample_name:
        metadata['anion'] = 'FeSO4'
        metadata['iron_source'] = 'FeSO4'
        metadata['sulfate'] = True
    
    # Extract ligand (organic acid)
    if 'Malic' in sample_name or 'MA' in sample_name:
        metadata['ligand'] = 'malic_acid'
        metadata['organic_acid'] = 'malic_acid'
    elif 'Tataric' in sample_name or 'Tartaric' in sample_name or 'TA' in sample_name:
        metadata['ligand'] = 'tartaric_acid'
        metadata['organic_acid'] = 'tartaric_acid'
    
    # Extract concentrations (e.g., 0.1M, 0.5, 1)
    # Look for patterns like (0.1M-0.1M) or (0.5-0.5) or _0_5_0_5_
    conc_pattern1 = r'\((\d+\.?\d*)[M\-]+(\d+\.?\d*)[M\)]'  # (0.1M-0.1M) or (0.5-0.5)
    conc_pattern2 = r'_(\d+)_(\d+)_(?:pH|\d)'  # _0_5_0_5_pH or _1_0_5_4
    
    conc_match = re.search(conc_pattern1, sample_name)
    if conc_match:
        try:
            metadata['anion_concentration'] = float(conc_match.group(1))
            metadata['ligand_concentration'] = float(conc_match.group(2))
            if metadata['anion_concentration'] > 0:
                metadata['concentration_ratio'] = metadata['ligand_concentration'] / metadata['anion_concentration']
        except ValueError:
            pass
    else:
        # Try underscore-separated pattern
        # For FeCl2_MA_0_5_0_5_2_A: extract 0.5 and 0.5
        parts = sample_name.split('_')
        numbers = []
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:  # Skip first number if it exists
                # Check if this might be a concentration (not pH value)
                if i < len(parts) - 1 and not parts[i-1].startswith('pH'):
                    numbers.append(int(part))
        
        # Heuristic: first two numbers after MA/TA are concentrations
        if len(numbers) >= 2:
            # Convert e.g., 0_5 to 0.5, or 1_0 to 1.0
            try:
                if numbers[0] == 0:
                    metadata['anion_concentration'] = float(f"{numbers[0]}.{numbers[1]}")
                    if len(numbers) >= 4 and numbers[2] == 0:
                        metadata['ligand_concentration'] = float(f"{numbers[2]}.{numbers[3]}")
                    elif len(numbers) >= 3:
                        metadata['ligand_concentration'] = float(numbers[2])
                else:
                    metadata['anion_concentration'] = float(numbers[0])
                    if len(numbers) >= 3 and numbers[1] == 0:
                        metadata['ligand_concentration'] = float(f"{numbers[1]}.{numbers[2]}")
                    elif len(numbers) >= 2:
                        metadata['ligand_concentration'] = float(numbers[1])
                
                if 'anion_concentration' in metadata and 'ligand_concentration' in metadata:
                    if metadata['anion_concentration'] > 0:
                        metadata['concentration_ratio'] = metadata['ligand_concentration'] / metadata['anion_concentration']
            except (ValueError, IndexError):
                pass
    
    # Extract pH
    ph_match = re.search(r'pH(\d+\.?\d*)', sample_name)
    if ph_match:
        try:
            metadata['pH'] = float(ph_match.group(1))
        except ValueError:
            pass
    
    # Extract gel/solution state
    if 'gel' in sample_name.lower():
        metadata['state'] = 'gel'
    else:
        metadata['state'] = 'solution'
    
    # Extract replicate info (A, B, D markers)
    replicate_match = re.search(r'_([A-Z])$', sample_name)
    if replicate_match:
        metadata['replicate'] = replicate_match.group(1)
    
    return metadata


def create_sample_wrapper(name: str, energy: np.ndarray, mu_norm: np.ndarray):
    """Create a minimal sample object for feature extraction."""
    
    class SampleWrapper:
        def __init__(self, sample_name, energy_arr, mu_arr):
            self.sample_name = sample_name
            self.energy = energy_arr
            self.normalized_mu = mu_arr
            self.features = None
            
            # Extract E0 (edge position - max derivative)
            deriv = np.gradient(mu_arr, energy_arr)
            e0_idx = np.argmax(deriv)
            self.e0 = float(energy_arr[e0_idx])
            
            # Extract experimental metadata from sample name
            self.user_metadata = parse_sample_metadata(sample_name)
    
    return SampleWrapper(name, energy, mu_norm)


def save_results(output_dir, samples, batch_data, pca_results, cluster_results, trend_results):
    """Save all analysis results to JSON and CSV files."""
    
    # 0. XANES Feature Interpretation Guide
    feature_interpretation = {
        'e0': {
            'physical_meaning': 'Edge energy position (eV)',
            'structural_info': 'Fe oxidation state (Fe2+/Fe3+)',
            'electrolyte_context': 'Redox state of iron in solution/gel - higher energy indicates more oxidized Fe'
        },
        'edge_step': {
            'physical_meaning': 'Magnitude of absorption edge jump',
            'structural_info': 'Concentration of Fe atoms',
            'electrolyte_context': 'Iron content and sample homogeneity'
        },
        'white_line_intensity': {
            'physical_meaning': 'Intensity of white line peak (1s→4p transition)',
            'structural_info': 'Density of unoccupied 3d states',
            'electrolyte_context': 'Fe coordination environment strength - higher for stronger Fe-O/Fe-ligand bonding'
        },
        'white_line_energy': {
            'physical_meaning': 'Energy of white line maximum (eV)',
            'structural_info': 'Ligand field splitting energy',
            'electrolyte_context': 'Type and strength of coordinating ligands (carboxylate vs halide vs water)'
        },
        'white_line_fwhm': {
            'physical_meaning': 'Full-width at half-maximum of white line (eV)',
            'structural_info': 'Distribution of coordination environments',
            'electrolyte_context': 'Structural disorder - broader peaks indicate heterogeneous Fe coordination shells'
        },
        'pre_edge_area': {
            'physical_meaning': 'Area of pre-edge peak (1s→3d quadrupole transition)',
            'structural_info': 'Local symmetry and 3d-4p orbital mixing',
            'electrolyte_context': 'Coordination geometry - larger area for distorted/tetrahedral vs ideal octahedral Fe'
        },
        'xanes_centroid': {
            'physical_meaning': 'Center of mass of XANES region (eV)',
            'structural_info': 'Overall electronic structure',
            'electrolyte_context': 'Average oxidation state and ligand field effects in Fe complexes'
        },
        'xanes_area': {
            'physical_meaning': 'Total integrated intensity in XANES region',
            'structural_info': 'Total absorption cross-section',
            'electrolyte_context': 'Overall Fe-ligand interaction strength and coordination number'
        },
        'edge_slope': {
            'physical_meaning': 'Steepness of edge rise (normalized units/eV)',
            'structural_info': 'Degree of structural order',
            'electrolyte_context': 'Crystallinity - sharper for ordered complexes, gradual for amorphous/disordered'
        },
        'first_derivative_max': {
            'physical_meaning': 'Inflection point energy of edge (eV)',
            'structural_info': 'True edge position',
            'electrolyte_context': 'Fe oxidation state determination (same as e0, more precise)'
        },
        'spectral_mean': {
            'physical_meaning': 'Average energy of XANES spectrum (eV)',
            'structural_info': 'Overall absorption profile center',
            'electrolyte_context': 'Weighted average of all Fe electronic transitions'
        },
        'spectral_std': {
            'physical_meaning': 'Standard deviation of XANES spectrum (eV)',
            'structural_info': 'Spread of absorption features',
            'electrolyte_context': 'Diversity of Fe coordination states and bonding environments'
        },
        'spectral_skewness': {
            'physical_meaning': 'Asymmetry of spectral distribution',
            'structural_info': 'Shape of absorption envelope',
            'electrolyte_context': 'Presence of secondary Fe species or multiple coordination geometries'
        },
        'spectral_kurtosis': {
            'physical_meaning': 'Tailedness/peakedness of spectral distribution',
            'structural_info': 'Sharpness vs broadness of features',
            'electrolyte_context': 'Uniformity of Fe sites - high = uniform, low = heterogeneous mixtures'
        },
        'post_edge_oscillations': {
            'physical_meaning': 'EXAFS-like oscillations beyond edge',
            'structural_info': 'Extended coordination shells',
            'electrolyte_context': 'Second/third shell ligands and Fe-Fe distances (oligomerization/precipitation)'
        }
    }
    
    # Save interpretation guide as JSON
    import json
    with open(output_dir / "xanes_feature_interpretation_guide.json", 'w') as f:
        json.dump(feature_interpretation, f, indent=2)
    print(f"      ✓ Saved: xanes_feature_interpretation_guide.json")
    
    # Create human-readable interpretation table
    interp_table = []
    for feature, info in feature_interpretation.items():
        if feature in batch_data.feature_names:  # Only include features actually extracted
            interp_table.append({
                'Feature': feature,
                'Physical_Meaning': info['physical_meaning'],
                'Structural_Info': info['structural_info'],
                'Electrolyte_Context': info['electrolyte_context']
            })
    
    interp_df = pd.DataFrame(interp_table)
    interp_df.to_csv(output_dir / "feature_interpretation_guide.csv", index=False)
    print(f"      ✓ Saved: feature_interpretation_guide.csv")
    
    # 1. Sample metadata
    metadata_list = []
    for i, sample in enumerate(samples):
        meta = sample.user_metadata.copy()
        meta['sample_index'] = i
        meta['cluster'] = int(cluster_results.labels[i]) if cluster_results.labels is not None else -1
        metadata_list.append(meta)
    
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(output_dir / "sample_metadata.csv", index=False)
    print(f"      ✓ Saved: sample_metadata.csv")
    
    # 2. Feature matrix
    feature_df = pd.DataFrame(
        batch_data.feature_matrix,
        columns=batch_data.feature_names,
        index=[s.sample_name for s in samples]
    )
    feature_df.to_csv(output_dir / "feature_matrix.csv")
    print(f"      ✓ Saved: feature_matrix.csv")
    
    # 3. PCA results
    pca_dict = {
        'n_components': pca_results.n_components,
        'variance_explained': pca_results.explained_variance,
        'cumulative_variance': pca_results.cumulative_variance,
        'total_variance_captured': pca_results.variance_captured,
        'feature_importance': pca_results.feature_importance
    }
    with open(output_dir / "pca_analysis.json", 'w') as f:
        json.dump(pca_dict, f, indent=2)
    print(f"      ✓ Saved: pca_analysis.json")
    
    # PCA scores
    if pca_results.scores is not None:
        scores_df = pd.DataFrame(
            pca_results.scores,
            columns=[f'PC{i+1}' for i in range(pca_results.n_components)],
            index=[s.sample_name for s in samples]
        )
        scores_df.to_csv(output_dir / "pca_scores.csv")
        print(f"      ✓ Saved: pca_scores.csv")
    
    # 4. Clustering results
    cluster_dict = {
        'method': cluster_results.method,
        'n_clusters': cluster_results.n_clusters,
        'silhouette_score': cluster_results.silhouette_score,
        'labels': cluster_results.labels.tolist() if cluster_results.labels is not None else []
    }
    with open(output_dir / "clustering_results.json", 'w') as f:
        json.dump(cluster_dict, f, indent=2)
    print(f"      ✓ Saved: clustering_results.json")
    
    # Cluster assignments
    if cluster_results.labels is not None:
        cluster_df = pd.DataFrame({
            'sample_name': [s.sample_name for s in samples],
            'cluster': cluster_results.labels
        })
        cluster_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
        print(f"      ✓ Saved: cluster_assignments.csv")
    
    # 5. Trend analysis results
    if hasattr(trend_results, 'significant_correlations') and trend_results.significant_correlations:
        corr_df = pd.DataFrame(trend_results.significant_correlations)
        corr_df.to_csv(output_dir / "feature_metadata_correlations.csv", index=False)
        print(f"      ✓ Saved: feature_metadata_correlations.csv")
        
        # Add structural interpretation to correlations
        corr_interp = []
        for _, row in corr_df.iterrows():
            feature = row.get('feature', 'unknown')
            condition = row.get('metadata', 'unknown')
            r_val = row.get('correlation', 0)
            
            # Get feature interpretation
            feature_info = feature_interpretation.get(feature, {})
            struct_meaning = feature_info.get('electrolyte_context', 'Unknown structural meaning')
            
            # Interpret correlation direction
            if r_val > 0:
                trend = f"increases with higher {condition}"
                implication = f"Higher {condition} → more {struct_meaning.lower()}"
            else:
                trend = f"decreases with higher {condition}"
                implication = f"Higher {condition} → less {struct_meaning.lower()}"
            
            corr_interp.append({
                'Feature': feature,
                'Condition': condition,
                'Correlation': round(r_val, 3),
                'P_value': row.get('p_value', 1.0),
                'Trend': trend,
                'Structural_Implication': implication,
                'Feature_Represents': struct_meaning
            })
        
        interp_corr_df = pd.DataFrame(corr_interp)
        # Filter to experimental conditions only (exclude e0 self-correlation)
        exp_conditions = ['pH', 'anion_concentration', 'ligand_concentration', 'anion_ligand_ratio']
        interp_corr_df_filtered = interp_corr_df[interp_corr_df['Condition'].isin(exp_conditions)]
        
        if not interp_corr_df_filtered.empty:
            interp_corr_df_filtered.to_csv(output_dir / "correlation_structural_interpretation.csv", index=False)
            print(f"      ✓ Saved: correlation_structural_interpretation.csv")

    
    if hasattr(trend_results, 'outlier_indices') and trend_results.outlier_indices:
        outlier_df = pd.DataFrame({
            'sample_index': trend_results.outlier_indices,
            'sample_name': [samples[i].sample_name for i in trend_results.outlier_indices if i < len(samples)]
        })
        outlier_df.to_csv(output_dir / "outlier_detection.csv", index=False)
        print(f"      ✓ Saved: outlier_detection.csv")
    
    # 6. Summary report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        'analysis_timestamp': timestamp,
        'dataset': {
            'n_samples': len(samples),
            'n_features': batch_data.feature_matrix.shape[1],
            'feature_names': batch_data.feature_names
        },
        'pca': {
            'n_components': pca_results.n_components,
            'variance_captured': pca_results.variance_captured,
            'kaiser_criterion': pca_results.kaiser_criterion
        },
        'clustering': {
            'method': cluster_results.method,
            'n_clusters': cluster_results.n_clusters,
            'silhouette_score': cluster_results.silhouette_score
        },
        'trends': {
            'n_correlations': len(trend_results.significant_correlations) if hasattr(trend_results, 'significant_correlations') else 0,
            'n_outliers': len(trend_results.outlier_indices) if hasattr(trend_results, 'outlier_indices') else 0
        }
    }
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"      ✓ Saved: analysis_summary.json")


def generate_plots(output_dir, samples, batch_data, pca_results, cluster_results, trend_results):
    """Generate all visualization plots."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # 0. Feature Matrix Visualization (Quality Check)
    print(f"      📊 Creating feature extraction quality check plots...")
    
    # 0a. Feature Heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Normalize features for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(batch_data.feature_matrix)
    
    im = ax.imshow(normalized_features.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_yticks(range(len(batch_data.feature_names)))
    ax.set_yticklabels(batch_data.feature_names, fontsize=13)
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([f"S{i+1}" for i in range(len(samples))], fontsize=11, rotation=90)
    
    ax.set_xlabel('Sample Index', fontsize=16, fontweight='bold')
    ax.set_ylabel('XANES Feature', fontsize=16, fontweight='bold')
    ax.set_title('Feature Matrix Heatmap (Normalized)', fontsize=18, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value (Z-score)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_matrix_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: feature_matrix_heatmap.png")
    
    # 0b. Feature Distribution Box Plots (Raw Values)
    n_features = len(batch_data.feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*3.5))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(batch_data.feature_names):
        ax = axes[idx]
        feat_values = batch_data.feature_matrix[:, idx]
        
        bp = ax.boxplot([feat_values], vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][0].set_alpha(0.7)
        
        # Overlay actual data points
        x_pos = np.random.normal(1, 0.04, size=len(feat_values))
        ax.scatter(x_pos, feat_values, alpha=0.6, s=40, c='darkblue', edgecolors='black', linewidths=0.5)
        
        ax.set_ylabel(feat_name, fontsize=13, fontweight='bold')
        ax.set_title(f'{feat_name}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.tick_params(labelsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(feat_values)
        std_val = np.std(feat_values)
        ax.text(0.98, 0.98, f'μ={mean_val:.2e}\nσ={std_val:.2e}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Extracted XANES Features - Distribution Check', fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions_raw.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: feature_distributions_raw.png")
    
    # 0c. Feature Values by Sample (Line plots for key features)
    key_features = ['e0', 'edge_slope', 'white_line_intensity', 'pre_edge_area', 
                   'xanes_centroid', 'post_edge_slope']
    key_features = [f for f in key_features if f in batch_data.feature_names]
    
    if len(key_features) >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        sample_indices = np.arange(len(samples))
        
        for idx, feat_name in enumerate(key_features[:6]):
            ax = axes[idx]
            feat_idx = batch_data.feature_names.index(feat_name)
            feat_values = batch_data.feature_matrix[:, feat_idx]
            
            ax.plot(sample_indices, feat_values, 'o-', linewidth=2, markersize=8, 
                   color='steelblue', markeredgecolor='black', markeredgewidth=1)
            
            ax.set_xlabel('Sample Index', fontsize=15, fontweight='bold')
            ax.set_ylabel(feat_name, fontsize=15, fontweight='bold')
            ax.set_title(f'{feat_name} Across Samples', fontsize=16, fontweight='bold')
            ax.tick_params(labelsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(sample_indices[::2])
            
            # Highlight outliers (beyond 2 std)
            mean = np.mean(feat_values)
            std = np.std(feat_values)
            outliers = np.abs(feat_values - mean) > 2 * std
            if np.any(outliers):
                ax.scatter(sample_indices[outliers], feat_values[outliers], 
                          s=200, c='red', marker='*', edgecolors='black', 
                          linewidths=1.5, zorder=5, label='Outlier (>2σ)')
                ax.legend(fontsize=12)
        
        plt.suptitle('Key XANES Features vs Sample Index', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_vs_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: feature_vs_samples.png")
    
    # 1. PCA Variance Explained (Scree Plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(range(1, len(pca_results.explained_variance)+1), pca_results.explained_variance)
    ax1.set_xlabel('Principal Component', fontsize=16)
    ax1.set_ylabel('Variance Explained', fontsize=16)
    ax1.set_title('PCA Scree Plot', fontsize=18)
    ax1.set_xticks(range(1, len(pca_results.explained_variance)+1))
    ax1.tick_params(labelsize=14)
    
    ax2.plot(range(1, len(pca_results.cumulative_variance)+1), pca_results.cumulative_variance, 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=16)
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=16)
    ax2.set_title('Cumulative Variance', fontsize=18)
    ax2.legend(fontsize=15)
    ax2.tick_params(labelsize=14)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_variance_explained.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: pca_variance_explained.png")
    
    # 2. PCA Scores Plot (PC1 vs PC2) - IMPROVED with Chemical Grouping
    if pca_results.scores is not None:
        # Extract categorical variables for coloring
        anion_types = []
        ligand_types = []
        for sample in samples:
            meta = sample.user_metadata
            anion = meta.get('anion', '').lower()
            ligand = meta.get('ligand', '').lower()
            
            # Anion type
            if 'fecl2' in anion or 'fecl' in anion:
                anion_types.append('FeCl₂')
            elif 'feso4' in anion or 'feso' in anion:
                anion_types.append('FeSO₄')
            else:
                anion_types.append('Unknown')
            
            # Ligand type
            if 'malic' in ligand:
                ligand_types.append('Malic')
            elif 'tartar' in ligand:
                ligand_types.append('Tartaric')
            else:
                ligand_types.append('Unknown')
        
        # Create 2x2 grid: Original + Anion + Ligand + Combined
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Top-left: Original K-means clusters
        if cluster_results.labels is not None:
            # Plot each cluster separately with labels for legend
            n_clusters = cluster_results.n_clusters
            cluster_colors = plt.cm.tab10(np.arange(n_clusters))
            
            for cluster_id in range(n_clusters):
                mask = cluster_results.labels == cluster_id
                axes[0, 0].scatter(
                    pca_results.scores[mask, 0], 
                    pca_results.scores[mask, 1],
                    c=[cluster_colors[cluster_id]],
                    label=f'Cluster {cluster_id}',
                    s=150, 
                    alpha=0.6, 
                    edgecolors='black', 
                    linewidth=1.5
                )
            
            axes[0, 0].legend(fontsize=16, loc='best')
        else:
            axes[0, 0].scatter(pca_results.scores[:, 0], pca_results.scores[:, 1], 
                              c='blue', s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        axes[0, 0].set_xlabel(f'PC1 ({pca_results.explained_variance[0]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[0, 0].set_ylabel(f'PC2 ({pca_results.explained_variance[1]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[0, 0].set_title('K-Means Clustering (Unsupervised)', fontsize=20, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(labelsize=16)
        
        # Top-right: Colored by Anion Type
        anion_color_map = {'FeCl₂': '#2E86AB', 'FeSO₄': '#A23B72', 'Unknown': '#888888'}
        anion_marker_map = {'FeCl₂': 'o', 'FeSO₄': 's', 'Unknown': 'x'}
        for anion_type in set(anion_types):
            mask = np.array([a == anion_type for a in anion_types])
            axes[0, 1].scatter(
                pca_results.scores[mask, 0],
                pca_results.scores[mask, 1],
                c=anion_color_map[anion_type],
                marker=anion_marker_map[anion_type],
                label=anion_type,
                s=180,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )
        axes[0, 1].set_xlabel(f'PC1 ({pca_results.explained_variance[0]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[0, 1].set_ylabel(f'PC2 ({pca_results.explained_variance[1]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[0, 1].set_title('Grouped by Anion Type', fontsize=20, fontweight='bold')
        axes[0, 1].legend(fontsize=17, loc='best')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(labelsize=16)
        
        # Bottom-left: Colored by Ligand Type
        ligand_color_map = {'Malic': '#F77F00', 'Tartaric': '#06A77D', 'Unknown': '#888888'}
        ligand_marker_map = {'Malic': '^', 'Tartaric': 'v', 'Unknown': 'x'}
        for ligand_type in set(ligand_types):
            mask = np.array([l == ligand_type for l in ligand_types])
            axes[1, 0].scatter(
                pca_results.scores[mask, 0],
                pca_results.scores[mask, 1],
                c=ligand_color_map[ligand_type],
                marker=ligand_marker_map[ligand_type],
                label=ligand_type,
                s=180,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )
        axes[1, 0].set_xlabel(f'PC1 ({pca_results.explained_variance[0]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[1, 0].set_ylabel(f'PC2 ({pca_results.explained_variance[1]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[1, 0].set_title('Grouped by Ligand Type', fontsize=20, fontweight='bold')
        axes[1, 0].legend(fontsize=17, loc='best')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(labelsize=16)
        
        # Bottom-right: Combined Chemical Groups
        combined_groups = []
        for anion, ligand in zip(anion_types, ligand_types):
            combined_groups.append(f"{anion}+{ligand}")
        
        unique_groups = sorted(set(combined_groups))
        group_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        group_color_dict = dict(zip(unique_groups, group_colors))
        
        for group in unique_groups:
            mask = np.array([g == group for g in combined_groups])
            axes[1, 1].scatter(
                pca_results.scores[mask, 0],
                pca_results.scores[mask, 1],
                c=[group_color_dict[group]],
                label=group.replace('+', ' + '),
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.2
            )
        axes[1, 1].set_xlabel(f'PC1 ({pca_results.explained_variance[0]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[1, 1].set_ylabel(f'PC2 ({pca_results.explained_variance[1]*100:.1f}%)', fontsize=18, fontweight='bold')
        axes[1, 1].set_title('Chemical Groups (Anion + Ligand)', fontsize=20, fontweight='bold')
        axes[1, 1].legend(fontsize=15, loc='best', ncol=2)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(labelsize=16)
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_scores_pc1_pc2.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: pca_scores_pc1_pc2.png (4-panel classification)")
    
    # 3. PCA Loadings (Top features for PC1 and PC2)
    if pca_results.feature_importance:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PC1 loadings
        pc1_features = pca_results.feature_importance.get('PC1', [])[:10]
        if pc1_features:
            features = [f['feature'] for f in pc1_features]
            loadings = [f['loading'] for f in pc1_features]
            ax1.barh(features, loadings, height=0.7)
            ax1.set_xlabel('Loading', fontsize=16)
            ax1.set_title('PC1 Feature Loadings (Top 10)', fontsize=18)
            ax1.tick_params(labelsize=14)
            ax1.invert_yaxis()
        
        # PC2 loadings
        pc2_features = pca_results.feature_importance.get('PC2', [])[:10]
        if pc2_features:
            features = [f['feature'] for f in pc2_features]
            loadings = [f['loading'] for f in pc2_features]
            ax2.barh(features, loadings, height=0.7)
            ax2.set_xlabel('Loading', fontsize=16)
            ax2.set_title('PC2 Feature Loadings (Top 10)', fontsize=18)
            ax2.tick_params(labelsize=14)
            ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_loadings.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: pca_loadings.png")
    
    # 4. Hierarchical Clustering Dendrogram
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    
    # Use PCA scores for hierarchical clustering
    if pca_results.scores is not None:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Compute linkage matrix
        linkage_matrix = linkage(pca_results.scores[:, :3], method='ward')
        
        # Create dendrogram with anion type colors
        sample_labels = []
        for i, sample in enumerate(samples):
            anion = sample.user_metadata.get('anion', '').lower()
            if 'fecl' in anion:
                prefix = 'FeCl₂'
            elif 'feso' in anion:
                prefix = 'FeSO₄'
            else:
                prefix = 'Unk'
            sample_labels.append(f"{prefix}-{i+1}")
        
        dend = dendrogram(
            linkage_matrix,
            labels=sample_labels,
            ax=ax,
            leaf_font_size=15,
            leaf_rotation=90,
            color_threshold=0.7*max(linkage_matrix[:, 2])
        )
        
        ax.set_xlabel('Sample ID', fontsize=18, fontweight='bold')
        ax.set_ylabel('Distance (Ward)', fontsize=18, fontweight='bold')
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=15)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "hierarchical_dendrogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: hierarchical_dendrogram.png")
    
    # 5. Feature Distribution Box Plots by Chemical Groups
    if pca_results.scores is not None and batch_data.feature_matrix is not None:
        # Select top 6 most important features from PC1
        top_features_idx = []
        if pca_results.feature_importance and 'PC1' in pca_results.feature_importance:
            pc1_features = pca_results.feature_importance['PC1'][:6]
            for feat_dict in pc1_features:
                feat_name = feat_dict['feature']
                if feat_name in batch_data.feature_names:
                    top_features_idx.append(batch_data.feature_names.index(feat_name))
        
        if len(top_features_idx) >= 4:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()
            
            # Prepare data
            anion_types = []
            ligand_types = []
            for sample in samples:
                meta = sample.user_metadata
                anion = meta.get('anion', '').lower()
                ligand = meta.get('ligand', '').lower()
                
                if 'fecl' in anion:
                    anion_types.append('FeCl₂')
                elif 'feso' in anion:
                    anion_types.append('FeSO₄')
                else:
                    anion_types.append('Unknown')
                
                if 'malic' in ligand:
                    ligand_types.append('Malic')
                elif 'tartar' in ligand:
                    ligand_types.append('Tartaric')
                else:
                    ligand_types.append('Unknown')
            
            combined_groups = [f"{a}+{l}" for a, l in zip(anion_types, ligand_types)]
            
            for plot_idx, feat_idx in enumerate(top_features_idx):
                if plot_idx >= 6:
                    break
                
                ax = axes[plot_idx]
                feat_name = batch_data.feature_names[feat_idx]
                feat_values = batch_data.feature_matrix[:, feat_idx]
                
                # Create box plot data
                unique_groups = sorted(set(combined_groups))
                data_by_group = []
                for group in unique_groups:
                    group_values = [feat_values[i] for i, g in enumerate(combined_groups) if g == group]
                    data_by_group.append(group_values)
                
                bp = ax.boxplot(data_by_group, labels=[g.replace('+', '\n+\n') for g in unique_groups],
                               patch_artist=True, widths=0.6)
                
                # Color boxes
                colors_box = ['#2E86AB', '#06A77D', '#A23B72', '#F77F00']
                for patch, color in zip(bp['boxes'], colors_box[:len(unique_groups)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                ax.set_ylabel(feat_name, fontsize=16, fontweight='bold')
                ax.set_title(f'{feat_name} Distribution', fontsize=17, fontweight='bold')
                ax.tick_params(labelsize=15)
                ax.grid(axis='y', alpha=0.3)
            
            plt.suptitle('Key XANES Features by Chemical Group', fontsize=20, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(output_dir / "feature_distributions_boxplot.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ Saved: feature_distributions_boxplot.png")
    
    # 6. t-SNE Visualization (Non-linear dimensionality reduction)
    if pca_results.scores is not None:
        print(f"      🔬 Running t-SNE analysis...")
        
        # Run t-SNE on PCA scores for better performance
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(samples)-1), 
                   init='pca', learning_rate='auto')
        tsne_coords = tsne.fit_transform(pca_results.scores)
        
        # Extract categorical variables
        anion_types = []
        ligand_types = []
        for sample in samples:
            meta = sample.user_metadata
            anion = meta.get('anion', '').lower()
            ligand = meta.get('ligand', '').lower()
            
            if 'fecl' in anion:
                anion_types.append('FeCl₂')
            elif 'feso' in anion:
                anion_types.append('FeSO₄')
            else:
                anion_types.append('Unknown')
            
            if 'malic' in ligand:
                ligand_types.append('Malic')
            elif 'tartar' in ligand:
                ligand_types.append('Tartaric')
            else:
                ligand_types.append('Unknown')
        
        combined_groups = [f"{a}+{l}" for a, l in zip(anion_types, ligand_types)]
        
        # Create 2x2 t-SNE plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: By K-means clusters
        if cluster_results.labels is not None:
            # Plot each cluster separately with labels for legend
            n_clusters = cluster_results.n_clusters
            cluster_colors = plt.cm.tab10(np.arange(n_clusters))
            
            for cluster_id in range(n_clusters):
                mask = cluster_results.labels == cluster_id
                axes[0, 0].scatter(
                    tsne_coords[mask, 0], 
                    tsne_coords[mask, 1],
                    c=[cluster_colors[cluster_id]],
                    label=f'Cluster {cluster_id}',
                    s=150, 
                    alpha=0.6, 
                    edgecolors='black', 
                    linewidth=1.5
                )
            
            axes[0, 0].legend(fontsize=16, loc='best')
        else:
            axes[0, 0].scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                              c='blue', s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        axes[0, 0].set_xlabel('t-SNE 1', fontsize=18, fontweight='bold')
        axes[0, 0].set_ylabel('t-SNE 2', fontsize=18, fontweight='bold')
        axes[0, 0].set_title('t-SNE: K-Means Clusters', fontsize=20, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(labelsize=16)
        
        # Plot 2: By Anion Type
        anion_color_map = {'FeCl₂': '#2E86AB', 'FeSO₄': '#A23B72', 'Unknown': '#888888'}
        anion_marker_map = {'FeCl₂': 'o', 'FeSO₄': 's', 'Unknown': 'x'}
        for anion_type in set(anion_types):
            mask = np.array([a == anion_type for a in anion_types])
            axes[0, 1].scatter(
                tsne_coords[mask, 0],
                tsne_coords[mask, 1],
                c=anion_color_map[anion_type],
                marker=anion_marker_map[anion_type],
                label=anion_type,
                s=180,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )
        axes[0, 1].set_xlabel('t-SNE 1', fontsize=18, fontweight='bold')
        axes[0, 1].set_ylabel('t-SNE 2', fontsize=18, fontweight='bold')
        axes[0, 1].set_title('t-SNE: By Anion Type', fontsize=20, fontweight='bold')
        axes[0, 1].legend(fontsize=17, loc='best')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(labelsize=16)
        
        # Plot 3: By Ligand Type
        ligand_color_map = {'Malic': '#F77F00', 'Tartaric': '#06A77D', 'Unknown': '#888888'}
        ligand_marker_map = {'Malic': '^', 'Tartaric': 'v', 'Unknown': 'x'}
        for ligand_type in set(ligand_types):
            mask = np.array([l == ligand_type for l in ligand_types])
            axes[1, 0].scatter(
                tsne_coords[mask, 0],
                tsne_coords[mask, 1],
                c=ligand_color_map[ligand_type],
                marker=ligand_marker_map[ligand_type],
                label=ligand_type,
                s=180,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )
        axes[1, 0].set_xlabel('t-SNE 1', fontsize=18, fontweight='bold')
        axes[1, 0].set_ylabel('t-SNE 2', fontsize=18, fontweight='bold')
        axes[1, 0].set_title('t-SNE: By Ligand Type', fontsize=20, fontweight='bold')
        axes[1, 0].legend(fontsize=17, loc='best')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(labelsize=16)
        
        # Plot 4: Combined Chemical Groups
        unique_groups = sorted(set(combined_groups))
        group_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        group_color_dict = dict(zip(unique_groups, group_colors))
        
        for group in unique_groups:
            mask = np.array([g == group for g in combined_groups])
            axes[1, 1].scatter(
                tsne_coords[mask, 0],
                tsne_coords[mask, 1],
                c=[group_color_dict[group]],
                label=group.replace('+', ' + '),
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidths=1.2
            )
        axes[1, 1].set_xlabel('t-SNE 1', fontsize=18, fontweight='bold')
        axes[1, 1].set_ylabel('t-SNE 2', fontsize=18, fontweight='bold')
        axes[1, 1].set_title('t-SNE: Chemical Groups', fontsize=20, fontweight='bold')
        axes[1, 1].legend(fontsize=15, loc='best', ncol=2)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(labelsize=16)
        
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_classification.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: tsne_classification.png")
    
    # 7. ANOVA Statistical Testing
    if batch_data.feature_matrix is not None:
        print(f"      📊 Running ANOVA tests...")
        
        # Extract groups
        anion_types = []
        ligand_types = []
        combined_groups = []
        for sample in samples:
            meta = sample.user_metadata
            anion = meta.get('anion', '').lower()
            ligand = meta.get('ligand', '').lower()
            
            if 'fecl' in anion:
                anion_types.append('FeCl2')
            elif 'feso' in anion:
                anion_types.append('FeSO4')
            else:
                anion_types.append('Unknown')
            
            if 'malic' in ligand:
                ligand_types.append('Malic')
            elif 'tartar' in ligand:
                ligand_types.append('Tartaric')
            else:
                ligand_types.append('Unknown')
            
            combined_groups.append(f"{anion_types[-1]}+{ligand_types[-1]}")
        
        # Perform ANOVA for each feature across combined groups
        anova_results = []
        unique_groups = sorted(set(combined_groups))
        
        for feat_idx, feat_name in enumerate(batch_data.feature_names):
            feat_values = batch_data.feature_matrix[:, feat_idx]
            
            # Group data
            group_data = []
            for group in unique_groups:
                group_values = [feat_values[i] for i, g in enumerate(combined_groups) if g == group]
                group_data.append(group_values)
            
            # Run ANOVA
            f_stat, p_val = f_oneway(*group_data)
            
            # Calculate effect size (eta-squared)
            grand_mean = np.mean(feat_values)
            ss_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in group_data])
            ss_total = sum((feat_values - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            anova_results.append({
                'Feature': feat_name,
                'F_statistic': round(f_stat, 4),
                'p_value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No',
                'eta_squared': round(eta_squared, 4),
                'Effect_size': 'Large' if eta_squared > 0.14 else ('Medium' if eta_squared > 0.06 else 'Small')
            })
        
        # Save ANOVA results
        anova_df = pd.DataFrame(anova_results).sort_values('p_value')
        anova_df.to_csv(output_dir / "anova_results.csv", index=False)
        print(f"      ✓ Saved: anova_results.csv")
        
        # Create ANOVA visualization
        sig_features = anova_df[anova_df['Significant'] == 'Yes'].head(10)
        
        if not sig_features.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            
            # Plot 1: F-statistics
            ax1.barh(sig_features['Feature'], sig_features['F_statistic'], 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
            ax1.set_xlabel('F-statistic', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Feature', fontsize=16, fontweight='bold')
            ax1.set_title('ANOVA F-Statistics (Top 10 Features)', fontsize=18, fontweight='bold')
            ax1.tick_params(labelsize=14)
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
            
            # Plot 2: Effect sizes
            colors = ['#d73027' if es == 'Large' else ('#fc8d59' if es == 'Medium' else '#fee090') 
                     for es in sig_features['Effect_size']]
            ax2.barh(sig_features['Feature'], sig_features['eta_squared'], 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
            ax2.set_xlabel('Effect Size (η²)', fontsize=16, fontweight='bold')
            ax2.set_ylabel('Feature', fontsize=16, fontweight='bold')
            ax2.set_title('Effect Sizes by Feature', fontsize=18, fontweight='bold')
            ax2.tick_params(labelsize=14)
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
            
            # Add legend for effect sizes
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d73027', edgecolor='black', label='Large (η² > 0.14)'),
                Patch(facecolor='#fc8d59', edgecolor='black', label='Medium (0.06 < η² < 0.14)'),
                Patch(facecolor='#fee090', edgecolor='black', label='Small (η² < 0.06)')
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=13)
            
            plt.tight_layout()
            plt.savefig(output_dir / "anova_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ Saved: anova_visualization.png")
            
            # Create Structure-Condition Linkage Plot
            print(f"      🔗 Creating feature → structure → condition linkage...")
            
            # Feature structural interpretation mapping
            feature_structure_map = {
                'e0': 'Fe Oxidation State',
                'edge_slope': 'Coordination Order/Disorder',
                'white_line_intensity': 'Fe-Ligand Covalency',
                'pre_edge_area': 'Coordination Geometry',
                'xanes_centroid': 'Average Electronic State',
                'post_edge_slope': 'Extended Coordination',
                'second_derivative_zero': 'Edge Shape/Curvature',
                'edge_step': 'Fe Concentration',
                'white_line_energy': 'Ligand Field Strength',
                'white_line_fwhm': 'Site Heterogeneity'
            }
            
            # Get top 6 significant features with their effect sizes and correlations
            top_features = sig_features.head(6)
            
            # Create comprehensive linkage table
            linkage_data = []
            for _, row in top_features.iterrows():
                feat = row['Feature']
                
                # Get structural meaning
                struct_meaning = feature_structure_map.get(feat, 'Unknown')
                
                # Get which conditions affect this feature (from correlation analysis)
                feat_correlations = []
                if hasattr(trend_results, 'significant_correlations') and trend_results.significant_correlations:
                    for corr in trend_results.significant_correlations:
                        if corr.get('feature') == feat:
                            condition = corr.get('metadata', '')
                            if condition not in ['e0']:  # Exclude self-correlation
                                r_val = corr.get('correlation', 0)
                                feat_correlations.append({
                                    'condition': condition,
                                    'r': r_val,
                                    'direction': 'increases' if r_val > 0 else 'decreases'
                                })
                
                # Sort by absolute correlation
                feat_correlations.sort(key=lambda x: abs(x['r']), reverse=True)
                
                # Get top affecting condition
                if feat_correlations:
                    top_cond = feat_correlations[0]
                    condition_effect = f"{top_cond['condition']} (r={top_cond['r']:.2f}, {top_cond['direction']})"
                else:
                    condition_effect = "No significant correlations"
                
                linkage_data.append({
                    'XANES_Feature': feat,
                    'Structure_Information': struct_meaning,
                    'ANOVA_Effect_Size': f"{row['eta_squared']:.3f} ({row['Effect_size']})",
                    'Primary_Condition_Effect': condition_effect,
                    'F_statistic': row['F_statistic'],
                    'p_value': row['p_value']
                })
            
            linkage_df = pd.DataFrame(linkage_data)
            linkage_df.to_csv(output_dir / "feature_structure_condition_linkage.csv", index=False)
            print(f"      ✓ Saved: feature_structure_condition_linkage.csv")
            
            # Create visual linkage diagram
            fig, ax = plt.subplots(figsize=(18, 10))
            
            # Prepare data for visualization
            y_positions = np.arange(len(linkage_data))
            features = [d['XANES_Feature'] for d in linkage_data]
            structures = [d['Structure_Information'] for d in linkage_data]
            effect_sizes = [float(d['ANOVA_Effect_Size'].split()[0]) for d in linkage_data]
            
            # Create horizontal bars colored by effect size
            colors_map = []
            for eta in effect_sizes:
                if eta > 0.14:
                    colors_map.append('#d73027')  # Red for large
                elif eta > 0.06:
                    colors_map.append('#fc8d59')  # Orange for medium
                else:
                    colors_map.append('#fee090')  # Yellow for small
            
            bars = ax.barh(y_positions, effect_sizes, color=colors_map, alpha=0.7, 
                          edgecolor='black', linewidth=1.5, height=0.6)
            
            # Add feature names on the left
            for i, (feat, struct) in enumerate(zip(features, structures)):
                ax.text(-0.02, i, feat, ha='right', va='center', fontsize=13, fontweight='bold')
                ax.text(effect_sizes[i] + 0.02, i, f'→ {struct}', ha='left', va='center', 
                       fontsize=12, style='italic', color='darkblue')
            
            ax.set_yticks([])
            ax.set_xlabel('Effect Size (η²) - Chemical Group Discrimination', fontsize=16, fontweight='bold')
            ax.set_title('XANES Features → Fe Structure Information\n(How much does chemistry affect each structural property?)', 
                        fontsize=18, fontweight='bold', pad=20)
            ax.set_xlim(-0.15, max(effect_sizes) * 1.5)
            ax.grid(axis='x', alpha=0.3)
            ax.tick_params(labelsize=14)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#d73027', edgecolor='black', label='Large Effect (η² > 0.14)'),
                Patch(facecolor='#fc8d59', edgecolor='black', label='Medium Effect (0.06 < η² < 0.14)'),
                Patch(facecolor='#fee090', edgecolor='black', label='Small Effect (η² < 0.06)')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=13, framealpha=0.9)
            
            # Add interpretive text box
            textstr = ('Interpretation:\n'
                      '• Larger bars = feature is more sensitive to chemical group changes\n'
                      '• These structural properties are most affected by anion/ligand choice\n'
                      '• Use these features as quality control metrics for electrolyte synthesis')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(output_dir / "feature_structure_linkage.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ Saved: feature_structure_linkage.png")
            
            # Create Condition-to-Structure Impact Network
            print(f"      🌐 Creating condition → structure impact network...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
            
            # Left panel: Categorical condition effects (Anion and Ligand Type)
            categorical_impacts = []
            
            # Check for categorical condition correlations
            if hasattr(trend_results, 'significant_correlations') and trend_results.significant_correlations:
                for corr in trend_results.significant_correlations:
                    feat = corr.get('feature', '')
                    meta = corr.get('metadata', '')
                    r_val = corr.get('correlation', 0)
                    
                    if meta in ['anion_type', 'ligand_type'] and feat in feature_structure_map:
                        categorical_impacts.append({
                            'Condition': meta.replace('_', ' ').title(),
                            'Structure': feature_structure_map[feat],
                            'Feature': feat,
                            'Correlation': r_val,
                            'Direction': '↑ Increases' if r_val > 0 else '↓ Decreases'
                        })
            
            if categorical_impacts:
                cat_df = pd.DataFrame(categorical_impacts)
                
                # Group by condition
                for idx, condition in enumerate(['Anion Type', 'Ligand Type']):
                    subset = cat_df[cat_df['Condition'] == condition]
                    
                    if len(subset) > 0:
                        y_pos = np.arange(len(subset))
                        colors = ['green' if r > 0 else 'red' for r in subset['Correlation'].values]
                        
                        ax1.barh(y_pos + idx * (len(subset) + 1), [abs(r) for r in subset['Correlation'].values], 
                                color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
                        
                        # Add labels
                        for i, (_, row) in enumerate(subset.iterrows()):
                            y = i + idx * (len(subset) + 1)
                            label = f"{row['Structure']}\n({row['Feature']})"
                            ax1.text(-0.01, y, label, ha='right', va='center', fontsize=11)
                            
                            direction_text = row['Direction']
                            ax1.text(abs(row['Correlation']) + 0.01, y, direction_text, 
                                    ha='left', va='center', fontsize=10, 
                                    color='darkgreen' if row['Correlation'] > 0 else 'darkred',
                                    fontweight='bold')
                        
                        # Add condition label
                        mid_y = (len(subset) - 1) / 2 + idx * (len(subset) + 1)
                        ax1.text(0.5, mid_y, condition, ha='center', va='center',
                                fontsize=14, fontweight='bold', 
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                ax1.set_yticks([])
                ax1.set_xlabel('|Correlation Coefficient|', fontsize=15, fontweight='bold')
                ax1.set_title('Chemical Type Effects on Fe Structure\n(FeCl₂ vs FeSO₄, Malic vs Tartaric Acid)',
                             fontsize=16, fontweight='bold', pad=15)
                ax1.grid(axis='x', alpha=0.3)
                ax1.axvline(0, color='black', linewidth=1.5)
                ax1.set_xlim(-0.2, 0.7)
            else:
                ax1.text(0.5, 0.5, 'No significant categorical\ncondition correlations found\n(p > 0.05)',
                        ha='center', va='center', fontsize=14, transform=ax1.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                ax1.set_xticks([])
                ax1.set_yticks([])
            
            # Right panel: Continuous condition effects (pH, Concentrations)
            continuous_impacts = []
            
            if hasattr(trend_results, 'significant_correlations') and trend_results.significant_correlations:
                for corr in trend_results.significant_correlations:
                    feat = corr.get('feature', '')
                    meta = corr.get('metadata', '')
                    r_val = corr.get('correlation', 0)
                    
                    if meta in ['pH', 'anion_concentration', 'ligand_concentration', 'anion_ligand_ratio'] \
                       and feat in feature_structure_map:
                        continuous_impacts.append({
                            'Condition': meta.replace('_', ' ').title(),
                            'Structure': feature_structure_map[feat],
                            'Feature': feat,
                            'Correlation': r_val
                        })
            
            if continuous_impacts:
                cont_df = pd.DataFrame(continuous_impacts)
                cont_df = cont_df.sort_values('Correlation', key=abs, ascending=False)
                
                # Take top 10
                cont_df = cont_df.head(10)
                
                y_pos = np.arange(len(cont_df))
                colors = ['green' if r > 0 else 'red' for r in cont_df['Correlation'].values]
                
                ax2.barh(y_pos, [abs(r) for r in cont_df['Correlation'].values], color=colors, alpha=0.6,
                        edgecolor='black', linewidth=1.5)
                
                # Add labels
                for i, (_, row) in enumerate(cont_df.iterrows()):
                    label = f"{row['Condition']} → {row['Structure']}"
                    ax2.text(-0.01, i, label, ha='right', va='center', fontsize=11)
                    
                    direction = '↑' if row['Correlation'] > 0 else '↓'
                    ax2.text(abs(row['Correlation']) + 0.01, i, 
                            f"{direction} r={row['Correlation']:.2f}", 
                            ha='left', va='center', fontsize=10,
                            color='darkgreen' if row['Correlation'] > 0 else 'darkred',
                            fontweight='bold')
                
                ax2.set_yticks([])
                ax2.set_xlabel('|Correlation Coefficient|', fontsize=15, fontweight='bold')
                ax2.set_title('Solution Parameter Effects on Fe Structure\n(pH, Concentrations, Ratio)',
                             fontsize=16, fontweight='bold', pad=15)
                ax2.grid(axis='x', alpha=0.3)
                ax2.axvline(0, color='black', linewidth=1.5)
            else:
                ax2.text(0.5, 0.5, 'No significant continuous\ncondition correlations found',
                        ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                ax2.set_xticks([])
                ax2.set_yticks([])
            
            plt.suptitle('Experiment Design → Fe Electronic/Coordination Structure', 
                        fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(output_dir / "condition_structure_impact.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ Saved: condition_structure_impact.png")
    
    # 8. Random Forest Feature Importance
    if batch_data.feature_matrix is not None:
        print(f"      🌲 Running Random Forest feature importance...")
        
        # Extract categorical labels
        anion_labels = []
        ligand_labels = []
        for sample in samples:
            meta = sample.user_metadata
            anion = meta.get('anion', '').lower()
            ligand = meta.get('ligand', '').lower()
            
            if 'fecl' in anion:
                anion_labels.append(0)  # FeCl2
            elif 'feso' in anion:
                anion_labels.append(1)  # FeSO4
            else:
                anion_labels.append(-1)
            
            if 'malic' in ligand:
                ligand_labels.append(0)  # Malic
            elif 'tartar' in ligand:
                ligand_labels.append(1)  # Tartaric
            else:
                ligand_labels.append(-1)
        
        anion_labels = np.array(anion_labels)
        ligand_labels = np.array(ligand_labels)
        
        # Train Random Forest for Anion Type
        valid_anion = anion_labels >= 0
        if np.sum(valid_anion) > 5:
            rf_anion = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            rf_anion.fit(batch_data.feature_matrix[valid_anion], anion_labels[valid_anion])
            anion_importance = rf_anion.feature_importances_
            anion_score = rf_anion.score(batch_data.feature_matrix[valid_anion], anion_labels[valid_anion])
        else:
            anion_importance = np.zeros(len(batch_data.feature_names))
            anion_score = 0
        
        # Train Random Forest for Ligand Type
        valid_ligand = ligand_labels >= 0
        if np.sum(valid_ligand) > 5:
            rf_ligand = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            rf_ligand.fit(batch_data.feature_matrix[valid_ligand], ligand_labels[valid_ligand])
            ligand_importance = rf_ligand.feature_importances_
            ligand_score = rf_ligand.score(batch_data.feature_matrix[valid_ligand], ligand_labels[valid_ligand])
        else:
            ligand_importance = np.zeros(len(batch_data.feature_names))
            ligand_score = 0
        
        # Save feature importance
        importance_data = []
        for i, feat_name in enumerate(batch_data.feature_names):
            importance_data.append({
                'Feature': feat_name,
                'Anion_Importance': round(anion_importance[i], 4),
                'Ligand_Importance': round(ligand_importance[i], 4),
                'Combined_Importance': round((anion_importance[i] + ligand_importance[i]) / 2, 4)
            })
        
        importance_df = pd.DataFrame(importance_data).sort_values('Combined_Importance', ascending=False)
        importance_df.to_csv(output_dir / "rf_feature_importance.csv", index=False)
        print(f"      ✓ Saved: rf_feature_importance.csv")
        
        # Visualize feature importance
        top_features = importance_df.head(10)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Anion Type Classification
        ax1.barh(top_features['Feature'], top_features['Anion_Importance'], 
                color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax1.set_xlabel('Feature Importance', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Feature', fontsize=16, fontweight='bold')
        ax1.set_title(f'RF Importance: Anion Type (Accuracy={anion_score:.2%})', 
                     fontsize=18, fontweight='bold')
        ax1.tick_params(labelsize=14)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Plot 2: Ligand Type Classification
        ax2.barh(top_features['Feature'], top_features['Ligand_Importance'], 
                color='#F77F00', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax2.set_xlabel('Feature Importance', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Feature', fontsize=16, fontweight='bold')
        ax2.set_title(f'RF Importance: Ligand Type (Accuracy={ligand_score:.2%})', 
                     fontsize=18, fontweight='bold')
        ax2.tick_params(labelsize=14)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / "rf_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: rf_feature_importance.png")
        
        print(f"      ✓ Saved: pca_loadings.png")
    
    # 4. Cluster Assignments
    if cluster_results.labels is not None:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        cluster_counts = pd.Series(cluster_results.labels).value_counts().sort_index()
        bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                     color=plt.cm.tab10(range(len(cluster_counts))), 
                     width=0.6, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Cluster ID', fontsize=16)
        ax.set_ylabel('Number of Samples', fontsize=16)
        ax.set_title(f'Cluster Distribution ({cluster_results.n_clusters} clusters, silhouette={cluster_results.silhouette_score:.3f})', 
                    fontsize=18)
        ax.set_xticks(cluster_counts.index)
        ax.tick_params(labelsize=14)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / "cluster_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: cluster_distribution.png")
    
    # 5. Feature-Metadata Correlations
    if hasattr(trend_results, 'significant_correlations') and trend_results.significant_correlations:
        corr_df = pd.DataFrame(trend_results.significant_correlations)
        top_corr = corr_df.head(15) if not corr_df.empty else corr_df
        
        if not top_corr.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            labels = [f"{row.get('feature', 'N/A')} ↔ {row.get('metadata', 'N/A')}" 
                     for _, row in top_corr.iterrows()]
            values = top_corr.get('correlation', [0]*len(top_corr)).values
            
            colors = ['green' if v > 0 else 'red' for v in values]
            bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.7, height=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=13)
            ax.set_xlabel('Correlation Coefficient', fontsize=16)
            ax.set_title('Top Feature-Metadata Correlations', fontsize=18)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.tick_params(axis='x', labelsize=14)
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(output_dir / "feature_metadata_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ Saved: feature_metadata_correlations.png")
    
    # 6. Sample metadata distribution
    metadata_df = pd.DataFrame([s.user_metadata for s in samples])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # pH distribution
    if 'pH' in metadata_df.columns:
        metadata_df['pH'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], 
                                                            color='skyblue', width=0.6, edgecolor='black')
        axes[0, 0].set_title('pH Distribution', fontsize=18)
        axes[0, 0].set_xlabel('pH', fontsize=16)
        axes[0, 0].set_ylabel('Count', fontsize=16)
        axes[0, 0].tick_params(labelsize=14)
    
    # Anion distribution
    if 'anion' in metadata_df.columns:
        metadata_df['anion'].value_counts().plot(kind='bar', ax=axes[0, 1], 
                                                  color='lightcoral', width=0.6, edgecolor='black')
        axes[0, 1].set_title('Anion Distribution', fontsize=18)
        axes[0, 1].set_xlabel('Anion', fontsize=16)
        axes[0, 1].set_ylabel('Count', fontsize=16)
        axes[0, 1].tick_params(labelsize=14)
    
    # Ligand distribution
    if 'ligand' in metadata_df.columns:
        metadata_df['ligand'].value_counts().plot(kind='bar', ax=axes[1, 0], 
                                                   color='lightgreen', width=0.6, edgecolor='black')
        axes[1, 0].set_title('Ligand Distribution', fontsize=18)
        axes[1, 0].set_xlabel('Ligand', fontsize=16)
        axes[1, 0].set_ylabel('Count', fontsize=16)
        axes[1, 0].tick_params(labelsize=14)
    
    # State distribution
    if 'state' in metadata_df.columns:
        metadata_df['state'].value_counts().plot(kind='bar', ax=axes[1, 1], 
                                                  color='plum', width=0.6, edgecolor='black')
        axes[1, 1].set_title('Sample State Distribution', fontsize=18)
        axes[1, 1].set_xlabel('State', fontsize=16)
        axes[1, 1].set_ylabel('Count', fontsize=16)
        axes[1, 1].tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_metadata_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved: sample_metadata_distribution.png")
    
    # 7. Experimental Condition Impact Analysis
    if hasattr(trend_results, 'significant_correlations') and trend_results.significant_correlations:
        corr_df = pd.DataFrame(trend_results.significant_correlations)
        
        if not corr_df.empty and 'metadata' in corr_df.columns:
            # Calculate impact metrics for each metadata field
            impact_analysis = {}
            
            for metadata_field in corr_df['metadata'].unique():
                field_corrs = corr_df[corr_df['metadata'] == metadata_field]
                
                # Count significant correlations
                n_significant = len(field_corrs)
                
                # Average absolute correlation strength
                avg_correlation = field_corrs['correlation'].abs().mean()
                
                # Maximum correlation strength
                max_correlation = field_corrs['correlation'].abs().max()
                
                # Weighted impact score (combines frequency and strength)
                impact_score = n_significant * avg_correlation
                
                impact_analysis[metadata_field] = {
                    'count': n_significant,
                    'avg_abs_correlation': avg_correlation,
                    'max_abs_correlation': max_correlation,
                    'impact_score': impact_score
                }
            
            # Create impact DataFrame and sort by impact score
            impact_df = pd.DataFrame(impact_analysis).T
            impact_df = impact_df.sort_values('impact_score', ascending=True)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot 1: Impact Score (count × average correlation)
            colors_impact = plt.cm.RdYlGn(impact_df['impact_score'] / impact_df['impact_score'].max())
            bars1 = ax1.barh(impact_df.index, impact_df['impact_score'], 
                           color=colors_impact, height=0.65, edgecolor='black', linewidth=1.5)
            ax1.set_xlabel('Impact Score (Count × Avg |r|)', fontsize=16)
            ax1.set_ylabel('Experimental Condition', fontsize=16)
            ax1.set_title('Condition Impact on XANES Features', fontsize=18, fontweight='bold')
            ax1.tick_params(labelsize=14)
            
            # Add value labels
            for i, (idx, row) in enumerate(impact_df.iterrows()):
                ax1.text(row['impact_score'], i, f"  {row['impact_score']:.2f}", 
                        va='center', fontsize=13, fontweight='bold')
            
            # Plot 2: Correlation strength breakdown
            x = range(len(impact_df))
            width = 0.35
            
            bars_avg = ax2.barh([i - width/2 for i in x], impact_df['avg_abs_correlation'], 
                               width, label='Avg |Correlation|', color='steelblue', 
                               edgecolor='black', linewidth=1)
            bars_max = ax2.barh([i + width/2 for i in x], impact_df['max_abs_correlation'], 
                               width, label='Max |Correlation|', color='coral', 
                               edgecolor='black', linewidth=1)
            
            ax2.set_yticks(x)
            ax2.set_yticklabels(impact_df.index, fontsize=14)
            ax2.set_xlabel('Correlation Strength', fontsize=16)
            ax2.set_title('Correlation Strength by Condition', fontsize=18, fontweight='bold')
            ax2.legend(fontsize=14, loc='lower right')
            ax2.tick_params(labelsize=14)
            ax2.set_xlim(0, 1.0)
            
            plt.tight_layout()
            plt.savefig(output_dir / "condition_impact_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ Saved: condition_impact_analysis.png")
            
            # Save impact analysis to CSV
            impact_df['condition'] = impact_df.index
            impact_df = impact_df.reset_index(drop=True)
            impact_df = impact_df[['condition', 'count', 'avg_abs_correlation', 'max_abs_correlation', 'impact_score']]
            impact_df.columns = ['Condition', 'Num_Correlations', 'Avg_Abs_Correlation', 'Max_Abs_Correlation', 'Impact_Score']
            impact_df = impact_df.sort_values('Impact_Score', ascending=False)
            impact_df.to_csv(output_dir / "condition_impact_ranking.csv", index=False)
            print(f"      ✓ Saved: condition_impact_ranking.csv")


def generate_grouped_xanes_plots(output_dir, data, samples):
    """Generate grouped XANES plots by anion type (FeCl2 vs FeSO4)."""
    
    # Group samples by anion type
    fecl2_samples = {}
    feso4_samples = {}
    
    for sample_name, (energy, mu_norm) in data.items():
        if 'FeCl2' in sample_name or 'FeCl' in sample_name:
            fecl2_samples[sample_name] = {
                'processed_data': {
                    'energy': energy,
                    'mu_norm': mu_norm
                },
                'sample_name': sample_name
            }
        elif 'FeSO4' in sample_name or 'FeSO' in sample_name:
            feso4_samples[sample_name] = {
                'processed_data': {
                    'energy': energy,
                    'mu_norm': mu_norm
                },
                'sample_name': sample_name
            }
    
    print(f"      Grouped: {len(fecl2_samples)} FeCl2 samples, {len(feso4_samples)} FeSO4 samples")
    
    # Create plotter instance
    plotter = XASPlotter(figsize=(14, 10), dpi=300)
    
    # Plot FeCl2 group
    if fecl2_samples:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(fecl2_samples)))
        for i, (sample_name, result) in enumerate(fecl2_samples.items()):
            energy = result['processed_data']['energy']
            mu_norm = result['processed_data']['mu_norm']
            
            # Shorten label for legend
            label = sample_name.replace('FeCl2_', '').replace('Malic_acid', 'MA').replace('Tataric_acid', 'TA').replace('_', ' ')
            ax.plot(energy, mu_norm, color=colors[i], linewidth=2.5, alpha=0.8, label=label)
        
        ax.set_xlabel('Energy (eV)', fontsize=18)
        ax.set_ylabel('Normalized μ(E)', fontsize=18)
        ax.set_title(r'XANES Spectra - FeCl$_2$ + Organic Ligands', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=15)
        ax.grid(True, alpha=0.3, linewidth=1)
        ax.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.9)
        
        plt.tight_layout()
        save_path = output_dir / "xanes_group_FeCl2_ligands.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: xanes_group_FeCl2_ligands.png")
    
    # Plot FeSO4 group
    if feso4_samples:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(feso4_samples)))
        for i, (sample_name, result) in enumerate(feso4_samples.items()):
            energy = result['processed_data']['energy']
            mu_norm = result['processed_data']['mu_norm']
            
            # Shorten label for legend
            label = sample_name.replace('FeSO4_', '').replace('Malic_acid', 'MA').replace('Tataric_acid', 'TA').replace('_', ' ')
            ax.plot(energy, mu_norm, color=colors[i], linewidth=2.5, alpha=0.8, label=label)
        
        ax.set_xlabel('Energy (eV)', fontsize=18)
        ax.set_ylabel('Normalized μ(E)', fontsize=18)
        ax.set_title(r'XANES Spectra - FeSO$_4$ + Organic Ligands', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=15)
        ax.grid(True, alpha=0.3, linewidth=1)
        ax.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.9)
        
        plt.tight_layout()
        save_path = output_dir / "xanes_group_FeSO4_ligands.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: xanes_group_FeSO4_ligands.png")
    
    # Create side-by-side comparison
    if fecl2_samples and feso4_samples:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
        
        # FeCl2 panel
        colors1 = plt.cm.tab20(np.linspace(0, 1, len(fecl2_samples)))
        for i, (sample_name, result) in enumerate(fecl2_samples.items()):
            energy = result['processed_data']['energy']
            mu_norm = result['processed_data']['mu_norm']
            label = sample_name.replace('FeCl2_', '').replace('Malic_acid', 'MA').replace('Tataric_acid', 'TA').replace('_', ' ')[:30]
            ax1.plot(energy, mu_norm, color=colors1[i], linewidth=2, alpha=0.7, label=label)
        
        ax1.set_xlabel('Energy (eV)', fontsize=16)
        ax1.set_ylabel('Normalized μ(E)', fontsize=16)
        ax1.set_title(r'FeCl$_2$ + Ligands', fontsize=18, fontweight='bold')
        ax1.tick_params(labelsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
        
        # FeSO4 panel
        colors2 = plt.cm.tab20(np.linspace(0, 1, len(feso4_samples)))
        for i, (sample_name, result) in enumerate(feso4_samples.items()):
            energy = result['processed_data']['energy']
            mu_norm = result['processed_data']['mu_norm']
            label = sample_name.replace('FeSO4_', '').replace('Malic_acid', 'MA').replace('Tataric_acid', 'TA').replace('_', ' ')[:30]
            ax2.plot(energy, mu_norm, color=colors2[i], linewidth=2, alpha=0.7, label=label)
        
        ax2.set_xlabel('Energy (eV)', fontsize=16)
        ax2.set_ylabel('Normalized μ(E)', fontsize=16)
        ax2.set_title(r'FeSO$_4$ + Ligands', fontsize=18, fontweight='bold')
        ax2.tick_params(labelsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
        
        fig.suptitle('XANES Comparison: Effect of Anion Type on Fe K-edge', fontsize=20, fontweight='bold')
        plt.tight_layout()
        save_path = output_dir / "xanes_comparison_anion_types.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved: xanes_comparison_anion_types.png")


def main():
    """Run ML analysis on .prj file data."""
    
    print("="*80)
    print("XAS ML WORKFLOW - ATHENA PROJECT FILE")
    print("="*80)
    
    # Configuration
    prj_file = r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\Jiang samples-03032026.prj"
    output_dir = Path(r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_results\prj_ml_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Read .prj file
    print("\n[1/6] Reading Athena project file...")
    data = read_prj_file(prj_file, exclude_refs=True, exclude_smoothed=True)
    print(f"      Loaded {len(data)} samples")
    
    # Step 2: Extract features
    print("\n[2/6] Extracting XAS features...")
    feature_extractor = XASFeatureExtractor()
    
    samples = []
    for name, (energy, mu_norm) in data.items():
        try:
            sample = create_sample_wrapper(name, energy, mu_norm)
            features = feature_extractor.extract_features(sample)
            sample.features = features
            samples.append(sample)
            print(f"      ✓ {name}")
        except Exception as e:
            print(f"      ✗ {name}: {e}")
    
    print(f"      Successfully extracted features from {len(samples)}/{len(data)} samples")
    
    if len(samples) < 2:
        print("\n❌ Error: Need at least 2 samples for ML analysis")
        return
    
    # Step 3: Assemble batch
    print("\n[3/6] Assembling feature matrix...")
    batch_assembler = XASBatchAssembler()
    
    # Extract experimental metadata for correlation analysis
    experimental_metadata = {
        'pH': [],
        'anion_concentration': [],
        'ligand_concentration': [],
        'anion_ligand_ratio': [],
        'anion_type': [],      # Categorical: 0=FeCl2, 1=FeSO4
        'ligand_type': []      # Categorical: 0=malic_acid, 1=tartaric_acid
    }
    
    for sample in samples:
        metadata = sample.user_metadata
        # Extract numeric pH
        try:
            ph_val = float(metadata.get('pH', 0))
            experimental_metadata['pH'].append(ph_val if ph_val > 0 else None)
        except (ValueError, TypeError):
            experimental_metadata['pH'].append(None)
        
        # Extract numeric anion concentration
        try:
            anion_conc = float(metadata.get('anion_concentration', 0))
            experimental_metadata['anion_concentration'].append(anion_conc if anion_conc > 0 else None)
        except (ValueError, TypeError):
            anion_conc = None
            experimental_metadata['anion_concentration'].append(None)
        
        # Extract numeric ligand concentration  
        try:
            ligand_conc = float(metadata.get('ligand_concentration', 0))
            experimental_metadata['ligand_concentration'].append(ligand_conc if ligand_conc > 0 else None)
        except (ValueError, TypeError):
            ligand_conc = None
            experimental_metadata['ligand_concentration'].append(None)
        
        # Calculate anion/ligand concentration ratio
        try:
            if anion_conc and ligand_conc and ligand_conc > 0:
                ratio = anion_conc / ligand_conc
                experimental_metadata['anion_ligand_ratio'].append(ratio)
            else:
                experimental_metadata['anion_ligand_ratio'].append(None)
        except (ValueError, TypeError, ZeroDivisionError):
            experimental_metadata['anion_ligand_ratio'].append(None)
        
        # Extract categorical anion type (label encoding)
        anion = metadata.get('anion', '').lower()
        if 'fecl2' in anion or 'fecl' in anion:
            experimental_metadata['anion_type'].append(0)  # FeCl2
        elif 'feso4' in anion or 'feso' in anion:
            experimental_metadata['anion_type'].append(1)  # FeSO4
        else:
            experimental_metadata['anion_type'].append(None)
        
        # Extract categorical ligand type (label encoding)
        ligand = metadata.get('ligand', '').lower()
        if 'malic' in ligand:
            experimental_metadata['ligand_type'].append(0)  # Malic acid
        elif 'tartar' in ligand or 'tatari' in ligand:
            experimental_metadata['ligand_type'].append(1)  # Tartaric acid
        else:
            experimental_metadata['ligand_type'].append(None)
    
    # Assemble dataset with experimental metadata
    batch_data = batch_assembler.assemble_dataset(samples, metadata=experimental_metadata)
    print(f"      Shape: {batch_data.feature_matrix.shape}")
    print(f"      Features: {len(batch_data.feature_names)}")
    
    # Step 4: PCA
    print("\n[4/6] Running PCA...")
    pca_analyzer = XASPCAAnalyzer()
    pca_results = pca_analyzer.analyze(batch_data, n_components=5)
    print(f"      Variance explained: {pca_results.variance_captured*100:.1f}%")
    
    # Step 5: Clustering
    print("\n[5/6] Running clustering...")
    clusterer = XASClusterer()
    cluster_results = clusterer.cluster(batch_data, use_pca_scores=pca_results.scores)
    print(f"      Found {cluster_results.n_clusters} clusters")
    print(f"      Silhouette score: {cluster_results.silhouette_score:.3f}")
    
    # Step 6: Trend analysis
    print("\n[6/6] Analyzing trends...")
    trend_analyzer = XASTrendAnalyzer()
    trend_results = trend_analyzer.analyze(batch_data, cluster_results)
    n_correlations = len(trend_results.significant_correlations) if hasattr(trend_results, 'significant_correlations') else 0
    n_outliers = len(trend_results.outlier_indices) if hasattr(trend_results, 'outlier_indices') else 0
    print(f"      Found {n_correlations} significant correlations")
    print(f"      Detected {n_outliers} outliers")
    
    # Save results
    print("\n[7/8] Saving results...")
    save_results(output_dir, samples, batch_data, pca_results, cluster_results, trend_results)
    
    # Generate plots
    print("\n[8/9] Generating plots...")
    generate_plots(output_dir, samples, batch_data, pca_results, cluster_results, trend_results)
    
    # Generate grouped XANES plots
    print("\n[9/9] Generating grouped XANES plots...")
    generate_grouped_xanes_plots(output_dir, data, samples)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nDataset: {len(samples)} samples")
    print(f"PCA: {pca_results.n_components} components, {pca_results.variance_captured*100:.1f}% variance")
    print(f"Clustering: {cluster_results.n_clusters} clusters, silhouette={cluster_results.silhouette_score:.3f}")
    print(f"Trends: {n_correlations} significant correlations, {n_outliers} outliers")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
