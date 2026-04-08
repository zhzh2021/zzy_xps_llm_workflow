"""
Test Whole-Spectrum PCA and Experiment Planning

Runs spectrum-based PCA on normalized XAS data and suggests next experiments.

Workflow:
1. Load normalized spectra from 02_analyzed_data
2. Run whole-spectrum PCA using XASSpectrumPCA
3. Interpret PC components (what chemical features they capture)
4. Map experimental conditions to PCA space
5. Suggest next experiments using XASExperimentPlanner

Author: XAS Workflow Team
Date: March 6, 2026
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import re

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add paths
tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))

# Import modules directly using importlib to bypass __init__.py
import importlib.util

# Load spectrum_pca module
spec_pca = importlib.util.spec_from_file_location(
    "xas_spectrum_pca",
    tools_dir / "xas_ml_modules" / "xas_spectrum_pca.py"
)
spectrum_pca_module = importlib.util.module_from_spec(spec_pca)
spec_pca.loader.exec_module(spectrum_pca_module)

# Load experiment_planner module
spec_planner = importlib.util.spec_from_file_location(
    "xas_experiment_planner",
    tools_dir / "xas_ml_modules" / "xas_experiment_planner.py"
)
exp_planner_module = importlib.util.module_from_spec(spec_planner)
spec_planner.loader.exec_module(exp_planner_module)

XASSpectrumPCA = spectrum_pca_module.XASSpectrumPCA
XASExperimentPlanner = exp_planner_module.XASExperimentPlanner


def load_normalized_spectra(data_dir: Path):
    """
    Load normalized XAS spectra from CSV files.
    
    Returns:
        energies: List of energy arrays
        spectra: List of mu_normalized arrays
        sample_names: List of sample names
    """
    csv_files = sorted(data_dir.glob("*_analyzed.csv"))
    
    energies = []
    spectra = []
    sample_names = []
    
    for csv_file in csv_files:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Get sample name (remove _dat_analyzed.csv)
        name = csv_file.stem.replace('_dat_analyzed', '').replace('_analyzed', '')
        sample_names.append(name)
        
        # Extract columns
        energy = df['energy'].values
        mu_norm = df['mu_normalized'].values
        
        energies.append(energy)
        spectra.append(mu_norm)
    
    return energies, spectra, sample_names


def parse_metadata_from_name(sample_name: str):
    """Extract experimental parameters from sample name."""
    metadata = {}
    
    # Iron source
    if 'FeCl2' in sample_name:
        metadata['iron_source'] = 'FeCl2'
        metadata['anion'] = 'Cl'
    elif 'FeSO4' in sample_name:
        metadata['iron_source'] = 'FeSO4'
        metadata['anion'] = 'SO4'
    elif 'FeAcetate' in sample_name:
        metadata['iron_source'] = 'FeAcetate'
        metadata['anion'] = 'Acetate'
    
    # Ligand
    if 'Malic' in sample_name:
        metadata['ligand'] = 'Malic_acid'
    elif 'Tartaric' in sample_name:
        metadata['ligand'] = 'Tartaric_acid'
    
    # pH
    ph_match = re.search(r'pH(\d+)', sample_name)
    if ph_match:
        metadata['pH'] = int(ph_match.group(1))
    
    # State
    if 'gel' in sample_name.lower():
        metadata['state'] = 'gel'
    else:
        metadata['state'] = 'solution'
    
    # Replicate
    rep_match = re.search(r'_R(\d+)', sample_name)
    if rep_match:
        metadata['replicate'] = int(rep_match.group(1))
    
    # Concentrations
    conc_match = re.search(r'\((\d+(?:_\d+)?)-(\d+(?:_\d+)?)\)', sample_name)
    if conc_match:
        anion_str = conc_match.group(1).replace('_', '.')
        ligand_str = conc_match.group(2).replace('_', '.')
        try:
            metadata['anion_conc'] = float(anion_str)
            metadata['ligand_conc'] = float(ligand_str)
        except:
            pass
    
    return metadata


def extract_experimental_params(sample_names):
    """Extract experimental parameters for all samples."""
    params = {
        'pH': [],
        'anion_conc': [],
        'ligand_conc': [],
        'iron_source': [],
        'ligand': [],
        'state': []
    }
    
    for name in sample_names:
        meta = parse_metadata_from_name(name)
        params['pH'].append(meta.get('pH'))
        params['anion_conc'].append(meta.get('anion_conc'))
        params['ligand_conc'].append(meta.get('ligand_conc'))
        params['iron_source'].append(meta.get('iron_source', 'Unknown'))
        params['ligand'].append(meta.get('ligand', 'Unknown'))
        params['state'].append(meta.get('state', 'Unknown'))
    
    return params


def save_results(output_dir: Path, pca_result, interpretations, suggestions, experimental_params):
    """Save spectrum PCA and experiment planning results."""
    
    results_dir = output_dir / 'spectrum_pca_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save PCA summary
    with open(results_dir / 'spectrum_pca_summary.txt', 'w') as f:
        f.write(pca_result.summary())
    
    # 2. Save scores
    df_scores = pd.DataFrame(
        pca_result.scores,
        columns=[f'PC{i+1}' for i in range(pca_result.n_components)],
        index=pca_result.sample_names
    )
    df_scores.to_csv(results_dir / 'spectrum_pca_scores.csv')
    
    # 3. Save loadings
    df_loadings = pd.DataFrame(
        pca_result.loadings.T,
        columns=[f'PC{i+1}' for i in range(pca_result.n_components)],
        index=pca_result.energy_grid
    )
    df_loadings.index.name = 'energy_eV'
    df_loadings.to_csv(results_dir / 'spectrum_pca_loadings.csv')
    
    # 4. Save component interpretations
    interp_data = []
    for interp in interpretations:
        interp_data.append({
            'component': f'PC{interp.pc_number}',
            'variance_explained': float(interp.variance_explained),
            'interpretation': interp.interpretation,
            'peak_energies': [float(e) for e in interp.peak_energies],
            'peak_regions': interp.peak_regions
        })
    
    with open(results_dir / 'component_interpretations.json', 'w') as f:
        json.dump(interp_data, f, indent=2)
    
    # 5. Save experiment suggestions
    sug_data = []
    for i, sug in enumerate(suggestions):
        sug_data.append({
            'suggestion_id': i + 1,
            'strategy': sug.strategy,
            'pc_scores': sug.predicted_scores.tolist(),
            'distance_to_nearest': float(sug.distance_to_nearest),
            'priority': float(sug.priority),
            'reason': sug.reason,
            'suggested_conditions': sug.suggested_conditions
        })
    
    with open(results_dir / 'experiment_suggestions.json', 'w') as f:
        json.dump(sug_data, f, indent=2)
    
    # 6. Save experimental parameters mapping
    df_params = pd.DataFrame(experimental_params, index=pca_result.sample_names)
    df_params.to_csv(results_dir / 'experimental_parameters.csv')
    
    print(f"\n[OK] Results saved to: {results_dir}")
    print(f"  - spectrum_pca_summary.txt")
    print(f"  - spectrum_pca_scores.csv")
    print(f"  - spectrum_pca_loadings.csv")
    print(f"  - component_interpretations.json")
    print(f"  - experiment_suggestions.json")
    print(f"  - experimental_parameters.csv")


def main():
    """Run whole-spectrum PCA and experiment planning."""
    
    print("=" * 80)
    print("XAS WHOLE-SPECTRUM PCA & EXPERIMENT PLANNING")
    print("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent / "project_root"
    data_dir = project_root / "xas_results" / "02_analyzed_data" / "normalized_data"
    output_dir = project_root / "xas_results" / "04_ml_analysis"
    
    # Step 1: Load normalized spectra
    print("\n[1/6] Loading normalized spectra...")
    energies, spectra, sample_names = load_normalized_spectra(data_dir)
    print(f"  Loaded {len(spectra)} normalized spectra")
    print(f"  Energy range: {energies[0][0]:.1f} - {energies[0][-1]:.1f} eV")
    
    # Step 2: Run whole-spectrum PCA
    print("\n[2/6] Running whole-spectrum PCA...")
    analyzer = XASSpectrumPCA(
        n_components=5,  # Keep first 5 components
        normalization='standard',  # Standardize spectra
        energy_range=(7100, 7200),  # Focus on XANES region
        n_grid_points=500
    )
    
    pca_result = analyzer.analyze_spectra(energies, spectra, sample_names)
    
    print(f"  Components: {pca_result.n_components}")
    print(f"  Variance explained:")
    for i in range(pca_result.n_components):
        print(f"    PC{i+1}: {pca_result.variance_ratio[i]*100:.1f}% "
              f"(cumulative: {pca_result.cumulative_variance[i]*100:.1f}%)")
    
    # Step 3: Interpret PC components
    print("\n[3/6] Interpreting principal components...")
    planner = XASExperimentPlanner(
        edge_energy=7120.0,  # Fe K-edge
        xanes_range=(7100, 7160),
        exafs_range=(7160, 7500)
    )
    
    interpretations = planner.interpret_components(pca_result, peak_threshold=0.15)
    
    print(f"  Component interpretations:")
    for interp in interpretations:
        print(f"    PC{interp.pc_number} ({interp.variance_explained*100:.1f}%): {interp.interpretation}")
        print(f"      Peak energies: {[f'{e:.1f}' for e in interp.peak_energies[:3]]}")
        print(f"      Regions: {interp.peak_regions[:3]}")
    
    # Step 4: Extract experimental parameters
    print("\n[4/6] Mapping experimental conditions...")
    experimental_params = extract_experimental_params(sample_names)
    
    # Count unique conditions
    unique_pH = len(set(p for p in experimental_params['pH'] if p is not None))
    unique_sources = len(set(experimental_params['iron_source']))
    unique_ligands = len(set(experimental_params['ligand']))
    
    print(f"  Experimental space:")
    print(f"    pH values: {unique_pH} unique")
    print(f"    Iron sources: {unique_sources} unique ({set(experimental_params['iron_source'])})")
    print(f"    Ligands: {unique_ligands} unique ({set(experimental_params['ligand'])})")
    
    # Step 5: Suggest experiments
    print("\n[5/6] Generating experiment suggestions...")
    
    # Try multiple strategies
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
    
    # Sort by priority
    all_suggestions.sort(key=lambda x: x.priority, reverse=True)
    top_suggestions = all_suggestions[:5]  # Top 5
    
    print(f"\n  Top {len(top_suggestions)} experiment suggestions:")
    for i, sug in enumerate(top_suggestions):
        print(f"\n  Suggestion {i+1}:")
        print(f"    Strategy: {sug.strategy}")
        print(f"    PC1 score: {sug.predicted_scores[0]:.2f}")
        print(f"    PC2 score: {sug.predicted_scores[1]:.2f}")
        print(f"    Distance to nearest: {sug.distance_to_nearest:.2f}")
        print(f"    Priority: {sug.priority:.2f}")
        print(f"    Reason: {sug.reason}")
        if sug.suggested_conditions:
            print(f"    Suggested conditions: {sug.suggested_conditions}")
    
    # Step 6: Save results
    print("\n[6/6] Saving results...")
    save_results(output_dir, pca_result, interpretations, top_suggestions, experimental_params)
    
    # Generate plots
    print("\n[7/7] Generating plots...")
    plots_dir = output_dir / 'analysis_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Scree plot
        analyzer.plot_scree(pca_result, save_path=plots_dir / 'spectrum_pca_scree.png')
        print("  [OK] Scree plot saved")
    except Exception as e:
        print(f"  Warning: Scree plot failed: {e}")
    
    try:
        # Scores plot (colored by pH)
        pH_values = [p if p is not None else 0 for p in experimental_params['pH']]
        analyzer.plot_scores(
            pca_result, pc_x=1, pc_y=2,
            color_by=pH_values,
            save_path=plots_dir / 'spectrum_pca_scores_pH.png'
        )
        print("  [OK] Scores plot (pH) saved")
    except Exception as e:
        print(f"  Warning: Scores plot failed: {e}")
    
    try:
        # Loadings plot
        analyzer.plot_loadings(
            pca_result,
            components=[1, 2, 3],
            save_path=plots_dir / 'spectrum_pca_loadings.png'
        )
        print("  [OK] Loadings plot saved")
    except Exception as e:
        print(f"  Warning: Loadings plot failed: {e}")
    
    try:
        # Conditions overlay plot (NEW - multi-panel view!)
        planner.plot_conditions_overlay(
            pca_result,
            experimental_params=experimental_params,
            pc_x=1, pc_y=2,
            figsize=(16, 10),
            save_path=plots_dir / 'conditions_overlay.png'
        )
        print("  [OK] Conditions overlay plot saved")
    except Exception as e:
        print(f"  Warning: Conditions overlay plot failed: {e}")
    
    try:
        # Experiment planning plot
        planner.plot_experiment_planning(
            pca_result,
            experimental_params=experimental_params,
            suggestions=top_suggestions,
            pc_x=1, pc_y=2,
            color_by='pH',
            save_path=plots_dir / 'experiment_planning.png'
        )
        print("  [OK] Experiment planning plot saved")
    except Exception as e:
        print(f"  Warning: Experiment planning plot failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SPECTRUM PCA & EXPERIMENT PLANNING COMPLETE")
    print("=" * 80)
    print(f"\nAnalyzed {len(sample_names)} spectra")
    print(f"PCA: {pca_result.n_components} components, {pca_result.cumulative_variance[-1]*100:.1f}% variance")
    print(f"Experiment suggestions: {len(top_suggestions)} high-priority suggestions")
    print(f"\nOutput: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
