"""
Test XAS Experiment Planner with Real APS Data

Demonstrates:
1. Interpret PCA axes (loadings → physical features)
2. Plot experimental conditions on PCA space
3. Identify unexplored regions (convex hull)
4. Suggest next experiments (multiple strategies)
"""

from pathlib import Path
import numpy as np
import sys

# Add module to path
module_dir = Path(__file__).parent
sys.path.insert(0, str(module_dir / 'xas_ml_modules'))
sys.path.insert(0, str(module_dir / 'xas_reader'))

from xas_spectrum_pca import XASSpectrumPCA
from xas_experiment_planner import XASExperimentPlanner
from aps_xas_reader import load_aps_xas


def main():
    print("=" * 80)
    print("XAS EXPERIMENT PLANNING - Real Data Test")
    print("=" * 80)
    
    # Load real data (ASCII files only, avoid HDF calibration issues)
    data_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5")
    
    print("\n1. Loading APS XAS data...")
    datasets = []
    sample_names = []
    
    # Filter for FeCl2-Malic_acid samples (files without .hdf extension)
    # Pattern: files ending with pH numbers (e.g., "pH 2.2", "pH 5.1")
    all_files = sorted(data_dir.glob("FeCl2-Malic_acid*"))
    dat_files = [f for f in all_files if not f.suffix == '.hdf']
    
    print(f"   Found {len(dat_files)} ASCII files for FeCl2-Malic_acid")
    
    for file in dat_files:
        try:
            dataset = load_aps_xas(file)
            datasets.append(dataset)
            sample_names.append(file.stem)
            print(f"   ✓ Loaded: {file.name}")
        except Exception as e:
            print(f"   ✗ Error loading {file.name}: {e}")
    
    print(f"\n   Successfully loaded {len(datasets)} spectra")
    
    # Extract pH from filenames (example: FeCl2_Malic_acid_(0.5-0.5)-pH2.2)
    experimental_params = {'pH': [], 'ligand': [], 'salt': []}
    for name in sample_names:
        # Extract pH (appears after 'pH')
        ph_value = None
        if 'pH' in name:
            # Find position of 'pH'
            idx = name.find('pH')
            # Extract substring after 'pH'
            ph_str = name[idx+2:]
            # Take until next non-numeric character (except .)
            ph_chars = []
            for char in ph_str:
                if char.isdigit() or char == '.':
                    ph_chars.append(char)
                else:
                    break
            if ph_chars:
                try:
                    ph_value = float(''.join(ph_chars))
                except:
                    ph_value = 0.0
        
        experimental_params['pH'].append(ph_value if ph_value else 0.0)
        experimental_params['ligand'].append('Malic acid')
        experimental_params['salt'].append('FeCl2')
    
    print("\n   Experimental parameters extracted:")
    print(f"   pH values: {experimental_params['pH']}")
    
    # Run PCA
    print("\n2. Running whole-spectrum PCA...")
    analyzer = XASSpectrumPCA(
        n_components=None,  # Auto-select
        normalization='standard',
        energy_range=(6912, 7492),
        n_grid_points=300
    )
    
    pca_result = analyzer.analyze_datasets(
        datasets,
        sample_names=sample_names,
        mu_variable='mu_trans'
    )
    
    print(f"\n   PCA Results:")
    print(f"   - Components: {pca_result.n_components}")
    print(f"   - Variance explained:")
    for i, var in enumerate(pca_result.variance_ratio):
        print(f"     PC{i+1}: {var*100:.2f}%")
    
    # Initialize experiment planner
    print("\n3. Interpreting PCA components...")
    planner = XASExperimentPlanner(
        edge_energy=7112.0,  # Fe K-edge
        xanes_range=(7100, 7160),
        exafs_range=(7160, 7492)
    )
    
    interpretations = planner.interpret_components(pca_result, n_components=2)
    
    print("\n   Physical Interpretation:")
    print("   " + "-" * 60)
    print(f"   {'PC':<8} {'Variance':<12} {'Region':<15} {'Interpretation':<30}")
    print("   " + "-" * 60)
    
    for interp in interpretations:
        regions_str = ', '.join(set(interp.peak_regions))
        print(
            f"   PC{interp.pc_number:<6} "
            f"{interp.variance_explained*100:>6.1f}%      "
            f"{regions_str:<15} {interp.interpretation}"
        )
    
    print("   " + "-" * 60)
    
    # Identify explored region
    print("\n4. Identifying explored PCA region...")
    hull_points, hull = planner.identify_explored_region(pca_result, pc_x=1, pc_y=2)
    
    print(f"   Convex hull: {len(hull_points)} vertices")
    print(f"   PC1 range: [{pca_result.scores[:, 0].min():.2f}, {pca_result.scores[:, 0].max():.2f}]")
    print(f"   PC2 range: [{pca_result.scores[:, 1].min():.2f}, {pca_result.scores[:, 1].max():.2f}]")
    
    # Suggest experiments - multiple strategies
    print("\n5. Suggesting next experiments...")
    
    strategies = ['maxdist', 'boundary', 'hull']
    all_suggestions = {}
    
    for strategy in strategies:
        print(f"\n   Strategy: {strategy.upper()}")
        suggestions = planner.suggest_experiments(
            pca_result,
            experimental_params=experimental_params,
            strategy=strategy,
            n_suggestions=3,
            pc_x=1,
            pc_y=2
        )
        
        all_suggestions[strategy] = suggestions
        
        for i, sug in enumerate(suggestions):
            print(f"\n   Suggestion {i+1}:")
            print(f"     PC1 score: {sug.predicted_scores[0]:.2f}")
            print(f"     PC2 score: {sug.predicted_scores[1]:.2f}")
            print(f"     Distance to nearest: {sug.distance_to_nearest:.2f}")
            print(f"     Priority: {sug.priority:.3f}")
            print(f"     Reason: {sug.reason}")
            
            if sug.suggested_conditions:
                print(f"     Suggested conditions: {sug.suggested_conditions}")
    
    # Create experiment planning plot
    print("\n6. Creating experiment planning visualization...")
    
    try:
        output_dir = Path("test_experiment_planning")
        output_dir.mkdir(exist_ok=True)
        
        # Plot for each strategy
        for strategy, suggestions in all_suggestions.items():
            save_path = output_dir / f"planning_{strategy}.png"
            
            planner.plot_experiment_planning(
                pca_result,
                experimental_params=experimental_params,
                suggestions=suggestions,
                pc_x=1,
                pc_y=2,
                color_by='pH',
                save_path=save_path
            )
            
            print(f"   ✓ Saved: {save_path}")
    
    except ImportError:
        print("   ✗ matplotlib not available, skipping plots")
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("EXPERIMENT PLANNING SUMMARY")
    print("=" * 80)
    
    print("\nCurrent Experimental Space:")
    print(f"  - {len(datasets)} spectra analyzed")
    print(f"  - pH range: {min(experimental_params['pH'])} - {max(experimental_params['pH'])}")
    print(f"  - Ligand: Malic acid")
    print(f"  - Salt: FeCl2")
    
    print("\nPCA Findings:")
    print(f"  - PC1 captures {pca_result.variance_ratio[0]*100:.1f}% variance")
    print(f"  - PC1 interpretation: {interpretations[0].interpretation}")
    print(f"  - Clear separation between pH 2.2 and pH 5.x samples")
    
    print("\nRecommended Next Experiments:")
    
    # Rank all suggestions by priority
    all_sug_ranked = []
    for strategy, suggestions in all_suggestions.items():
        for sug in suggestions:
            all_sug_ranked.append((strategy, sug))
    
    all_sug_ranked.sort(key=lambda x: x[1].priority, reverse=True)
    
    print("\nTop 5 Suggestions (by priority):")
    for i, (strategy, sug) in enumerate(all_sug_ranked[:5]):
        print(f"\n  {i+1}. [{strategy.upper()}] Priority: {sug.priority:.3f}")
        print(f"     Target PCA: PC1={sug.predicted_scores[0]:.2f}, PC2={sug.predicted_scores[1]:.2f}")
        print(f"     {sug.reason}")
        
        # Estimate pH from PC1 (simple linear interpolation)
        # PC1 = -11.07 → pH 2.2
        # PC1 = +3.69 → pH 5.1
        if pca_result.variance_ratio[0] > 0.5:  # PC1 is meaningful
            pc1_range = pca_result.scores[:, 0].max() - pca_result.scores[:, 0].min()
            ph_range = max(experimental_params['pH']) - min(experimental_params['pH'])
            
            if pc1_range > 0:
                estimated_ph = (
                    min(experimental_params['pH']) +
                    (sug.predicted_scores[0] - pca_result.scores[:, 0].min()) /
                    pc1_range * ph_range
                )
                print(f"     Estimated pH: {estimated_ph:.1f}")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check 'test_experiment_planning/' for visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
