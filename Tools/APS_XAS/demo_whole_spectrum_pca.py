"""
Demo: Whole-Spectrum PCA on APS XAS Data

Demonstrates using xas_spectrum_pca module with real XAS data from APS beamline.
Shows how to:
1. Load spectra from xarray datasets
2. Run whole-spectrum PCA
3. Visualize results (scores, loadings, scree plots)  
4. Export results

Author: ZZY Lab
Date: March 5, 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, r"N:\zhenzhen\C-Steel\Data\XAS data")

try:
    from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA
    from aps_xas_reader import load_aps_dataset
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you have:")
    print("1. xas_spectrum_pca.py in xas_ml_modules/")
    print("2. aps_xas_reader.py available")
    sys.exit(1)


def demo_whole_spectrum_pca():
    """Demonstrate whole-spectrum PCA on real XAS data."""
    
    print("=" * 80)
    print("WHOLE-SPECTRUM PCA DEMO")
    print("Analysis of APS XAS Data")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: Load XAS data
    # =========================================================================
    print("\n[Step 1] Loading XAS spectra from APS data...")
    
    data_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Using synthetic data instead...")
        return demo_synthetic_data()
    
    # Load FeCl2 samples
    print(f"Loading from: {data_dir}")
    datasets = load_aps_dataset(data_dir, pattern="FeCl2-Malic*", prefer_ascii=True)
    
    if len(datasets) < 2:
        print(f"Not enough datasets found ({len(datasets)}). Need at least 2 for PCA.")
        return demo_synthetic_data()
    
    print(f"✓ Loaded {len(datasets)} spectra")
    
    # Show sample info
    print("\nSamples:")
    for i, ds in enumerate(datasets[:5]):
        filename = ds.attrs.get('filename', f'sample_{i}')
        n_points = ds.attrs.get('n_points', len(ds['energy']))
        e_range = ds.attrs.get('energy_range', (0, 0))
        print(f"  {i+1}. {filename}: {n_points} points, {e_range[0]:.1f}-{e_range[1]:.1f} eV")
    
    if len(datasets) > 5:
        print(f"  ... and {len(datasets) - 5} more")
    
    # =========================================================================
    # Step 2: Initialize PCA analyzer
    # =========================================================================
    print("\n[Step 2] Initializing whole-spectrum PCA analyzer...")
    
    analyzer = XASSpectrumPCA(
        n_components=None,           # Auto-select based on variance
        variance_threshold=0.95,      # Keep 95% of variance
        normalization='standard',     # Standard normalization
        energy_range=None,            # Auto-detect overlapping range
        n_grid_points=400             # Common grid resolution
    )
    
    print("✓ Analyzer configured:")
    print(f"  - Normalization: standard (zero mean, unit variance)")
    print(f"  - Variance threshold: 95%")
    print(f"  - Grid points: 400")
    
    # =========================================================================
    # Step 3: Run PCA
    # =========================================================================
    print("\n[Step 3] Running whole-spectrum PCA...")
    
    result = analyzer.analyze_datasets(
        datasets=datasets,
        mu_variable='mu_trans',  # Use transmission mu
        sample_names=None        # Use filenames from datasets
    )
    
    print("✓ PCA complete!")
    
    # =========================================================================
    # Step 4: Display results
    # =========================================================================
    print("\n" + result.summary())
    
    # Show scores for first few samples
    print("\n[Step 4] PCA Scores (first 5 samples, first 3 components):")
    print("-" * 60)
    print(f"{'Sample':<40} {'PC1':>8} {'PC2':>8} {'PC3':>8}")
    print("-" * 60)
    
    n_show = min(5, result.n_spectra)
    n_pc_show = min(3, result.n_components)
    
    for i in range(n_show):
        name = result.sample_names[i][:38]  # Truncate long names
        scores = [f"{result.scores[i, j]:8.3f}" for j in range(n_pc_show)]
        print(f"{name:<40} {' '.join(scores)}")
    
    # Show loadings at key energies
    print("\n[Step 5] PCA Loadings at key energies:")
    print("-" * 60)
    
    # Find indices for key energies
    energy_grid = result.energy_grid
    key_energies = [7100, 7112, 7130, 7150]  # Pre-edge, edge, post-edge
    
    print(f"{'Energy (eV)':<15} {'PC1':>12} {'PC2':>12} {'PC3':>12}")
    print("-" * 60)
    
    for e_target in key_energies:
        # Find closest energy point
        idx = np.argmin(np.abs(energy_grid - e_target))
        e_actual = energy_grid[idx]
        
        loadings = [f"{result.loadings[j, idx]:12.4f}" for j in range(n_pc_show)]
        print(f"{e_actual:<15.1f} {' '.join(loadings)}")
    
    # =========================================================================
    # Step 6: Interpret components
    # =========================================================================
    print("\n[Step 6] Component Interpretation:")
    print("-" * 80)
    
    for i in range(min(3, result.n_components)):
        loading = result.loadings[i, :]
        
        # Find peak features in loading
        abs_loading = np.abs(loading)
        max_idx = np.argmax(abs_loading)
        max_energy = energy_grid[max_idx]
        max_value = loading[max_idx]
        
        print(f"\nPC{i+1} ({result.variance_ratio[i]:.1%} variance):")
        print(f"  - Maximum contribution at {max_energy:.1f} eV (value: {max_value:.4f})")
        
        # Identify regions (simplified)
        if max_energy < 7110:
            print(f"  - Interpretation: Pre-edge features / oxidation state")
        elif 7110 <= max_energy < 7125:
            print(f"  - Interpretation: Edge position / valence state")
        else:
            print(f"  - Interpretation: XANES features / coordination environment")
    
    # =========================================================================
    # Step 7: Plot results
    # =========================================================================
    print("\n[Step 7] Generating plots...")
    
    try:
        # Scree plot
        print("  - Scree plot (variance explained)...")
        analyzer.plot_scree(result, save_path='whole_spectrum_pca_scree.png')
        
        # Scores plot
        print("  - Scores plot (sample trajectories)...")
        analyzer.plot_scores(result, pc_x=1, pc_y=2, save_path='whole_spectrum_pca_scores.png')
        
        # Loadings plot
        print("  - Loadings plot (spectral interpretation)...")
        analyzer.plot_loadings(result, components=[1, 2, 3], save_path='whole_spectrum_pca_loadings.png')
        
        print("✓ Plots generated!")
        
    except Exception as e:
        print(f"  ⚠ Plotting failed: {e}")
        print("  (matplotlib may not be available)")
    
    # =========================================================================
    # Step 8: Export results
    # =========================================================================
    print("\n[Step 8] Exporting results...")
    
    output_dir = Path('whole_spectrum_pca_results')
    
    try:
        analyzer.export_results(result, output_dir)
        print(f"✓ Results exported to: {output_dir.absolute()}")
        print(f"  - pca_scores.csv: Sample scores (clustering/trajectories)")
        print(f"  - pca_loadings.csv: Spectral loadings (interpretation)")
        print(f"  - pca_variance.csv: Variance explained")
        print(f"  - pca_summary.txt: Summary report")
        
    except Exception as e:
        print(f"  ⚠ Export failed: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  - Analyzed {result.n_spectra} spectra")
    print(f"  - Found {result.n_components} principal components")
    print(f"  - Captured {result.cumulative_variance[-1]:.1%} of total variance")
    print(f"  - Top 3 PCs explain: {result.cumulative_variance[min(2, result.n_components-1)]:.1%}")
    
    print(f"\nApplications:")
    print(f"  • Scores → Sample clustering & reaction trajectories")
    print(f"  • Loadings → Spectral features driving variation")
    print(f"  • Variance → Component importance")
    
    print(f"\n✨ Whole-spectrum PCA reveals structure without predefined features!")
    print("=" * 80)
    
    return result


def demo_synthetic_data():
    """Demo with synthetic data if real data not available."""
    
    print("\n[Running demo with synthetic data...]")
    print("=" * 80)
    
    # Create synthetic spectra with evolving features
    n_spectra = 15
    n_points = 350
    energy_base = np.linspace(7050, 7250, n_points)
    
    spectra_list = []
    energy_list = []
    sample_names = []
    
    print(f"Generating {n_spectra} synthetic XAS spectra...")
    
    for i in range(n_spectra):
        # Simulate reaction progression: edge shift + white line change
        progress = i / (n_spectra - 1)
        
        energy = energy_base + np.random.randn(n_points) * 0.05
        
        # Edge shift (Fe(II) → Fe(III) simulation)
        e0 = 7112 + progress * 3  # Edge shifts up by 3 eV
        edge = 1 / (1 + np.exp(-(energy - e0) / 2.5))
        
        # White line intensity change
        white_line_pos = e0 + 8
        white_line_intensity = 0.3 + progress * 0.4
        white_line = white_line_intensity * np.exp(-((energy - white_line_pos) / 3) ** 2)
        
        # XANES oscillations
        k = 2 * np.pi * (energy - e0) / 25
        oscillations = 0.08 * np.sin(k + progress * np.pi / 2) * edge
        
        # Noise
        noise = 0.02 * np.random.randn(n_points)
        
        mu = edge + white_line + oscillations + noise
        
        energy_list.append(energy)
        spectra_list.append(mu)
        sample_names.append(f'Reaction_t{i*5:02d}min')
    
    print(f"✓ Generated spectra simulating Fe(II) → Fe(III) oxidation")
    
    # Run PCA
    print("\nRunning whole-spectrum PCA...")
    analyzer = XASSpectrumPCA(variance_threshold=0.95, n_grid_points=300)
    
    result = analyzer.analyze_spectra(
        energies=energy_list,
        spectra=spectra_list,
        sample_names=sample_names
    )
    
    print("\n" + result.summary())
    
    # Plot results
    try:
        print("\nGenerating plots...")
        analyzer.plot_scree(result, save_path='synthetic_pca_scree.png')
        analyzer.plot_scores(result, pc_x=1, pc_y=2, color_by=np.arange(n_spectra), 
                           save_path='synthetic_pca_scores.png')
        analyzer.plot_loadings(result, components=[1, 2], save_path='synthetic_pca_loadings.png')
        print("✓ Plots saved!")
    except Exception as e:
        print(f"  ⚠ Plotting skipped: {e}")
    
    # Export
    try:
        output_dir = Path('synthetic_pca_results')
        analyzer.export_results(result, output_dir)
        print(f"\n✓ Results exported to: {output_dir.absolute()}")
    except Exception as e:
        print(f"  ⚠ Export failed: {e}")
    
    print("\n" + "=" * 80)
    print("Synthetic demo complete!")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    print("\nXAS Whole-Spectrum PCA Demonstration")
    print("Discovering spectral structure without predefined features\n")
    
    try:
        result = demo_whole_spectrum_pca()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
