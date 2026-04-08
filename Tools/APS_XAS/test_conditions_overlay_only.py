"""
Quick test: Generate conditions overlay plot only
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import using importlib to bypass __init__.py issues
import importlib.util

def load_module_from_file(module_name, file_path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
xas_base = Path(__file__).parent
spectrum_pca_path = xas_base / "xas_ml_modules" / "xas_spectrum_pca.py"
planner_path = xas_base / "xas_ml_modules" / "xas_experiment_planner.py"

spectrum_pca_module = load_module_from_file("xas_spectrum_pca", str(spectrum_pca_path))
planner_module = load_module_from_file("xas_experiment_planner", str(planner_path))

XASSpectrumPCA = spectrum_pca_module.XASSpectrumPCA
XASExperimentPlanner = planner_module.XASExperimentPlanner


def main():
    print("=" * 80)
    print("TESTING CONDITIONS OVERLAY PLOT")
    print("=" * 80)
    
    # Paths
    data_dir = Path(r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_results\02_analyzed_data\normalized_data")
    output_dir = Path(r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_results\04_ml_analysis\analysis_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load normalized spectra
    print("\n[1/3] Loading normalized spectra...")
    csv_files = sorted(data_dir.glob("*_analyzed.csv"))
    
    energies = []
    spectra = []
    sample_names = []
    
    for csv_file in csv_files:
        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            energy = data[:, 0]
            mu_norm = data[:, 2]  # mu_normalized column (3rd column)
            
            energies.append(energy)
            spectra.append(mu_norm)
            sample_names.append(csv_file.stem.replace('_analyzed', ''))
        except Exception as e:
            print(f"  Warning: Skipped {csv_file.name}: {e}")
    
    print(f"  Loaded {len(sample_names)} spectra")
    
    # Run PCA
    print("\n[2/3] Running PCA...")
    analyzer = XASSpectrumPCA(
        n_components=5,
        normalization='standard',
        energy_range=(7100, 7200),
        n_grid_points=500
    )
    
    pca_result = analyzer.analyze_spectra(energies, spectra, sample_names)
    print(f"  PC1: {pca_result.variance_ratio[0]*100:.1f}%")
    print(f"  PC2: {pca_result.variance_ratio[1]*100:.1f}%")
    
    # Extract experimental parameters
    print("\n[3/3] Creating conditions overlay plot...")
    
    experimental_params = {
        'pH': [],
        'iron_source': [],
        'ligand': [],
        'ligand_concentration': [],
        'state': []
    }
    
    for name in sample_names:
        # Parse sample name (e.g., "FeCl2_Malic_acid_pH2.2_gel")
        parts = name.split('_')
        
        # Iron source
        if 'FeCl2' in name:
            experimental_params['iron_source'].append('FeCl2')
        elif 'FeSO4' in name:
            experimental_params['iron_source'].append('FeSO4')
        else:
            experimental_params['iron_source'].append('Unknown')
        
        # Ligand
        if 'Malic' in name:
            experimental_params['ligand'].append('Malic acid')
        elif 'Tartaric' in name:
            experimental_params['ligand'].append('Tartaric acid')
        else:
            experimental_params['ligand'].append('Unknown')
        
        # pH
        ph_value = None
        for part in parts:
            if part.startswith('pH'):
                try:
                    ph_value = float(part[2:])
                except:
                    pass
        experimental_params['pH'].append(ph_value if ph_value else 0.0)
        
        # Ligand concentration
        conc_value = 0.0
        for i, part in enumerate(parts):
            if 'acid' in part.lower() and i + 1 < len(parts):
                try:
                    conc_value = float(parts[i + 1])
                except:
                    pass
        experimental_params['ligand_concentration'].append(conc_value)
        
        # State
        if 'gel' in name.lower():
            experimental_params['state'].append('gel')
        elif 'solution' in name.lower():
            experimental_params['state'].append('solution')
        else:
            experimental_params['state'].append('unknown')
    
    # Create overlay plot
    planner = XASExperimentPlanner()
    
    save_path = output_dir / "conditions_overlay.png"
    
    planner.plot_conditions_overlay(
        pca_result,
        experimental_params=experimental_params,
        pc_x=1,
        pc_y=2,
        figsize=(18, 12),
        save_path=save_path
    )
    
    print(f"\n  [OK] Plot saved to: {save_path}")
    print("\n" + "=" * 80)
    print("SUCCESS! Conditions overlay plot created.")
    print("=" * 80)


if __name__ == "__main__":
    main()
