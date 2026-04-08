"""
Quick test of whole-spectrum PCA with real APS data (ASCII files only).
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"N:\zhenzhen\C-Steel\Data\XAS data")

from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA
from aps_xas_reader import load_aps_dataset

print("=" * 80)
print("QUICK TEST: Whole-Spectrum PCA on Real Data")
print("=" * 80)

# Load only ASCII files (properly calibrated)
data_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5")

print(f"\nLoading FeCl2-Malic_acid samples (ASCII only)...")
datasets = load_aps_dataset(data_dir, pattern="FeCl2-Malic_acid*", prefer_ascii=True)

# Filter out HDF files manually
datasets = [ds for ds in datasets if not ds.attrs.get('filename', '').endswith('.hdf')]

print(f"✓ Loaded {len(datasets)} ASCII spectra\n")

# Show samples
for i, ds in enumerate(datasets[:5]):
    name = ds.attrs.get('filename', f'sample_{i}')
    n = ds.attrs.get('n_points', len(ds['energy']))
    e_range = ds.attrs.get('energy_range', (0, 0))
    print(f"  {i+1}. {name}: {n} pts, {e_range[0]:.0f}-{e_range[1]:.0f} eV")

# Initialize analyzer
print(f"\nRunning whole-spectrum PCA...")
analyzer = XASSpectrumPCA(
    variance_threshold=0.95,
    normalization='standard',
    n_grid_points=300
)

# Run PCA
result = analyzer.analyze_datasets(
    datasets=datasets,
    mu_variable='mu_trans'
)

# Results
print("\n" + "=" * 80)
print(result.summary())

# Show key results
print("\n📊 Key Results:")
print(f"  - PC1 captures: {result.variance_ratio[0]:.1%} of variance")
print(f"  - PC2 captures: {result.variance_ratio[1]:.1%} of variance")
print(f"  - Total captured: {result.cumulative_variance[-1]:.1%}")

print(f"\n🔬 Scores (sample positions in PC space):")
for i in range(min(5, len(result.sample_names))):
    name = result.sample_names[i][:35]
    pc1 = result.scores[i, 0]
    pc2 = result.scores[i, 1] if result.n_components > 1 else 0
    print(f"  {name:<35s}  PC1={pc1:7.3f}  PC2={pc2:7.3f}")

# Export
output_dir = Path('test_pca_output')
analyzer.export_results(result, output_dir)
print(f"\n✓ Results exported to: {output_dir.absolute()}")

print("\n" + "=" * 80)
print("✅ Test complete! Whole-spectrum PCA working on real data.")
print("=" * 80)
