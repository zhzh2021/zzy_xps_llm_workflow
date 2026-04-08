"""
Demonstration script for APS XAS data reader

Shows how to:
1. Load individual ASCII files (processed data - recommended)
2. Load individual HDF files (raw data - requires calibration)
3. Batch load multiple files from a directory
4. Extract data for different measurement modes
"""

from aps_xas_reader import (
    load_aps_xas, 
    get_transmission_mu, 
    get_fluorescence_mu,
    get_reference_mu,
    load_aps_dataset
)
from pathlib import Path
import numpy as np

# Data directory
data_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5")

print("=" * 80)
print("APS XAS Reader Demonstration")
print("=" * 80)

# Example 1: Load processed ASCII file (RECOMMENDED)
print("\n1. Loading processed ASCII file (RECOMMENDED)")
print("-" * 80)
ascii_file = data_dir / "FeCl2-Malic_acid_(0.5-0.5)-pH2.2"
data_ascii = load_aps_xas(ascii_file)

print(f"File: {data_ascii.metadata['filename']}")
print(f"Mode: {data_ascii.mode}")
print(f"Points: {len(data_ascii.energy)}")
print(f"Energy range: {data_ascii.energy[0]:.1f} - {data_ascii.energy[-1]:.1f} eV")

# Get transmission data
energy_trans, mu_trans = get_transmission_mu(data_ascii)
print(f"\nTransmission mu range: {mu_trans.min():.4f} - {mu_trans.max():.4f}")

# Get fluorescence data if available
if data_ascii.fluorescence is not None:
    energy_fluor, mu_fluor = get_fluorescence_mu(data_ascii, normalize=True)
    print(f"Fluorescence (normalized): {mu_fluor.min():.4f} - {mu_fluor.max():.4f}")

# Get reference data if available
if data_ascii.mu_ref is not None:
    energy_ref, mu_ref = get_reference_mu(data_ascii)
    print(f"Reference mu range: {mu_ref.min():.4f} - {mu_ref.max():.4f}")

# Example 2: Load raw HDF file
print("\n2. Loading raw HDF file (requires calibration)")
print("-" * 80)
hdf_file = data_dir / "FeCl2-Malic_acid_(0.5-0.5)-pH2_1.hdf"
data_hdf = load_aps_xas(hdf_file)

print(f"File: {data_hdf.metadata['filename']}")
print(f"Mode: {data_hdf.mode}")
print(f"Points: {len(data_hdf.energy)}")
if 'warning' in data_hdf.metadata:
    print(f"WARNING: {data_hdf.metadata['warning']}")
print(f"Encoder range: {data_hdf.energy[0]:.1f} - {data_hdf.energy[-1]:.1f}")

# Example 3: Batch load multiple files
print("\n3. Batch loading multiple files")
print("-" * 80)
# Load all FeCl2 samples (ASCII only, which is preferred)
dataset = load_aps_dataset(data_dir, pattern="FeCl2*", prefer_ascii=True)

print(f"Loaded {len(dataset)} files:")
for i, data in enumerate(dataset[:5], 1):  # Show first 5
    print(f"  {i}. {data.metadata['filename']}: "
          f"{data.mode}, {len(data.energy)} points, "
          f"E={data.energy[0]:.0f}-{data.energy[-1]:.0f} eV")
if len(dataset) > 5:
    print(f"  ... and {len(dataset) - 5} more")

# Example 4: Load with metadata CSV
print("\n4. Loading with sample metadata")
print("-" * 80)
try:
    import pandas as pd
    metadata_csv = data_dir / "metadata.csv"
    
    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
        
        # Load a specific sample with its metadata
        sample_row = df[df['label'] == 'JL1'].iloc[0]
        
        print(f"Sample: {sample_row['sample_name']}")
        print(f"  Iron source: {sample_row['iron_source']}")
        print(f"  Anion: {sample_row['anion_type']}")
        print(f"  Ligand: {sample_row['ligand_type']}")
        print(f"  pH: {sample_row['solution_pH']}")
        print(f"  State: {sample_row['solution_state']}")
        
        # Load the data file
        # Find a file that matches this label
        matching_files = [f for f in data_dir.glob(f"*{sample_row['sample_name']}*") 
                         if f.is_file() and not f.suffix == '.hdf']
        if matching_files:
            data = load_aps_xas(matching_files[0])
            print(f"  Loaded: {data.metadata['filename']}")
            print(f"  Energy range: {data.energy[0]:.1f} - {data.energy[-1]:.1f} eV")
    else:
        print("Metadata CSV not found")
except ImportError:
    print("pandas not available - skipping metadata demo")

# Example 5: Data extraction for analysis
print("\n5. Extracting data for XANES analysis")
print("-" * 80)
# Load a sample
sample_file = data_dir / "FeSO4-Malic_acid_(0.5-0.5)-pH2.1"
if sample_file.exists():
    data = load_aps_xas(sample_file)
    
    # Extract energy and mu for analysis
    energy, mu = get_transmission_mu(data)
    
    print(f"Sample: {data.metadata['filename']}")
    print(f"Data arrays ready for analysis:")
    print(f"  energy.shape = {energy.shape}")
    print(f"  mu.shape = {mu.shape}")
    print(f"  Energy step: {np.diff(energy).mean():.2f} eV")
    
    # You can now use this data with larch, scipy, or other analysis tools
    print(f"\nReady for XANES analysis:")
    print(f"  Pre-edge region: E < 7110 eV")
    print(f"  Edge region: 7110 - 7130 eV")
    print(f"  Post-edge region: E > 7130 eV")
else:
    print(f"Sample file not found: {sample_file}")

print("\n" + "=" * 80)
print("Demonstration complete!")
print("=" * 80)
print("\nKey points:")
print("  - Use ASCII files (no .hdf extension) for processed, calibrated data")
print("  - HDF files contain raw detector data and need energy calibration")
print("  - load_aps_dataset() can batch load multiple files")
print("  - Use metadata.csv to link XAS data with experimental conditions")
