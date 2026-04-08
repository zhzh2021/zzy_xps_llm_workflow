"""
Comprehensive test and data extraction for APS XAS files

This script:
1. Tests that the reader can load both ASCII and HDF files
2. Extracts data from all samples
3. Saves extracted data to CSV files for analysis
4. Validates data integrity
"""

from aps_xas_reader import load_aps_xas, get_transmission_mu, load_aps_dataset
from pathlib import Path
import numpy as np
import pandas as pd

# Data directory
data_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5")
output_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\extracted_data")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("APS XAS Data Extraction Test")
print("=" * 80)

# Test 1: Load single ASCII file
print("\n1. Testing single ASCII file loading...")
print("-" * 80)
test_file = data_dir / "FeCl2-Malic_acid_(0.5-0.5)-pH2.2"
try:
    data = load_aps_xas(test_file)
    energy, mu = get_transmission_mu(data)
    
    print(f"✓ Successfully loaded: {data.metadata['filename']}")
    print(f"  - Points: {len(energy)}")
    print(f"  - Energy range: {energy[0]:.2f} - {energy[-1]:.2f} eV")
    print(f"  - Mu range: {mu.min():.4f} - {mu.max():.4f}")
    
    # Save to CSV
    output_file = output_dir / f"{data.metadata['filename']}_extracted.csv"
    df = pd.DataFrame({
        'energy_eV': energy,
        'mu_transmission': mu,
        'i0': data.i0,
        'i1': data.i1
    })
    df.to_csv(output_file, index=False)
    print(f"  - Saved to: {output_file.name}")
    
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Load single HDF file
print("\n2. Testing single HDF file loading...")
print("-" * 80)
test_hdf = data_dir / "FeCl2-Malic_acid_(0.5-0.5)-pH2_1.hdf"
try:
    data = load_aps_xas(test_hdf)
    energy, mu = get_transmission_mu(data)
    
    print(f"✓ Successfully loaded: {data.metadata['filename']}")
    print(f"  - Points: {len(energy)}")
    print(f"  - WARNING: {data.metadata.get('warning', 'None')}")
    print(f"  - Raw encoder range: {energy[0]:.2f} - {energy[-1]:.2f}")
    print(f"  - Mu range: {mu.min():.4f} - {mu.max():.4f}")
    
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Batch load all samples
print("\n3. Testing batch loading of all samples...")
print("-" * 80)
# Load only ASCII files (processed data)
dataset = load_aps_dataset(data_dir, pattern="*", prefer_ascii=True)

# Filter to only non-HDF files
dataset = [d for d in dataset if not d.metadata['filename'].endswith('.hdf')]

print(f"✓ Loaded {len(dataset)} samples")

# Test 4: Extract and save all data
print("\n4. Extracting and saving all sample data...")
print("-" * 80)

extraction_summary = []

for i, data in enumerate(dataset, 1):
    try:
        energy, mu = get_transmission_mu(data)
        
        # Create output filename
        output_file = output_dir / f"{data.metadata['filename']}_extracted.csv"
        
        # Prepare data frame
        df_data = {
            'energy_eV': energy,
            'mu_transmission': mu,
            'i0': data.i0,
            'i1': data.i1
        }
        
        # Add I2 if available
        if data.i2 is not None:
            df_data['i2'] = data.i2
            df_data['mu_reference'] = data.mu_ref
        
        # Add fluorescence if available
        if data.fluorescence is not None:
            df_data['fluorescence_total'] = data.fluorescence['total']
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False)
        
        extraction_summary.append({
            'filename': data.metadata['filename'],
            'mode': data.mode,
            'n_points': len(energy),
            'energy_min': energy.min(),
            'energy_max': energy.max(),
            'mu_min': mu.min(),
            'mu_max': mu.max(),
            'has_reference': data.i2 is not None,
            'has_fluorescence': data.fluorescence is not None,
            'output_file': output_file.name
        })
        
        if i <= 5:  # Show first 5
            print(f"  {i}. {data.metadata['filename']}: {len(energy)} points → {output_file.name}")
    
    except Exception as e:
        print(f"  ✗ Failed on {data.metadata['filename']}: {e}")

if len(dataset) > 5:
    print(f"  ... and {len(dataset) - 5} more files")

# Test 5: Create summary file
print("\n5. Creating extraction summary...")
print("-" * 80)
summary_df = pd.DataFrame(extraction_summary)
summary_file = output_dir / "extraction_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary saved to: {summary_file.name}")
print(f"\nSummary statistics:")
print(f"  - Total samples extracted: {len(summary_df)}")
print(f"  - Transmission mode: {(summary_df['mode'] == 'transmission').sum()}")
print(f"  - Fluorescence mode: {(summary_df['mode'] == 'fluorescence').sum()}")
print(f"  - With reference data: {summary_df['has_reference'].sum()}")
print(f"  - With fluorescence data: {summary_df['has_fluorescence'].sum()}")

# Test 6: Link with metadata
print("\n6. Linking extracted data with sample metadata...")
print("-" * 80)
metadata_csv = data_dir / "metadata.csv"

if metadata_csv.exists():
    meta_df = pd.read_csv(metadata_csv)
    
    # Filter to mapped samples only
    meta_df = meta_df[meta_df['sample_name'] != 'NOT_MAPPED'].copy()
    
    # Create a mapping from filename to sample info
    print(f"✓ Loaded metadata for {len(meta_df)} files")
    
    # Link extraction summary with metadata
    # Match by finding the sample_name in the extracted filename
    linked_data = []
    for _, row in summary_df.iterrows():
        # Try to match with metadata
        matches = meta_df[meta_df['new_filename'] == row['filename']]
        if len(matches) == 0:
            # Try matching original filename
            matches = meta_df[meta_df['original_filename'].str.contains(row['filename'].split('.')[0], na=False)]
        
        if len(matches) > 0:
            match = matches.iloc[0]
            linked_data.append({
                'extracted_file': row['output_file'],
                'sample_name': match['sample_name'],
                'iron_source': match['iron_source'],
                'anion_type': match['anion_type'],
                'ligand_type': match['ligand_type'],
                'anion_concentration': match['anion_concentration'],
                'ligand_concentration': match['ligand_concentration'],
                'solution_pH': match['solution_pH'],
                'solution_state': match['solution_state'],
                'anion_ligand_ratio': match['anion_ligand_ratio'],
                'mode': row['mode'],
                'n_points': row['n_points'],
                'energy_range': f"{row['energy_min']:.1f}-{row['energy_max']:.1f}"
            })
    
    if linked_data:
        linked_df = pd.DataFrame(linked_data)
        linked_file = output_dir / "extracted_data_with_metadata.csv"
        linked_df.to_csv(linked_file, index=False)
        print(f"✓ Linked data saved to: {linked_file.name}")
        print(f"\nLinked samples by condition:")
        print(linked_df.groupby(['iron_source', 'ligand_type', 'solution_pH']).size())
else:
    print("✗ Metadata CSV not found")

# Test 7: Data validation
print("\n7. Validating extracted data...")
print("-" * 80)
validation_issues = []

for _, row in summary_df.iterrows():
    # Check for reasonable energy range (Fe K-edge ~7112 eV)
    if row['energy_min'] < 6000 or row['energy_max'] > 8000:
        validation_issues.append(f"{row['filename']}: Energy range outside Fe K-edge region")
    
    # Check for sufficient data points
    if row['n_points'] < 50:
        validation_issues.append(f"{row['filename']}: Too few data points ({row['n_points']})")
    
    # Check for reasonable mu values
    if row['mu_min'] < 0 or row['mu_max'] > 5:
        validation_issues.append(f"{row['filename']}: Unusual mu range ({row['mu_min']:.2f} - {row['mu_max']:.2f})")

if validation_issues:
    print(f"⚠ Found {len(validation_issues)} validation issues:")
    for issue in validation_issues[:10]:
        print(f"  - {issue}")
    if len(validation_issues) > 10:
        print(f"  ... and {len(validation_issues) - 10} more")
else:
    print("✓ All data passed validation checks")

print("\n" + "=" * 80)
print("Data Extraction Complete!")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print(f"Extracted {len(dataset)} samples to CSV files")
print("\nFiles created:")
print(f"  - {len(dataset)} individual sample CSV files")
print(f"  - extraction_summary.csv (overview of all extractions)")
if metadata_csv.exists() and linked_data:
    print(f"  - extracted_data_with_metadata.csv (linked with experimental conditions)")
print("\nReady for XANES analysis!")
