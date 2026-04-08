"""
Test script for feature extraction from normalized XAS data.

This script reads normalized CSV files from the analyzer output
and extracts features using the XASFeatureExtractor.
"""

import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'xas_analyzer'))

from xas_feature_extractor import XASFeatureExtractor, generate_feature_comparison_plots_from_dicts


def load_normalized_csv(csv_path: Path) -> dict:
    """Load normalized CSV and return data for feature extraction."""
    df = pd.read_csv(csv_path)
    
    # Print columns for debugging (first file only)
    if not hasattr(load_normalized_csv, 'columns_printed'):
        print(f"\nCSV columns: {df.columns.tolist()}")
        load_normalized_csv.columns_printed = True
    
    # Extract metadata - check for different possible column names
    e0 = None
    edge_step = None
    
    # Try to get e0 from columns
    for col in df.columns:
        if 'e0' in col.lower() and len(df) > 0:
            e0 = df[col].iloc[0]
            break
    
    # Try to get edge_step from columns  
    for col in df.columns:
        if 'edge_step' in col.lower() and len(df) > 0:
            edge_step = df[col].iloc[0]
            break
    
    # Get energy - try different possible column names
    energy = None
    for col_name in ['energy', 'Energy', 'E', 'e']:
        if col_name in df.columns:
            energy = df[col_name].values
            break
    
    if energy is None:
        raise ValueError(f"Could not find energy column in {csv_path.name}")
    
    # Get mu_normalized - try different possible column names
    mu_normalized = None
    for col_name in ['mu_normalized', 'norm', 'normalized', 'mu_norm', 'flat']:
        if col_name in df.columns:
            mu_normalized = df[col_name].values
            break
    
    # Fallback to mu if no normalized column
    if mu_normalized is None:
        for col_name in ['mu', 'absorption', 'xmu']:
            if col_name in df.columns:
                mu_normalized = df[col_name].values
                break
    
    if mu_normalized is None:
        raise ValueError(f"Could not find mu/normalized column in {csv_path.name}. Columns: {df.columns.tolist()}")
    
    # Return data dictionary
    return {
        'sample_name': csv_path.stem.replace('_analyzed', ''),
        'energy': energy,
        'mu_normalized': mu_normalized,
        'e0': e0 if e0 is not None else 7120.0,
        'edge_step': edge_step if edge_step is not None else 1.0
    }


def main():
    """Main test function."""
    # Directories
    project_root = Path(__file__).parent.parent.parent.parent / "project_root"
    data_dir = project_root / "xas_results" / "02_analyzed_data" / "normalized_data"
    output_dir = project_root / "xas_results" / "03_feature_extraction" / "extracted_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all normalized CSV files
    csv_files = sorted(data_dir.glob("*_analyzed.csv"))
    
    print("=" * 80)
    print("XAS Feature Extraction Test")
    print("=" * 80)
    print(f"\nInput directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(csv_files)} normalized files\n")
    
    if not csv_files:
        print("No normalized CSV files found!")
        return
    
    # Initialize extractor
    extractor = XASFeatureExtractor()
    
    # Process each file
    all_features = []
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"[{i}/{len(csv_files)}] Processing: {csv_file.name}")
            
            # Load data
            data = load_normalized_csv(csv_file)
            
            # Extract features - pass energy and mu arrays directly
            features = extractor.extract_features_from_arrays(
                energy=data['energy'],
                mu_normalized=data['mu_normalized'],
                sample_name=data['sample_name'],
                e0=data['e0'],
                edge_step=data['edge_step']
            )
            
            # Save individual feature file
            output_file = output_dir / f"{data['sample_name']}_features.json"
            with open(output_file, 'w') as f:
                json.dump(features.model_dump(), f, indent=2)
            
            all_features.append(features.model_dump())
            successful += 1
            
            print(f"  [OK] Features extracted and saved")
            print(f"    E0: {features.e0:.2f} eV")
            print(f"    Edge step: {features.edge_step:.4f}")
            
        except Exception as e:
            failed += 1
            print(f"  [FAIL] Failed: {e}")
    
    # Save batch summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(csv_files),
        'successful': successful,
        'failed': failed,
        'features': all_features
    }
    
    summary_file = output_dir / "batch_features_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate feature comparison plots via feature extractor module
    try:
        plots_dir = generate_feature_comparison_plots_from_dicts(all_features)
        if plots_dir:
            print(f"\nFeature plots saved to: {plots_dir}")
    except Exception as e:
        print(f"\n[WARN] Feature plot generation failed: {e}")

    
    print("\n" + "=" * 80)
    print("Feature Extraction Complete")
    print("=" * 80)
    print(f"Successful: {successful}/{len(csv_files)}")
    print(f"Failed: {failed}/{len(csv_files)}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
