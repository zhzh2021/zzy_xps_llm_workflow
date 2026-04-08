"""
Convert Excel XAS data to .dat files for workflow processing.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def convert_excel_xas_to_dat(excel_path, output_dir, sheet_name=0):
    """
    Convert Excel file with XAS data to individual .dat files.
    
    Expects Excel format with alternating columns:
    - Sample1_Energy, Sample1_Mu, Sample2_Energy, Sample2_Mu, ...
    
    Parameters
    ----------
    excel_path : str or Path
        Path to Excel file
    output_dir : str or Path
        Directory to save .dat files
    sheet_name : int or str
        Sheet name or index to read
        
    Returns
    -------
    file_paths : list
        List of created .dat file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # First row contains headers, use them as sample names
    # Every pair of columns is (Energy, Mu)
    file_paths = []
    
    col_idx = 0
    sample_count = 0
    sample_name_counts = {}  # Track duplicate sample names
    
    while col_idx < len(df.columns) - 1:
        # Get column names
        energy_col = df.columns[col_idx]
        mu_col = df.columns[col_idx + 1]
        
        # Skip if column name suggests it's not a sample (Unnamed without data)
        if 'Unnamed' in str(energy_col) and col_idx > 0:
            col_idx += 1
            continue
            
        # Extract data (skip first row which is header)
        energy_data = df.iloc[1:, col_idx].values
        mu_data = df.iloc[1:, col_idx + 1].values
        
        # Clean NaN values
        mask = ~(pd.isna(energy_data) | pd.isna(mu_data))
        energy_clean = energy_data[mask].astype(float)
        mu_clean = mu_data[mask].astype(float)
        
        if len(energy_clean) > 10:  # Only save if we have enough data points
            # Create safe filename from sample name
            sample_name = str(energy_col).replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            sample_name = sample_name[:100]  # Limit length
            
            # Handle duplicates by adding R2, R3 suffix for replicate measurements
            if sample_name in sample_name_counts:
                sample_name_counts[sample_name] += 1
                replicate_num = sample_name_counts[sample_name]
                final_name = f"{sample_name}_R{replicate_num}"
            else:
                sample_name_counts[sample_name] = 1
                final_name = sample_name
            
            # Save as .dat file
            dat_file = output_dir / f"{final_name}.dat"
            
            with open(dat_file, 'w') as f:
                f.write("# XAS data converted from Excel\n")
                f.write(f"# Sample: {energy_col}\n")
                f.write("# Column.1: energy (eV)\n")
                f.write("# Column.2: mu (normalized)\n")
                f.write("#----\n")
                for e, m in zip(energy_clean, mu_clean):
                    f.write(f"{e:.6f}  {m:.6f}\n")
            
            file_paths.append(dat_file)
            sample_count += 1
            print(f"Created: {dat_file.name} ({len(energy_clean)} points)")
        
        # Move to next pair (skip the mu column)
        col_idx += 2
    
    print(f"\nConverted {sample_count} samples from sheet '{sheet_name}'")
    return file_paths


if __name__ == "__main__":
    # Convert the Excel file
    excel_file = Path(r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\XAS data for FeCl2-FeSO4-MA-TA-pH2-5.xlsx")
    output_dir = Path(r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\excel_converted")
    
    print("="*80)
    print("CONVERTING EXCEL XAS DATA TO .DAT FILES")
    print("="*80)
    print(f"\nSource: {excel_file.name}")
    print(f"Output: {output_dir}\n")
    
    # Convert both sheets
    all_files = []
    
    print("\n--- Sheet: Malic acid ---")
    files_ma = convert_excel_xas_to_dat(excel_file, output_dir / "malic_acid", sheet_name="Malic acid")
    all_files.extend(files_ma)
    
    print("\n--- Sheet: Tartaric Acid ---")
    files_ta = convert_excel_xas_to_dat(excel_file, output_dir / "tartaric_acid", sheet_name="Tartaric Acid")
    all_files.extend(files_ta)
    
    print("\n" + "="*80)
    print(f"TOTAL: {len(all_files)} samples converted")
    print("="*80)
