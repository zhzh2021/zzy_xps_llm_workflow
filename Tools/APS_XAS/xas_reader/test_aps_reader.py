"""Test reading APS XAS data files"""
import h5py
import numpy as np
from pathlib import Path

# Test HDF file
hdf_file = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2_1.hdf")
ascii_file = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2.2")

print("=" * 80)
print("HDF File Structure")
print("=" * 80)
with h5py.File(hdf_file, 'r') as f:
    print(f"Keys: {list(f.keys())}\n")
    for key in f.keys():
        dataset = f[key]
        print(f"{key}:")
        print(f"  Shape: {dataset.shape}")
        print(f"  Dtype: {dataset.dtype}")
        print(f"  Min: {dataset[:].min():.4f}, Max: {dataset[:].max():.4f}")
        print()

print("=" * 80)
print("ASCII File Structure")
print("=" * 80)
# Read first few lines to see header
with open(ascii_file, 'r') as f:
    header_lines = []
    for i, line in enumerate(f):
        if line.startswith('#'):
            header_lines.append(line.strip())
        else:
            break
    print("Header:")
    for line in header_lines:
        print(line)

# Load data
data = np.loadtxt(ascii_file)
print(f"\nData shape: {data.shape}")
print(f"Columns: {data.shape[1]}")
print("\nColumn ranges:")
col_names = ['E_eV', 'I0', 'I1', 'I2', 'I3', 'mu01', 'mu12', 'flatot', 'fla1', 
             'fla2', 'fla3', 'fla4', 'fla5', 'fla6', 'fla7', 'enc']
for i, name in enumerate(col_names):
    if i < data.shape[1]:
        print(f"  {i}: {name:10s} - min: {data[:, i].min():12.4f}, max: {data[:, i].max():12.4f}")

print("\n" + "=" * 80)
print("Comparison")
print("=" * 80)
print(f"HDF has {196} points (raw detector data)")
print(f"ASCII has {data.shape[0]} points (processed XAS data)")
print("\nConclusion:")
print("- HDF: Raw beamline data (encoder counts, detector currents)")
print("- ASCII: Processed XAS spectra (calibrated energy, calculated mu)")
