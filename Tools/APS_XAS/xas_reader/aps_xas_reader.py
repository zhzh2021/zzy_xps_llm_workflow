"""
<<<<<<< Updated upstream
APS 12-BM-B XAS Data Reader Module (Standardized xarray Format)
=======
APS 12-BM-B XAS Data Reader Module
>>>>>>> Stashed changes

Handles loading XAS data from APS beamline 12-BM-B, which provides data in two formats:
1. HDF5 files (.hdf) - Raw beamline data (encoder positions, detector currents)
2. ASCII files (no extension or .dat) - Processed XAS spectra (calibrated energy, calculated mu)

The ASCII files are recommended for XAS analysis as they contain:
- Calibrated energy in eV
- Calculated absorption coefficients (mu)
- Fluorescence data
- Reference foil data
<<<<<<< Updated upstream

Conforms to Standardized XAS Data Format Specification v3.0.
"""

__version__ = '3.0'
__reader_name__ = 'APS_12BM-B_Reader'

import numpy as np
import pandas as pd
=======
"""

import h5py
import numpy as np
>>>>>>> Stashed changes
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

<<<<<<< Updated upstream
# Make h5py optional
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available - HDF5 files will not be supported")


def _load_ascii_table(file_path: Path) -> np.ndarray:
    df = pd.read_csv(
        file_path,
        comment='#',
        sep=None,
        engine='python',
        header=None,
    )
    if df.empty:
        raise ValueError("No numeric data found in ASCII file")
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    if df.shape[1] < 2:
        raise ValueError("ASCII file must have at least 2 columns (energy, intensity)")
    df = df.dropna(subset=[0, 1])
    return df.to_numpy()


def _clean_and_sort(energy: np.ndarray,
                    mu: np.ndarray,
                    i0: Optional[np.ndarray] = None,
                    i1: Optional[np.ndarray] = None,
                    i2: Optional[np.ndarray] = None,
                    fluor: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    energy = np.asarray(energy)
    mu = np.asarray(mu)

    mask = np.isfinite(energy) & np.isfinite(mu)
    if i0 is not None:
        mask &= np.isfinite(i0)
    if i1 is not None:
        mask &= np.isfinite(i1)
    if i2 is not None:
        mask &= np.isfinite(i2)
    if fluor is not None:
        mask &= np.isfinite(fluor)

    energy = energy[mask]
    mu = mu[mask]
    i0 = i0[mask] if i0 is not None else None
    i1 = i1[mask] if i1 is not None else None
    i2 = i2[mask] if i2 is not None else None
    fluor = fluor[mask] if fluor is not None else None

    order = np.argsort(energy)
    energy = energy[order]
    mu = mu[order]
    if i0 is not None:
        i0 = i0[order]
    if i1 is not None:
        i1 = i1[order]
    if i2 is not None:
        i2 = i2[order]
    if fluor is not None:
        fluor = fluor[order]

    return energy, mu, i0, i1, i2, fluor


def read_aps_ascii(file_path: str | Path, output_dir: Optional[str | Path] = None,
                   save_output: bool = True) -> xr.Dataset:
    """
    Read APS 12-BM-B ASCII format XAS data file.

    Expected format (from header):
    # 1_E_eV 2_I0 3_I1 4_I2 5_I3 6_mu01 7_mu12 8_flatot 9_fla1 10_fla2 11_fla3
      12_fla4 13_fla5 14_fla6 15_fla7 16_enc

=======

def read_aps_ascii(file_path: str | Path) -> xr.Dataset:
    """
    Read APS 12-BM-B ASCII format XAS data file.
    
    Expected format (from header):
    # 1_E_eV 2_I0 3_I1 4_I2 5_I3 6_mu01 7_mu12 8_flatot 9_fla1 10_fla2 11_fla3 
      12_fla4 13_fla5 14_fla6 15_fla7 16_enc
    
>>>>>>> Stashed changes
    Where:
    - E_eV: Energy in eV
    - I0, I1, I2, I3: Detector currents
    - mu01: ln(I0/I1) - absorption coefficient of sample
    - mu12: ln(I1/I2) - absorption coefficient of reference
    - flatot: Total fluorescence (sum of all channels)
    - fla1-fla7: Individual fluorescence channels
    - enc: Encoder position
<<<<<<< Updated upstream

    Parameters
    ----------
    file_path : str or Path
        Path to the ASCII data file
    output_dir : str or Path, optional
        Directory to save standardized output. If None and save_output=True,
        saves to project_root/xas_results/01_standardized_data
    save_output : bool, optional
        Whether to save the dataset to netCDF format (default: True)

=======
    
>>>>>>> Stashed changes
    Parameters
    ----------
    file_path : str or Path
        Path to the APS ASCII data file
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Returns
    -------
    xarray.Dataset
        Standardized XAS dataset with energy coordinate and data variables
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
<<<<<<< Updated upstream

    # Read header information
    metadata = {
        'filename': file_path.name,
        'beamline': 'APS 12-BM-B',
        'reader_version': __version__,
        'reader_name': __reader_name__
    }

=======
    
    # Read header information
    metadata = {
        'filename': file_path.name,
        'beamline': 'APS 12-BM-B'
    }
    
>>>>>>> Stashed changes
    header_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.strip())
                # Parse specific metadata
                if 'Data collected at' in line:
                    metadata['source'] = line.strip('# ').strip()
                elif any(date_sep in line for date_sep in ['/', '-']) and ',' in line:
                    metadata['date'] = line.strip('# ').strip()
            else:
                break
<<<<<<< Updated upstream

    metadata['header'] = header_lines

    # Load numeric data
    data = _load_ascii_table(file_path)

=======
    
    metadata['header'] = header_lines
    
    # Load numeric data
    data = np.loadtxt(file_path)
    
>>>>>>> Stashed changes
    # Extract columns based on expected format
    # Columns: E_eV, I0, I1, I2, I3, mu01, mu12, flatot, fla1-fla7, enc
    energy = data[:, 0]
    i0 = data[:, 1]
    i1 = data[:, 2]
    i2 = data[:, 3] if data.shape[1] > 3 else None
    mu_trans = data[:, 5] if data.shape[1] > 5 else np.log(i0 / i1)
    mu_ref = data[:, 6] if data.shape[1] > 6 else None
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    # Extract fluorescence data if available
    fluorescence = None
    if data.shape[1] >= 15:  # Has fluorescence channels
        fluorescence = {
            'total': data[:, 7],
            'channel_1': data[:, 8],
            'channel_2': data[:, 9],
            'channel_3': data[:, 10],
            'channel_4': data[:, 11],
            'channel_5': data[:, 12],
            'channel_6': data[:, 13],
            'channel_7': data[:, 14]
        }
<<<<<<< Updated upstream

=======
        
>>>>>>> Stashed changes
        # Determine if this is primarily fluorescence data
        # Check if fluorescence signal is significant
        fluor_max = fluorescence['total'].max()
        mode = 'fluorescence' if fluor_max > 100 else 'transmission'
    else:
        mode = 'transmission'
<<<<<<< Updated upstream

    # Clean and sort
    fluor_total = fluorescence['total'] if fluorescence is not None else None
    energy, mu_trans, i0, i1, i2, fluor_total = _clean_and_sort(
        energy, mu_trans, i0, i1, i2, fluor_total
    )

    metadata['mode'] = mode
    metadata['n_points'] = len(energy)
    metadata['energy_range'] = (float(energy.min()), float(energy.max()))

=======
    
    metadata['mode'] = mode
    metadata['n_points'] = len(energy)
    metadata['energy_range'] = (energy.min(), energy.max())
    
>>>>>>> Stashed changes
    # Create xarray Dataset with standardized structure
    data_vars = {
        'i0': ('point', i0, {'description': 'Incident beam intensity'}),
        'i1': ('point', i1, {'description': 'Transmitted beam intensity'}),
        'mu_trans': ('point', mu_trans, {'description': 'Transmission absorption coefficient ln(I0/I1)'})
    }
<<<<<<< Updated upstream

    # Add optional data variables
    if i2 is not None:
        data_vars['i2'] = ('point', i2, {'description': 'Reference detector intensity'})

    if mu_ref is not None:
        data_vars['mu_ref'] = ('point', mu_ref, {'description': 'Reference absorption coefficient ln(I1/I2)'})

    # Add fluorescence channels
    if fluorescence is not None:
        for channel_name, channel_data in fluorescence.items():
            var_name = f'fluor_{channel_name}'
            data_vars[var_name] = ('point', channel_data, {'description': f'Fluorescence {channel_name}'})

    # Create Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'energy': ('point', energy, {'units': 'eV', 'description': 'Photon energy', 'calibrated': True})},
        attrs=metadata
    )

    return ds


def read_aps_hdf(file_path: str | Path,
                 energy_calibration: Optional[Tuple[float, float]] = None) -> xr.Dataset:
    """
    Read APS 12-BM-B HDF5 format XAS data file.

    HDF files contain raw beamline data:
    - INENC1.VAL.Mean: Encoder position (raw counts)
    - FMC_IN.VAL1.Mean: I0 detector
    - FMC_IN.VAL2.Mean: I1 detector
=======
    
    # Add optional data variables
    if i2 is not None:
        data_vars['i2'] = ('point', i2, {'description': 'Reference detector intensity'})
    
    if mu_ref is not None:
        data_vars['mu_ref'] = ('point', mu_ref, {'description': 'Reference absorption coefficient ln(I1/I2)'})
    
    # Add fluorescence channels
    if fluorescence is not None:
        for channel_name, channel_data in fluorescence.items():
            var_name = f'fluor_{channel_name}' if 'channel' in channel_name else f'fluor_{channel_name}'
            data_vars[var_name] = ('point', channel_data, {'description': f'Fluorescence {channel_name}'})
    
    # Create Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'energy': ('point', energy, {'units': 'eV', 'description': 'Photon energy'})},
        attrs=metadata
    )
    
    return ds


def read_aps_hdf(file_path: str | Path, 
                 energy_calibration: Optional[Tuple[float, float]] = None) -> xr.Dataset:
    """
    Read APS 12-BM-B HDF5 format XAS data file.
    
    HDF files contain raw beamline data:
    - INENC1.VAL.Mean: Encoder position (raw counts)
    - FMC_IN.VAL1.Mean: I0 detector
    - FMC_IN.VAL2.Mean: I1 detector  
>>>>>>> Stashed changes
    - FMC_IN.VAL3.Mean: I2 detector (reference)
    - FMC_IN.VAL4.Mean: Additional detector
    - COUNTER1.OUT.Value: Counter output
    - PCAP.SAMPLES.Value: Sample count
<<<<<<< Updated upstream

    Note: HDF files contain raw data and require calibration to convert encoder
    positions to energy. If calibration is not provided, energy will be in
    arbitrary encoder units.

=======
    
    Note: HDF files contain raw data and require calibration to convert encoder 
    positions to energy. If calibration is not provided, energy will be in 
    arbitrary encoder units.
    
>>>>>>> Stashed changes
    Parameters
    ----------
    file_path : str or Path
        Path to the APS HDF5 data file
    energy_calibration : tuple of (offset, scale), optional
        Calibration to convert encoder position to energy in eV:
        energy_eV = offset + scale * encoder_position
        If None, uses encoder positions as pseudo-energy
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Returns
    -------
    xarray.Dataset
        Standardized XAS dataset with energy coordinate and data variables
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    metadata = {
        'filename': file_path.name,
        'beamline': 'APS 12-BM-B',
        'format': 'HDF5 (raw)',
<<<<<<< Updated upstream
        'calibrated': energy_calibration is not None,
        'reader_version': __version__,
        'reader_name': __reader_name__
    }

    if not HAS_H5PY:
        raise ImportError("HDF5 format requires h5py library which is not available")

    with h5py.File(file_path, 'r') as f:
        # Read detector data
        if 'INENC1.VAL.Mean' not in f or 'FMC_IN.VAL1.Mean' not in f or 'FMC_IN.VAL2.Mean' not in f:
            raise ValueError("HDF5 file missing required datasets")

        encoder = f['INENC1.VAL.Mean'][:]
        i0 = f['FMC_IN.VAL1.Mean'][:]
        i1 = f['FMC_IN.VAL2.Mean'][:]
        i2 = f['FMC_IN.VAL3.Mean'][:] if 'FMC_IN.VAL3.Mean' in f else None

        # Store raw encoder values
        metadata['encoder_range'] = (float(np.min(encoder)), float(np.max(encoder)))

=======
        'calibrated': energy_calibration is not None
    }
    
    with h5py.File(file_path, 'r') as f:
        # Read detector data
        encoder = f['INENC1.VAL.Mean'][:]
        i0 = f['FMC_IN.VAL1.Mean'][:]
        i1 = f['FMC_IN.VAL2.Mean'][:]
        i2 = f['FMC_IN.VAL3.Mean'][:]
        
        # Store raw encoder values
        metadata['encoder_range'] = (encoder.min(), encoder.max())
    
>>>>>>> Stashed changes
    # Convert encoder to energy if calibration provided
    if energy_calibration is not None:
        offset, scale = energy_calibration
        energy = offset + scale * encoder
        metadata['energy_calibration'] = {'offset': offset, 'scale': scale}
    else:
        # Use normalized encoder position as pseudo-energy
        energy = encoder
        metadata['warning'] = 'No energy calibration provided - using raw encoder values'
<<<<<<< Updated upstream

    # Calculate mu from I0 and I1
    with np.errstate(divide='ignore', invalid='ignore'):
        mu_trans = np.log(i0 / i1)
        mu_ref = np.log(i1 / i2) if i2 is not None else None

    energy, mu_trans, i0, i1, i2, _ = _clean_and_sort(energy, mu_trans, i0, i1, i2, None)

    metadata['n_points'] = len(energy)
    metadata['energy_range'] = (float(np.min(energy)), float(np.max(energy)))
    metadata['mode'] = 'transmission'

=======
    
    # Calculate mu from I0 and I1
    mu_trans = np.log(i0 / i1)
    mu_ref = np.log(i1 / i2) if i2 is not None else None
    
    metadata['n_points'] = len(energy)
    metadata['energy_range'] = (energy.min(), energy.max())
    metadata['mode'] = 'transmission'
    
>>>>>>> Stashed changes
    # Create xarray Dataset with standardized structure
    data_vars = {
        'i0': ('point', i0, {'description': 'Incident beam intensity'}),
        'i1': ('point', i1, {'description': 'Transmitted beam intensity'}),
        'mu_trans': ('point', mu_trans, {'description': 'Transmission absorption coefficient ln(I0/I1)'})
    }
<<<<<<< Updated upstream

    if i2 is not None:
        data_vars['i2'] = ('point', i2, {'description': 'Reference detector intensity'})

    if mu_ref is not None:
        data_vars['mu_ref'] = ('point', mu_ref, {'description': 'Reference absorption coefficient ln(I1/I2)'})

    # Create Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'energy': ('point', energy, {'units': 'eV' if energy_calibration else 'encoder_units',
                                             'description': 'Photon energy' if energy_calibration else 'Encoder position (uncalibrated)',
                                             'calibrated': energy_calibration is not None})},
        attrs=metadata
    )

    return ds


def load_aps_xas(file_path: str | Path,
=======
    
    if i2 is not None:
        data_vars['i2'] = ('point', i2, {'description': 'Reference detector intensity'})
    
    if mu_ref is not None:
        data_vars['mu_ref'] = ('point', mu_ref, {'description': 'Reference absorption coefficient ln(I1/I2)'})
    
    # Create Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'energy': ('point', energy, {'units': 'eV' if energy_calibration else 'encoder_units', 
                                             'description': 'Photon energy' if energy_calibration else 'Encoder position (uncalibrated)'})},
        attrs=metadata
    )
    
    return ds


def load_aps_xas(file_path: str | Path, 
>>>>>>> Stashed changes
                 prefer_ascii: bool = True,
                 energy_calibration: Optional[Tuple[float, float]] = None) -> xr.Dataset:
    """
    Load XAS data from APS 12-BM-B files.
<<<<<<< Updated upstream

    Automatically detects file format and loads appropriate data.
    For samples with both HDF and ASCII files, prefers ASCII (processed) data
    unless specified otherwise.

=======
    
    Automatically detects file format and loads appropriate data.
    For samples with both HDF and ASCII files, prefers ASCII (processed) data
    unless specified otherwise.
    
>>>>>>> Stashed changes
    Parameters
    ----------
    file_path : str or Path
        Path to XAS data file (.hdf for HDF5, or ASCII file)
    prefer_ascii : bool, default True
        If both .hdf and ASCII versions exist, prefer ASCII (processed) data
    energy_calibration : tuple of (offset, scale), optional
        Only used for HDF files - calibration for encoder to energy conversion
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Returns
    -------
    xarray.Dataset
        Standardized XAS dataset with energy coordinate and data variables
<<<<<<< Updated upstream

=======
        
>>>>>>> Stashed changes
    Examples
    --------
    >>> # Load processed ASCII data
    >>> data = load_aps_xas('FeCl2-Malic_acid_(0.5-0.5)-pH2.2')
    >>> energy = data.energy
    >>> mu = data.mu_trans
<<<<<<< Updated upstream
    >>>
    >>> # Load raw HDF data with calibration
    >>> data = load_aps_xas('sample.hdf', energy_calibration=(7000.0, 1e-7))
    >>>
=======
    >>> 
    >>> # Load raw HDF data with calibration
    >>> data = load_aps_xas('sample.hdf', energy_calibration=(7000.0, 1e-7))
    >>> 
>>>>>>> Stashed changes
    >>> # Access fluorescence data if available
    >>> if data.fluorescence is not None:
    >>>     fluor_signal = data.fluorescence['total']
    """
    file_path = Path(file_path)
<<<<<<< Updated upstream

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type
    if file_path.suffix.lower() in ['.hdf', '.h5', '.hdf5']:
        return read_aps_hdf(file_path, energy_calibration)
    return read_aps_ascii(file_path)
=======
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type
    if file_path.suffix.lower() == '.hdf':
        return read_aps_hdf(file_path, energy_calibration)
    else:
        return read_aps_ascii(file_path)
>>>>>>> Stashed changes


def get_transmission_mu(data: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract transmission mode XANES data (energy, mu).
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Parameters
    ----------
    data : xarray.Dataset
        XAS dataset
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Returns
    -------
    energy : np.ndarray
        Energy in eV
    mu : np.ndarray
        Transmission absorption coefficient
    """
    return data['energy'].values, data['mu_trans'].values


<<<<<<< Updated upstream
def get_fluorescence_mu(data: xr.Dataset,
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fluorescence mode XANES data (energy, mu).

=======
def get_fluorescence_mu(data: xr.Dataset, 
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fluorescence mode XANES data (energy, mu).
    
>>>>>>> Stashed changes
    Parameters
    ----------
    data : xarray.Dataset
        XAS dataset
    normalize : bool, default True
        If True, normalize fluorescence by I0
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Returns
    -------
    energy : np.ndarray
        Energy in eV
    mu : np.ndarray
        Fluorescence signal (normalized by I0 if requested)
<<<<<<< Updated upstream

=======
        
>>>>>>> Stashed changes
    Raises
    ------
    ValueError
        If fluorescence data is not available
    """
    if 'fluor_total' not in data:
        raise ValueError("No fluorescence data available in this file")
<<<<<<< Updated upstream

    fluorescence_signal = data['fluor_total'].values

=======
    
    fluorescence_signal = data['fluor_total'].values
    
>>>>>>> Stashed changes
    if normalize and 'i0' in data:
        # Normalize fluorescence by I0
        mu = fluorescence_signal / data['i0'].values
    else:
        mu = fluorescence_signal
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    return data['energy'].values, mu


def get_reference_mu(data: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract reference foil absorption data (energy, mu_ref).
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Parameters
    ----------
    data : xarray.Dataset
        XAS dataset
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Returns
    -------
    energy : np.ndarray
        Energy in eV
    mu_ref : np.ndarray
        Reference absorption coefficient
<<<<<<< Updated upstream

=======
        
>>>>>>> Stashed changes
    Raises
    ------
    ValueError
        If reference data is not available
    """
    if 'mu_ref' not in data:
        raise ValueError("No reference data available in this file")
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    return data['energy'].values, data['mu_ref'].values


# Convenience function for batch loading
<<<<<<< Updated upstream
def load_aps_dataset(data_dir: str | Path,
=======
def load_aps_dataset(data_dir: str | Path, 
>>>>>>> Stashed changes
                     pattern: str = "*",
                     prefer_ascii: bool = True) -> List[xr.Dataset]:
    """
    Load multiple APS XAS files from a directory.
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Parameters
    ----------
    data_dir : str or Path
        Directory containing XAS data files
    pattern : str, default "*"
        Glob pattern to match files (e.g., "*.hdf", "FeCl2*")
    prefer_ascii : bool, default True
        Prefer ASCII files over HDF when both exist
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    Returns
    -------
    list of xarray.Dataset
        List of loaded XAS datasets
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
<<<<<<< Updated upstream

    # Filter out backup directories
    files = [f for f in files if f.is_file() and 'backup' not in str(f)]

    # If prefer_ascii, remove .hdf files that have corresponding ASCII versions
    if prefer_ascii:
        hdf_files = [f for f in files if f.suffix.lower() in ['.hdf', '.h5', '.hdf5']]
=======
    
    # Filter out backup directories
    files = [f for f in files if f.is_file() and 'backup' not in str(f)]
    
    # If prefer_ascii, remove .hdf files that have corresponding ASCII versions
    if prefer_ascii:
        hdf_files = [f for f in files if f.suffix.lower() == '.hdf']
>>>>>>> Stashed changes
        for hdf_file in hdf_files:
            # Check for corresponding ASCII file (same stem)
            ascii_candidates = [
                hdf_file.with_suffix(''),  # No extension
                hdf_file.with_suffix('.dat'),
                hdf_file.with_suffix('.txt')
            ]
            if any(f in files for f in ascii_candidates):
                files.remove(hdf_file)
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    dataset = []
    for file_path in files:
        try:
            data = load_aps_xas(file_path, prefer_ascii=prefer_ascii)
            dataset.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    return dataset


if __name__ == "__main__":
    # Example usage
    import sys
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default test file
        file_path = r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2.2"
<<<<<<< Updated upstream

    print(f"Loading: {file_path}\n")
    data = load_aps_xas(file_path)

=======
    
    print(f"Loading: {file_path}\n")
    data = load_aps_xas(file_path)
    
>>>>>>> Stashed changes
    print("=" * 80)
    print("Loaded APS XAS Data (xarray.Dataset)")
    print("=" * 80)
    print(f"Filename: {data.attrs['filename']}")
    print(f"Beamline: {data.attrs['beamline']}")
    print(f"Mode: {data.attrs['mode']}")
    print(f"Points: {data.attrs['n_points']}")
    print(f"Energy range: {data.attrs['energy_range'][0]:.2f} - {data.attrs['energy_range'][1]:.2f} eV")
    print(f"\nDataset structure:")
    print(data)
    print(f"\nDetectors:")
    print(f"  I0: {data['i0'].min().values:.2f} - {data['i0'].max().values:.2f}")
    print(f"  I1: {data['i1'].min().values:.2f} - {data['i1'].max().values:.2f}")
    if 'i2' in data:
        print(f"  I2: {data['i2'].min().values:.2f} - {data['i2'].max().values:.2f}")
    print(f"\nAbsorption coefficient (mu):")
    print(f"  Transmission: {data['mu_trans'].min().values:.4f} - {data['mu_trans'].max().values:.4f}")
    if 'mu_ref' in data:
        print(f"  Reference: {data['mu_ref'].min().values:.4f} - {data['mu_ref'].max().values:.4f}")
<<<<<<< Updated upstream

    if 'fluor_total' in data:
        print(f"\nFluorescence:")
        print(f"  Total: {data['fluor_total'].min().values:.2f} - {data['fluor_total'].max().values:.2f}")

=======
    
    if 'fluor_total' in data:
        print(f"\nFluorescence:")
        print(f"  Total: {data['fluor_total'].min().values:.2f} - {data['fluor_total'].max().values:.2f}")
    
>>>>>>> Stashed changes
    print("\n" + "=" * 80)
