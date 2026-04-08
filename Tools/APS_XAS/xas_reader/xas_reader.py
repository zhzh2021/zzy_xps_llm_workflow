"""
XAS Data Reader Module (Standardized xarray Format)

Handles loading XAS data from various file formats commonly used at beamlines.
Outputs standardized xarray.Dataset format compatible with all downstream analyzers.

Supports XDI format and generic ASCII formats with energy and intensity columns.
Conforms to Standardized XAS Data Format Specification v3.0.
"""

__version__ = '3.0'
__reader_name__ = 'Generic_XAS_Reader'

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# Make larch optional
try:
    from larch import Group
    from larch.io import read_ascii, read_xdi
    HAS_LARCH = True
except ImportError:
    HAS_LARCH = False
    print("Warning: larch not available - XDI format will not be supported")

# Import reference functions (also optional)
try:
    from .xas_reference_loader import calibrate_energy_with_reference, create_reference_diagnostic_plot
except ImportError:
    try:
        from xas_reference_loader import calibrate_energy_with_reference, create_reference_diagnostic_plot
    except ImportError:
        pass  # Reference functions not available


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


def save_standardized_dataset(dataset: xr.Dataset, original_file: Path, output_dir: Optional[Path] = None):
    """
    Save standardized xarray.Dataset to netCDF format.

    Parameters
    ----------
    dataset : xr.Dataset
        Standardized XAS dataset to save
    original_file : Path
        Original file path (used for naming)
    output_dir : Path, optional
        Directory to save to. If None, saves to project_root/xas_results/01_standardized_data
    """
    if output_dir is None:
        # Find project_root by going up from current file location
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent  # Tools/APS_XAS/xas_reader -> project_root
        output_dir = project_root / "xas_results" / "01_standardized_data"

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    output_filename = original_file.stem + "_standardized.nc"
    output_path = output_dir / output_filename

    # Save to netCDF format
    dataset.to_netcdf(output_path)
    print(f"Saved standardized data to: {output_path}")
    return output_path


def load_xas_file(file_path: str | Path, beamline: str = "Generic",
                  output_dir: Optional[str | Path] = None, save_output: bool = True) -> xr.Dataset:
    """
    Load XAS data from a single file into standardized xarray.Dataset format.

    Supports:
    - XDI format (.xdi)
    - Generic ASCII with columns: energy, i0, it, ir (transmission with reference)
    - Generic ASCII with columns: energy, i0, it (transmission)
    - Generic ASCII with columns: energy, i0, iff (fluorescence)
    - Generic ASCII with columns: energy, mu (direct absorption)

    For 4-column data (energy, i0, it, ir), the reference channel (ir) is used to calculate
    mu_ref = ln(It/Ir). Energy calibration using the reference foil is NOT applied.

    Fluorescence mode is detected automatically from file headers containing
    keywords like 'fluorescence', 'fluor', 'iff', or 'i_ff'.

    Parameters
    ----------
    file_path : str or Path
        Path to the XAS data file
    beamline : str, optional
        Beamline identifier (default: "Generic")
    output_dir : str or Path, optional
        Directory to save standardized output. If None and save_output=True,
        saves to project_root/xas_results/01_standardized_data
    save_output : bool, optional
        Whether to save the dataset to netCDF format (default: True)

    Returns
    -------
    xarray.Dataset
        Standardized XAS dataset with energy coordinate and data variables
        Conforms to Standardized XAS Data Format Specification v3.0

    Raises
    ------
    ValueError
        If file format is not supported or data cannot be parsed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"XAS file not found: {file_path}")

    ext = file_path.suffix.lower()

    # Initialize metadata
    metadata = {
        'filename': file_path.name,
        'beamline': beamline,
        'reader_version': __version__,
        'reader_name': __reader_name__
    }

    # Initialize optional data containers
    i0_data = None
    i1_data = None
    i2_data = None
    fluor_data = None

    try:
        if ext == '.xdi':
            # XDI format (standard beamline format)
            if not HAS_LARCH:
                raise ImportError("XDI format requires larch library which is not available")

            xdi_data = read_xdi(str(file_path))
            energy = xdi_data.energy
            mu = xdi_data.mutrans  # transmission mu(E)
            energy_calibrated = energy  # XDI files typically already calibrated

            # Extract intensities if available
            i0_data = getattr(xdi_data, 'i0', None)
            i1_data = getattr(xdi_data, 'it', None)
            i2_data = getattr(xdi_data, 'ir', None)

            # If intensities available, recalculate mu
            if i0_data is not None and i1_data is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu = np.log(i0_data / i1_data)

            metadata['format'] = 'XDI'
            metadata['mode'] = 'transmission'

        elif ext in ['.dat', '.txt', '.csv', '.asc']:
            # Generic ASCII format
            # First check for header information to detect fluorescence mode
            is_fluorescence = False
            header_lines = []

            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#') or line.strip() == '':
                            header_lines.append(line.strip())
                        else:
                            break

                # Check header for fluorescence indicators
                header_text = '\n'.join(header_lines).lower()
                if any(keyword in header_text for keyword in ['fluorescence', 'flourescence', 'fluor', 'iff', 'i_ff']):
                    is_fluorescence = True
            except Exception:
                pass

            raw = _load_ascii_table(file_path)

            if raw.shape[1] >= 4:
                # Assume columns: energy, i0, it, ir (transmission with reference)
                energy = raw[:, 0]
                i0_data = raw[:, 1]
                i1_data = raw[:, 2]
                i2_data = raw[:, 3]
                # Calculate absorption coefficient: mu = ln(I0/It)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu = np.log(i0_data / i1_data)

                # Reference calibration is not applied - only used for diagnostics
                energy_calibrated = energy
                metadata['format'] = 'ASCII (4-column with reference)'
                metadata['mode'] = 'transmission'

            elif raw.shape[1] >= 3 and is_fluorescence:
                # Fluorescence mode: energy, i0, iff
                energy = raw[:, 0]
                i0_data = raw[:, 1]
                fluor_data = raw[:, 2]
                # Calculate absorption coefficient for fluorescence: mu = I_ff / I_0
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu = fluor_data / i0_data
                energy_calibrated = energy  # No calibration for fluorescence mode
                metadata['format'] = 'ASCII (fluorescence)'
                metadata['mode'] = 'fluorescence'

            elif raw.shape[1] >= 3:
                # Assume columns: energy, i0, it (transmission mode)
                energy = raw[:, 0]
                i0_data, i1_data = raw[:, 1], raw[:, 2]
                # Calculate absorption coefficient: mu = ln(I0/It)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu = np.log(i0_data / i1_data)
                energy_calibrated = energy  # No reference for calibration
                metadata['format'] = 'ASCII (3-column transmission)'
                metadata['mode'] = 'transmission'

            elif raw.shape[1] >= 2:
                # Assume columns: energy, mu (direct absorption)
                energy = raw[:, 0]
                mu = raw[:, 1]
                energy_calibrated = energy  # No calibration for direct absorption
                metadata['format'] = 'ASCII (2-column direct mu)'
                metadata['mode'] = 'transmission'
            else:
                raise ValueError("ASCII file must have at least 2 columns (energy, intensity)")

        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .xdi, .dat, .txt, .csv, .asc")

        # Basic validation
        if len(energy) == 0 or len(mu) == 0:
            raise ValueError("No data found in file")

        if len(energy) != len(mu):
            raise ValueError(f"Energy and intensity arrays have different lengths: {len(energy)} vs {len(mu)}")

        # Clean data and sort by energy
        energy_calibrated, mu, i0_data, i1_data, i2_data, fluor_data = _clean_and_sort(
            energy_calibrated, mu, i0_data, i1_data, i2_data, fluor_data
        )

        if len(energy_calibrated) < 10:
            raise ValueError("Insufficient data points after cleaning")

        # Build standardized dataset
        metadata['n_points'] = len(energy_calibrated)
        metadata['energy_range'] = (float(energy_calibrated.min()), float(energy_calibrated.max()))

        # Create data variables dict - REQUIRED variables
        data_vars = {}

        # If we have intensities, use them; otherwise create from mu
        if i0_data is not None and i1_data is not None:
            data_vars['i0'] = ('point', i0_data, {'description': 'Incident beam intensity'})
            data_vars['i1'] = ('point', i1_data, {'description': 'Transmitted beam intensity'})
        else:
            # Reconstruct approximate intensities from mu
            mu_safe = np.clip(mu, -50, 50)
            i0_data = np.ones_like(mu_safe) * 1e6  # Arbitrary normalization
            i1_data = i0_data / np.exp(mu_safe)
            data_vars['i0'] = ('point', i0_data, {'description': 'Incident beam intensity (reconstructed)'})
            data_vars['i1'] = ('point', i1_data, {'description': 'Transmitted beam intensity (reconstructed)'})

        data_vars['mu_trans'] = ('point', mu, {'description': 'Transmission absorption coefficient ln(I0/I1)'})

        # OPTIONAL variables
        if i2_data is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                mu_ref = np.log(i1_data / i2_data)
            data_vars['i2'] = ('point', i2_data, {'description': 'Reference detector intensity'})
            data_vars['mu_ref'] = ('point', mu_ref, {'description': 'Reference absorption coefficient ln(I1/I2)'})

        if fluor_data is not None:
            data_vars['fluor_total'] = ('point', fluor_data, {'description': 'Total fluorescence signal'})

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={'energy': ('point', energy_calibrated, {
                'units': 'eV',
                'description': 'Photon energy',
                'calibrated': True
            })},
            attrs=metadata
        )

        return ds

    except Exception as e:
        raise ValueError(f"Failed to load XAS file {file_path}: {str(e)}")


def load_xas_batch(data_dir: str | Path, pattern: str = "*", beamline: str = "Generic") -> List[xr.Dataset]:
    """
    Load multiple XAS files from a directory into standardized xarray.Dataset format.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing XAS files
    pattern : str
        Glob pattern for file matching (default: "*" for all files)
    beamline : str, optional
        Beamline identifier (default: "Generic")

    Returns
    -------
    list of xarray.Dataset
        List of loaded XAS datasets in standardized format
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    datasets = []

    # Find all matching files
    for file_path in sorted(data_dir.glob(pattern)):
        if not file_path.is_file():
            continue

        try:
            data = load_xas_file(file_path, beamline=beamline)
            datasets.append(data)
            print(f"Loaded {file_path.name}")
        except Exception as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")
            continue

    if not datasets:
        raise ValueError(f"No valid XAS files found in {data_dir}")

    print(f"Successfully loaded {len(datasets)} XAS files")
    return datasets


# Helper functions for data extraction (matching aps_xas_reader API)

def get_transmission_mu(data: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract transmission mode XANES data (energy, mu).

    Parameters
    ----------
    data : xarray.Dataset
        XAS dataset

    Returns
    -------
    energy : np.ndarray
        Energy in eV
    mu : np.ndarray
        Transmission absorption coefficient
    """
    return data['energy'].values, data['mu_trans'].values


def get_fluorescence_mu(data: xr.Dataset,
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fluorescence mode XANES data (energy, mu).

    Parameters
    ----------
    data : xarray.Dataset
        XAS dataset
    normalize : bool, default True
        If True, normalize fluorescence by I0

    Returns
    -------
    energy : np.ndarray
        Energy in eV
    mu : np.ndarray
        Fluorescence signal (normalized by I0 if requested)

    Raises
    ------
    ValueError
        If fluorescence data is not available
    """
    if 'fluor_total' not in data:
        raise ValueError("No fluorescence data available in this file")

    fluorescence_signal = data['fluor_total'].values

    if normalize and 'i0' in data:
        # Normalize fluorescence by I0
        mu = fluorescence_signal / data['i0'].values
    else:
        mu = fluorescence_signal

    return data['energy'].values, mu


def get_reference_mu(data: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract reference foil absorption data (energy, mu_ref).

    Parameters
    ----------
    data : xarray.Dataset
        XAS dataset

    Returns
    -------
    energy : np.ndarray
        Energy in eV
    mu_ref : np.ndarray
        Reference absorption coefficient

    Raises
    ------
    ValueError
        If reference data is not available
    """
    if 'mu_ref' not in data:
        raise ValueError("No reference data available in this file")

    return data['energy'].values, data['mu_ref'].values


def detect_xas_file_type(file_path: str | Path) -> Dict[str, Any]:
    """
    Detect XAS file type and extract metadata.

    Parameters
    ----------
    file_path : str or Path
        Path to the XAS file

    Returns
    -------
    info : dict
        File type information and metadata
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    info = {
        'file_path': str(file_path),
        'extension': ext,
        'format': 'unknown',
        'technique': 'unknown',
        'beamline': 'unknown',
        'energy_calibration': 'unknown',
        'facility': None,
        'element': None,
        'edge': None
    }

    if ext == '.xdi':
        info['format'] = 'XDI'
        # Parse XDI header for metadata
        try:
            with open(file_path, 'r') as f:
                header_lines = []
                for line in f:
                    if line.startswith('#'):
                        header_lines.append(line.strip())
                    else:
                        break

                header_text = '\n'.join(header_lines)

                # Extract beamline information
                if 'Beamline.' in header_text:
                    for line in header_lines:
                        if 'Beamline.' in line and 'name' in line.lower():
                            parts = line.split(':', 1)
                            if len(parts) > 1:
                                info['beamline'] = parts[1].strip()

                # Extract energy calibration
                if 'Mono.' in header_text:
                    for line in header_lines:
                        if 'Mono.' in line and 'calibration' in line.lower():
                            parts = line.split(':', 1)
                            if len(parts) > 1:
                                info['energy_calibration'] = parts[1].strip()

                # Detect technique from header
                header_lower = header_text.lower()
                if 'fluorescence' in header_lower or 'fluor' in header_lower or 'iff' in header_lower:
                    info['technique'] = 'XAS-fluorescence'
                elif 'xanes' in header_lower:
                    info['technique'] = 'XANES'
                elif 'exafs' in header_lower:
                    info['technique'] = 'EXAFS'
                else:
                    info['technique'] = 'XAS'

        except Exception as e:
            print(f"Warning: Could not parse XDI header for {file_path}: {e}")

    elif ext in ['.dat', '.txt', '.csv', '.asc']:
        info['format'] = 'ASCII'
        # Parse header for beamline metadata
        try:
            with open(file_path, 'r') as f:
                header_lines = []
                line_count = 0
                for line in f:
                    if line.startswith('#') and line_count < 30:  # Read first 30 lines for metadata
                        header_lines.append(line.strip())
                        line_count += 1
                    elif not line.startswith('#'):
                        break

                # Parse metadata from header
                for line in header_lines:
                    line = line.lstrip('#').strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()

                        if key == 'facility':
                            info['facility'] = value
                        elif key == 'beamline':
                            info['beamline'] = value
                        elif key == 'year':
                            info['year'] = int(value) if value.isdigit() else value
                        elif key == 'cycle':
                            info['cycle'] = int(value) if value.isdigit() else value
                        elif key == 'pi':
                            info['pi'] = value
                        elif key == 'proposal':
                            info['proposal'] = value
                        elif key == 'scan_id':
                            info['scan_id'] = int(value) if value.isdigit() else value
                        elif key == 'element':
                            # Extract element symbol (e.g., "Iron ( 26)" -> "Fe")
                            if '(' in value:
                                element_part = value.split('(')[0].strip()
                                # Convert element name to symbol if needed
                                element_name_to_symbol = {
                                    'iron': 'Fe', 'copper': 'Cu', 'nickel': 'Ni',
                                    'zinc': 'Zn', 'cobalt': 'Co', 'chromium': 'Cr'
                                }
                                element_lower = element_part.lower()
                                info['element'] = element_name_to_symbol.get(element_lower, element_part)
                            else:
                                info['element'] = value
                        elif key == 'edge':
                            info['edge'] = value
                        elif key == 'e0':
                            try:
                                info['e0'] = float(value)
                            except ValueError:
                                info['e0'] = value

                # Determine technique from metadata
                header_text = '\n'.join(header_lines).lower()
                if 'fluorescence' in header_text or 'fluor' in header_text or 'iff' in header_text:
                    info['technique'] = 'XAS-fluorescence'
                elif info.get('edge') and info.get('element'):
                    info['technique'] = f"{info['element']}-{info['edge']}"
                elif 'xanes' in file_path.name.lower():
                    info['technique'] = 'XANES'
                elif 'exafs' in file_path.name.lower():
                    info['technique'] = 'EXAFS'
                else:
                    info['technique'] = 'XAS'

        except Exception as e:
            print(f"Warning: Could not parse header metadata for {file_path}: {e}")
            # Fallback to filename-based detection
            name_lower = file_path.name.lower()
            if 'fluorescence' in name_lower or 'fluor' in name_lower:
                info['technique'] = 'XAS-fluorescence'
            elif 'xanes' in name_lower:
                info['technique'] = 'XANES'
            elif 'exafs' in name_lower:
                info['technique'] = 'EXAFS'
            else:
                info['technique'] = 'XAS'

    return info


def read_xas_file(file_path: str | Path) -> Dict[str, Any]:
    """
    Load XAS data from a single file and return structured data.

    Returns a dictionary with energy, mu, and metadata for workflow compatibility.

    Parameters
    ----------
    file_path : str or Path
        Path to the XAS data file

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'energy': np.ndarray of energy values
        - 'mu': np.ndarray of absorption coefficients
        - 'metadata': dict of file metadata
        - 'flags': list of validation flags
    """
    ds = load_xas_file(file_path)
    energy = ds['energy'].values
    mu = ds['mu_trans'].values
    metadata = detect_xas_file_type(file_path)

    # Basic validation checks
    flags = []
    if not np.all(np.diff(energy) > 0):
        flags.append("non_monotonic_energy")
    if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
        flags.append("invalid_intensity")
    if len(energy) < 10:
        flags.append("insufficient_data_points")

    return {
        "energy": energy,
        "mu": mu,
        "metadata": metadata,
        "flags": flags
    }
