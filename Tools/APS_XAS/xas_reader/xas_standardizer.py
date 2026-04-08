"""
XAS Data Standardizer - Loads and saves XAS data in standardized format

This module provides a high-level interface to load XAS data from various formats
and automatically save it to a standardized netCDF format in the project output directory.

Usage:
    from xas_standardizer import standardize_file, standardize_batch

    # Single file
    ds = standardize_file("path/to/file.dat")

    # Multiple files
    datasets = standardize_batch("path/to/directory", "*.dat")
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import xarray as xr

# Import the readers
try:
    from .xas_reader import load_xas_file, load_xas_batch
    from .aps_xas_reader import read_aps_ascii, load_aps_dataset, load_aps_xas
    from .prj_reader import read_prj_file, __reader_name__ as PRJ_READER_NAME, __version__ as PRJ_READER_VERSION
except ImportError:
    from xas_reader import load_xas_file, load_xas_batch
    from aps_xas_reader import read_aps_ascii, load_aps_dataset, load_aps_xas
    from prj_reader import read_prj_file, __reader_name__ as PRJ_READER_NAME, __version__ as PRJ_READER_VERSION


def get_default_output_dir():
    """Get the default output directory for standardized data."""
    # Find project_root by going up from current file location
    # xas_standardizer.py is in: zzy_llm/Tools/APS_XAS/xas_reader/
    # project_root is at: zzy_llm/project_root/
    current_file = Path(__file__).resolve()
    zzy_llm_dir = current_file.parent.parent.parent.parent  # xas_reader -> APS_XAS -> Tools -> zzy_llm
    return zzy_llm_dir / "project_root" / "xas_results" / "01_standardized_data"


def save_dataset(dataset: xr.Dataset, original_file: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Save standardized xarray.Dataset to CSV format.

    Parameters
    ----------
    dataset : xr.Dataset
        Standardized XAS dataset to save
    original_file : Path
        Original file path (used for naming)
    output_dir : Path, optional
        Directory to save to. If None, saves to project_root/xas_results/01_standardized_data

    Returns
    -------
    Path
        Path to the saved CSV file
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    else:
        output_dir = Path(output_dir)

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename - preserve full filename including extensions like .1, .2, .3
    # Replace dots with underscores to avoid confusion with CSV extension
    safe_name = original_file.name.replace('.', '_')
    output_filename = safe_name + "_standardized.csv"
    output_path = output_dir / output_filename

    # Convert to pandas DataFrame and save as CSV
    import pandas as pd

    # Build DataFrame with energy and all variables
    data_dict = {'energy': dataset.energy.values}
    for var_name in dataset.data_vars:
        data_dict[var_name] = dataset[var_name].values

    df = pd.DataFrame(data_dict)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved standardized data to: {output_path}")

    return output_path


def _prj_sample_to_dataset(sample_name: str, energy: np.ndarray, mu_norm: np.ndarray, source_file: Path) -> xr.Dataset:
    mu_norm = np.asarray(mu_norm)
    energy = np.asarray(energy)

    # Reconstruct approximate intensities from normalized mu
    mu_safe = np.clip(mu_norm, -50, 50)
    i0 = np.ones_like(mu_safe) * 1e6
    i1 = i0 / np.exp(mu_safe)

    metadata = {
        'filename': sample_name,
        'beamline': 'Athena/PRJ',
        'mode': 'transmission',
        'n_points': len(energy),
        'energy_range': (float(np.min(energy)), float(np.max(energy))),
        'reader_version': PRJ_READER_VERSION,
        'reader_name': PRJ_READER_NAME,
        'format': 'Athena PRJ',
        'source_prj': source_file.name,
        'normalized': True,
    }

    ds = xr.Dataset(
        data_vars={
            'i0': ('point', i0, {'description': 'Incident beam intensity (reconstructed)'}),
            'i1': ('point', i1, {'description': 'Transmitted beam intensity (reconstructed)'}),
            'mu_trans': ('point', mu_norm, {'description': 'Normalized absorption coefficient'})
        },
        coords={'energy': ('point', energy, {'units': 'eV', 'description': 'Photon energy', 'calibrated': True})},
        attrs=metadata
    )

    return ds


def standardize_prj(prj_path: str | Path,
                    output_dir: Optional[str | Path] = None,
                    save: bool = True,
                    exclude_refs: bool = True,
                    exclude_smoothed: bool = True,
                    name_filter: Optional[str] = None) -> List[xr.Dataset]:
    """
    Load an Athena .prj file and return a list of standardized datasets.
    """
    prj_path = Path(prj_path)
    data = read_prj_file(prj_path, exclude_refs=exclude_refs,
                         exclude_smoothed=exclude_smoothed, name_filter=name_filter)

    datasets: List[xr.Dataset] = []
    for sample_name, (energy, mu_norm) in data.items():
        ds = _prj_sample_to_dataset(sample_name, energy, mu_norm, prj_path)
        datasets.append(ds)
        if save:
            safe_sample = sample_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            pseudo_file = Path(f"{prj_path.stem}__{safe_sample}")
            save_dataset(ds, pseudo_file, output_dir)

    return datasets


def standardize_file(file_path: str | Path,
                    beamline: str = "auto",
                    output_dir: Optional[str | Path] = None,
                    save: bool = True) -> Union[xr.Dataset, List[xr.Dataset]]:
    """
    Load a single XAS file and optionally save it in standardized format.

    Parameters
    ----------
    file_path : str or Path
        Path to the XAS data file
    beamline : str, optional
        Beamline identifier. Use "auto" to detect from file format:
        - APS ASCII files (no extension) -> "APS 12-BM-B"
        - APS HDF5 files (.hdf/.h5) -> "APS 12-BM-B"
        - Athena .prj -> batch of datasets
        - Other formats -> "Generic"
    output_dir : str or Path, optional
        Directory to save standardized output. If None, saves to
        project_root/xas_results/01_standardized_data
    save : bool, optional
        Whether to save the standardized dataset (default: True)

    Returns
    -------
    xr.Dataset or list of xr.Dataset
        Standardized XAS dataset(s)
    """
    file_path = Path(file_path)

    # Handle Athena project files
    if file_path.suffix.lower() == '.prj':
        return standardize_prj(file_path, output_dir=output_dir, save=save)

    # Auto-detect beamline from file format
    if beamline == "auto":
        # Check if it's an APS ASCII file (no true extension, possibly ends with pH values)
        if file_path.suffix == "" or (file_path.suffix.startswith('.') and len(file_path.suffix) <= 3 and any(c.isdigit() for c in file_path.suffix)):
            # APS ASCII format (no extension or ends with numbers like .2 from pH2.2)
            beamline = "APS 12-BM-B"
            dataset = read_aps_ascii(file_path)
        elif file_path.suffix.lower() in ['.hdf', '.h5', '.hdf5']:
            # APS HDF5 format
            beamline = "APS 12-BM-B"
            dataset = load_aps_xas(file_path)
        else:
            # Generic format
            beamline = "Generic"
            dataset = load_xas_file(file_path, beamline=beamline)
    elif beamline == "APS 12-BM-B":
        dataset = load_aps_xas(file_path)
    else:
        dataset = load_xas_file(file_path, beamline=beamline)

    # Save if requested
    if save:
        save_dataset(dataset, file_path, output_dir)

    return dataset


def standardize_batch(data_dir: str | Path,
                     pattern: str = "*",
                     beamline: str = "auto",
                     output_dir: Optional[str | Path] = None,
                     save: bool = True) -> List[xr.Dataset]:
    """
    Load multiple XAS files and optionally save them in standardized format.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing XAS data files
    pattern : str, optional
        Glob pattern for file selection (default: "*" for all files)
    beamline : str, optional
        Beamline identifier. Use "auto" to detect from file format.
    output_dir : str or Path, optional
        Directory to save standardized output. If None, saves to
        project_root/xas_results/01_standardized_data
    save : bool, optional
        Whether to save the standardized datasets (default: True)

    Returns
    -------
    List[xr.Dataset]
        List of standardized XAS datasets
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))

    datasets: List[xr.Dataset] = []
    for file_path in files:
        try:
            ds_or_list = standardize_file(file_path, beamline=beamline,
                                         output_dir=output_dir, save=save)
            if isinstance(ds_or_list, list):
                datasets.extend(ds_or_list)
            else:
                datasets.append(ds_or_list)
        except Exception as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")

    print(f"\nSuccessfully loaded {len(datasets)} datasets from {len(files)} files")
    return datasets


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        file_or_dir = Path(sys.argv[1])

        if file_or_dir.is_file():
            print(f"Standardizing file: {file_or_dir}")
            ds = standardize_file(file_or_dir)
            if isinstance(ds, list):
                print(f"Loaded {len(ds)} samples from PRJ")
            else:
                print(f"Dataset loaded: {len(ds.energy)} points, {ds.attrs['energy_range']}")
        elif file_or_dir.is_dir():
            pattern = sys.argv[2] if len(sys.argv) > 2 else "*"
            print(f"Standardizing directory: {file_or_dir} (pattern: {pattern})")
            datasets = standardize_batch(file_or_dir, pattern)
            print(f"Loaded {len(datasets)} datasets")
        else:
            print(f"Error: {file_or_dir} not found")
    else:
        print("Usage: python xas_standardizer.py <file_or_directory> [pattern]")
        print("Example: python xas_standardizer.py data.dat")
        print("Example: python xas_standardizer.py data_folder '*.dat'")
