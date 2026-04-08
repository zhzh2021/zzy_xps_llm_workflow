"""
Reader for Athena project (.prj) files.

Provides functions to read XAS data from Athena/Demeter project files
which contain pre-normalized spectra.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

__version__ = '3.0'
__reader_name__ = 'Athena_PRJ_Reader'

try:
    from larch.io import read_athena
    HAS_LARCH = True
except ImportError:
    HAS_LARCH = False


def read_prj_file(prj_path: str,
                  exclude_refs: bool = True,
                  exclude_smoothed: bool = True,
                  name_filter: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Read XAS data from Athena project file.

    Parameters
    ----------
    prj_path : str
        Path to .prj file
    exclude_refs : bool, default=True
        Exclude reference samples (names starting with 'd_Ref_')
    exclude_smoothed : bool, default=True
        Exclude smoothed samples (names containing 'boxcar')
    name_filter : str, optional
        Only include samples containing this string in their name

    Returns
    -------
    data : dict
        Dictionary mapping sample names to (energy, mu_norm) tuples

    Examples
    --------
    >>> data = read_prj_file('samples.prj')
    >>> for name, (energy, mu) in data.items():
    ...     print(f"{name}: {len(energy)} points")
    """
    if not HAS_LARCH:
        raise ImportError("Athena PRJ reading requires larch. Install with: pip install xraylarch")

    prj_path = Path(prj_path)
    if not prj_path.exists():
        raise FileNotFoundError(f"Project file not found: {prj_path}")

    # Read all groups from Athena project
    groups = read_athena(str(prj_path))

    data = {}
    excluded_count = 0

    for name, group in groups.items():
        # Apply filters
        if exclude_refs and name.startswith('d_Ref_'):
            excluded_count += 1
            continue

        if exclude_smoothed and 'boxcar' in name.lower():
            excluded_count += 1
            continue

        if name_filter and name_filter.lower() not in name.lower():
            excluded_count += 1
            continue

        # Extract energy and normalized mu
        if hasattr(group, 'energy') and hasattr(group, 'norm'):
            energy = np.array(group.energy)
            mu_norm = np.array(group.norm)
            data[name] = (energy, mu_norm)
        else:
            print(f"Warning: Group '{name}' missing energy or norm data, skipping")

    print(f"\nRead {len(data)} samples from {prj_path.name}")
    if excluded_count > 0:
        print(f"Excluded {excluded_count} groups (refs/smoothed/filtered)")

    return data


def get_prj_sample_info(prj_path: str) -> List[Dict]:
    """
    Get information about all samples in a .prj file without loading full data.

    Parameters
    ----------
    prj_path : str
        Path to .prj file

    Returns
    -------
    info : list of dict
        List of sample information dictionaries
    """
    if not HAS_LARCH:
        raise ImportError("Athena PRJ reading requires larch. Install with: pip install xraylarch")

    groups = read_athena(str(prj_path))

    info = []
    for name, group in groups.items():
        sample_info = {
            'name': name,
            'has_energy': hasattr(group, 'energy'),
            'has_mu': hasattr(group, 'mu'),
            'has_norm': hasattr(group, 'norm'),
            'is_reference': name.startswith('d_Ref_'),
            'is_smoothed': 'boxcar' in name.lower()
        }

        if hasattr(group, 'energy'):
            sample_info['n_points'] = len(group.energy)
            sample_info['energy_range'] = (float(group.energy[0]), float(group.energy[-1]))

        if hasattr(group, 'e0'):
            sample_info['e0'] = float(group.e0)

        info.append(sample_info)

    return info


def save_prj_samples_to_dat(prj_path: str,
                            output_dir: str,
                            exclude_refs: bool = True,
                            exclude_smoothed: bool = True) -> List[Path]:
    """
    Extract samples from .prj file and save as individual .dat files.

    Parameters
    ----------
    prj_path : str
        Path to .prj file
    output_dir : str
        Directory to save .dat files
    exclude_refs : bool, default=True
        Exclude reference samples
    exclude_smoothed : bool, default=True
        Exclude smoothed samples

    Returns
    -------
    file_paths : list of Path
        Paths to created .dat files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read data
    data = read_prj_file(prj_path, exclude_refs=exclude_refs, exclude_smoothed=exclude_smoothed)

    file_paths = []
    for name, (energy, mu_norm) in data.items():
        # Create safe filename
        safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        dat_file = output_dir / f"{safe_name}.dat"

        # Write .dat file
        with open(dat_file, 'w') as f:
            f.write("# XAS data extracted from Athena project file\n")
            f.write(f"# Sample: {name}\n")
            f.write(f"# Source: {Path(prj_path).name}\n")
            f.write("# Column.1: energy (eV)\n")
            f.write("# Column.2: mu (normalized)\n")
            f.write("#----\n")
            for e, m in zip(energy, mu_norm):
                f.write(f"{e:.6f}  {m:.6f}\n")

        file_paths.append(dat_file)
        print(f"Created: {dat_file.name} ({len(energy)} points)")

    print(f"\nExtracted {len(file_paths)} samples to {output_dir}")
    return file_paths


if __name__ == "__main__":
    # Example usage
    prj_file = r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\Jiang samples-03032026.prj"

    print("="*80)
    print("READING ATHENA PROJECT FILE")
    print("="*80)

    # Get sample info
    print("\nGetting sample information...")
    info = get_prj_sample_info(prj_file)

    print(f"\nTotal groups in file: {len(info)}")
    print(f"References: {sum(1 for s in info if s['is_reference'])}")
    print(f"Smoothed: {sum(1 for s in info if s['is_smoothed'])}")
    print(f"Regular samples: {sum(1 for s in info if not s['is_reference'] and not s['is_smoothed'])}")

    # Read data (excluding refs and smoothed)
    print("\n" + "="*80)
    print("READING SAMPLE DATA (excluding references and smoothed)")
    print("="*80)
    data = read_prj_file(prj_file, exclude_refs=True, exclude_smoothed=True)

    print("\nSample names:")
    for i, name in enumerate(data.keys(), 1):
        energy, mu = data[name]
        print(f"{i:2d}. {name} ({len(energy)} points)")
