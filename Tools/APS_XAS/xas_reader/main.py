"""
XAS Data Reader - Main Entry Point

Single entry point for loading and standardizing XAS data from any format.
Automatically detects file format, loads data, and saves to standardized CSV.

Usage:
    # From Python
    from main import load_and_standardize, batch_process

    # Single file
    dataset = load_and_standardize("data.dat")

    # Batch processing
    datasets = batch_process("data_folder/", "*.dat")

    # Command line
    python main.py path/to/file.dat
    python main.py path/to/folder/ --pattern "*.dat"
"""

import sys
from pathlib import Path
from typing import List, Optional, Union
import xarray as xr

# Import all reader components so the entry point registers all formats
try:
    from aps_xas_reader import load_aps_xas  # noqa: F401
    from prj_reader import read_prj_file  # noqa: F401
    from xas_reader import load_xas_file  # noqa: F401
except ImportError:
    # If used as a package
    from .aps_xas_reader import load_aps_xas  # type: ignore  # noqa: F401
    from .prj_reader import read_prj_file  # type: ignore  # noqa: F401
    from .xas_reader import load_xas_file  # type: ignore  # noqa: F401

# Import standardizer (dispatch layer)
try:
    from xas_standardizer import standardize_file, standardize_batch, get_default_output_dir
except ImportError:
    from .xas_standardizer import standardize_file, standardize_batch, get_default_output_dir  # type: ignore

__version__ = '3.0'


def load_and_standardize(file_path: Union[str, Path],
                         beamline: str = "auto",
                         output_dir: Optional[Union[str, Path]] = None,
                         save: bool = True) -> Union[xr.Dataset, List[xr.Dataset]]:
    """
    Load and standardize a single XAS file (main entry point).

    Automatically detects file format:
    - APS ASCII files (no extension or .2/.5 pH values)
    - APS HDF5 files (.hdf, .h5)
    - Generic XAS files (.dat, .txt, .csv, .xdi)
    - Athena project files (.prj) -> returns list of datasets

    Parameters
    ----------
    file_path : str or Path
        Path to XAS data file
    beamline : str, default "auto"
        Beamline identifier ("auto", "APS 12-BM-B", "Generic")
    output_dir : str or Path, optional
        Output directory. Default: project_root/xas_results/01_standardized_data
    save : bool, default True
        Whether to save standardized CSV file

    Returns
    -------
    xr.Dataset or List[xr.Dataset]
        Standardized XAS dataset(s)

    Examples
    --------
    >>> ds = load_and_standardize("FeCl2_sample.dat")
    >>> print(ds.energy.values)
    >>> print(ds.mu_trans.values)
    """
    return standardize_file(file_path, beamline=beamline,
                           output_dir=output_dir, save=save)


def batch_process(data_dir: Union[str, Path],
                 pattern: str = "*",
                 beamline: str = "auto",
                 output_dir: Optional[Union[str, Path]] = None,
                 save: bool = True) -> List[xr.Dataset]:
    """
    Process multiple XAS files from a directory (batch entry point).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing XAS files
    pattern : str, default "*"
        Glob pattern for file selection (e.g., "*.dat", "FeCl2*")
    beamline : str, default "auto"
        Beamline identifier
    output_dir : str or Path, optional
        Output directory. Default: project_root/xas_results/01_standardized_data
    save : bool, default True
        Whether to save standardized CSV files

    Returns
    -------
    List[xr.Dataset]
        List of standardized XAS datasets

    Examples
    --------
    >>> datasets = batch_process("xas_raw_data/", "*.dat")
    >>> print(f"Loaded {len(datasets)} datasets")
    """
    return standardize_batch(data_dir, pattern=pattern, beamline=beamline,
                           output_dir=output_dir, save=save)


def get_output_directory() -> Path:
    """
    Get the default output directory for standardized data.

    Returns
    -------
    Path
        Default output directory: project_root/xas_results/01_standardized_data
    """
    return get_default_output_dir()


def get_raw_data_directory() -> Path:
    """
    Get the default raw data directory.

    Returns
    -------
    Path
        Default raw data directory: project_root/xas_raw_data
    """
    # Find project_root by going up from current file location
    # main.py is in: zzy_llm/Tools/APS_XAS/xas_reader/main.py
    # project_root is at: zzy_llm/project_root
    current_file = Path(__file__).resolve()
    zzy_llm_dir = current_file.parent.parent.parent.parent  # xas_reader -> APS_XAS -> Tools -> zzy_llm
    return zzy_llm_dir / "project_root" / "xas_raw_data"


def main():
    """Command-line interface for XAS data reader."""
    import argparse

    parser = argparse.ArgumentParser(
        description='XAS Data Reader - Load and standardize XAS data from any format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in default raw data directory
  python main.py

  # Process specific file
  python main.py --file data.dat

  # Process specific subdirectory
  python main.py --dir 021526-FeCL2-FeSO4-MA-TA-pH2-5

  # Batch with pattern
  python main.py --pattern "FeCl2*"

  # Specify beamline
  python main.py --file data.dat --beamline "APS 12-BM-B"

  # Custom output directory
  python main.py --output results/

  # Load only (no save)
  python main.py --no-save
        """
    )

    parser.add_argument('--file', type=str, default=None,
                       help='Specific file to process (relative to raw data dir or absolute path)')
    parser.add_argument('--dir', type=str, default=None,
                       help='Specific directory to process (relative to raw data dir or absolute path)')
    parser.add_argument('--pattern', type=str, default='*',
                       help='Glob pattern for batch processing (default: "*")')
    parser.add_argument('--beamline', type=str, default='auto',
                       choices=['auto', 'APS 12-BM-B', 'Generic'],
                       help='Beamline identifier (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: project_root/xas_results/01_standardized_data)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save standardized files')
    parser.add_argument('--version', action='version',
                       version=f'XAS Data Reader v{__version__}')

    args = parser.parse_args()

    save = not args.no_save
    raw_data_dir = get_raw_data_directory()

    # Determine what to process
    if args.file:
        # Single file specified
        path = Path(args.file)
        if not path.is_absolute():
            path = raw_data_dir / path
    elif args.dir:
        # Directory specified
        path = Path(args.dir)
        if not path.is_absolute():
            path = raw_data_dir / path
    else:
        # Default: process all files in raw data directory
        path = raw_data_dir

    # Print header
    print("=" * 70)
    print("XAS DATA READER v{} - Standardization Tool".format(__version__))
    print("=" * 70)
    print(f"\nDefault raw data directory: {raw_data_dir}")

    if not path.exists():
        print(f"\nError: Path not found: {path}")
        return 1

    if path.is_file():
        # Single file processing
        print(f"\nProcessing file: {path.name}")
        print(f"Beamline: {args.beamline}")
        print(f"Save output: {save}")
        if save:
            output_dir = Path(args.output) if args.output else get_default_output_dir()
            print(f"Output directory: {output_dir}")

        print("\nLoading data...")
        result = load_and_standardize(path, beamline=args.beamline,
                                      output_dir=args.output, save=save)

        if isinstance(result, list):
            print("\nDataset Summary:")
            print(f"  Samples loaded: {len(result)}")
            if result:
                print(f"  First sample: {result[0].attrs.get('filename', 'Unknown')}")
                print(f"  Energy range: {result[0].energy.values[0]:.2f} to {result[0].energy.values[-1]:.2f} eV")
                print(f"  Variables: {', '.join(result[0].data_vars.keys())}")
                print(f"  Format: {result[0].attrs.get('format', 'Unknown')}")
        else:
            ds = result
            print("\nDataset Summary:")
            print(f"  Energy points: {len(ds.energy)}")
            print(f"  Energy range: {ds.energy.values[0]:.2f} to {ds.energy.values[-1]:.2f} eV")
            print(f"  Variables: {', '.join(ds.data_vars.keys())}")
            print(f"  Format: {ds.attrs.get('format', 'Unknown')}")
            print(f"  Reader: {ds.attrs.get('reader_name', 'Unknown')} v{ds.attrs.get('reader_version', '?')}")

    elif path.is_dir():
        # Batch processing
        print(f"\nProcessing directory: {path}")
        print(f"Pattern: {args.pattern}")
        print(f"Beamline: {args.beamline}")
        print(f"Save output: {save}")
        if save:
            output_dir = Path(args.output) if args.output else get_default_output_dir()
            print(f"Output directory: {output_dir}")

        print("\nLoading files...")
        datasets = batch_process(path, pattern=args.pattern,
                                beamline=args.beamline,
                                output_dir=args.output, save=save)

        print("\nBatch Summary:")
        print(f"  Files processed: {len(datasets)}")
        if datasets:
            total_points = sum(len(ds.energy) for ds in datasets)
            print(f"  Total energy points: {total_points}")
            print(f"  Average points per file: {total_points / len(datasets):.0f}")

            # Show first and last files
            print(f"\n  First file: {datasets[0].attrs.get('filename', 'Unknown')}")
            print(f"    Energy: {datasets[0].energy.values[0]:.1f} to {datasets[0].energy.values[-1]:.1f} eV")
            if len(datasets) > 1:
                print(f"  Last file: {datasets[-1].attrs.get('filename', 'Unknown')}")
                print(f"    Energy: {datasets[-1].energy.values[0]:.1f} to {datasets[-1].energy.values[-1]:.1f} eV")

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
