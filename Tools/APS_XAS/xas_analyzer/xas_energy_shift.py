"""
XAS Energy Shift Correction Module

Energy calibration and shift correction for XAS spectra.
Adapted from autoXAS library (https://github.com/UlrikFriisJensen/autoXAS).

Functions:
- calculate_edge_shift: Calculate energy shifts by comparing theoretical vs measured edge energies
- apply_energy_correction: Apply calculated shifts to align spectra
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import warnings

# Larch imports for XAS processing
try:
    from larch.xafs import find_e0
    from larch.xray import xray_edge
except ImportError:
    raise ImportError("Larch library required for XAS energy shift correction. Install with: pip install larch")


class XASEnergyCalibrator:
    """
    XAS Energy Shift Correction and Calibration.

    Calculates and applies energy shifts to align XAS spectra based on
    theoretical edge energies vs measured edge positions.

    Adapted from autoXAS library energy correction methods.
    """

    def __init__(self,
                 edge_energies: Optional[Dict[str, str]] = None,
                 energy_unit: str = 'eV'):
        """
        Initialize energy calibrator.

        Parameters
        ----------
        edge_energies : dict, optional
            Dictionary mapping element symbols to edge types (e.g., {'Cu': 'K', 'Fe': 'K'})
            If None, defaults to common XAS edges
        energy_unit : str
            Energy unit ('eV' or 'keV')
        """
        # Default edge configurations for common XAS elements
        self.edge_energies = edge_energies or {
            'Ti': 'K', 'V': 'K', 'Cr': 'K', 'Mn': 'K', 'Fe': 'K', 'Co': 'K', 'Ni': 'K', 'Cu': 'K', 'Zn': 'K',
            'Zr': 'K', 'Nb': 'K', 'Mo': 'K', 'Ru': 'K', 'Rh': 'K', 'Pd': 'K', 'Ag': 'K', 'Cd': 'K', 'In': 'K', 'Sn': 'K',
            'Ce': 'L3', 'Pr': 'L3', 'Nd': 'L3', 'Sm': 'L3', 'Eu': 'L3', 'Gd': 'L3', 'Tb': 'L3', 'Dy': 'L3',
            'Ho': 'L3', 'Er': 'L3', 'Tm': 'L3', 'Yb': 'L3', 'Lu': 'L3'
        }

        self.energy_unit = energy_unit
        self.edge_correction_energies = {}  # Stores calculated shifts

    def calculate_edge_shift(self,
                           energy: np.ndarray,
                           mu: np.ndarray,
                           element: str,
                           reference_measurement: Optional[int] = None) -> float:
        """
        Calculate energy shift for a single spectrum.

        Compares theoretical edge energy with measured edge position.

        Parameters
        ----------
        energy : np.ndarray
            Energy values
        mu : np.ndarray
            Absorption coefficient μ(E)
        element : str
            Element symbol (e.g., 'Cu', 'Fe')
        reference_measurement : int, optional
            Specific measurement number to use for shift calculation

        Returns
        -------
        shift : float
            Energy shift in eV (positive = measured edge higher than theoretical)
        """
        if element not in self.edge_energies:
            warnings.warn(f"No edge configuration for element {element}. Using K-edge as default.")
            edge = 'K'
        else:
            edge = self.edge_energies[element]

        # Get theoretical edge energy
        try:
            edge_energy_theoretical = xray_edge(element, edge, energy_only=True)
            if self.energy_unit == 'keV':
                edge_energy_theoretical /= 1000  # Convert eV to keV
        except Exception as e:
            raise ValueError(f"Could not get theoretical edge energy for {element} {edge}: {e}")

        # Find measured edge energy
        try:
            edge_energy_measured = find_e0(energy, mu)
        except Exception as e:
            raise ValueError(f"Could not find edge energy in spectrum for {element}: {e}")

        # Calculate shift (theoretical - measured)
        shift = edge_energy_theoretical - edge_energy_measured

        return shift

    def calibrate_batch(self,
                       spectra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       element_assignments: Dict[str, str],
                       reference_measurement: int = 1) -> Dict[str, float]:
        """
        Calculate energy shifts for a batch of spectra.

        Parameters
        ----------
        spectra_data : dict
            Dictionary mapping sample names to (energy, mu) tuples
        element_assignments : dict
            Dictionary mapping sample names to element symbols
        reference_measurement : int
            Measurement number to use for shift calculation

        Returns
        -------
        shifts : dict
            Dictionary mapping sample names to calculated shifts
        """
        shifts = {}

        for sample_name, (energy, mu) in spectra_data.items():
            if sample_name not in element_assignments:
                warnings.warn(f"No element assignment for {sample_name}. Skipping shift calculation.")
                continue

            element = element_assignments[sample_name]

            try:
                shift = self.calculate_edge_shift(energy, mu, element, reference_measurement)
                shifts[sample_name] = shift
                print(f"✓ Calculated shift for {sample_name} ({element}): {shift:.2f} eV")
            except Exception as e:
                warnings.warn(f"Failed to calculate shift for {sample_name}: {e}")
                continue

        # Store shifts for later use
        self.edge_correction_energies = shifts

        return shifts

    def apply_energy_correction(self,
                              energy: np.ndarray,
                              mu: np.ndarray,
                              shift: float,
                              interpolate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply energy shift correction to a spectrum.

        Parameters
        ----------
        energy : np.ndarray
            Original energy values
        mu : np.ndarray
            Original absorption values
        shift : float
            Energy shift to apply (in same units as energy)
        interpolate : bool
            Whether to interpolate data to corrected energy grid

        Returns
        -------
        energy_corrected, mu_corrected : tuple of np.ndarray
            Corrected energy and absorption arrays
        """
        if shift == 0:
            return energy.copy(), mu.copy()

        # Apply shift to energy
        energy_corrected = energy + shift

        if not interpolate:
            return energy_corrected, mu.copy()

        # Create corrected energy grid (same as original)
        # Interpolate mu values to corrected energy points
        mu_corrected = np.interp(energy, energy_corrected, mu)

        return energy, mu_corrected

    def apply_batch_correction(self,
                             spectra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                             shifts: Optional[Dict[str, float]] = None,
                             interpolate: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply energy correction to a batch of spectra.

        Parameters
        ----------
        spectra_data : dict
            Dictionary mapping sample names to (energy, mu) tuples
        shifts : dict, optional
            Dictionary mapping sample names to shifts. If None, uses stored shifts.
        interpolate : bool
            Whether to interpolate data to corrected energy grids

        Returns
        -------
        corrected_data : dict
            Dictionary mapping sample names to corrected (energy, mu) tuples
        """
        if shifts is None:
            shifts = self.edge_correction_energies

        corrected_data = {}

        for sample_name, (energy, mu) in spectra_data.items():
            shift = shifts.get(sample_name, 0)

            try:
                energy_corr, mu_corr = self.apply_energy_correction(
                    energy, mu, shift, interpolate
                )
                corrected_data[sample_name] = (energy_corr, mu_corr)
                print(f"✓ Applied {shift:.2f} eV shift to {sample_name}")
            except Exception as e:
                warnings.warn(f"Failed to correct {sample_name}: {e}")
                corrected_data[sample_name] = (energy.copy(), mu.copy())

        return corrected_data

    def get_shift_statistics(self, shifts: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate statistics for calculated shifts.

        Parameters
        ----------
        shifts : dict
            Dictionary of calculated shifts

        Returns
        -------
        stats : dict
            Statistics dictionary
        """
        shift_values = list(shifts.values())

        if not shift_values:
            return {}

        stats = {
            'mean_shift': float(np.mean(shift_values)),
            'std_shift': float(np.std(shift_values)),
            'min_shift': float(np.min(shift_values)),
            'max_shift': float(np.max(shift_values)),
            'n_samples': len(shift_values)
        }

        return stats

    def save_calibration(self, filepath: Union[str, Path]) -> None:
        """
        Save calibration data to file.

        Parameters
        ----------
        filepath : str or Path
            Output file path
        """
        data = {
            'edge_correction_energies': self.edge_correction_energies,
            'edge_energies': self.edge_energies,
            'energy_unit': self.energy_unit
        }

        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            import json
            json.dump(data, f, indent=2)

    def load_calibration(self, filepath: Union[str, Path]) -> None:
        """
        Load calibration data from file.

        Parameters
        ----------
        filepath : str or Path
            Input file path
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            import json
            data = json.load(f)

        self.edge_correction_energies = data.get('edge_correction_energies', {})
        self.edge_energies = data.get('edge_energies', self.edge_energies)
        self.energy_unit = data.get('energy_unit', self.energy_unit)


# Convenience functions for direct use
def calculate_energy_shifts(spectra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          element_assignments: Dict[str, str],
                          edge_config: Optional[Dict[str, str]] = None,
                          energy_unit: str = 'eV') -> Dict[str, float]:
    """
    Convenience function to calculate energy shifts for a batch of spectra.

    Parameters
    ----------
    spectra_data : dict
        Dictionary mapping sample names to (energy, mu) tuples
    element_assignments : dict
        Dictionary mapping sample names to element symbols
    edge_config : dict, optional
        Edge configuration dictionary
    energy_unit : str
        Energy unit

    Returns
    -------
    shifts : dict
        Calculated shifts
    """
    calibrator = XASEnergyCalibrator(edge_config, energy_unit)
    return calibrator.calibrate_batch(spectra_data, element_assignments)


def apply_energy_corrections(spectra_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           shifts: Dict[str, float],
                           interpolate: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to apply energy corrections to a batch of spectra.

    Parameters
    ----------
    spectra_data : dict
        Dictionary mapping sample names to (energy, mu) tuples
    shifts : dict
        Dictionary mapping sample names to shifts
    interpolate : bool
        Whether to interpolate data

    Returns
    -------
    corrected_data : dict
        Corrected spectra data
    """
    calibrator = XASEnergyCalibrator()
    return calibrator.apply_batch_correction(spectra_data, shifts, interpolate)