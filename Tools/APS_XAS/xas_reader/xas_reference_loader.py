"""
XAS Reference Loader Module (Optional)

Loads reference spectrum (foil/standard) for energy alignment.
Validates energy range overlap.

Outputs:
- reference_energy
- reference_mu
- reference_metadata
- flags for mismatches
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import reader functionality (avoid circular import)
# from .xas_reader import load_xas_file  # Commented out to avoid circular import

# Import at function level to avoid circular imports
def _load_xas_file(file_path):
    """Helper function to load XAS file, avoiding circular imports."""
    try:
        from .xas_reader import load_xas_file
    except ImportError:
        from xas_reader import load_xas_file
    return load_xas_file(file_path)


def calibrate_energy_with_reference(energy: np.ndarray, mu_ref: np.ndarray, reference_edge: float = 8333.0) -> np.ndarray:
    """
    Calibrate energy using reference foil absorption spectrum.

    Parameters
    ----------
    energy : np.ndarray
        Raw energy values
    mu_ref : np.ndarray
        Reference foil absorption coefficient μ_ref = ln(It/Ir)
    reference_edge : float
        Known edge energy of reference foil (default: 8333.0 eV for Ni K-edge)

    Returns
    -------
    energy_calibrated : np.ndarray
        Calibrated energy values
    """
    # Create diagnostic plot to show reference signal
    create_reference_diagnostic_plot(energy, mu_ref, reference_edge)

    try:
        # Find the energy position of the reference edge
        # Look for the maximum derivative (edge position) in the reference spectrum
        from scipy.signal import find_peaks

        # Smooth the reference spectrum to reduce noise
        mu_ref_smooth = np.convolve(mu_ref, np.ones(5)/5, mode='valid')
        energy_smooth = energy[len(energy) - len(mu_ref_smooth):]

        # Calculate derivative
        derivative = np.gradient(mu_ref_smooth, energy_smooth)

        # Find peaks in the derivative (edge positions)
        peaks, _ = find_peaks(derivative, height=np.max(derivative)*0.1)

        if len(peaks) > 0:
            # Use the first significant peak as the edge position
            edge_index = peaks[0]
            measured_edge_energy = energy_smooth[edge_index]

            # Calculate energy shift
            energy_shift = reference_edge - measured_edge_energy

            # Apply calibration
            energy_calibrated = energy + energy_shift

            print(".1f")
            return energy_calibrated
        else:
            print(f"Warning: Could not find reference edge in spectrum, using original energy")
            return energy

    except ImportError:
        print("Warning: scipy not available for energy calibration, using original energy")
        return energy
    except Exception as e:
        print(f"Warning: Energy calibration failed: {e}, using original energy")
        return energy


def create_reference_diagnostic_plot(energy: np.ndarray, mu_ref: np.ndarray, reference_edge: float = 8333.0):
    """
    Create diagnostic plot to visualize reference foil signal and check for edges.

    Parameters
    ----------
    energy : np.ndarray
        Energy values
    mu_ref : np.ndarray
        Reference absorption coefficient
    reference_edge : float
        Expected reference edge energy
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # Plot reference signal
        plt.subplot(2, 1, 1)
        plt.plot(energy, mu_ref, 'b-', linewidth=1.5, label='Reference μ(E)')
        plt.axvline(x=7112, color='r', linestyle='--', linewidth=2, label='Fe K-edge (7112 eV)')
        plt.axvline(x=reference_edge, color='g', linestyle='--', linewidth=2, label=f'Ni K-edge ({reference_edge} eV)')
        plt.xlabel('Energy (eV)')
        plt.ylabel('μ(E) reference')
        plt.title('Reference Foil Signal - Edge Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot derivative to show edge positions
        plt.subplot(2, 1, 2)
        try:
            from scipy.signal import find_peaks
            # Smooth the reference spectrum
            mu_ref_smooth = np.convolve(mu_ref, np.ones(5)/5, mode='valid')
            energy_smooth = energy[len(energy) - len(mu_ref_smooth):]

            # Calculate derivative
            derivative = np.gradient(mu_ref_smooth, energy_smooth)

            plt.plot(energy_smooth, derivative, 'orange', linewidth=1.5, label='Derivative')
            plt.axvline(x=7112, color='r', linestyle='--', linewidth=2, label='Fe K-edge (7112 eV)')
            plt.axvline(x=reference_edge, color='g', linestyle='--', linewidth=2, label=f'Ni K-edge ({reference_edge} eV)')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)

            # Mark detected peaks
            peaks, _ = find_peaks(derivative, height=np.max(derivative)*0.1)
            if len(peaks) > 0:
                for peak_idx in peaks[:3]:  # Show first 3 peaks
                    peak_energy = energy_smooth[peak_idx]
                    plt.axvline(x=peak_energy, color='purple', linestyle=':', alpha=0.7,
                              label=f'Detected edge: {peak_energy:.1f} eV')

            plt.title('Reference Signal Derivative - Edge Detection')
            plt.legend()
        except ImportError:
            plt.text(0.5, 0.5, 'Scipy not available for derivative analysis',
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title('Reference Signal Derivative (Scipy not available)')

        plt.xlabel('Energy (eV)')
        plt.ylabel('dμ/dE')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        import os
        from pathlib import Path
        plot_dir = Path("reference_diagnostic_plots")
        plot_dir.mkdir(exist_ok=True)
        plot_file = plot_dir / "reference_foil_diagnostic.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Reference diagnostic plot saved to: {plot_file}")

        plt.show()

    except Exception as e:
        print(f"Warning: Could not create reference diagnostic plot: {e}")


class XASReferenceLoader:
    """
    Loads and validates XAS reference spectra.
    """

    def __init__(self,
                 reference_file: Optional[str] = None,
                 reference_data: Optional[Dict] = None):
        """
        Initialize reference loader.

        Parameters
        ----------
        reference_file : str, optional
            Path to reference file
        reference_data : dict, optional
            Pre-loaded reference data
        """
        self.reference_file = reference_file
        self.reference_data = reference_data

    def load_reference(self) -> Dict[str, Any]:
        """
        Load reference spectrum.

        Returns
        -------
        result : dict
            Reference data with energy, mu, metadata, flags
        """
        flags = []
        confidence = 1.0

        if self.reference_data:
            # Use provided data
            energy = self.reference_data.get("energy")
            mu = self.reference_data.get("mu")
            metadata = self.reference_data.get("metadata", {})
        elif self.reference_file:
            # Load from file
            try:
                data = _load_xas_file(self.reference_file)
                energy = data["energy"]
                mu = data["mu"]
                metadata = data["metadata"]
            except Exception as e:
                return {
                    "reference_energy": None,
                    "reference_mu": None,
                    "reference_metadata": {},
                    "flags": ["reference_load_failed"],
                    "confidence": 0.0
                }
        else:
            return {
                "reference_energy": None,
                "reference_mu": None,
                "reference_metadata": {},
                "flags": ["no_reference_provided"],
                "confidence": 0.0
            }

        # Validate reference data
        if energy is None or mu is None:
            flags.append("invalid_reference_data")
            confidence = 0.0
        else:
            # Check data quality
            if len(energy) < 50:
                flags.append("reference_insufficient_data")
                confidence *= 0.7

            if np.any(~np.isfinite(mu)):
                flags.append("reference_invalid_values")
                confidence *= 0.5

            # Check monotonicity
            if not np.all(np.diff(energy) > 0):
                flags.append("reference_non_monotonic_energy")
                confidence *= 0.8

        result = {
            "reference_energy": energy,
            "reference_mu": mu,
            "reference_metadata": metadata,
            "flags": flags,
            "confidence": confidence
        }

        return result


def load_xas_reference(reference_file: Optional[str] = None,
                      reference_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function for loading XAS reference.

    Parameters
    ----------
    reference_file : str, optional
        Path to reference file
    reference_data : dict, optional
        Pre-loaded reference data

    Returns
    -------
    result : dict
        Reference loading results
    """
    loader = XASReferenceLoader(reference_file, reference_data)
    return loader.load_reference()