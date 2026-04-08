"""
XAS Energy Alignment Module

Estimates E₀ from derivative analysis and aligns sample E₀ to reference if provided.
Quantifies confidence in alignment.

Outputs:
- ΔE applied
- method used
- alignment confidence
- flags for ambiguous edge or multiple edges
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from larch.xafs import find_e0


class XASEnergyAligner:
    """
    XAS energy alignment using E₀ estimation and reference alignment.
    """

    def __init__(self,
                 alignment_method: str = 'derivative',
                 reference_energy: Optional[np.ndarray] = None,
                 reference_mu: Optional[np.ndarray] = None):
        """
        Initialize energy aligner.

        Parameters
        ----------
        alignment_method : str
            Method for alignment ('derivative' or 'reference')
        reference_energy : np.ndarray, optional
            Reference spectrum energy
        reference_mu : np.ndarray, optional
            Reference spectrum μ(E)
        """
        self.alignment_method = alignment_method
        self.reference_energy = reference_energy
        self.reference_mu = reference_mu

        if alignment_method == 'reference' and (reference_energy is None or reference_mu is None):
            raise ValueError("Reference energy and mu required for reference alignment")

    def estimate_e0(self, energy: np.ndarray, mu: np.ndarray) -> Tuple[float, float]:
        """
        Estimate E₀ from derivative analysis.

        Parameters
        ----------
        energy : np.ndarray
            Energy values
        mu : np.ndarray
            Absorption coefficient

        Returns
        -------
        e0, confidence : tuple
            Estimated edge energy and confidence score
        """
        try:
            # Use Larch's find_e0 for robust E₀ estimation
            e0 = find_e0(energy, mu)

            # Calculate confidence based on derivative sharpness
            dmu = np.gradient(mu, energy)
            peak_height = np.max(dmu)
            noise_level = np.std(dmu)

            if noise_level > 0:
                confidence = min(1.0, peak_height / (10 * noise_level))
            else:
                confidence = 0.5

            return e0, confidence

        except Exception as e:
            # Fallback to simple derivative maximum
            dmu = np.gradient(mu, energy)
            e0 = energy[np.argmax(dmu)]
            confidence = 0.3  # Low confidence for fallback

            return e0, confidence

    def align_to_reference(self,
                          energy: np.ndarray,
                          mu: np.ndarray,
                          reference_e0: float) -> Tuple[float, float]:
        """
        Align spectrum E₀ to reference E₀.

        Parameters
        ----------
        energy : np.ndarray
            Sample energy
        mu : np.ndarray
            Sample μ(E)
        reference_e0 : float
            Reference edge energy

        Returns
        -------
        delta_e, confidence : tuple
            Energy shift and confidence
        """
        sample_e0, sample_confidence = self.estimate_e0(energy, mu)

        delta_e = reference_e0 - sample_e0
        confidence = sample_confidence * 0.9  # Slightly reduce confidence for alignment

        return delta_e, confidence

    def align_spectrum(self,
                      energy: np.ndarray,
                      mu: np.ndarray) -> Dict[str, Any]:
        """
        Align energy axis of XAS spectrum.

        Parameters
        ----------
        energy : np.ndarray
            Energy values
        mu : np.ndarray
            Absorption coefficient

        Returns
        -------
        result : dict
            Alignment results with delta_e, method, confidence, flags
        """
        flags = []
        confidence = 1.0
        delta_e = 0.0
        method = self.alignment_method

        if self.alignment_method == 'derivative':
            # No alignment needed, just estimate E₀
            e0, conf = self.estimate_e0(energy, mu)
            confidence = conf
            method = 'derivative_estimation'

        elif self.alignment_method == 'reference':
            # Align to reference
            ref_e0, ref_conf = self.estimate_e0(self.reference_energy, self.reference_mu)
            delta_e, confidence = self.align_to_reference(energy, mu, ref_e0)
            method = 'reference_alignment'

            # Check energy range overlap
            energy_overlap = (np.min(energy) <= ref_e0 <= np.max(energy))
            if not energy_overlap:
                flags.append("energy_range_mismatch")
                confidence *= 0.5

        else:
            flags.append("unknown_alignment_method")
            confidence = 0.0

        # Check for ambiguous edge
        dmu = np.gradient(mu, energy)
        peaks = np.where((dmu[:-1] > 0) & (dmu[1:] < 0))[0]
        if len(peaks) > 1:
            flags.append("multiple_edges_detected")
            confidence *= 0.7

        result = {
            "delta_e": delta_e,
            "method": method,
            "confidence": confidence,
            "flags": flags
        }

        return result


def align_xas_energy(energy: np.ndarray,
                    mu: np.ndarray,
                    method: str = 'derivative',
                    reference_energy: Optional[np.ndarray] = None,
                    reference_mu: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Convenience function for energy alignment.

    Parameters
    ----------
    energy : np.ndarray
        Energy values
    mu : np.ndarray
        Absorption coefficient
    method : str
        Alignment method
    reference_energy : np.ndarray, optional
        Reference energy
    reference_mu : np.ndarray, optional
        Reference μ(E)

    Returns
    -------
    result : dict
        Alignment results
    """
    aligner = XASEnergyAligner(method, reference_energy, reference_mu)
    return aligner.align_spectrum(energy, mu)