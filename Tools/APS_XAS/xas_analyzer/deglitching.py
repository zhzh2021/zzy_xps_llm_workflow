"""
XAS Deglitching Module

Removes detector spikes, Bragg peaks, and glitches from XAS spectra.
Uses Larch's conservative deglitching algorithm.

Outputs:
- cleaned μ(E)
- glitch mask
- points removed count
- flags for excessive deglitching or localized artifacts
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from larch.math import deglitch


class XASDeglitcher:
    """
    XAS spectrum deglitching using Larch's deglitch function.

    Removes spikes and artifacts while being conservative and reversible.
    """

    def __init__(self,
                 deglitch_params: Optional[Dict] = None):
        """
        Initialize deglitcher with parameters.

        Parameters
        ----------
        deglitch_params : dict, optional
            Parameters for larch.math.deglitch
        """
        self.deglitch_params = deglitch_params or {
            'expon': 2,      # exponent for deglitch
            'nsigma': 3,     # sigma threshold for spike detection
            'replace': 'linear'  # replacement method
        }

    def deglitch_spectrum(self,
                         energy: np.ndarray,
                         mu: np.ndarray) -> Dict[str, Any]:
        """
        Remove glitches from XAS spectrum.

        Parameters
        ----------
        energy : np.ndarray
            Energy values
        mu : np.ndarray
            Absorption coefficient μ(E)

        Returns
        -------
        result : dict
            Contains cleaned_mu, glitch_mask, points_removed, flags, confidence
        """
        flags = []
        confidence = 1.0

        try:
            # Apply deglitching using Larch
            mu_cleaned, glitch_mask = deglitch(mu, **self.deglitch_params)

            # Count removed points
            points_removed = np.sum(glitch_mask)

            # Check for excessive deglitching
            total_points = len(mu)
            removal_fraction = points_removed / total_points

            if removal_fraction > 0.05:  # More than 5% removed
                flags.append("excessive_deglitching")
                confidence *= 0.7

            # Check for localized artifacts (clusters of glitches)
            if points_removed > 0:
                # Find consecutive glitches
                diff_mask = np.diff(glitch_mask.astype(int))
                start_indices = np.where(diff_mask == 1)[0] + 1
                end_indices = np.where(diff_mask == -1)[0]

                if len(start_indices) > 0 and len(end_indices) > 0:
                    # Check if any glitch cluster is > 10 points
                    cluster_sizes = end_indices - start_indices
                    if np.any(cluster_sizes > 10):
                        flags.append("localized_artifacts")
                        confidence *= 0.8

            result = {
                "cleaned_mu": mu_cleaned,
                "glitch_mask": glitch_mask,
                "points_removed": points_removed,
                "flags": flags,
                "confidence": confidence
            }

        except Exception as e:
            result = {
                "cleaned_mu": mu.copy(),  # Return original if deglitching fails
                "glitch_mask": np.zeros_like(mu, dtype=bool),
                "points_removed": 0,
                "flags": ["deglitching_failed"],
                "confidence": 0.0
            }

        return result


def deglitch_xas_spectrum(energy: np.ndarray,
                         mu: np.ndarray,
                         deglitch_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function for deglitching a single XAS spectrum.

    Parameters
    ----------
    energy : np.ndarray
        Energy values
    mu : np.ndarray
        Absorption coefficient μ(E)
    deglitch_params : dict, optional
        Deglitching parameters

    Returns
    -------
    result : dict
        Deglitching results
    """
    deglitcher = XASDeglitcher(deglitch_params)
    return deglitcher.deglitch_spectrum(energy, mu)