"""
XAS Normalization Module

Proposes normalization parameters and performs pre-edge subtraction and edge step normalization.
Uses physics-informed defaults and records parameters explicitly.

Outputs:
- normalized μ(E)
- normalization parameters
- confidence score
- flags
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from larch.xafs import pre_edge

# Import parameter proposal from validator
try:
    from .xas_normalization_validator import propose_normalization_params
except ImportError:
    from xas_normalization_validator import propose_normalization_params


class XASNormalizer:
    """
    XAS spectrum normalization using pre-edge subtraction and post-edge normalization.
    """

    def __init__(self,
                 normalization_params: Optional[Dict] = None,
                 use_proposed_params: bool = True):
        """
        Initialize normalizer.

        Parameters
        ----------
        normalization_params : dict, optional
            Fixed normalization parameters
        use_proposed_params : bool
            Whether to propose parameters automatically
        """
        self.fixed_params = normalization_params
        self.use_proposed_params = use_proposed_params

    def normalize_spectrum(self,
                          energy: np.ndarray,
                          mu: np.ndarray) -> Dict[str, Any]:
        """
        Normalize XAS spectrum.

        Parameters
        ----------
        energy : np.ndarray
            Energy values
        mu : np.ndarray
            Absorption coefficient

        Returns
        -------
        result : dict
            Normalization results with normalized_mu, parameters, confidence, flags
        """
        flags = []
        confidence = 1.0

        # Propose or use fixed parameters
        if self.use_proposed_params and self.fixed_params is None:
            params, param_flags = propose_normalization_params(energy, mu)
            flags.extend(param_flags)
        else:
            params = self.fixed_params or {
                "e0": None,  # Will be estimated
                "pre1": -150,
                "pre2": -30,
                "norm1": 150,
                "norm2": 800,
                "nnorm": 2
            }

        try:
            # Create Larch group
            from larch import Group
            g = Group(energy=energy, mu=mu)

            # Apply pre_edge normalization
            pre_edge(g, **params)

            # Extract results
            normalized_mu = g.norm
            edge_step = g.edge_step
            e0 = g.e0

            # Store final parameters
            final_params = {
                "e0": e0,
                "pre1": params.get("pre1"),
                "pre2": params.get("pre2"),
                "norm1": params.get("norm1"),
                "norm2": params.get("norm2"),
                "nnorm": params.get("nnorm", 2),
                "edge_step": edge_step
            }

            # Basic validation
            if not np.isfinite(edge_step) or edge_step <= 0:
                flags.append("invalid_edge_step")
                confidence *= 0.5

            if np.any(~np.isfinite(normalized_mu)):
                flags.append("normalization_failed")
                confidence = 0.0

            result = {
                "normalized_mu": normalized_mu,
                "parameters": final_params,
                "confidence": confidence,
                "flags": flags
            }

        except Exception as e:
            # Return original data if normalization fails
            result = {
                "normalized_mu": mu.copy(),
                "parameters": params,
                "confidence": 0.0,
                "flags": ["normalization_error"]
            }

        return result


def normalize_xas_spectrum(energy: np.ndarray,
                          mu: np.ndarray,
                          normalization_params: Optional[Dict] = None,
                          use_proposed_params: bool = True) -> Dict[str, Any]:
    """
    Convenience function for XAS normalization.

    Parameters
    ----------
    energy : np.ndarray
        Energy values
    mu : np.ndarray
        Absorption coefficient
    normalization_params : dict, optional
        Fixed parameters
    use_proposed_params : bool
        Whether to propose parameters

    Returns
    -------
    result : dict
        Normalization results
    """
    normalizer = XASNormalizer(normalization_params, use_proposed_params)
    return normalizer.normalize_spectrum(energy, mu)