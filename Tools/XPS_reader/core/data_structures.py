"""Core data structures for XPS data handling."""

from dataclasses import dataclass, field
from typing import Any, Dict

import logging
import numpy as np

from .metadata import XPSMetadata


@dataclass
class Spectrum:
    """Standardized representation of an XPS spectrum."""

    name: str
    energy: np.ndarray
    intensity: np.ndarray
    energy_units: str = "eV"
    intensity_units: str = "counts"
    source_format: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert array-like inputs to NumPy arrays for consistency."""
        self.energy = np.asarray(self.energy)
        self.intensity = np.asarray(self.intensity)

    def is_valid_xps(self, allow_zero: bool = False) -> bool:
        """Validate spectrum data quality for downstream processing."""
        try:
            if len(self.energy) != len(self.intensity):
                logging.debug(
                    "Length mismatch: energy=%s, intensity=%s",
                    len(self.energy),
                    len(self.intensity),
                )
                return False

            if len(self.energy) < 2:
                logging.debug("Too few points: %s", len(self.energy))
                return False

            energy_min = float(np.min(self.energy))
            energy_max = float(np.max(self.energy))

            logging.debug(
                "Energy range: %.2f to %.2f eV",
                energy_min,
                energy_max,
            )

            if not (0 <= energy_min <= 1500 and 0 <= energy_max <= 1500):
                logging.debug("Energy values outside valid XPS range (0-1500 eV)")
                return False

            if not allow_zero and np.all(self.intensity == 0):
                logging.debug("All intensity values are zero")
                return False

            intensity_std = float(np.std(self.intensity))
            intensity_mean = float(np.mean(self.intensity))

            if intensity_mean > 0:
                snr = intensity_mean / intensity_std if intensity_std > 0 else 0
                logging.debug("Signal-to-noise ratio: %.2f", snr)
                if snr < 0.1:
                    logging.debug("Very low signal-to-noise ratio")
                    return False

            return True

        except Exception as exc:  # pylint: disable=broad-except
            logging.debug("Validation error: %s", exc)
            return False


@dataclass
class XPSData:
    """Standard container for parsed XPS data arrays and metadata."""

    data: np.ndarray
    metadata: XPSMetadata


__all__ = ["Spectrum", "XPSData", "XPSMetadata"]
