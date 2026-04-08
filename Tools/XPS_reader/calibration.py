"""
Energy calibration module for XPS data processing.

This module handles energy axis calibration by aligning reference peaks
(e.g., C1s) to target binding energies using YAML-driven configuration.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from core.data_structures import Spectrum


@dataclass
class CalibrationResult:
    """Summary of the calibration step applied to a group of spectra."""

    status: str = "skipped"
    applied_shift_ev: Optional[float] = None
    attempted_shift_ev: Optional[float] = None
    reference_region: Optional[str] = None
    peak_energy_ev: Optional[float] = None
    target_energy_ev: Optional[float] = None
    source_spectrum: Optional[str] = None
    warning: Optional[str] = None


class EnergyCalibrator:
    """Calibrate energy axis by aligning a reference peak (e.g., C1s) to a target BE with YAML-driven configuration."""

    def __init__(self, config: Dict):
        """
        Initialize with YAML configuration.

        Args:
            config: Full YAML configuration dictionary
        """
        cal = (config or {}).get('energy_calibration', {})

        self.enable = bool(cal.get('enable', True))
        self.reference_region = str(cal.get('reference_region', "C1s"))
        self.target_be = float(cal.get('target_binding_energy_ev', 284.8))
        self.max_allowed_shift = float(cal.get('max_allowed_shift_ev', 5.0))
        self.smooth_window = int(cal.get('smooth_window_points', 7))
        self.smooth_poly = int(cal.get('smooth_poly_order', 2))
        self.min_prom_frac = float(cal.get('min_prominence_fraction', 0.10))
        self.min_points = int(cal.get('min_points_required', 20))

    def _ensure_ascending(self, E: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure energy array is in ascending order."""
        if len(E) >= 2 and E[0] > E[-1]:
            return E[::-1], I[::-1]
        return E, I

    def detect_peak(self, spectrum: Spectrum, erange: Tuple[float, float]) -> Optional[float]:
        """
        Detect the dominant peak center in the given energy range; return its BE.

        Args:
            spectrum: Input spectrum
            erange: Energy range tuple (min, max)

        Returns:
            Peak binding energy or None if not found
        """
        e_min, e_max = float(min(erange)), float(max(erange))
        mask = (spectrum.energy >= e_min) & (spectrum.energy <= e_max)
        if not np.any(mask):
            return None

        E = spectrum.energy[mask].astype(float)
        I = spectrum.intensity[mask].astype(float)

        if len(E) < self.min_points:
            return None

        # Ensure ascending energy for consistent smoothing/peak finding
        E, I = self._ensure_ascending(E, I)

        # Smooth intensity to reduce noise, respecting odd window requirement
        win = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
        win = min(win, len(I) - (len(I) % 2 == 0)) if len(I) > 3 else 3
        try:
            I_smooth = savgol_filter(I, window_length=max(
                3, win), polyorder=self.smooth_poly, mode='interp')
        except Exception:
            I_smooth = I

        # Dynamic prominence threshold relative to max in range
        local_max = float(np.nanmax(I_smooth)) if len(I_smooth) else 0.0
        if local_max <= 0:
            return None
        prom = max(1e-6, self.min_prom_frac * local_max)

        peaks, props = find_peaks(I_smooth, prominence=prom)
        if peaks is None or len(peaks) == 0:
            return None

        # Choose strongest peak by prominence, fallback to highest intensity
        idx = int(np.argmax(props.get('prominences', I_smooth[peaks])))
        p = peaks[idx]
        peak_be = float(E[p])
        return peak_be

    def calibrate_spectra(self,
                          spectra: List[Spectrum],
                          region_defs: Dict,
                          reference_region_name: Optional[str] = None) -> Tuple[List[Spectrum], CalibrationResult]:
        """
        Calibrate all spectra from a file by a single shift determined from the reference region.

        Args:
            spectra: List of spectra to calibrate
            region_defs: Region definitions dictionary
            reference_region_name: Reference region name (uses YAML config if None)

        Returns:
            Tuple of (calibrated_spectra, calibration_result)
        """
        result = CalibrationResult()
        if not self.enable:
            print("   ℹ️ Energy calibration disabled via YAML settings.")
            result.status = "disabled"
            return spectra, result

        # Use YAML config value if not provided
        reference_region_name = reference_region_name or self.reference_region
        result.reference_region = reference_region_name

        # Find reference region energy range from region definitions
        ref = region_defs.get(reference_region_name)
        if not ref or 'energy_range' not in ref:
            print(
                f"   ⚠️ Calibration skipped: reference region '{reference_region_name}' not defined")
            result.status = "missing_reference"
            return spectra, result

        erange = tuple(ref['energy_range'])

        # Try to detect peak on any spectrum containing the reference range; pick the best one
        candidates = []
        for s in spectra:
            mask = (s.energy >= min(erange)) & (s.energy <= max(erange))
            if np.count_nonzero(mask) >= self.min_points:
                peak = self.detect_peak(s, erange)
                if peak is not None:
                    # Score by local max intensity in the window
                    local_max = float(np.nanmax(s.intensity[mask]))
                    candidates.append((s.name, peak, local_max))

        if not candidates:
            print("   ⚠️ Calibration: no valid reference peaks detected")
            result.status = "peak_not_found"
            return spectra, result

        # Choose candidate with highest local max intensity
        # (name, peak_be, local_max)
        best = max(candidates, key=lambda x: x[2])
        peak_be = best[1]
        delta = float(self.target_be - peak_be)
        result.attempted_shift_ev = delta
        result.peak_energy_ev = peak_be
        result.target_energy_ev = self.target_be
        result.source_spectrum = best[0]

        summary = (
            f"shift = {delta:.3f} eV (ref={reference_region_name}, "
            f"peak={peak_be:.3f} eV, target={self.target_be:.3f} eV)"
        )

        if abs(delta) > self.max_allowed_shift:
            print(
                f"   ⚠️ Calibration shift {delta:.3f} eV exceeds max allowed ({self.max_allowed_shift} eV). Skipping.")
            warning = (
                "Instrument or charging issue found: shift "
                f"{delta:.3f} eV exceeds limit {self.max_allowed_shift:.3f} eV"
            )
            result.status = "exceeds_limit"
            result.warning = warning

            for s in spectra:
                s.metadata = {
                    **s.metadata,
                    'calibration_summary': summary,
                    'calibration_warning': warning,
                    'attempted_shift_ev': delta,
                    'calibration_reference': reference_region_name,
                    'calibration_target_ev': self.target_be,
                    'calibration_peak_ev': peak_be,
                    'calibration_source_spectrum': best[0]
                }

            return spectra, result

        # Apply shift to all spectra in the file
        calibrated = []
        for s in spectra:
            try:
                s.energy = s.energy + delta
                s.metadata = {
                    **s.metadata,
                    'energy_shift_ev': delta,
                    'calibration_reference': reference_region_name,
                    'calibration_target_ev': self.target_be,
                    'calibration_peak_ev': peak_be,
                    'calibration_source_spectrum': best[0],
                    'calibration_summary': summary
                }
                calibrated.append(s)
            except Exception as e:
                print(f"   ⚠️ Calibration failed for {s.name}: {e}")
                calibrated.append(s)

        print(f"   🎯 Energy calibration applied: {summary}")
        result.status = "applied"
        result.applied_shift_ev = delta
        result.warning = None
        return calibrated, result