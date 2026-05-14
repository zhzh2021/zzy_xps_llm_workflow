"""
Energy calibration module for XPS data processing.

This module handles energy axis calibration by aligning reference peaks
(e.g., C1s) to target binding energies using YAML-driven configuration.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

from core.data_structures import Spectrum


class CalibrationMode(Enum):
    """Supported energy calibration strategies."""
    adventitious_carbon = "adventitious_carbon"   # default: C1s C-C at 284.8 eV
    metallic_reference  = "metallic_reference"    # ISO 15472 metal foil (Au, Ag, Cu, …)
    fermi_edge          = "fermi_edge"            # conducting samples: Ef = 0 eV
    known_peak          = "known_peak"            # user-specified element/BE pair
    disable             = "disable"               # no calibration


# ISO 15472 / NIST SRD 20 reference binding energies (eV)
METALLIC_REFERENCES: Dict[str, float] = {
    "Au4f": 83.96,
    "Ag3d": 368.21,
    "Cu2p": 932.62,
    "Si2p":  99.30,
    "Ir4f":  60.83,
}


@dataclass
class CalibrationResult:
    """Summary of the calibration step applied to a group of spectra."""

    status: str = "skipped"
    mode: Optional[str] = None
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
        self.mode = str(cal.get('mode', 'adventitious_carbon'))
        self.reference_region = str(cal.get('reference_region', "C1s"))
        self.metallic_region = str(cal.get('metallic_reference_region', "Au4f"))
        self.fermi_window_ev = float(cal.get('fermi_window_ev', 1.0))
        self.fermi_broadening_ev = float(cal.get('fermi_broadening_ev', 0.1))
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

    # ------------------------------------------------------------------
    # Multi-mode calibration methods (Q5 additions)
    # ------------------------------------------------------------------

    def calibrate_metallic_reference(
        self,
        spectra: List[Spectrum],
        region_defs: Dict,
    ) -> Tuple[List[Spectrum], CalibrationResult]:
        """
        Calibrate using a metallic reference foil (ISO 15472 / NIST SRD 20).

        The region whose name matches `self.metallic_region` (e.g. "Au4f") is
        located in the data; its peak is shifted to the ISO 15472 target BE.
        """
        result = CalibrationResult(mode="metallic_reference")
        target_be = METALLIC_REFERENCES.get(self.metallic_region)
        if target_be is None:
            result.status = "missing_reference"
            result.warning = (
                f"Metallic reference '{self.metallic_region}' not in METALLIC_REFERENCES dict. "
                f"Available: {list(METALLIC_REFERENCES.keys())}"
            )
            print(f"   ⚠️ {result.warning}")
            return spectra, result

        ref = region_defs.get(self.metallic_region)
        if not ref or 'energy_range' not in ref:
            result.status = "missing_reference"
            result.warning = f"Region '{self.metallic_region}' not found in region_defs."
            print(f"   ⚠️ {result.warning}")
            return spectra, result

        erange = tuple(ref['energy_range'])
        candidates = []
        for s in spectra:
            mask = (s.energy >= min(erange)) & (s.energy <= max(erange))
            if np.count_nonzero(mask) >= self.min_points:
                peak = self.detect_peak(s, erange)
                if peak is not None:
                    local_max = float(np.nanmax(s.intensity[mask]))
                    candidates.append((s.name, peak, local_max))

        if not candidates:
            result.status = "peak_not_found"
            print(f"   ⚠️ No {self.metallic_region} peak detected for metallic reference calibration.")
            return spectra, result

        best = max(candidates, key=lambda x: x[2])
        peak_be = best[1]
        delta = float(target_be - peak_be)
        result.reference_region = self.metallic_region
        result.peak_energy_ev = peak_be
        result.target_energy_ev = target_be
        result.source_spectrum = best[0]
        result.attempted_shift_ev = delta

        if abs(delta) > self.max_allowed_shift:
            result.status = "exceeds_limit"
            result.warning = (
                f"Metallic ref shift {delta:.3f} eV exceeds max allowed {self.max_allowed_shift} eV."
            )
            print(f"   ⚠️ {result.warning}")
            return spectra, result

        for s in spectra:
            s.energy = s.energy + delta
            s.metadata = {**s.metadata, 'energy_shift_ev': delta,
                          'calibration_mode': 'metallic_reference',
                          'calibration_reference': self.metallic_region,
                          'calibration_target_ev': target_be}
        print(
            f"   🎯 Metallic reference calibration applied: "
            f"{self.metallic_region} shift={delta:+.3f} eV (ISO 15472 target={target_be} eV)"
        )
        result.status = "applied"
        result.applied_shift_ev = delta
        return spectra, result

    def calibrate_fermi_edge(
        self,
        spectra: List[Spectrum],
    ) -> Tuple[List[Spectrum], CalibrationResult]:
        """
        Calibrate by fitting the Fermi-Dirac edge so that E_F = 0 eV.

        Requires a valence-band or Fermi-level spectrum in `spectra`
        spanning the window ±`self.fermi_window_ev` around 0 eV.
        """
        result = CalibrationResult(mode="fermi_edge")

        def fermi_dirac(E: np.ndarray, ef: float, T_broad: float, scale: float, bg: float):
            """Fermi-Dirac distribution broadened by Gaussian (instrumental)."""
            kT = max(T_broad, 0.01)
            return scale / (np.exp((E - ef) / kT) + 1.0) + bg

        best_candidate = None
        best_score = -np.inf

        for s in spectra:
            mask = np.abs(s.energy) <= self.fermi_window_ev
            if np.count_nonzero(mask) < self.min_points:
                continue
            E_w = s.energy[mask].astype(float)
            I_w = s.intensity[mask].astype(float)
            E_w, I_w = self._ensure_ascending(E_w, I_w)
            try:
                p0 = [0.0, self.fermi_broadening_ev, float(np.max(I_w) - np.min(I_w)), float(np.min(I_w))]
                popt, _ = curve_fit(fermi_dirac, E_w, I_w, p0=p0, maxfev=5000)
                ef_fit = float(popt[0])
                score = float(np.max(I_w))
                if score > best_score:
                    best_score = score
                    best_candidate = (s.name, ef_fit)
            except Exception:
                continue

        if best_candidate is None:
            result.status = "peak_not_found"
            print("   ⚠️ Fermi edge not detected within ±{:.2f} eV window.".format(self.fermi_window_ev))
            return spectra, result

        source_name, ef_fit = best_candidate
        delta = -ef_fit  # shift so that E_F = 0 eV
        result.reference_region = "Fermi edge"
        result.peak_energy_ev = ef_fit
        result.target_energy_ev = 0.0
        result.source_spectrum = source_name
        result.attempted_shift_ev = delta

        if abs(delta) > self.max_allowed_shift:
            result.status = "exceeds_limit"
            result.warning = f"Fermi edge shift {delta:.3f} eV exceeds max allowed {self.max_allowed_shift} eV."
            print(f"   ⚠️ {result.warning}")
            return spectra, result

        for s in spectra:
            s.energy = s.energy + delta
            s.metadata = {**s.metadata, 'energy_shift_ev': delta,
                          'calibration_mode': 'fermi_edge',
                          'fermi_edge_fitted_ev': ef_fit}
        print(f"   🎯 Fermi-edge calibration applied: E_F fitted at {ef_fit:+.3f} eV → shift={delta:+.3f} eV")
        result.status = "applied"
        result.applied_shift_ev = delta
        return spectra, result

    def calibrate(
        self,
        spectra: List[Spectrum],
        region_defs: Dict,
        override_mode: Optional[str] = None,
    ) -> Tuple[List[Spectrum], CalibrationResult]:
        """
        Unified calibration dispatch. Reads `self.mode` (or `override_mode`)
        and calls the appropriate method.

        Modes
        -----
        adventitious_carbon  →  calibrate_spectra()  (original AdC method)
        metallic_reference   →  calibrate_metallic_reference()
        fermi_edge           →  calibrate_fermi_edge()
        known_peak           →  calibrate_spectra() with user-specified region
        disable              →  no-op, returns (spectra, CalibrationResult(status='disabled'))
        """
        mode = override_mode or self.mode
        if not self.enable or mode == CalibrationMode.disable.value:
            res = CalibrationResult(status="disabled", mode=mode)
            print("   ℹ️ Energy calibration disabled.")
            return spectra, res

        if mode == CalibrationMode.metallic_reference.value:
            return self.calibrate_metallic_reference(spectra, region_defs)
        elif mode == CalibrationMode.fermi_edge.value:
            return self.calibrate_fermi_edge(spectra)
        else:
            # adventitious_carbon or known_peak — use the original AdC method
            return self.calibrate_spectra(spectra, region_defs)


def select_calibration_mode(sample_description: str) -> str:
    """
    Heuristic helper for the LLM router: given a natural-language sample
    description, return the most appropriate CalibrationMode string.

    Rules
    -----
    - mentions metal foil / Au / Ag / Cu / Ir  →  metallic_reference
    - mentions valence band / Fermi / conductor  →  fermi_edge
    - mentions 'no calibration' / 'skip'        →  disable
    - default                                   →  adventitious_carbon
    """
    desc = sample_description.lower()
    if any(kw in desc for kw in ("foil", "au4f", "ag3d", "cu2p", "ir4f", "metallic ref")):
        return CalibrationMode.metallic_reference.value
    if any(kw in desc for kw in ("fermi", "valence", "conductor", "metal film")):
        return CalibrationMode.fermi_edge.value
    if any(kw in desc for kw in ("no calibration", "skip calibration", "disable")):
        return CalibrationMode.disable.value
    return CalibrationMode.adventitious_carbon.value