"""Utility helpers for optional Savitzky-Golay smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from scipy.signal import savgol_filter

from core.data_structures import Spectrum


@dataclass
class SmoothingSettings:
    """Container describing how (and if) spectra should be smoothed."""

    enable: bool = False
    smooth_window_points: int = 7
    smooth_poly_order: int = 2
    min_points_required: int = 20


def load_smoothing_settings(config: Dict) -> SmoothingSettings:
    """
    Derive smoothing settings from the project configuration.

    The `processing.smoothing` section takes precedence. If it is missing we fall
    back to the legacy keys that previously lived under `energy_calibration`.
    """

    processing_cfg = config.get("processing", {})
    smoothing_cfg = processing_cfg.get("smoothing", {})

    if not smoothing_cfg:
        # Legacy fallback to keep existing YAML files working without edits.
        legacy = config.get("energy_calibration", {})
        smoothing_cfg = {
            "enable": False,
            "smooth_window_points": legacy.get("smooth_window_points", 7),
            "smooth_poly_order": legacy.get("smooth_poly_order", 2),
            "min_points_required": legacy.get("min_points_required", 20),
        }

    return SmoothingSettings(
        enable=smoothing_cfg.get("enable", False),
        smooth_window_points=smoothing_cfg.get("smooth_window_points", 7),
        smooth_poly_order=smoothing_cfg.get("smooth_poly_order", 2),
        min_points_required=smoothing_cfg.get("min_points_required", 20),
    )


def _sanitize_window_length(window_points: int, data_length: int) -> int:
    """Ensure the Savitzky-Golay window is odd, <= data length, and valid."""
    window = max(1, min(window_points, data_length))
    if window % 2 == 0:
        window -= 1
    if window < 3 and data_length >= 3:
        window = 3  # Smallest valid odd window for savgol_filter
    if window > data_length:
        window = data_length if data_length % 2 == 1 else data_length - 1
    return window


def smooth_spectrum(spectrum: Spectrum, settings: SmoothingSettings) -> Spectrum:
    """Apply smoothing in-place for a single spectrum if it meets the criteria."""
    if not settings.enable:
        return spectrum

    if len(spectrum.intensity) < max(settings.min_points_required, settings.smooth_poly_order + 2):
        return spectrum

    window = _sanitize_window_length(settings.smooth_window_points, len(spectrum.intensity))
    if window <= settings.smooth_poly_order:
        return spectrum

    spectrum.intensity = savgol_filter(
        spectrum.intensity,
        window_length=window,
        polyorder=settings.smooth_poly_order,
        mode="interp",
    )
    return spectrum


def smooth_spectra(spectra: Iterable[Spectrum], settings: SmoothingSettings):
    """Apply smoothing to a collection of spectra (mutates the provided spectra)."""
    if not settings.enable:
        return spectra

    for spectrum in spectra:
        smooth_spectrum(spectrum, settings)

    return spectra


__all__ = ["SmoothingSettings", "load_smoothing_settings", "smooth_spectrum", "smooth_spectra"]
