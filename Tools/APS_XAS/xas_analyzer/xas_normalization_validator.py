import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from larch import Group
from larch.xafs import pre_edge


# =========================
# Data containers
# =========================

@dataclass
class NormalizationResult:
    success: bool
    confidence: float
    parameters: Dict
    metrics: Dict
    flags: List[str]


# =========================
# Utility functions
# =========================

def estimate_noise(y):
    """Estimate noise using high-frequency residuals"""
    return np.std(np.diff(y))


def detect_large_bump(energy, mu, threshold=5.0):
    """
    Detect unphysical bumps (glitches, Bragg peaks, detector issues)
    """
    d2 = np.diff(mu, n=2)
    return np.max(np.abs(d2)) > threshold * np.median(np.abs(d2))


def estimate_edge_energy(energy, mu):
    """Crude E0 estimate from max derivative"""
    dmu = np.gradient(mu, energy)
    return energy[np.argmax(dmu)]


# =========================
# Parameter proposal
# =========================

def propose_normalization_params(energy, mu):
    flags = []

    noise = estimate_noise(mu)
    bump_detected = detect_large_bump(energy, mu)

    if bump_detected:
        flags.append("large_unphysical_bump_detected")

    e0 = estimate_edge_energy(energy, mu)

    # Physics-based defaults
    params = {
        "e0": e0,
        "pre1": -150,
        "pre2": -30,
        "norm1": 150,
        "norm2": 800,
        "nnorm": 2
    }

    # Adjust if noisy
    if noise > 0.02:
        params["pre1"] = -200
        params["norm2"] = 1000
        flags.append("high_noise_adjusted_windows")

    return params, flags


# =========================
# Physics validation checks
# =========================

def validate_normalization(g):
    flags = []
    metrics = {}

    # --- Pre-edge flatness ---
    pre_mask = g.energy < (g.e0 - 50)
    slope = np.polyfit(g.energy[pre_mask], g.norm[pre_mask], 1)[0]

    metrics["pre_edge_slope"] = slope
    if abs(slope) > 1e-3:
        flags.append("pre_edge_not_flat")

    # --- Edge step ---
    edge_step = g.edge_step
    metrics["edge_step"] = edge_step

    if not (0.7 <= edge_step <= 1.3):
        flags.append("edge_step_out_of_range")

    # --- Post-edge smoothness ---
    post_mask = g.energy > (g.e0 + 200)
    curvature = np.mean(np.abs(np.diff(g.norm[post_mask], n=2)))
    metrics["post_edge_curvature"] = curvature

    if curvature > 0.01:
        flags.append("post_edge_not_smooth")

    return metrics, flags


# =========================
# Uncertainty estimation
# =========================

def normalization_uncertainty(energy, mu, base_params):
    """
    Perturb normalization windows and evaluate feature stability
    """
    wl_peaks = []

    for shift in [-20, 0, 20]:
        g = Group(energy=energy, mu=mu)
        pre_edge(
            g,
            e0=base_params["e0"],
            pre1=base_params["pre1"] + shift,
            pre2=base_params["pre2"],
            norm1=base_params["norm1"],
            norm2=base_params["norm2"],
            nnorm=base_params["nnorm"]
        )
        wl_peaks.append(np.max(g.norm))

    wl_peaks = np.array(wl_peaks)
    return np.std(wl_peaks) / np.mean(wl_peaks)


# =========================
# Main driver
# =========================

def normalize_and_validate(energy, mu) -> NormalizationResult:
    flags = []

    # ---- Step 1: propose params ----
    params, param_flags = propose_normalization_params(energy, mu)
    flags.extend(param_flags)

    # ---- Step 2: normalize ----
    g = Group(energy=energy, mu=mu)
    pre_edge(g, **params)

    # ---- Step 3: physics validation ----
    metrics, validation_flags = validate_normalization(g)
    flags.extend(validation_flags)

    # ---- Step 4: uncertainty ----
    wl_uncertainty = normalization_uncertainty(energy, mu, params)
    metrics["white_line_uncertainty"] = wl_uncertainty

    if wl_uncertainty > 0.05:
        flags.append("normalization_sensitive_to_window")

    # ---- Step 5: confidence score ----
    confidence = 1.0
    confidence -= 0.2 * len(flags)
    confidence -= min(wl_uncertainty, 0.2)
    confidence = max(confidence, 0.0)

    success = confidence > 0.6

    return NormalizationResult(
        success=success,
        confidence=confidence,
        parameters=params,
        metrics=metrics,
        flags=flags
    )
