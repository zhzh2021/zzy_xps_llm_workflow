"""
XAS Linear Combination Fitting (LCF) Module

Quantitative analysis using Linear Combination Fitting for XAS spectra.
Used for determining oxidation state fractions (e.g., Fe²⁺/Fe³⁺ ratios).

FOR EXAMPLE, (if Fe²⁺ reference available):
• Linear Combination Fitting
• Report: % Fe²⁺, % Fe³⁺ for each sample
• Include error estimates
• Check fit quality (R-factor < 0.02)
• Compare with other techniques if available
• Report uncertainties

Using larch for LCF implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Using larch for LCF
from larch import Group
from larch.xafs import pre_edge, autobk
from larch.fitting import guess, param, minimize

import json
import matplotlib.pyplot as plt
import yaml


def _default_output_dirs() -> Tuple[Path, Path]:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent / "project_root"
    data_dir = project_root / "xas_results" / "05_LCF_fitting" / "fitting_data"
    plots_dir = project_root / "xas_results" / "05_LCF_fitting" / "fitting_plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plots_dir


def _get_reference_library_paths(config_path: Optional[Path] = None) -> Tuple[Path, Optional[Path]]:
    """
    Return (default_library_dir, project_override_dir if exists).
    """
    current_file = Path(__file__).resolve()
    tools_dir = current_file.parent.parent
    project_root = current_file.parent.parent.parent.parent / "project_root"

    if config_path is None:
        config_path = tools_dir / "xas_config" / "reference_library.yaml"

    default_dir = tools_dir / "xas_config" / "standards_library"
    override_dir = project_root / "standards_library"

    if Path(config_path).exists():
        try:
            cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            lib_cfg = cfg.get("reference_library", {}) if isinstance(cfg, dict) else {}
            default_dir = tools_dir / "xas_config" / lib_cfg.get("default_dir", "standards_library")
            override_dir = project_root / lib_cfg.get("project_override_dir", "project_root/standards_library")
        except Exception:
            pass

    if not override_dir.exists():
        override_dir = None
    return default_dir, override_dir


def load_reference_library(config_path: Optional[Path] = None,
                           override_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load references defined in reference_library.yaml.
    Returns dict: {label: {energy, mu, shift, source_path}}
    """
    current_file = Path(__file__).resolve()
    tools_dir = current_file.parent.parent
    if config_path is None:
        config_path = tools_dir / "xas_config" / "reference_library.yaml"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Reference library config not found: {config_path}")

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    refs_cfg = cfg.get("references", {}) if isinstance(cfg, dict) else {}

    default_dir, project_override = _get_reference_library_paths(config_path)
    if override_dir is not None:
        project_override = Path(override_dir)

    library = {}
    for label, entry in refs_cfg.items():
        filename = entry.get("filename")
        if not filename:
            continue
        energy_col = entry.get("energy_column", "energy")
        mu_col = entry.get("mu_column", "mu_normalized")
        shift = float(entry.get("shift_eV", 0.0))

        # Prefer project override if present
        candidate = (project_override / filename) if project_override else None
        if candidate is None or not candidate.exists():
            candidate = default_dir / filename

        if not candidate.exists():
            continue

        df = pd.read_csv(candidate)
        if energy_col not in df.columns:
            raise ValueError(f"Energy column '{energy_col}' not found in {candidate}")
        if mu_col not in df.columns:
            # fallback if normalized not available
            if 'mu_cleaned' in df.columns:
                mu_col = 'mu_cleaned'
            else:
                raise ValueError(f"Mu column '{mu_col}' not found in {candidate}")

        library[label] = {
            'energy': df[energy_col].values,
            'mu': df[mu_col].values,
            'shift': shift,
            'source_path': str(candidate)
        }

    return library


def _save_lcf_outputs(sample_name: str,
                      results: Dict[str, Any],
                      model_fit: np.ndarray,
                      residuals: np.ndarray,
                      sample_mu_fit: np.ndarray,
                      energy_fit: np.ndarray,
                      output_data_dir: Optional[Path] = None,
                      output_plots_dir: Optional[Path] = None) -> None:
    data_dir, plots_dir = _default_output_dirs()
    if output_data_dir is not None:
        data_dir = Path(output_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    if output_plots_dir is not None:
        plots_dir = Path(output_plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results (strip large arrays / non-serializable)
    out_json = data_dir / f"{sample_name}_lcf_results.json"
    json_ready = dict(results)
    for key in ['model_fit', 'residuals', 'sample_mu_fit', 'energy_fit']:
        if key in json_ready:
            json_ready.pop(key)
    with open(out_json, 'w') as f:
        json.dump(json_ready, f, indent=2)

    # Save fit curves
    np.savez(
        data_dir / f"{sample_name}_lcf_fit_arrays.npz",
        energy=energy_fit,
        sample=sample_mu_fit,
        model=model_fit,
        residuals=residuals
    )

    # Save CSV table
    out_csv = data_dir / f"{sample_name}_lcf_fit_table.csv"
    np.savetxt(out_csv,
               np.column_stack([energy_fit, sample_mu_fit, model_fit, residuals]),
               delimiter=',',
               header='energy,sample_mu,model_mu,residual',
               comments='')

    # Plot fit
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energy_fit, sample_mu_fit, label='Sample', linewidth=2.2)
    ax.plot(energy_fit, model_fit, label='LCF Fit', linewidth=2.2)
    ax.set_xlabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized μ(E)', fontsize=14, fontweight='bold')
    ax.set_title(f'LCF Fit: {sample_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{sample_name}_lcf_fit.png", dpi=300)
    plt.close(fig)

    # Plot residuals
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(energy_fit, residuals, color='darkred', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Residual', fontsize=14, fontweight='bold')
    ax.set_title(f'LCF Residuals: {sample_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{sample_name}_lcf_residuals.png", dpi=300)
    plt.close(fig)


def perform_xas_lcf_fitting(sample_mu: np.ndarray,
                            energy: np.ndarray,
                            ref_mu_fe2: np.ndarray,
                            ref_mu_fe3: np.ndarray,
                            ref_energy_fe2: Optional[np.ndarray] = None,
                            ref_energy_fe3: Optional[np.ndarray] = None,
                            energy_shift_fe2: float = 0.0,
                            energy_shift_fe3: float = 0.0,
                            e0: float = 7112.0,
                            fit_range: Optional[Tuple[float, float]] = None,
                            fit_region: str = "xanes",
                            pre_edge_range: Tuple[float, float] = (-50.0, -10.0),
                            xanes_range: Tuple[float, float] = (-20.0, 100.0),
                            exafs_range: Tuple[float, float] = (150.0, 800.0),
                            weight_mode: str = "uniform",
                            weight_window: Optional[Tuple[float, float]] = None,
                            weight_factor: float = 3.0,
                            custom_weights: Optional[np.ndarray] = None,
                            bootstrap_samples: int = 200,
                            bootstrap_seed: int = 42,
                            diagnostics: bool = True,
                            allow_scale_offset: bool = True,
                            scale_bounds: Tuple[float, float] = (0.9, 1.1),
                            offset_bounds: Tuple[float, float] = (-0.05, 0.05),
                            adaptive_window: bool = False,
                            window_percentiles: Tuple[float, float] = (5.0, 95.0),
                            sample_name: str = "sample",
                            output_data_dir: Optional[Path] = None,
                            output_plots_dir: Optional[Path] = None,
                            save_outputs: bool = True) -> Dict[str, Any]:
    """
    Perform Linear Combination Fitting (LCF) of XAS spectrum using Fe²⁺ and Fe³⁺ references.

    The LCF equation is:
    μ_sample = f₂₊ × μ_Fe²⁺ + f₃₊ × μ_Fe³⁺
    where f₂₊ + f₃₊ = 1

    Parameters
    ----------
    sample_mu : np.ndarray
        Normalized absorption coefficient of the sample
    energy : np.ndarray
        Energy values corresponding to the sample_mu
    ref_mu_fe2 : np.ndarray
        Normalized absorption coefficient of Fe²⁺ reference
    ref_mu_fe3 : np.ndarray
        Normalized absorption coefficient of Fe³⁺ reference
    e0 : float
        Edge energy for pre-edge subtraction (default: 7112 eV for Fe K-edge)
    fit_range : tuple of float, optional
        Energy range for fitting (e_min, e_max). If None, uses full range.

    Returns
    -------
    result : dict
        Fitting results including fractions, uncertainties, and fit quality metrics
    """
    try:
        # Align reference energy grids to sample energy (interpolation)
        if ref_energy_fe2 is None:
            ref_energy_fe2 = energy
        if ref_energy_fe3 is None:
            ref_energy_fe3 = energy

        ref_energy_fe2 = np.asarray(ref_energy_fe2, dtype=float) + float(energy_shift_fe2)
        ref_energy_fe3 = np.asarray(ref_energy_fe3, dtype=float) + float(energy_shift_fe3)

        ref_mu_fe2 = np.asarray(ref_mu_fe2, dtype=float)
        ref_mu_fe3 = np.asarray(ref_mu_fe3, dtype=float)

        # Interpolate references onto sample energy grid
        ref_mu_fe2_interp = np.interp(energy, ref_energy_fe2, ref_mu_fe2)
        ref_mu_fe3_interp = np.interp(energy, ref_energy_fe3, ref_mu_fe3)

        # Create larch groups for sample and references
        sample = Group(energy=energy, mu=sample_mu)
        ref_fe2 = Group(energy=energy, mu=ref_mu_fe2_interp)
        ref_fe3 = Group(energy=energy, mu=ref_mu_fe3_interp)

        # Pre-edge subtraction and normalization
        pre_edge(sample, e0=e0, pre1=e0-112, pre2=e0-12, norm1=e0+38, norm2=e0+188)
        pre_edge(ref_fe2, e0=e0, pre1=e0-112, pre2=e0-12, norm1=e0+38, norm2=e0+188)
        pre_edge(ref_fe3, e0=e0, pre1=e0-112, pre2=e0-12, norm1=e0+38, norm2=e0+188)

        # Define fitting range
        if fit_range is None:
            region = fit_region.lower()
            if region == "pre_edge":
                fit_range = (e0 + pre_edge_range[0], e0 + pre_edge_range[1])
            elif region == "exafs":
                fit_range = (e0 + exafs_range[0], e0 + exafs_range[1])
            else:
                fit_range = (e0 + xanes_range[0], e0 + xanes_range[1])  # Default XANES
            if adaptive_window:
                p_lo, p_hi = window_percentiles
                fit_range = (np.percentile(energy, p_lo), np.percentile(energy, p_hi))

        # Get indices for fitting range
        fit_mask = (energy >= fit_range[0]) & (energy <= fit_range[1])
        if not np.any(fit_mask):
            raise ValueError(f"No data points in fitting range {fit_range}")

        # Extract data for fitting
        sample_mu_fit = sample.norm[fit_mask]
        ref_fe2_mu_fit = ref_fe2.norm[fit_mask]
        ref_fe3_mu_fit = ref_fe3.norm[fit_mask]
        energy_fit = energy[fit_mask]

        # Build weights
        if custom_weights is not None:
            weights = np.asarray(custom_weights, dtype=float)
            if weights.shape[0] != len(energy_fit):
                raise ValueError("custom_weights length must match fit range length")
        else:
            weights = np.ones_like(energy_fit, dtype=float)
            mode = weight_mode.lower()
            if mode != "uniform":
                if weight_window is None:
                    if mode == "pre_edge":
                        weight_window = (e0 + pre_edge_range[0], e0 + pre_edge_range[1])
                    elif mode == "exafs":
                        weight_window = (e0 + exafs_range[0], e0 + exafs_range[1])
                    else:
                        weight_window = (e0 + xanes_range[0], e0 + xanes_range[1])
                wmask = (energy_fit >= weight_window[0]) & (energy_fit <= weight_window[1])
                weights[wmask] = float(weight_factor)

        # Define LCF function
        def lcf_func(f2: float) -> np.ndarray:
            """Linear combination: f2 * Fe2+ + (1-f2) * Fe3+"""
            return f2 * ref_fe2_mu_fit + (1.0 - f2) * ref_fe3_mu_fit

        # Define residual function for minimization
        def residual(params: np.ndarray) -> np.ndarray:
            if allow_scale_offset:
                f2, scale, offset = params
            else:
                f2 = params[0]
                scale = 1.0
                offset = 0.0
            model = lcf_func(f2) * scale + offset
            return (sample_mu_fit - model) * weights

        # Perform least squares fitting
        from scipy.optimize import least_squares

        # Initial guess: 50% Fe²⁺
        if allow_scale_offset:
            x0 = np.array([0.5, 1.0, 0.0], dtype=float)
            bounds = ([0.0, scale_bounds[0], offset_bounds[0]],
                      [1.0, scale_bounds[1], offset_bounds[1]])
        else:
            x0 = np.array([0.5], dtype=float)
            bounds = (0.0, 1.0)
        result_fit = least_squares(residual, x0, bounds=bounds)

        f2_opt = float(result_fit.x[0])
        f3_opt = 1.0 - f2_opt
        scale_opt = float(result_fit.x[1]) if allow_scale_offset else 1.0
        offset_opt = float(result_fit.x[2]) if allow_scale_offset else 0.0

        # Calculate fit quality metrics
        model_fit = lcf_func(f2_opt) * scale_opt + offset_opt
        residuals = sample_mu_fit - model_fit

        # R-factor (goodness of fit)
        r_factor = np.sum(residuals**2) / np.sum(sample_mu_fit**2)

        # Reduced chi-squared
        n_points = len(sample_mu_fit)
        n_params = 3 if allow_scale_offset else 1
        chi_squared = np.sum(residuals**2)
        dof = max(1, n_points - n_params)
        reduced_chi_squared = chi_squared / dof

        # Estimate uncertainties using Jacobian
        if hasattr(result_fit, 'jac') and result_fit.jac is not None:
            # Calculate parameter uncertainties from covariance matrix
            try:
                # Jacobian is df/df2, residuals are function of f2
                jac = result_fit.jac
                cov_matrix = np.linalg.inv(jac.T @ jac) * (chi_squared / (n_points - n_params))
                f2_uncertainty = np.sqrt(cov_matrix[0, 0])
            except:
                f2_uncertainty = 0.05  # Default uncertainty
        else:
            f2_uncertainty = 0.05  # Default uncertainty

        f3_uncertainty = f2_uncertainty  # Same uncertainty for complementary fraction

        # Bootstrap uncertainty (residual resampling)
        bootstrap_results = None
        if bootstrap_samples and bootstrap_samples > 0:
            try:
                rng = np.random.default_rng(bootstrap_seed)
                resids = residuals.copy()
                f2_boot = []
                for _ in range(int(bootstrap_samples)):
                    resampled = rng.choice(resids, size=resids.shape[0], replace=True)
                    sample_boot = model_fit + resampled

                    def residual_boot(f2: float) -> np.ndarray:
                        model = lcf_func(f2)
                        return (sample_boot - model) * weights

                    if allow_scale_offset:
                        x0b = np.array([f2_opt, scale_opt, offset_opt], dtype=float)
                        bounds_b = ([0.0, scale_bounds[0], offset_bounds[0]],
                                    [1.0, scale_bounds[1], offset_bounds[1]])
                        fit_boot = least_squares(residual_boot, x0b, bounds=bounds_b)
                    else:
                        fit_boot = least_squares(residual_boot, f2_opt, bounds=(0.0, 1.0))
                    f2_boot.append(fit_boot.x[0])

                f2_boot = np.array(f2_boot, dtype=float)
                if f2_boot.size > 5:
                    f2_uncertainty = float(np.std(f2_boot))
                    f3_uncertainty = f2_uncertainty
                bootstrap_results = {
                    'n_samples': int(bootstrap_samples),
                    'f2_mean': float(np.mean(f2_boot)),
                    'f2_std': float(np.std(f2_boot)),
                    'f2_ci95': [float(np.percentile(f2_boot, 2.5)), float(np.percentile(f2_boot, 97.5))]
                }
            except Exception:
                bootstrap_results = None

        # Diagnostics
        diagnostics_report = None
        if diagnostics:
            diagnostics_report = {
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'max_abs_residual': float(np.max(np.abs(residuals))),
                'weighted_residual_std': float(np.std((sample_mu_fit - model_fit) * weights))
            }

        # Prepare results
        results = {
            'success': result_fit.success,
            'f2_fraction': f2_opt,
            'f3_fraction': f3_opt,
            'f2_uncertainty': f2_uncertainty,
            'f3_uncertainty': f3_uncertainty,
            'r_factor': r_factor,
            'reduced_chi_squared': reduced_chi_squared,
            'degrees_of_freedom': dof,
            'fit_range': fit_range,
            'n_points_fitted': n_points,
            'scale_factor': scale_opt,
            'offset': offset_opt,
            'allow_scale_offset': allow_scale_offset,
            'weight_mode': weight_mode,
            'weight_window': weight_window,
            'weight_factor': weight_factor,
            'adaptive_window': adaptive_window,
            'window_percentiles': window_percentiles,
            'bootstrap': bootstrap_results,
            'diagnostics': diagnostics_report,
            'normalization': {
                'e0': e0,
                'pre1': e0 - 112,
                'pre2': e0 - 12,
                'norm1': e0 + 38,
                'norm2': e0 + 188
            },
            'config': {
                'fit_region': fit_region,
                'pre_edge_range': pre_edge_range,
                'xanes_range': xanes_range,
                'exafs_range': exafs_range,
                'weight_mode': weight_mode,
                'weight_window': weight_window,
                'weight_factor': weight_factor,
                'allow_scale_offset': allow_scale_offset,
                'scale_bounds': scale_bounds,
                'offset_bounds': offset_bounds,
                'adaptive_window': adaptive_window,
                'window_percentiles': window_percentiles,
                'bootstrap_samples': bootstrap_samples,
                'bootstrap_seed': bootstrap_seed,
                'diagnostics': diagnostics
            },
            'fit_quality': _assess_fit_quality(r_factor, reduced_chi_squared),
            'model_fit': model_fit,
            'residuals': residuals,
            'sample_mu_fit': sample_mu_fit,
            'energy_fit': energy_fit
        }

        if save_outputs:
            _save_lcf_outputs(
                sample_name=sample_name,
                results=results,
                model_fit=model_fit,
                residuals=residuals,
                sample_mu_fit=sample_mu_fit,
                energy_fit=energy_fit,
                output_data_dir=output_data_dir,
                output_plots_dir=output_plots_dir
            )

        return results

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'f2_fraction': None,
            'f3_fraction': None,
            'r_factor': None,
            'fit_quality': 'failed'
        }


def _assess_fit_quality(r_factor: float, reduced_chi_squared: float) -> str:
    """
    Assess the quality of the LCF fit based on statistical metrics.

    Parameters
    ----------
    r_factor : float
        R-factor (sum of squared residuals / sum of squared data)
    reduced_chi_squared : float
        Reduced chi-squared statistic

    Returns
    -------
    quality : str
        Fit quality assessment ('excellent', 'good', 'acceptable', 'poor')
    """
    if r_factor < 0.005 and reduced_chi_squared < 2.0:
        return 'excellent'
    elif r_factor < 0.01 and reduced_chi_squared < 5.0:
        return 'good'
    elif r_factor < 0.02 and reduced_chi_squared < 10.0:
        return 'acceptable'
    else:
        return 'poor'


def perform_xas_lcf_fitting_multi(sample_mu: np.ndarray,
                                  energy: np.ndarray,
                                  ref_spectra: List[Dict[str, Any]],
                                  e0: float = 7112.0,
                                  fit_range: Optional[Tuple[float, float]] = None,
                                  fit_region: str = "xanes",
                                  pre_edge_range: Tuple[float, float] = (-50.0, -10.0),
                                  xanes_range: Tuple[float, float] = (-20.0, 100.0),
                                  exafs_range: Tuple[float, float] = (150.0, 800.0),
                                  weight_mode: str = "uniform",
                                  weight_window: Optional[Tuple[float, float]] = None,
                                  weight_factor: float = 3.0,
                                  custom_weights: Optional[np.ndarray] = None,
                                  bootstrap_samples: int = 200,
                                  bootstrap_seed: int = 42,
                                  diagnostics: bool = True,
                                  allow_scale_offset: bool = True,
                                  scale_bounds: Tuple[float, float] = (0.9, 1.1),
                                  offset_bounds: Tuple[float, float] = (-0.05, 0.05),
                                  adaptive_window: bool = False,
                                  window_percentiles: Tuple[float, float] = (5.0, 95.0),
                                  sample_name: str = "sample",
                                  output_data_dir: Optional[Path] = None,
                                  output_plots_dir: Optional[Path] = None,
                                  save_outputs: bool = True) -> Dict[str, Any]:
    """
    Generalized LCF with N references and constraints (sum=1, non-negative).

    ref_spectra: list of dicts with keys:
      - 'mu' (required)
      - 'energy' (optional; if missing, uses sample energy)
      - 'shift' (optional eV shift)
      - 'label' (optional name)
    """
    try:
        if len(ref_spectra) < 2:
            raise ValueError("At least two references are required for LCF.")

        # Prepare reference matrix (interpolate onto sample energy grid)
        ref_labels = []
        ref_matrix = []
        for i, ref in enumerate(ref_spectra):
            ref_mu = np.asarray(ref.get('mu'), dtype=float)
            ref_energy = np.asarray(ref.get('energy', energy), dtype=float)
            shift = float(ref.get('shift', 0.0))
            ref_label = ref.get('label', f"ref_{i+1}")

            ref_energy = ref_energy + shift
            ref_mu_interp = np.interp(energy, ref_energy, ref_mu)
            ref_matrix.append(ref_mu_interp)
            ref_labels.append(ref_label)

        ref_matrix = np.vstack(ref_matrix)  # (n_refs, n_points)

        # Larch normalization
        sample = Group(energy=energy, mu=sample_mu)
        pre_edge(sample, e0=e0, pre1=e0-112, pre2=e0-12, norm1=e0+38, norm2=e0+188)

        ref_norms = []
        for ref_mu in ref_matrix:
            g = Group(energy=energy, mu=ref_mu)
            pre_edge(g, e0=e0, pre1=e0-112, pre2=e0-12, norm1=e0+38, norm2=e0+188)
            ref_norms.append(g.norm)
        ref_norms = np.vstack(ref_norms)

        # Fit range
        if fit_range is None:
            region = fit_region.lower()
            if region == "pre_edge":
                fit_range = (e0 + pre_edge_range[0], e0 + pre_edge_range[1])
            elif region == "exafs":
                fit_range = (e0 + exafs_range[0], e0 + exafs_range[1])
            else:
                fit_range = (e0 + xanes_range[0], e0 + xanes_range[1])
            if adaptive_window:
                p_lo, p_hi = window_percentiles
                fit_range = (np.percentile(energy, p_lo), np.percentile(energy, p_hi))

        fit_mask = (energy >= fit_range[0]) & (energy <= fit_range[1])
        if not np.any(fit_mask):
            raise ValueError(f"No data points in fitting range {fit_range}")

        energy_fit = energy[fit_mask]
        sample_mu_fit = sample.norm[fit_mask]
        ref_fit = ref_norms[:, fit_mask]

        # Weights
        if custom_weights is not None:
            weights = np.asarray(custom_weights, dtype=float)
            if weights.shape[0] != len(energy_fit):
                raise ValueError("custom_weights length must match fit range length")
        else:
            weights = np.ones_like(energy_fit, dtype=float)
            mode = weight_mode.lower()
            if mode != "uniform":
                if weight_window is None:
                    if mode == "pre_edge":
                        weight_window = (e0 + pre_edge_range[0], e0 + pre_edge_range[1])
                    elif mode == "exafs":
                        weight_window = (e0 + exafs_range[0], e0 + exafs_range[1])
                    else:
                        weight_window = (e0 + xanes_range[0], e0 + xanes_range[1])
                wmask = (energy_fit >= weight_window[0]) & (energy_fit <= weight_window[1])
                weights[wmask] = float(weight_factor)

        # Constrained optimization (SLSQP)
        from scipy.optimize import minimize

        n_refs = ref_fit.shape[0]
        x0 = np.ones(n_refs) / n_refs
        if allow_scale_offset:
            x0 = np.concatenate([x0, [1.0, 0.0]])

        def model_mix(fracs, scale=1.0, offset=0.0):
            return np.dot(fracs, ref_fit) * scale + offset

        def objective(params):
            if allow_scale_offset:
                fracs = params[:-2]
                scale = params[-2]
                offset = params[-1]
            else:
                fracs = params
                scale = 1.0
                offset = 0.0
            resid = (sample_mu_fit - model_mix(fracs, scale, offset)) * weights
            return np.sum(resid ** 2)

        if allow_scale_offset:
            constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[:-2]) - 1.0}]
            bounds = [(0.0, 1.0) for _ in range(n_refs)] + [scale_bounds, offset_bounds]
        else:
            constraints = [{'type': 'eq', 'fun': lambda f: np.sum(f) - 1.0}]
            bounds = [(0.0, 1.0) for _ in range(n_refs)]

        opt = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if allow_scale_offset:
            fracs_opt = opt.x[:-2]
            scale_opt = float(opt.x[-2])
            offset_opt = float(opt.x[-1])
        else:
            fracs_opt = opt.x
            scale_opt = 1.0
            offset_opt = 0.0

        model_fit = model_mix(fracs_opt, scale_opt, offset_opt)
        residuals = sample_mu_fit - model_fit

        r_factor = np.sum(residuals**2) / np.sum(sample_mu_fit**2)
        n_points = len(sample_mu_fit)
        n_params = (n_refs - 1) + (2 if allow_scale_offset else 0)
        chi_squared = np.sum(residuals**2)
        dof = max(1, (n_points - n_params))
        reduced_chi_squared = chi_squared / dof

        # Bootstrap uncertainty
        bootstrap_results = None
        frac_uncertainty = None
        if bootstrap_samples and bootstrap_samples > 0:
            try:
                rng = np.random.default_rng(bootstrap_seed)
                resids = residuals.copy()
                frac_boot = []
                for _ in range(int(bootstrap_samples)):
                    resampled = rng.choice(resids, size=resids.shape[0], replace=True)
                    sample_boot = model_fit + resampled

                    def obj_boot(params):
                        if allow_scale_offset:
                            fracs = params[:-2]
                            scale = params[-2]
                            offset = params[-1]
                        else:
                            fracs = params
                            scale = 1.0
                            offset = 0.0
                        resid = (sample_boot - model_mix(fracs, scale, offset)) * weights
                        return np.sum(resid ** 2)

                    x0b = fracs_opt
                    if allow_scale_offset:
                        x0b = np.concatenate([fracs_opt, [scale_opt, offset_opt]])
                    opt_b = minimize(obj_boot, x0b, method='SLSQP', bounds=bounds, constraints=constraints)
                    frac_boot.append(opt_b.x[:n_refs] if allow_scale_offset else opt_b.x)

                frac_boot = np.array(frac_boot)
                if frac_boot.shape[0] > 5:
                    frac_uncertainty = np.std(frac_boot, axis=0)
                bootstrap_results = {
                    'n_samples': int(bootstrap_samples),
                    'fraction_mean': np.mean(frac_boot, axis=0).tolist(),
                    'fraction_std': np.std(frac_boot, axis=0).tolist(),
                    'fraction_ci95': [
                        [float(np.percentile(frac_boot[:, i], 2.5)), float(np.percentile(frac_boot[:, i], 97.5))]
                        for i in range(n_refs)
                    ]
                }
            except Exception:
                bootstrap_results = None

        diagnostics_report = None
        if diagnostics:
            diagnostics_report = {
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'max_abs_residual': float(np.max(np.abs(residuals))),
                'weighted_residual_std': float(np.std((sample_mu_fit - model_fit) * weights))
            }

        results = {
            'success': bool(opt.success),
            'fractions': fracs_opt.tolist(),
            'fraction_labels': ref_labels,
            'fraction_uncertainty': None if frac_uncertainty is None else frac_uncertainty.tolist(),
            'r_factor': r_factor,
            'reduced_chi_squared': reduced_chi_squared,
            'degrees_of_freedom': dof,
            'fit_range': fit_range,
            'n_points_fitted': n_points,
            'scale_factor': scale_opt,
            'offset': offset_opt,
            'allow_scale_offset': allow_scale_offset,
            'weight_mode': weight_mode,
            'weight_window': weight_window,
            'weight_factor': weight_factor,
            'adaptive_window': adaptive_window,
            'window_percentiles': window_percentiles,
            'bootstrap': bootstrap_results,
            'diagnostics': diagnostics_report,
            'normalization': {
                'e0': e0,
                'pre1': e0 - 112,
                'pre2': e0 - 12,
                'norm1': e0 + 38,
                'norm2': e0 + 188
            },
            'config': {
                'fit_region': fit_region,
                'pre_edge_range': pre_edge_range,
                'xanes_range': xanes_range,
                'exafs_range': exafs_range,
                'weight_mode': weight_mode,
                'weight_window': weight_window,
                'weight_factor': weight_factor,
                'allow_scale_offset': allow_scale_offset,
                'scale_bounds': scale_bounds,
                'offset_bounds': offset_bounds,
                'adaptive_window': adaptive_window,
                'window_percentiles': window_percentiles,
                'bootstrap_samples': bootstrap_samples,
                'bootstrap_seed': bootstrap_seed,
                'diagnostics': diagnostics
            },
            'model_fit': model_fit,
            'residuals': residuals,
            'sample_mu_fit': sample_mu_fit,
            'energy_fit': energy_fit
        }

        if save_outputs:
            _save_lcf_outputs(
                sample_name=sample_name,
                results=results,
                model_fit=model_fit,
                residuals=residuals,
                sample_mu_fit=sample_mu_fit,
                energy_fit=energy_fit,
                output_data_dir=output_data_dir,
                output_plots_dir=output_plots_dir
            )

        return results

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'fractions': None,
            'r_factor': None
        }


def perform_batch_lcf_analysis(samples_data: Dict[str, Dict],
                              ref_fe2_data: Dict,
                              ref_fe3_data: Dict,
                              e0: float = 7112.0,
                              output_data_dir: Optional[Path] = None,
                              output_plots_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Perform LCF analysis on a batch of samples.

    Parameters
    ----------
    samples_data : dict
        Dictionary of sample data: {sample_name: {'energy': array, 'mu': array}}
    ref_fe2_data : dict
        Fe²⁺ reference data: {'energy': array, 'mu': array}
    ref_fe3_data : dict
        Fe³⁺ reference data: {'energy': array, 'mu': array}
    e0 : float
        Edge energy for fitting

    Returns
    -------
    batch_results : dict
        LCF results for all samples
    """
    batch_results = {}
    index_records = []

    for sample_name, sample_data in samples_data.items():
        try:
            result = perform_xas_lcf_fitting(
                sample_mu=sample_data['mu'],
                energy=sample_data['energy'],
                ref_mu_fe2=ref_fe2_data['mu'],
                ref_mu_fe3=ref_fe3_data['mu'],
                e0=e0,
                sample_name=sample_name,
                output_data_dir=output_data_dir,
                output_plots_dir=output_plots_dir,
                save_outputs=True
            )

            batch_results[sample_name] = result
            index_records.append({
                'sample_name': sample_name,
                'result_json': str((output_data_dir or _default_output_dirs()[0]) / f"{sample_name}_lcf_results.json"),
                'fit_arrays_npz': str((output_data_dir or _default_output_dirs()[0]) / f"{sample_name}_lcf_fit_arrays.npz"),
                'fit_table_csv': str((output_data_dir or _default_output_dirs()[0]) / f"{sample_name}_lcf_fit_table.csv"),
                'fit_plot': str((output_plots_dir or _default_output_dirs()[1]) / f"{sample_name}_lcf_fit.png"),
                'residual_plot': str((output_plots_dir or _default_output_dirs()[1]) / f"{sample_name}_lcf_residuals.png")
            })
            print(f"LCF analysis completed for {sample_name}: "
                  f"Fe²⁺ = {result.get('f2_fraction', 'N/A'):.1%}, "
                  f"R-factor = {result.get('r_factor', 'N/A'):.4f}")

        except Exception as e:
            print(f"LCF analysis failed for {sample_name}: {e}")
            batch_results[sample_name] = {
                'success': False,
                'error': str(e)
            }

    # Batch summary CSV + index JSON
    data_dir, plots_dir = _default_output_dirs()
    if output_data_dir is not None:
        data_dir = Path(output_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for sample_name, result in batch_results.items():
        if not result.get('success'):
            summary_rows.append({
                'sample_name': sample_name,
                'success': False,
                'r_factor': None,
                'reduced_chi_squared': None,
                'f2_fraction': None,
                'f3_fraction': None,
                'fit_quality': result.get('fit_quality')
            })
            continue
        summary_rows.append({
            'sample_name': sample_name,
            'success': True,
            'r_factor': result.get('r_factor'),
            'reduced_chi_squared': result.get('reduced_chi_squared'),
            'f2_fraction': result.get('f2_fraction'),
            'f3_fraction': result.get('f3_fraction'),
            'fit_quality': result.get('fit_quality'),
            'n_points_fitted': result.get('n_points_fitted'),
            'degrees_of_freedom': result.get('degrees_of_freedom')
        })

    summary_csv = data_dir / "lcf_batch_summary.csv"
    try:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    except Exception:
        with open(summary_csv, 'w') as f:
            headers = summary_rows[0].keys() if summary_rows else []
            f.write(",".join(headers) + "\n")
            for row in summary_rows:
                f.write(",".join([str(row.get(h, "")) for h in headers]) + "\n")

    index_json = data_dir / "lcf_outputs_index.json"
    with open(index_json, 'w') as f:
        json.dump(index_records, f, indent=2)

    return batch_results


def perform_batch_lcf_analysis_from_library(samples_data: Dict[str, Dict],
                                           ref_labels: Tuple[str, str] = ("Fe2+", "Fe3+"),
                                           e0: float = 7112.0,
                                           config_path: Optional[Path] = None,
                                           override_dir: Optional[Path] = None,
                                           output_data_dir: Optional[Path] = None,
                                           output_plots_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Batch LCF using references from the reference library config.
    """
    library = load_reference_library(config_path=config_path, override_dir=override_dir)
    if ref_labels[0] not in library or ref_labels[1] not in library:
        raise ValueError(f"Reference labels not found in library: {ref_labels}")

    ref_fe2_data = {
        'energy': library[ref_labels[0]]['energy'],
        'mu': library[ref_labels[0]]['mu']
    }
    ref_fe3_data = {
        'energy': library[ref_labels[1]]['energy'],
        'mu': library[ref_labels[1]]['mu']
    }

    return perform_batch_lcf_analysis(
        samples_data=samples_data,
        ref_fe2_data=ref_fe2_data,
        ref_fe3_data=ref_fe3_data,
        e0=e0,
        output_data_dir=output_data_dir,
        output_plots_dir=output_plots_dir
    )


def create_lcf_comparison_plots(batch_results: Dict[str, Dict],
                               output_dir: str | Path = "lcf_comparison_plots") -> Dict[str, str]:
    """
    Create comparison plots for LCF results across multiple samples.

    Parameters
    ----------
    batch_results : dict
        LCF results from perform_batch_lcf_analysis
    output_dir : str or Path
        Output directory for plots

    Returns
    -------
    plot_files : dict
        Dictionary mapping plot types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    plot_files = {}

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Extract successful results
        successful_samples = {}
        for sample_name, result in batch_results.items():
            if result.get('success', False):
                successful_samples[sample_name] = result

        if not successful_samples:
            print("No successful LCF results to plot")
            return plot_files

        # Plot 1: Fe²⁺/Fe³⁺ fractions comparison
        plt.figure(figsize=(12, 6))

        sample_names = list(successful_samples.keys())
        f2_values = [successful_samples[s]['f2_fraction'] for s in sample_names]
        f2_errors = [successful_samples[s]['f2_uncertainty'] for s in sample_names]

        bars = plt.bar(range(len(sample_names)), f2_values, yerr=f2_errors,
                      capsize=5, alpha=0.7, color='skyblue', label='Fe²⁺ fraction')
        plt.bar(range(len(sample_names)), [1-v for v in f2_values], bottom=f2_values,
               alpha=0.7, color='lightcoral', label='Fe³⁺ fraction')

        plt.xticks(range(len(sample_names)), sample_names, rotation=45, ha='right')
        plt.ylabel('Fraction')
        plt.title('LCF Results: Fe²⁺/Fe³⁺ Fractions Across Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fractions_plot = output_dir / "lcf_fractions_comparison.png"
        plt.savefig(fractions_plot, dpi=150, bbox_inches='tight')
        plot_files['fractions'] = str(fractions_plot)
        plt.close()

        # Plot 2: Fit quality metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        r_factors = [successful_samples[s]['r_factor'] for s in sample_names]
        chi_squared = [successful_samples[s]['reduced_chi_squared'] for s in sample_names]

        ax1.bar(range(len(sample_names)), r_factors, color='lightgreen', alpha=0.7)
        ax1.set_xticks(range(len(sample_names)))
        ax1.set_xticklabels(sample_names, rotation=45, ha='right')
        ax1.set_ylabel('R-factor')
        ax1.set_title('Fit Quality: R-factor')
        ax1.axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Acceptable limit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(range(len(sample_names)), chi_squared, color='gold', alpha=0.7)
        ax2.set_xticks(range(len(sample_names)))
        ax2.set_xticklabels(sample_names, rotation=45, ha='right')
        ax2.set_ylabel('Reduced χ²')
        ax2.set_title('Fit Quality: Reduced χ²')
        ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Acceptable limit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        quality_plot = output_dir / "lcf_fit_quality.png"
        plt.savefig(quality_plot, dpi=150, bbox_inches='tight')
        plot_files['quality'] = str(quality_plot)
        plt.close()

        print(f"Created {len(plot_files)} LCF comparison plots in {output_dir}")

    except ImportError:
        print("Warning: matplotlib/seaborn not available for LCF plots")
    except Exception as e:
        print(f"Warning: Could not create LCF comparison plots: {e}")

    return plot_files
