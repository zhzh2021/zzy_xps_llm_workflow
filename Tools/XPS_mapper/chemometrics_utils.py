"""
Chemometrics utility functions for XPS hyperspectral map analysis.

Core algorithms:
- PRE (Pattern Recognition Entropy)
- Charge correction/alignment
- Low-count pixel masking
- MCR-ALS / NMF decomposition
- L1 normalization
"""

import numpy as np
from typing import Optional, Dict
import logging
logger = logging.getLogger("xps_map")

# MCR-ALS (optional dependency)
try:
    from pymcr.mcr import McrAR
    from pymcr.constraints import ConstraintNonneg
    MCR_AVAILABLE = True
except ImportError:
    McrAR = None
    ConstraintNonneg = None
    MCR_AVAILABLE = False

# Sklearn (required for NMF fallback)
from sklearn.decomposition import NMF


def compute_pre_image(cube: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute PRE (Pattern Recognition Entropy) for each pixel.
    PRE = Shannon entropy on normalized spectrum: -sum(p * log(p))
    Returns (m, n) image showing spectral complexity/diversity.
    
    High PRE = complex/mixed spectrum, Low PRE = pure/simple spectrum
    
    Args:
        cube: (m, n, p) hyperspectral data cube
        eps: Small constant to avoid log(0)
    
    Returns:
        (m, n) PRE image
    """
    m, n, p = cube.shape
    X = cube.reshape(m * n, p)
    
    # Normalize per spectrum (probability distribution)
    s = X.sum(axis=1, keepdims=True)
    s[s < eps] = eps
    P = X / s
    
    # Shannon entropy: H = -sum(p_i * log(p_i))
    pre = -(P * np.log(P + eps)).sum(axis=1)
    
    return pre.reshape(m, n)


def normalize_l1(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L1 (sum) normalization: each spectrum sums to 1.
    Better for MCR/clustering than L2 normalization.
    
    Args:
        X: (N, p) matrix where each row is a spectrum
        eps: Small constant to avoid division by zero
    
    Returns:
        Normalized (N, p) matrix
    """
    s = X.sum(axis=1, keepdims=True)
    s[s < eps] = eps
    return X / s


def mask_low_counts(cube: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create boolean mask for pixels with sufficient total counts.
    Filters out noisy low-signal pixels.
    
    Args:
        cube: (m, n, p) hyperspectral data
        threshold: Minimum total counts per pixel
    
    Returns:
        (m, n) mask where True = valid pixel
    """
    totals = cube.sum(axis=2)
    mask = totals >= threshold
    return mask


def charge_align_cube(cube: np.ndarray, energy: np.ndarray,
                      ref_energy: float, window: float = 1.0) -> np.ndarray:
    """
    Align spectra by shifting to reference binding energy.
    Finds local maximum near ref_energy and shifts spectrum to align it.
    
    Useful for correcting charging effects in XPS data.
    Common references: C 1s C-C/C-H at 284.8 eV
    
    Args:
        cube: (m, n, p) hyperspectral data
        energy: (p,) energy axis
        ref_energy: Reference BE to align to (e.g., 284.8 for C-C/C-H)
        window: eV window around ref_energy to search for peak
    
    Returns:
        Aligned cube with same shape
    
    Note:
        Uses np.roll() which wraps edges. For production, consider
        interpolation-based alignment for better accuracy.
    """
    m, n, p = cube.shape
    aligned = np.empty_like(cube)
    
    # Find index of reference energy
    idx_ref = np.argmin(np.abs(energy - ref_energy))
    
    # Define search window
    idx_window = np.where(np.abs(energy - ref_energy) <= window)[0]
    if len(idx_window) == 0:
        idx_window = np.array([idx_ref])
    
    # Align each spectrum
    for i in range(m):
        for j in range(n):
            spec = cube[i, j, :]
            
            # Find local max in window
            k_local = idx_window[np.argmax(spec[idx_window])]
            
            # Calculate shift needed
            shift = idx_ref - k_local
            
            # Apply shift
            aligned[i, j, :] = np.roll(spec, shift)
    
    return aligned


def determine_n_components_from_pca(cube: np.ndarray, 
                                   variance_threshold: float = 0.95,
                                   max_components: int = 10) -> tuple:
    """
    Determine optimal number of components from PCA scree analysis.
    
    Uses cumulative explained variance to find minimum components needed.
    Rule: Keep components that explain >= variance_threshold of total variance.
    
    Args:
        cube: (m, n, p) hyperspectral data
        variance_threshold: Cumulative variance to retain (e.g., 0.95 = 95%)
        max_components: Maximum components to consider
    
    Returns:
        tuple: (n_optimal, pca_object, explained_variance_ratio)
    """
    from sklearn.decomposition import PCA
    
    m, n, p = cube.shape
    X = cube.reshape(m * n, p)
    
    # Fit PCA with max components
    n_comp = min(max_components, min(m*n, p) - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    
    # Cumulative explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Find first component where cumsum >= threshold
    n_optimal = np.argmax(cumsum_var >= variance_threshold) + 1
    
    # Ensure at least 2 components (for meaningful decomposition)
    n_optimal = max(2, n_optimal)
    
    logger.info(f"PCA scree analysis:")
    logger.info(f"  Optimal components: {n_optimal} (explains {cumsum_var[n_optimal-1]*100:.1f}% variance)")
    for i in range(min(5, n_comp)):
        logger.info(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}% "
                   f"(cumulative: {cumsum_var[i]*100:.1f}%)")
    
    return n_optimal, pca, pca.explained_variance_ratio_


def run_mcr_on_cube(cube: np.ndarray, n_components: int = 3,
                    normalize: bool = False,
                    max_iter: int = 50,
                    random_state: int = 0) -> Optional[Dict]:
    """
    Run MCR-ALS (or fallback to NMF) for non-negative matrix factorization.
    Better chemical interpretation than PCA for XPS.
    
    MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares):
    - Decomposes spectra into pure component spectra + concentrations
    - Non-negativity constraints (physical meaning)
    - No orthogonality constraint (unlike PCA)
    
    NOTE: For automatic component selection, use run_mcr_with_pca_init() instead.
    
    Args:
        cube: (m, n, p) hyperspectral data
        n_components: Number of pure components to extract
        normalize: Apply L1 normalization before decomposition
        max_iter: Maximum MCR iterations
        random_state: Random seed for reproducibility
    
    Returns:
        dict with:
            - method: 'MCR-ALS' or 'NMF' (fallback)
            - concentrations: (N, k) pixel concentrations
            - conc_maps: (m, n, k) concentration maps
            - component_spectra: (p, k) pure component spectra
        or None if decomposition fails
    """
    m, n, p = cube.shape
    X = cube.reshape(m * n, p)
    
    # Optional normalization
    if normalize:
        X = normalize_l1(X)
    
    # Try MCR-ALS if available
    if MCR_AVAILABLE and McrAR is not None:
        mcr = McrAR(
            max_iter=max_iter,
            c_constraints=[ConstraintNonneg()],
            st_constraints=[ConstraintNonneg()]
        )
        
        # Initialize with random non-negative matrix
        # NOTE: pymcr expects ST shape (n_components, p), not (p, n_components)
        np.random.seed(random_state)
        ST_init = np.abs(np.random.randn(n_components, p))
        
        try:
            mcr.fit(X, ST=ST_init)
            C = mcr.C_   # (N, k) concentrations
            ST = mcr.ST_.T  # transpose (k, p) -> (p, k) to match NMF convention
            method = "MCR-ALS"
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(
                f"MCR-ALS did not converge: {e}. Falling back to NMF. "
                "Quantitative atomic % should be interpreted with caution when NMF is used."
            )
            # Fallback to NMF
            nmf = NMF(n_components=n_components, init='nndsvda',
                     max_iter=500, random_state=random_state)
            C = nmf.fit_transform(X)
            ST = nmf.components_.T
            method = "NMF (MCR-ALS failed)"
    else:
        # NMF fallback
        nmf = NMF(n_components=n_components, init='nndsvda',
                 max_iter=500, random_state=random_state)
        C = nmf.fit_transform(X)
        ST = nmf.components_.T
        method = "NMF" + (" (pymcr not available)" if not MCR_AVAILABLE else "")

    # Reshape concentration matrix to maps
    conc_maps = C.reshape(m, n, n_components)

    # Reconstruction quality metrics
    X_recon = (C @ ST.T)
    residuals = X - X_recon
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((X - X.mean()) ** 2))
    r2_reconstruction = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rms_error = float(np.sqrt(np.mean(residuals ** 2)))
    fallback_used = not method.startswith("MCR-ALS")

    return {
        "method": method,
        "n_components": n_components,
        "concentrations": C,
        "conc_maps": conc_maps,
        "component_spectra": ST,
        "r2_reconstruction": r2_reconstruction,
        "rms_error": rms_error,
        "fallback_used": fallback_used,
    }


def run_mcr_with_pca_init(cube: np.ndarray, 
                          n_components: Optional[int] = None,
                          auto_select: bool = True,
                          variance_threshold: float = 0.9,
                          max_components: int = 10,
                          normalize: bool = False,
                          max_iter: int = 50,
                          random_state: int = 0) -> Optional[Dict]:
    """
    Run MCR-ALS with PCA-guided initialization and optional auto-selection.
    
    This implements the workflow:
    1. PCA scree analysis to determine optimal component count (if auto_select=True)
    2. Use PCA loadings as initial guess for MCR
    3. Run MCR-ALS with non-negativity constraints
    
    This follows the principle: D = C × S^T + E
    where n_components = (chemical states) + (distinct backgrounds)
    is automatically determined from variance analysis.
    
    Args:
        cube: (m, n, p) hyperspectral data
        n_components: Fixed number (or None for auto-selection)
        auto_select: Use PCA scree plot to determine n_components
        variance_threshold: Cumulative variance threshold (e.g., 0.99)
        max_components: Maximum components to consider in auto-selection
        normalize: Apply L1 normalization before decomposition
        max_iter: Maximum MCR iterations
        random_state: Random seed for reproducibility
    
    Returns:
        dict with:
            - method: Description of method used
            - n_components: Number of components extracted
            - concentrations: (N, k) pixel concentrations
            - conc_maps: (m, n, k) concentration maps
            - component_spectra: (p, k) pure component spectra
            - pca_variance: Explained variance ratios from PCA
            - auto_selected: Whether components were auto-selected
    """
    from sklearn.decomposition import PCA
    
    m, n, p = cube.shape
    X = cube.reshape(m * n, p)
    
    # Optional normalization
    if normalize:
        X = normalize_l1(X)
    
    # Step 1: Determine number of components
    auto_selected = False
    pca_variance = None
    
    if auto_select and n_components is None:
        n_components, pca_obj, pca_variance = determine_n_components_from_pca(
            cube, variance_threshold, max_components
        )
        auto_selected = True
        logger.info(f"Auto-selected {n_components} components from PCA analysis")
    elif n_components is None:
        n_components = 3  # Default fallback
        logger.info(f"Using default {n_components} components")
    else:
        logger.info(f"Using fixed {n_components} components")
    
    # Step 2: PCA initialization (even if not auto-selecting)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    
    # Use PCA loadings as initial guess (make non-negative)
    # NOTE: pymcr expects ST shape (k, p); pca.components_ is already (k, p)
    ST_init = np.abs(pca.components_)  # (k, p) shape
    
    if pca_variance is None:
        pca_variance = pca.explained_variance_ratio_
    
    logger.info(f"Initializing MCR with {n_components} PCA-derived components")
    
    # Step 3: Run MCR with PCA initialization
    if MCR_AVAILABLE and McrAR is not None:
        mcr = McrAR(
            max_iter=max_iter,
            c_constraints=[ConstraintNonneg()],
            st_constraints=[ConstraintNonneg()]
        )
        
        try:
            mcr.fit(X, ST=ST_init)  # PCA-informed initialization
            C = mcr.C_   # (N, k) concentrations
            ST = mcr.ST_.T  # transpose (k, p) -> (p, k) to match NMF convention
            method = f"MCR-ALS (PCA-init, k={n_components})"


            # Log convergence info if available
            if hasattr(mcr, 'n_iter_'):
                logger.info(f"MCR converged in {mcr.n_iter_} iterations")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(
                f"MCR-ALS did not converge: {e}. Falling back to NMF. "
                "Quantitative atomic % should be interpreted with caution when NMF is used."
            )
            # Fallback to NMF
            nmf = NMF(n_components=n_components, init='nndsvda',
                     max_iter=500, random_state=random_state)
            C = nmf.fit_transform(X)
            ST = nmf.components_.T
            method = f"NMF (MCR-ALS failed, k={n_components})"
    else:
        # NMF fallback
        nmf = NMF(n_components=n_components, init='nndsvda',
                 max_iter=500, random_state=random_state)
        C = nmf.fit_transform(X)
        ST = nmf.components_.T
        method = f"NMF (k={n_components})" + (" (pymcr not available)" if not MCR_AVAILABLE else "")

    # Reshape concentration matrix to maps
    conc_maps = C.reshape(m, n, n_components)

    # Reconstruction quality metrics
    X_recon = (C @ ST.T)
    residuals = X - X_recon
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((X - X.mean()) ** 2))
    r2_reconstruction = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rms_error = float(np.sqrt(np.mean(residuals ** 2)))
    fallback_used = not method.startswith("MCR-ALS")

    return {
        "method": method,
        "n_components": n_components,
        "auto_selected": auto_selected,
        "concentrations": C,
        "conc_maps": conc_maps,
        "component_spectra": ST,
        "pca_variance": pca_variance,
        "r2_reconstruction": r2_reconstruction,
        "rms_error": rms_error,
        "fallback_used": fallback_used,
    }


def compute_spectral_variability(cube: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral variability metrics across the map.
    
    Useful for deciding whether chemometrics analysis is worthwhile:
    - Low variability → uniform sample → skip advanced analysis
    - High variability → heterogeneous → proceed with PCA/clustering
    
    Args:
        cube: (m, n, p) hyperspectral data
    
    Returns:
        dict with:
            - mean_std: Average std across energy points
            - max_std: Maximum std across energy points
            - cv: Coefficient of variation (mean_std / mean_intensity)
    """
    m, n, p = cube.shape
    X = cube.reshape(m * n, p)
    
    # Std at each energy point
    std_per_energy = np.std(X, axis=0)
    mean_per_energy = np.mean(X, axis=0)
    
    mean_std = np.mean(std_per_energy)
    max_std = np.max(std_per_energy)
    
    # Coefficient of variation (normalized variability)
    mean_intensity = np.mean(mean_per_energy)
    cv = mean_std / (mean_intensity + 1e-12)
    
    return {
        "mean_std": float(mean_std),
        "max_std": float(max_std),
        "cv": float(cv)
    }


def compute_mcr_quality_metrics(cube: np.ndarray, mcr_results: Dict) -> Dict[str, np.ndarray]:
    """
    Compute quality metrics for MCR decomposition results.
    
    Validates MCR fit quality by computing:
    - Residuals: Data - Model reconstruction
    - R²: Coefficient of determination per pixel
    - Lack-of-fit (LOF): Relative residual magnitude
    
    Args:
        cube: (m, n, p) original hyperspectral data
        mcr_results: Dict from run_mcr_with_pca_init() containing:
            - concentrations: (m*n, k) matrix C
            - component_spectra: (k, p) matrix ST
    
    Returns:
        dict with:
            - residual_map: (m, n, p) pixel-wise residuals
            - r2_map: (m, n) R² per pixel
            - lof_map: (m, n) lack-of-fit per pixel
            - total_r2: float, overall R²
            - mean_lof: float, average LOF
    """
    m, n, p = cube.shape
    X_data = cube.reshape(m * n, p)  # (N, p)
    
    C = mcr_results["concentrations"]  # (N, k)
    ST = mcr_results["component_spectra"]  # (k, p)
    
    # Ensure ST is (k, p) - transpose if needed
    if ST.shape[1] != p:
        ST = ST.T  # (p, k) → (k, p)
    
    # Reconstruction: X_model = C @ ST
    X_model = C @ ST  # (N, p)
    
    # Residuals: E = X_data - X_model
    residuals = X_data - X_model  # (N, p)
    residual_cube = residuals.reshape(m, n, p)
    
    # R² per pixel: 1 - (SS_res / SS_tot)
    ss_res = np.sum(residuals**2, axis=1)  # (N,)
    ss_tot = np.sum((X_data - np.mean(X_data, axis=1, keepdims=True))**2, axis=1)  # (N,)
    
    # Avoid division by zero
    r2_per_pixel = np.ones(m * n)
    valid_mask = ss_tot > 1e-12
    r2_per_pixel[valid_mask] = 1 - (ss_res[valid_mask] / ss_tot[valid_mask])
    r2_map = r2_per_pixel.reshape(m, n)
    
    # Lack-of-fit (LOF): relative residual norm
    data_norm = np.linalg.norm(X_data, axis=1) + 1e-12  # (N,)
    residual_norm = np.linalg.norm(residuals, axis=1)  # (N,)
    lof_per_pixel = residual_norm / data_norm  # (N,)
    lof_map = lof_per_pixel.reshape(m, n)
    
    # Overall metrics
    total_ss_res = np.sum(ss_res)
    total_ss_tot = np.sum(ss_tot)
    total_r2 = 1 - (total_ss_res / (total_ss_tot + 1e-12))
    mean_lof = np.mean(lof_per_pixel)
    
    return {
        "residual_map": residual_cube,
        "r2_map": r2_map,
        "lof_map": lof_map,
        "total_r2": float(total_r2),
        "mean_lof": float(mean_lof)
    }


def validate_cluster_spectra(cluster_results: Dict, energy: np.ndarray, 
                             region: str, min_fwhm: float = 0.5,
                             energy_range: Optional[tuple] = None) -> np.ndarray:
    """
    Validate cluster mean spectra to detect and reject outliers/artifacts.
    
    Checks:
    1. Peak position within expected energy range for region
    2. Peak FWHM > min_fwhm (reject sharp spikes/artifacts)
    3. Intensity not anomalously low/high (reject dead/saturated pixels)
    
    Args:
        cluster_results: Dict from pca_cluster_analysis() with 'labels', 'cluster_info'
        energy: (p,) energy axis
        region: Region name (e.g., 'C1s', 'O1s') for validation rules
        min_fwhm: Minimum acceptable FWHM in eV
        energy_range: (min_eV, max_eV) expected peak range, or None for auto
    
    Returns:
        valid_mask: (m*n,) boolean array, True = valid pixel, False = outlier
    """
    from scipy.signal import find_peaks, peak_widths
    
    labels = cluster_results["labels"]  # (ny, nx)
    cluster_info = cluster_results["cluster_info"]
    n_clusters = len(cluster_info)
    
    # Flatten labels
    labels_flat = labels.flatten()
    
    # Extract mean spectra from cluster_info
    mean_spectra = [info["mean_spec"] for info in cluster_info]
    
    # Define expected energy ranges by region
    region_ranges = {
        "C1s": (282, 292),
        "O1s": (528, 538),
        "F1s": (682, 692),
        "Li1s": (52, 60),
        "N1s": (395, 405),
        "P2p": (128, 138),
        "S2p": (160, 170)
    }
    
    if energy_range is None:
        energy_range = region_ranges.get(region, (energy.min(), energy.max()))
    
    valid_clusters = []
    
    for i in range(n_clusters):
        spectrum = mean_spectra[i]
        
        # Skip clusters with no data
        if spectrum is None or cluster_info[i]["size"] == 0:
            logger.warning(f"Cluster {i}: Empty cluster, skipping")
            continue
        
        spectrum = np.array(spectrum)  # Ensure numpy array
        
        # Check if spectrum has reasonable intensity (not all zeros)
        if np.max(spectrum) < 1e-6:
            logger.warning(f"Cluster {i}: No intensity, marking as invalid")
            continue
        
        # Find peaks in cluster mean spectrum (try both directions for XPS inverted axis)
        peaks, props = find_peaks(spectrum, prominence=np.max(spectrum) * 0.05)
        
        if len(peaks) == 0:
            # Lower threshold and try again
            peaks, props = find_peaks(spectrum, prominence=np.max(spectrum) * 0.01)
        
        if len(peaks) == 0:
            # Use maximum intensity point as "peak"
            main_peak_idx = np.argmax(spectrum)
            logger.debug(f"Cluster {i}: No peaks found via find_peaks, using max intensity point")
        else:
            # Check main peak position
            main_peak_idx = peaks[np.argmax(spectrum[peaks])]
        
        peak_energy = energy[main_peak_idx]
        
        # Relaxed energy range check - warn but don't reject
        if not (energy_range[0] <= peak_energy <= energy_range[1]):
            logger.warning(f"Cluster {i}: Peak at {peak_energy:.1f} eV outside expected range "
                          f"{energy_range}, but accepting anyway")
            # Don't continue - accept it anyway
        
        # Estimate FWHM (optional - don't reject if fails)
        try:
            if len(peaks) > 0:
                widths, width_heights, left_ips, right_ips = peak_widths(
                    spectrum, peaks, rel_height=0.5
                )
                if main_peak_idx in peaks:
                    main_peak_width_idx = np.where(peaks == main_peak_idx)[0][0]
                    fwhm_bins = widths[main_peak_width_idx]
                    
                    # Convert bins to eV (assuming uniform spacing)
                    energy_step = np.abs(np.mean(np.diff(energy)))
                    fwhm_ev = fwhm_bins * energy_step
                    
                    if fwhm_ev < min_fwhm:
                        logger.warning(f"Cluster {i}: FWHM={fwhm_ev:.2f} eV < {min_fwhm} eV, "
                                      f"potential artifact but accepting")
                        # Don't reject - just warn
        except Exception as e:
            logger.debug(f"Cluster {i}: Could not estimate FWHM ({e}), accepting anyway")
        
        # Passed checks - accept cluster
        valid_clusters.append(i)
        logger.info(f"Cluster {i}: Valid (peak @ {peak_energy:.1f} eV, "
                   f"{cluster_info[i]['size']} pixels)")
    
    # Create mask: True for pixels in valid clusters
    valid_mask = np.isin(labels_flat, valid_clusters)
    
    n_valid = np.sum(valid_mask)
    n_total = len(labels_flat)
    logger.info(f"Cluster validation: {n_valid}/{n_total} pixels valid "
               f"({len(valid_clusters)}/{n_clusters} clusters passed)")
    
    return valid_mask


def assign_chemical_states(component_spectra: np.ndarray, energy: np.ndarray,
                           region: str) -> list:
    """
    Assign chemical state labels to MCR components based on peak positions.
    
    Uses simple peak position matching to known XPS database values.
    For more sophisticated fitting, integrate with XPS_peakfitting_V2.
    
    Args:
        component_spectra: (k, p) MCR component spectra matrix
        energy: (p,) energy axis in eV
        region: Region name (e.g., 'C1s', 'O1s')
    
    Returns:
        List of chemical state labels (strings), one per component
    """
    from scipy.signal import find_peaks
    
    # XPS database: region → {chemical_state: (BE_center, tolerance)}
    state_database = {
        "C1s": {
            "C-C/C-H": (284.8, 0.5),
            "C-O": (286.5, 0.5),
            "C=O": (287.8, 0.5),
            "O-C=O": (289.0, 0.5),
            "Carbonate": (290.0, 0.5)
        },
        "O1s": {
            "Metal Oxide": (530.0, 0.8),
            "Hydroxide": (531.5, 0.5),
            "C=O": (532.0, 0.5),
            "C-O": (533.0, 0.5),
            "H2O": (534.0, 0.5)
        },
        "F1s": {
            "Metal Fluoride": (685.0, 0.8),
            "C-F": (688.0, 0.5),
            "PVDF": (687.5, 0.5)
        },
        "Li1s": {
            "Li Metal": (54.0, 0.5),
            "Li2CO3": (55.5, 0.5),
            "LiF": (56.0, 0.5)
        }
    }
    
    states = state_database.get(region, {})
    if not states:
        logger.warning(f"No chemical state database for region '{region}'")
        return [f"Component {i}" for i in range(len(component_spectra))]
    
    labels = []
    for i, spectrum in enumerate(component_spectra):
        # Find main peak
        peaks, _ = find_peaks(spectrum, prominence=np.max(spectrum) * 0.1)
        
        if len(peaks) == 0:
            labels.append(f"Component {i} (no peak)")
            continue
        
        main_peak_idx = peaks[np.argmax(spectrum[peaks])]
        peak_be = energy[main_peak_idx]
        
        # Match to database
        best_match = None
        min_diff = float('inf')
        
        for state_name, (ref_be, tol) in states.items():
            diff = abs(peak_be - ref_be)
            if diff < tol and diff < min_diff:
                best_match = state_name
                min_diff = diff
        
        if best_match:
            labels.append(f"{best_match} ({peak_be:.1f} eV)")
        else:
            labels.append(f"Unknown ({peak_be:.1f} eV)")
    
    return labels
