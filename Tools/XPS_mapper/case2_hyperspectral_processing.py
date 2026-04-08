"""
Case 2: Hyperspectral Map Processing Module

Functions for processing 3D hyperspectral XPS maps including:
- Baseline correction (ALS)
- Peak fitting with Gaussian models
- Energy shift estimation
- PCA analysis and clustering
- NMF decomposition
- NNLS projection
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
from scipy.optimize import curve_fit, nnls
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.
    
    Args:
        y: 1D spectrum
        lam: Smoothness parameter (larger = smoother)
        p: Asymmetry parameter (0 < p < 1, smaller = more asymmetric)
        niter: Number of iterations
        
    Returns:
        Baseline array
    """
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = W + lam * D.T @ D
        try:
            b = np.linalg.solve(Z, W @ y)
        except np.linalg.LinAlgError:
            b = y.copy()
        w = p * (y > b) + (1 - p) * (y < b)
    return b


def estimate_energy_shift(E: np.ndarray, s: np.ndarray, ref_s: np.ndarray, max_shift: int = 10) -> float:
    """
    Estimate energy shift using cross-correlation.
    
    Args:
        E: Energy axis
        s: Spectrum to align
        ref_s: Reference spectrum
        max_shift: Maximum allowed shift in bins
        
    Returns:
        Energy shift in eV
    """
    s0 = (s - np.mean(s)) / (np.std(s) + 1e-12)
    r0 = (ref_s - np.mean(ref_s)) / (np.std(ref_s) + 1e-12)
    corr = np.correlate(s0, r0, mode="full")
    lag = np.argmax(corr) - (len(s0) - 1)
    lag = int(np.clip(lag, -max_shift, max_shift))
    dE = np.mean(np.diff(E)) if len(E) > 1 else 0.0
    return float(lag * dE)


def gaussian_model(E: np.ndarray, *params: float) -> np.ndarray:
    """
    Sum of Gaussian peaks with shared energy shift.
    
    Args:
        E: Energy axis
        *params: (amp1, ctr1, sig1, amp2, ctr2, sig2, ..., shift)
        
    Returns:
        Modeled spectrum
    """
    shift = params[-1]
    out = np.zeros_like(E)
    for i in range(0, len(params) - 1, 3):
        amp, ctr, sig = params[i:i+3]
        out += amp * np.exp(-0.5 * ((E - (ctr + shift)) / (sig + 1e-12))**2)
    return out


def fit_average_spectrum(
    E: np.ndarray,
    S_avg: np.ndarray,
    init_peaks: List[Tuple[float, float]],
    bounds_shift: Tuple[float, float] = (-0.5, 0.5)
) -> Optional[np.ndarray]:
    """
    Fit average spectrum with Gaussian model.
    
    Args:
        E: Energy axis
        S_avg: Average spectrum
        init_peaks: List of (center, sigma) initial values
        bounds_shift: Bounds for energy shift
        
    Returns:
        Fitted parameters or None if failed
    """
    params0 = []
    for ctr, sig in init_peaks:
        idx = np.argmin(np.abs(E - ctr))
        amp0 = max(1e-3, float(S_avg[idx]))
        params0.extend([amp0, ctr, max(sig, 1e-3)])
    params0.append(0.0)
    
    lower = []
    upper = []
    for ctr, sig in init_peaks:
        lower.extend([0.0, ctr - 0.5, 0.05])
        upper.extend([np.inf, ctr + 0.5, 3.0])
    lower.append(bounds_shift[0])
    upper.append(bounds_shift[1])
    
    try:
        popt, _ = curve_fit(gaussian_model, E, S_avg, p0=params0, bounds=(lower, upper), maxfev=5000)
        return popt
    except Exception:
        return None


def fit_pixel_spectrum(
    E: np.ndarray,
    S: np.ndarray,
    params_avg: np.ndarray,
    fix_centers_sigmas: bool = True
) -> Optional[Dict[str, float]]:
    """
    Fit single pixel spectrum with fixed centers and sigmas.
    
    Args:
        E: Energy axis
        S: Pixel spectrum
        params_avg: Parameters from average spectrum fit
        fix_centers_sigmas: Whether to fix peak positions and widths
        
    Returns:
        Dictionary with fitted areas, shift, and MSE
    """
    n = (len(params_avg) - 1) // 3
    centers_sigmas = []
    for i in range(n):
        centers_sigmas.extend(params_avg[3*i + 1:3*i + 3])

    def model_with_free_amps(E, *theta):
        shift = theta[-1]
        out = np.zeros_like(E)
        for i in range(n):
            amp = theta[i]
            ctr = centers_sigmas[2*i]
            sig = centers_sigmas[2*i + 1]
            out += amp * np.exp(-0.5 * ((E - (ctr + shift)) / (sig + 1e-12))**2)
        return out

    amps0 = [params_avg[3*i] for i in range(n)]
    p0 = amps0 + [0.0]
    lower = [0.0] * n + [-0.5]
    upper = [np.inf] * n + [0.5]
    
    try:
        popt, _ = curve_fit(model_with_free_amps, E, S, p0=p0, bounds=(lower, upper), maxfev=4000)
        pred = model_with_free_amps(E, *popt)
        mse = float(np.mean((S - pred)**2))
        out = {f"area_{i+1}": float(popt[i]) for i in range(n)}
        out["shift"] = float(popt[-1])
        out["mse"] = mse
        return out
    except Exception:
        return None


def process_hyperspectral(
    hmap,
    init_peaks: List[Tuple[float, float]],
    background_lam: float = 1e5,
    background_p: float = 0.01,
    niter_bg: int = 10,
    max_shift_bins: int = 10,
    do_pca: bool = True,
    n_pca: int = 3,
    do_nmf: bool = True,
    n_nmf: int = 3
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Complete Case 2 processing pipeline for hyperspectral maps.
    
    Args:
        hmap: HyperspectralMap object
        init_peaks: Initial peak positions and widths
        background_lam: ALS smoothness parameter
        background_p: ALS asymmetry parameter
        niter_bg: ALS iterations
        max_shift_bins: Maximum energy shift in bins
        do_pca: Whether to perform PCA
        n_pca: Number of PCA components
        do_nmf: Whether to perform NMF
        n_nmf: Number of NMF components
        
    Returns:
        Dictionary containing all analysis results
    """
    E = hmap.energy
    ny, nx, nE = hmap.shape
    
    # Fit average spectrum
    S_avg = np.mean(hmap.cube.reshape(ny*nx, nE), axis=0)
    params_avg = fit_average_spectrum(E, S_avg, init_peaks)
    if params_avg is None:
        raise RuntimeError("Average spectrum fitting failed or SciPy not available")

    # Initialize result arrays
    area_maps = {f"area_{i+1}": np.zeros((ny, nx), dtype=float) for i in range((len(params_avg)-1)//3)}
    shift_map = np.zeros((ny, nx), dtype=float)
    mse_map = np.zeros((ny, nx), dtype=float)

    # Baseline for average spectrum
    b_avg = baseline_als(S_avg, lam=background_lam, p=background_p, niter=niter_bg)
    S_avg_corr = S_avg - b_avg

    # Process each pixel
    for y in range(ny):
        for x in range(nx):
            s = hmap.cube[y, x, :].copy()
            b = baseline_als(s, lam=background_lam, p=background_p, niter=niter_bg)
            s_corr = s - b
            fit_res = fit_pixel_spectrum(E, s_corr, params_avg, fix_centers_sigmas=True)
            if fit_res is None:
                continue
            for k, v in fit_res.items():
                if k.startswith("area_"):
                    area_maps[k][y, x] = v
            shift_map[y, x] = fit_res["shift"]
            mse_map[y, x] = fit_res["mse"]

    results: Dict[str, Union[np.ndarray, Dict]] = {
        "area_maps": area_maps,
        "shift_map": shift_map,
        "mse_map": mse_map,
        "avg_params": params_avg,
        "avg_spectrum": S_avg,
        "avg_spectrum_baseline": b_avg,
        "avg_spectrum_corrected": S_avg_corr,
    }

    # PCA analysis
    if do_pca:
        X = hmap.cube.reshape(ny*nx, nE)
        pca = PCA(n_components=n_pca, whiten=False, random_state=0)
        scores = pca.fit_transform(X)
        components = pca.components_
        score_maps = scores.reshape(ny, nx, n_pca)
        results["pca"] = {
            "components": components,
            "score_maps": score_maps,
            "explained_variance": pca.explained_variance_ratio_,
        }

    # NMF analysis
    if do_nmf:
        X = hmap.cube.reshape(ny*nx, nE)
        X = np.clip(X, 0, None)
        nmf = NMF(n_components=n_nmf, init="nndsvd", random_state=0, max_iter=500)
        W = nmf.fit_transform(X)
        H = nmf.components_
        abundance_maps = W.reshape(ny, nx, n_nmf)
        results["nmf"] = {
            "components": H,
            "abundance_maps": abundance_maps,
            "reconstruction_error": nmf.reconstruction_err_,
        }

    return results


def nnls_project_pixel(
    E: np.ndarray,
    S: np.ndarray,
    basis: Dict[str, np.ndarray],
    fit_range: Tuple[float, float],
    subtract_baseline: bool = False,
    baseline_method: str = "als"
) -> Dict[str, float]:
    """
    Project a single pixel spectrum onto basis via NNLS.
    
    Args:
        E: Energy axis
        S: Pixel spectrum
        basis: Dictionary of basis spectra
        fit_range: (E_min, E_max) for fitting
        subtract_baseline: Whether to subtract baseline first
        baseline_method: 'als' or 'shirley'
        
    Returns:
        Dictionary of component areas
    """
    mask = (E >= fit_range[0]) & (E <= fit_range[1])
    y = S.copy()
    
    if subtract_baseline:
        if baseline_method == "shirley":
            try:
                import sys
                from pathlib import Path
                xps_utils_path = Path(__file__).parent.parent / "XPS_utils"
                if str(xps_utils_path) not in sys.path:
                    sys.path.insert(0, str(xps_utils_path))
                from background_correction import shirley_background
                yb = shirley_background(E, y)
            except (ImportError, Exception):
                yb = baseline_als(y, lam=1e5, p=0.01, niter=10)
        else:
            yb = baseline_als(y, lam=1e5, p=0.01, niter=10)
        y = y - yb

    # Build basis matrix
    names = list(basis.keys())
    B = np.column_stack([basis[n][mask] for n in names])
    yfit = y[mask]

    coeffs, _ = nnls(B, yfit)

    return {name: float(coeffs[i]) for i, name in enumerate(names)}


def pca_cluster_analysis(
    hmap,
    n_pca: int = 3,
    n_clusters: int = 4,
    use_minibatch: bool = True,
    normalize: str = "l2"
) -> Dict:
    """
    Perform PCA clustering analysis on hyperspectral map.
    
    Args:
        hmap: HyperspectralMap object
        n_pca: Number of PCA components
        n_clusters: Number of clusters
        use_minibatch: Use MiniBatchKMeans for speed
        normalize: Normalization method ('l2', 'mean', or None)
        
    Returns:
        Dictionary with PCA results, cluster labels, and cluster info
    """
    ny, nx, nE = hmap.shape
    X = hmap.cube.reshape(ny*nx, nE).astype(float)

    # Normalize spectra
    if normalize == "l2":
        norms = np.linalg.norm(X, axis=1) + 1e-12
        X = X / norms[:, None]
    elif normalize == "mean":
        means = np.mean(X, axis=1) + 1e-12
        X = X / means[:, None]

    # PCA
    pca = PCA(n_components=n_pca, whiten=False, random_state=0)
    scores = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    score_maps = scores.reshape(ny, nx, n_pca)

    # Clustering
    if use_minibatch:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=2048)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels_1d = km.fit_predict(scores)
    labels = labels_1d.reshape(ny, nx)

    # Cluster representatives
    cluster_info = []
    for k in range(n_clusters):
        idx = np.where(labels_1d == k)[0]
        if idx.size == 0:
            cluster_info.append({
                "cluster": k, "size": 0, "mean_spec": None, "medoid_index": None,
                "centroid_scores": None
            })
            continue
        mean_spec = np.mean(X[idx, :], axis=0)
        centroid = np.mean(scores[idx, :], axis=0)
        dists = np.linalg.norm(scores[idx, :] - centroid[None, :], axis=1)
        medoid_local = np.argmin(dists)
        medoid_index = int(idx[medoid_local])
        cluster_info.append({
            "cluster": k,
            "size": int(idx.size),
            "mean_spec": mean_spec,
            "medoid_index": medoid_index,
            "centroid_scores": centroid
        })

    return {
        "pca": {
            "components": pca.components_,
            "explained_variance": explained,
            "score_maps": score_maps
        },
        "labels": labels,
        "cluster_info": cluster_info
    }
