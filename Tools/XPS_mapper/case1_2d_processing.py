"""
Case 1: 2D Single-Energy Map Processing Module

Functions for processing 2D single-energy XPS maps including:
- Net/ratio computation
- Denoising
- Threshold segmentation
- Morphological cleanup
- ROI statistics
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
from skimage.filters import threshold_otsu


def compute_net_and_ratio(on_map, off_map: Optional[np.ndarray] = None, alpha: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Compute net and ratio maps from on-peak and off-peak measurements.
    
    Args:
        on_map: Map2D object containing on-peak data
        off_map: Optional Map2D object containing off-peak data
        alpha: Scaling factor for off-peak subtraction
        
    Returns:
        Dictionary with 'net' and 'ratio' arrays
    """
    eps = 1e-9
    if off_map is None:
        return {"net": on_map.data.copy(), "ratio": None}
    if on_map.shape != off_map.shape:
        raise ValueError("on_map and off_map must have the same shape")
    net = on_map.data - alpha * off_map.data
    ratio = net / (off_map.data + eps)
    return {"net": net, "ratio": ratio}


def denoise_map(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian denoising to a 2D map.
    
    Args:
        img: 2D array to denoise
        sigma: Gaussian kernel standard deviation
        
    Returns:
        Denoised 2D array
    """
    return gaussian_filter(img, sigma=sigma)


def threshold_segment(img: np.ndarray, method: str = "otsu", percentile: float = 95.0) -> Tuple[np.ndarray, float]:
    """
    Apply thresholding to segment the image.
    
    Args:
        img: 2D array to threshold
        method: 'otsu' or 'percentile'
        percentile: Percentile value if method='percentile'
        
    Returns:
        Tuple of (binary mask, threshold value)
    """
    if method == "otsu":
        thr = threshold_otsu(img)
    else:  # percentile-based thresholding
        thr = np.percentile(img, percentile)
    
    mask = img >= thr
    return mask.astype(bool), float(thr)


def morph_cleanup(mask: np.ndarray, op: str = "open", size: int = 2) -> np.ndarray:
    """
    Apply morphological operations to clean up binary mask.
    
    Args:
        mask: Binary mask array
        op: 'open' or 'close'
        size: Size of structural element
        
    Returns:
        Cleaned binary mask
    """
    structure = np.ones((size, size), dtype=bool)
    if op == "open":
        return binary_opening(mask, structure=structure)
    elif op == "close":
        return binary_closing(mask, structure=structure)
    return mask


def roi_stats(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for region of interest defined by mask.
    
    Args:
        img: 2D intensity array
        mask: Binary mask defining ROI
        
    Returns:
        Dictionary of statistics (area, mean, median, std, sum, min, max)
    """
    vals = img[mask.astype(bool)]
    if vals.size == 0:
        return {
            "area": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sum": 0.0,
            "min": np.nan,
            "max": np.nan
        }
    return {
        "area": float(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "sum": float(np.sum(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def pearson_correlation(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """
    Compute Pearson correlation between two maps.
    
    Args:
        map_a: First 2D array
        map_b: Second 2D array
        
    Returns:
        Correlation coefficient
    """
    a = map_a.flatten()
    b = map_b.flatten()
    if a.size == 0 or b.size == 0:
        return np.nan
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    cov = np.mean((a - mean_a) * (b - mean_b))
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a == 0 or std_b == 0:
        return np.nan
    return cov / (std_a * std_b)


def process_2d_map(
    on_map,
    off_map: Optional[np.ndarray] = None,
    sigma: float = 1.0,
    threshold_method: str = "otsu",
    percentile: float = 95.0,
    morph_op: str = "open",
    morph_size: int = 2,
) -> Dict:
    """
    Complete Case 1 processing pipeline for 2D single-energy maps.
    
    Args:
        on_map: Map2D object with on-peak data
        off_map: Optional Map2D object with off-peak data
        sigma: Gaussian denoising parameter
        threshold_method: 'otsu' or 'percentile'
        percentile: Percentile for thresholding
        morph_op: 'open' or 'close'
        morph_size: Morphological structuring element size
        
    Returns:
        Dictionary containing processed results
    """
    # Step 1: Compute net and ratio
    norm = compute_net_and_ratio(on_map, off_map)
    base_img = norm["net"] if norm["ratio"] is not None else on_map.data
    
    # Step 2: Denoise
    denoised = denoise_map(base_img, sigma=sigma)
    
    # Step 3: Threshold segmentation
    mask, thr = threshold_segment(denoised, method=threshold_method, percentile=percentile)
    
    # Step 4: Morphological cleanup
    mask_clean = morph_cleanup(mask, op=morph_op, size=morph_size)
    
    # Step 5: ROI statistics
    stats = roi_stats(denoised, mask_clean)
    
    return {
        "net_map": norm["net"],
        "ratio_map": norm["ratio"],
        "denoised": denoised,
        "threshold": thr,
        "mask": mask_clean,
        "roi_stats": stats,
    }
