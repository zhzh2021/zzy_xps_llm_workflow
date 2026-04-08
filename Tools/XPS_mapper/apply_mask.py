"""
Spatial masking module for XPS hyperspectral map analysis.

Implements three masking methods to improve data quality:
1. Intensity masking - Remove dead pixels below threshold
2. PCA-score masking - Isolate pixels with coherent spectral features  
3. Cluster masking - Focus on specific chemical phases

Reference: CLUSTER_MASKING_GUIDE.md
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class MaskingResults:
    """Container for masking results with metadata."""
    
    def __init__(self, mask: np.ndarray, method: str, n_masked: int, n_kept: int, **kwargs):
        self.mask = mask  # Boolean mask (True = keep)
        self.method = method
        self.n_masked = n_masked
        self.n_kept = n_kept
        self.metadata = kwargs
    
    def apply_to_cube(self, cube: np.ndarray) -> np.ndarray:
        """Apply mask to hyperspectral cube by zeroing masked pixels."""
        ny, nx = cube.shape[:2]
        cube_flat = cube.reshape(ny*nx, -1)
        cube_flat[~self.mask] = 0
        return cube_flat.reshape(cube.shape)
    
    def __repr__(self):
        return (f"MaskingResults(method='{self.method}', "
                f"kept={self.n_kept}, masked={self.n_masked}, "
                f"fraction={self.n_kept/(self.n_kept+self.n_masked):.2%})")


def apply_intensity_mask(
    cube: np.ndarray,
    threshold: float,
    method: str = 'total'
) -> MaskingResults:
    """
    Apply intensity-based masking to remove dead pixels.
    
    This is the baseline masking method that should ALWAYS be applied first.
    Removes pixels where total/mean intensity is below threshold.
    
    Args:
        cube: Hyperspectral cube (ny, nx, n_energy)
        threshold: Minimum intensity threshold
        method: 'total' (sum across energy) or 'mean' (mean across energy)
        
    Returns:
        MaskingResults object with boolean mask
    """
    ny, nx, nE = cube.shape
    
    if method == 'total':
        intensity_map = np.sum(cube, axis=2)
    elif method == 'mean':
        intensity_map = np.mean(cube, axis=2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'total' or 'mean'")
    
    mask = (intensity_map > threshold).flatten()
    n_kept = np.sum(mask)
    n_masked = len(mask) - n_kept
    
    logger.info(f"Intensity masking: threshold={threshold} ({method})")
    logger.info(f"  Kept: {n_kept}/{len(mask)} pixels ({n_kept/len(mask)*100:.1f}%)")
    logger.info(f"  Masked: {n_masked} pixels")
    
    return MaskingResults(
        mask=mask,
        method='intensity',
        n_masked=n_masked,
        n_kept=n_kept,
        threshold=threshold,
        intensity_method=method,
        intensity_map=intensity_map
    )


def apply_pca_score_mask(
    cube: np.ndarray,
    energy: np.ndarray,
    threshold: float = 0.5,
    pc_index: int = 0,
    normalize: str = 'l2'
) -> MaskingResults:
    """
    Apply PCA-score masking to isolate pixels with coherent spectral features.
    
    FIRST CHOICE FOR NOISY DATA. Uses PC1 (or specified PC) score map to identify 
    pixels that correlate with the main spectral peak shape. PCA is "noise-blind" 
    and extracts the peak direction from variance. Masking low scores removes pixels 
    lacking coherent spectral features.
    
    When to use:
    - MCR fails to converge (0 iterations)
    - PC1 variance explained < 20%
    - High noise-to-signal ratio visible in maps
    
    Args:
        cube: Hyperspectral cube (ny, nx, n_energy)
        energy: Energy axis
        threshold: Minimum PC score to keep (relative to max, 0-1 scale)
        pc_index: Which PC to use (0=PC1, 1=PC2, etc.)
        normalize: Normalization method ('l2', 'l1', or None)
        
    Returns:
        MaskingResults object with boolean mask and PCA info
    """
    ny, nx, nE = cube.shape
    X = cube.reshape(ny*nx, nE).astype(float)
    X_original = X.copy()
    
    # Normalize spectra for PCA stability
    if normalize == 'l2':
        norms = np.linalg.norm(X, axis=1) + 1e-12
        X = X / norms[:, None]
    elif normalize == 'l1':
        sums = np.sum(X, axis=1) + 1e-12
        X = X / sums[:, None]
    
    # Run PCA
    n_components = min(3, nE, ny*nx)
    pca = PCA(n_components=n_components, whiten=False, random_state=0)
    scores = pca.fit_transform(X)
    
    # Extract specified PC scores
    if pc_index >= n_components:
        logger.warning(f"PC{pc_index+1} requested but only {n_components} components available. Using PC1.")
        pc_index = 0
    
    pc_scores = scores[:, pc_index]
    score_map = pc_scores.reshape(ny, nx)
    
    # Normalize scores to 0-1 range for threshold comparison
    score_min = pc_scores.min()
    score_max = pc_scores.max()
    if score_max > score_min:
        normalized_scores = (pc_scores - score_min) / (score_max - score_min)
    else:
        normalized_scores = np.ones_like(pc_scores)
    
    # Apply threshold
    mask = normalized_scores > threshold
    n_kept = np.sum(mask)
    n_masked = len(mask) - n_kept
    
    logger.info(f"PCA-score masking: PC{pc_index+1}, threshold={threshold}")
    logger.info(f"  PC{pc_index+1} variance explained: {pca.explained_variance_ratio_[pc_index]*100:.1f}%")
    logger.info(f"  Score range: [{score_min:.3f}, {score_max:.3f}]")
    logger.info(f"  Kept: {n_kept}/{len(mask)} pixels ({n_kept/len(mask)*100:.1f}%)")
    logger.info(f"  Masked: {n_masked} pixels")
    
    return MaskingResults(
        mask=mask,
        method='pca_score',
        n_masked=n_masked,
        n_kept=n_kept,
        threshold=threshold,
        pc_index=pc_index,
        score_map=score_map,
        pca_components=pca.components_,
        pca_variance=pca.explained_variance_ratio_,
        score_range=(score_min, score_max)
    )


def apply_cluster_mask(
    cube: np.ndarray,
    energy: np.ndarray,
    cluster_labels: np.ndarray,
    focus_clusters: List[int],
    n_clusters: int,
    silhouette: Optional[float] = None
) -> MaskingResults:
    """
    Apply cluster-based masking to focus on specific chemical phases.
    
    ADVANCED - USE WITH CAUTION. Isolates pixels belonging to specific chemical 
    states identified by K-means clustering.
    
    When to use:
    - Silhouette score > 0.3 (confirms clusters are meaningful)
    - Clear spatial separation of chemical phases
    - After noise removal via intensity/PCA masking
    
    When NOT to use:
    - Silhouette score < 0.2 (clusters are arbitrary, not chemical)
    - Small datasets (<100 pixels)
    - Uncertain which clusters are signal vs. noise
    
    Args:
        cube: Hyperspectral cube (ny, nx, n_energy)
        energy: Energy axis
        cluster_labels: Cluster labels (ny*nx,) or (ny, nx)
        focus_clusters: List of cluster IDs to keep (e.g., [2] or [2, 3])
        n_clusters: Total number of clusters
        silhouette: Silhouette score (optional, for validation)
        
    Returns:
        MaskingResults object with boolean mask
    """
    ny, nx, nE = cube.shape
    labels_flat = cluster_labels.flatten()
    
    # Validate cluster IDs
    valid_ids = []
    for cluster_id in focus_clusters:
        if 0 <= cluster_id < n_clusters:
            valid_ids.append(cluster_id)
        else:
            logger.warning(f"Cluster ID {cluster_id} out of range (0-{n_clusters-1}), skipping")
    
    if not valid_ids:
        logger.error("No valid cluster IDs provided!")
        # Return all-True mask as fallback
        return MaskingResults(
            mask=np.ones(ny*nx, dtype=bool),
            method='cluster',
            n_masked=0,
            n_kept=ny*nx,
            focus_clusters=[],
            warning="No valid clusters"
        )
    
    # Check Silhouette score if provided
    if silhouette is not None:
        if silhouette < 0.2:
            logger.warning(f"LOW Silhouette score ({silhouette:.3f}) - clusters may not be chemically meaningful!")
            logger.warning("Consider using PCA-score masking instead of cluster masking.")
        elif silhouette > 0.3:
            logger.info(f"Good Silhouette score ({silhouette:.3f}) - clusters appear meaningful")
    
    # Create spatial mask
    mask = np.zeros(ny*nx, dtype=bool)
    for cluster_id in valid_ids:
        mask |= (labels_flat == cluster_id)
    
    n_kept = np.sum(mask)
    n_masked = len(mask) - n_kept
    
    logger.info(f"Cluster masking: keeping clusters {valid_ids}")
    if silhouette is not None:
        logger.info(f"  Silhouette score: {silhouette:.3f}")
    logger.info(f"  Kept: {n_kept}/{len(mask)} pixels ({n_kept/len(mask)*100:.1f}%)")
    logger.info(f"  Masked: {n_masked} pixels")
    
    # Log pixel counts per cluster
    for cluster_id in range(n_clusters):
        count = np.sum(labels_flat == cluster_id)
        status = "KEPT" if cluster_id in valid_ids else "masked"
        logger.info(f"    Cluster {cluster_id}: {count} pixels ({status})")
    
    return MaskingResults(
        mask=mask,
        method='cluster',
        n_masked=n_masked,
        n_kept=n_kept,
        focus_clusters=valid_ids,
        n_clusters=n_clusters,
        silhouette=silhouette,
        cluster_labels=labels_flat
    )


def validate_and_mask_clusters(
    cube: np.ndarray,
    energy: np.ndarray,
    cluster_results: Dict,
    region: str = "Unknown",
    min_fwhm: float = 0.5,
    silhouette_threshold: float = 0.2
) -> Tuple[np.ndarray, Dict]:
    """
    Validate cluster quality and create mask for outliers/artifacts.
    
    Checks cluster spectra for:
    - Sufficient spectral width (FWHM)
    - Non-negative features
    - Reasonable peak positions
    - Cluster meaningfulness (Silhouette score)
    
    Args:
        cube: Hyperspectral cube (ny, nx, n_energy)
        energy: Energy axis
        cluster_results: Dict with 'labels' and 'cluster_info'
        region: XPS region name for validation
        min_fwhm: Minimum FWHM (eV) for valid clusters
        silhouette_threshold: Minimum Silhouette score for meaningful clusters
        
    Returns:
        valid_mask: Boolean mask (True = valid pixels)
        validation_info: Dict with validation metrics
    """
    from chemometrics_utils import validate_cluster_spectra
    
    ny, nx = cube.shape[:2]
    labels_flat = cluster_results['labels'].flatten()
    
    # Compute Silhouette score
    X = cube.reshape(ny*nx, -1)
    try:
        sil_score = silhouette_score(
            X, labels_flat, 
            metric='euclidean', 
            sample_size=min(5000, len(labels_flat))
        )
    except Exception as e:
        logger.warning(f"Could not compute Silhouette score: {e}")
        sil_score = None
    
    # Validate cluster spectra (check FWHM, peak positions, etc.)
    valid_mask = validate_cluster_spectra(
        cluster_results,
        energy,
        region=region,
        min_fwhm=min_fwhm
    )
    
    n_invalid = np.sum(~valid_mask)
    n_valid = np.sum(valid_mask)
    
    validation_info = {
        'silhouette_score': sil_score,
        'n_valid': n_valid,
        'n_invalid': n_invalid,
        'valid_fraction': n_valid / len(valid_mask),
        'min_fwhm': min_fwhm,
        'clusters_meaningful': sil_score > silhouette_threshold if sil_score is not None else None
    }
    
    logger.info(f"Cluster validation:")
    if sil_score is not None:
        status = 'meaningful' if sil_score > silhouette_threshold else 'WARNING: low'
        logger.info(f"  Silhouette score: {sil_score:.3f} ({status})")
    logger.info(f"  Valid pixels: {n_valid}/{len(valid_mask)} ({n_valid/len(valid_mask)*100:.1f}%)")
    if n_invalid > 0:
        logger.info(f"  Invalid pixels (outliers/artifacts): {n_invalid}")
    
    return valid_mask, validation_info


def save_mask_visualization(
    mask: np.ndarray,
    shape: Tuple[int, int],
    output_dir: Path,
    base_name: str,
    method: str,
    show: bool = False
):
    """
    Save mask visualization as image.
    
    Args:
        mask: Boolean mask (flattened or 2D)
        shape: (ny, nx) shape for reshaping
        output_dir: Directory to save image
        base_name: Base filename
        method: Masking method name for title
        show: Whether to display plot
    """
    mask_img = mask.reshape(shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mask_img, cmap='gray', interpolation='nearest')
    ax.set_title(f'{method.upper()} Mask: {base_name}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im, ax=ax, label='Kept (1) / Masked (0)')
    
    mask_path = output_dir / f'{base_name}_{method}_mask.png'
    plt.savefig(mask_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"Saved {method} mask visualization: {mask_path.name}")


def combine_masks(*masks: np.ndarray, operation: str = 'and') -> np.ndarray:
    """
    Combine multiple masks with logical operations.
    
    Useful for applying multiple masking criteria sequentially:
    - Intensity + PCA-score (AND operation)
    - Multiple cluster selections (OR operation)
    
    Args:
        *masks: Variable number of boolean masks (same length)
        operation: 'and' (intersection) or 'or' (union)
        
    Returns:
        Combined boolean mask
    """
    if not masks:
        raise ValueError("At least one mask required")
    
    combined = masks[0].copy()
    for mask in masks[1:]:
        if len(mask) != len(combined):
            raise ValueError(f"All masks must have same length ({len(combined)} vs {len(mask)})")
        
        if operation == 'and':
            combined &= mask
        elif operation == 'or':
            combined |= mask
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'and' or 'or'")
    
    n_kept = np.sum(combined)
    logger.info(f"Combined {len(masks)} masks with '{operation}': {n_kept}/{len(combined)} pixels kept")
    
    return combined


def get_mask_from_config(
    config: Dict,
    region_name: str,
    mask_type: str
) -> Optional[float]:
    """
    Read masking parameters from config.
    
    Priority: Element-specific > Global
    
    Args:
        config: Configuration dictionary
        region_name: XPS region name (e.g., 'F1s')
        mask_type: Type of mask ('intensity_mask_threshold', 'pca_mask_threshold', 'focus_clusters')
        
    Returns:
        Masking parameter value or None
    """
    if not config:
        return None
    
    # Try element-specific first
    if region_name and 'regions' in config:
        region_def = config['regions'].get(region_name)
        if region_def and mask_type in region_def:
            value = region_def[mask_type]
            logger.debug(f"Using element-specific {mask_type} for {region_name}: {value}")
            return value
    
    # Fall back to global
    if 'global_processing' in config and mask_type in config['global_processing']:
        value = config['global_processing'][mask_type]
        logger.debug(f"Using global {mask_type}: {value}")
        return value
    
    # Direct top-level (legacy support)
    if mask_type in config:
        value = config[mask_type]
        logger.debug(f"Using top-level {mask_type}: {value}")
        return value
    
    return None
