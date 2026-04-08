"""
XAS Feature Extractor Module

Extracts comprehensive features from normalized XAS spectra for ML analysis.
Configuration-driven, agent-friendly, modular design.

Features extracted (18+ total):
- Edge features: e0, edge_step, edge_slope, pre_edge_area
- XANES features: white_line_intensity, white_line_energy, white_line_fwhm, 
                  xanes_area, xanes_centroid
- EXAFS features: chi_k_rms, ft_peak_r, ft_peak_amp
- Derivative features: first_derivative_max, second_derivative_zero
- Statistical features: spectral_mean, spectral_variance, spectral_skewness, spectral_kurtosis
- Plotting capabilities: feature comparison plots, VAE embedding plots
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats, signal, integrate
import sys

# Ensure APS_XAS tools directory is on sys.path for direct script execution
_current_dir = Path(__file__).resolve().parent
_tools_dir = _current_dir.parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

# XAS-specific imports
try:
    import larch
    from larch.xafs import pre_edge, autobk, xftf
    HAS_LARCH = True
except ImportError:
    HAS_LARCH = False
    print("Warning: larch not available. Feature extraction will be limited.")

# Local imports - XAS models from analyzer
try:
    from .xas_models import XASFeatures, XASSampleResult
except ImportError:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'xas_analyzer'))
        from xas_models import XASFeatures, XASSampleResult
    except ImportError:
        print("Warning: xas_models not available. Feature extraction will be limited.")

try:
    from xas_ml_modules.config_utils import ConfigLoader
except ImportError:
    # Config loader is optional for feature extraction
    ConfigLoader = None


logger = logging.getLogger(__name__)

# Optional feature plotting
try:
    from xas_plotter.xas_features_plotter import create_feature_comparison_plots
    HAS_FEATURE_PLOTTER = True
except Exception:
    create_feature_comparison_plots = None
    HAS_FEATURE_PLOTTER = False


class XASFeatureExtractor:
    """
    Extract comprehensive features from normalized XAS spectra.
    
    This class implements all feature extraction methods defined in the
    XASFeatures model. It's configuration-driven and designed for batch
    processing with minimal human intervention.
    
    Usage:
        extractor = XASFeatureExtractor()
        features = extractor.extract_features(sample_result)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize feature extractor.
        
        Args:
            config_path: Path to YAML config (optional, auto-detected if None)
        """
        if ConfigLoader is not None:
            self.config = ConfigLoader(config_path)
            self.feature_config = self.config.get_section('feature_extraction')
        else:
            self.config = None
            self.feature_config = {}
        
        if not HAS_LARCH:
            logger.warning("larch not available. Some features may not be extractable.")
        
        logger.info("XASFeatureExtractor initialized")
    
    def extract_features_simple(self,
                                energy: np.ndarray,
                                mu_normalized: np.ndarray,
                                sample_name: str = "unknown",
                                e0: Optional[float] = None,
                                edge_step: Optional[float] = None) -> XASFeatures:
        """
        Extract features directly from energy and normalized mu arrays (simplified).
        
        This is a simpler interface that doesn't require creating XASSampleResult objects.
        
        Args:
            energy: Energy array in eV
            mu_normalized: Normalized absorption coefficient
            sample_name: Sample identifier
            e0: Edge energy (optional, will be calculated if not provided)
            edge_step: Edge step (optional, will be calculated if not provided)
            
        Returns:
            XASFeatures object with all extracted features
        """
        # Create a simple object to hold the data
        class SimpleResult:
            def __init__(self):
                self.sample_name = sample_name
                self.energy = energy
                self.normalized_mu = mu_normalized
                self.e0 = e0
                self.edge_step_value = edge_step
        
        sample_result = SimpleResult()
        
        # Extract features using internal methods
        edge_features = self._extract_edge_features(sample_result)
        xanes_features = self._extract_xanes_features(sample_result)
        exafs_features = self._extract_exafs_features(sample_result)
        derivative_features = self._extract_derivative_features(sample_result)
        statistical_features = self._extract_statistical_features(sample_result)
        
        # Combine all features
        all_features = {
            'sample_name': sample_name,
            **edge_features,
            **xanes_features,
            **exafs_features,
            **derivative_features,
            **statistical_features
        }
        
        # Create XASFeatures object
        features = XASFeatures(**all_features)
        
        logger.info(f"Extracted {len(all_features)} features from {sample_name}")
        return features
    
    def extract_features_from_arrays(self, 
                                     energy: np.ndarray,
                                     mu_normalized: np.ndarray,
                                     sample_name: str = "unknown",
                                     e0: Optional[float] = None,
                                     edge_step: Optional[float] = None) -> XASFeatures:
        """
        Extract features directly from energy and normalized mu arrays.
        
        Convenience method for working with raw data arrays. Calls extract_features_simple().
        
        Args:
            energy: Energy array in eV
            mu_normalized: Normalized absorption coefficient
            sample_name: Sample identifier
            e0: Edge energy (optional, will be calculated if not provided)
            edge_step: Edge step (optional, will be calculated if not provided)
            
        Returns:
            XASFeatures object with all extracted features
        """
        return self.extract_features_simple(energy, mu_normalized, sample_name, e0, edge_step)
    
    def extract_features(self, sample_result: XASSampleResult) -> XASFeatures:
        """
        Extract all features from a processed XAS sample.
        
        Main entry point for feature extraction. Orchestrates extraction
        of all feature types.
        
        Args:
            sample_result: XASSampleResult object with normalized spectrum
            
        Returns:
            XASFeatures object with all extracted features
            
        Raises:
            ValueError: If required data is missing
        """
        if sample_result.normalized_mu is None:
            raise ValueError(f"Sample {sample_result.sample_name} has no normalized spectrum")
        
        logger.info(f"Extracting features from {sample_result.sample_name}")
        
        # Extract each feature category
        edge_features = self._extract_edge_features(sample_result)
        xanes_features = self._extract_xanes_features(sample_result)
        exafs_features = self._extract_exafs_features(sample_result)
        derivative_features = self._extract_derivative_features(sample_result)
        statistical_features = self._extract_statistical_features(sample_result)
        
        # Combine all features
        all_features = {
            **edge_features,
            **xanes_features,
            **exafs_features,
            **derivative_features,
            **statistical_features
        }
        
        # Add sample name
        all_features['sample_name'] = sample_result.sample_name
        
        # Create XASFeatures object
        features = XASFeatures(**all_features)
        
        logger.info(f"Extracted {len(all_features)} features from {sample_result.sample_name}")
        return features
    
    def _extract_edge_features(self, sample_result: XASSampleResult) -> Dict[str, float]:
        """
        Extract edge-related features.
        """
        features = {}

        energy = sample_result.energy
        mu = sample_result.normalized_mu

        if hasattr(sample_result, 'e0') and sample_result.e0 is not None:
            features['e0'] = float(sample_result.e0)
        else:
            features['e0'] = self._find_edge_inflection(energy, mu)

        e0 = features['e0']
        pre_edge_range, post_edge_range = self._get_pre_post_ranges()

        edge_step = self._calculate_edge_step(energy, mu, e0, pre_edge_range, post_edge_range)
        features['edge_step'] = edge_step

        edge_slope = self._calculate_edge_slope(energy, mu, e0)
        features['edge_slope'] = edge_slope

        pre_edge_area = self._calculate_pre_edge_area(energy, mu, e0, pre_edge_range)
        features['pre_edge_area'] = pre_edge_area

        return features

    def _extract_xanes_features(self, sample_result: XASSampleResult) -> Dict[str, float]:
        """
        Extract XANES region features.
        """
        features = {}

        energy = sample_result.energy
        mu = sample_result.normalized_mu

        if hasattr(sample_result, 'e0') and sample_result.e0 is not None:
            e0 = sample_result.e0
        else:
            e0 = self._find_edge_inflection(energy, mu)

        wl_range, xanes_range, centroid_range, wl_threshold = self._get_xanes_ranges()
        pre_edge_range, _ = self._get_pre_post_ranges()

        pre_mask = (energy >= (e0 + pre_edge_range[0])) & (energy <= (e0 + pre_edge_range[1]))
        baseline = float(np.nanmean(mu[pre_mask])) if np.sum(pre_mask) > 1 else float(np.nanmin(mu))

        wl_mask = (energy >= (e0 + wl_range[0])) & (energy <= (e0 + wl_range[1]))
        wl_energy = energy[wl_mask]
        wl_mu_raw = mu[wl_mask]

        if len(wl_energy) == 0:
            return {
                'white_line_intensity': np.nan,
                'white_line_prominence': np.nan,
                'white_line_energy': e0,
                'white_line_fwhm': np.nan,
                'xanes_area': np.nan,
                'xanes_centroid': e0
            }

        wl_mu = self._smooth(wl_mu_raw)
        wl_idx = int(np.argmax(wl_mu))
        peak_val = float(wl_mu[wl_idx])
        peak_energy = float(wl_energy[wl_idx])

        if (peak_val - baseline) < wl_threshold:
            features['white_line_intensity'] = np.nan
            features['white_line_prominence'] = np.nan
            features['white_line_energy'] = e0
            features['white_line_fwhm'] = np.nan
        else:
            features['white_line_intensity'] = peak_val
            features['white_line_prominence'] = float(peak_val - baseline)
            features['white_line_energy'] = peak_energy
            features['white_line_fwhm'] = self._calculate_fwhm(wl_energy, wl_mu, peak_energy, baseline=baseline)

        xanes_mask = (energy >= (e0 + xanes_range[0])) & (energy <= (e0 + xanes_range[1]))
        xanes_energy = energy[xanes_mask]
        xanes_mu = mu[xanes_mask]

        if len(xanes_energy) > 1:
            xanes_area = integrate.simpson(xanes_mu - baseline, xanes_energy)
            features['xanes_area'] = float(xanes_area)
        else:
            features['xanes_area'] = np.nan

        centroid_mask = (energy >= (e0 + centroid_range[0])) & (energy <= (e0 + centroid_range[1]))
        cen_energy = energy[centroid_mask]
        cen_mu = mu[centroid_mask] - baseline
        if len(cen_energy) > 1 and np.nansum(cen_mu) > 0:
            centroid = np.nansum(cen_energy * cen_mu) / np.nansum(cen_mu)
            features['xanes_centroid'] = float(centroid)
        else:
            features['xanes_centroid'] = e0

        return features

    def _extract_exafs_features(self, sample_result: XASSampleResult) -> Dict[str, float]:
        """
        Extract EXAFS features.
        
        Features: chi_k_rms, ft_peak_r, ft_peak_amp
        
        Args:
            sample_result: XASSampleResult object
            
        Returns:
            Dictionary of EXAFS features
        """
        features = {}
        
        # Check if EXAFS data is available
        if hasattr(sample_result, 'chi_k') and sample_result.chi_k is not None:
            chi_k = sample_result.chi_k
            k = sample_result.k if hasattr(sample_result, 'k') else None
            
            # chi_k_rms - RMS of chi(k)
            chi_rms = float(np.sqrt(np.mean(chi_k**2)))
            features['chi_k_rms'] = chi_rms
            
            # FT features (if R-space data available)
            if hasattr(sample_result, 'chi_r') and sample_result.chi_r is not None:
                chi_r = sample_result.chi_r
                r = sample_result.r if hasattr(sample_result, 'r') else None
                
                if r is not None and len(chi_r) > 0:
                    # Get magnitude
                    chi_r_mag = np.abs(chi_r)
                    
                    # ft_peak_r - position of main peak
                    peak_idx = np.argmax(chi_r_mag)
                    features['ft_peak_r'] = float(r[peak_idx])
                    
                    # ft_peak_amp - amplitude of main peak
                    features['ft_peak_amp'] = float(chi_r_mag[peak_idx])
                else:
                    features['ft_peak_r'] = 0.0
                    features['ft_peak_amp'] = 0.0
            else:
                features['ft_peak_r'] = 0.0
                features['ft_peak_amp'] = 0.0
        else:
            # No EXAFS data available
            logger.warning("No EXAFS data available, setting default values")
            features['chi_k_rms'] = 0.0
            features['ft_peak_r'] = 0.0
            features['ft_peak_amp'] = 0.0
        
        return features
    
    def _extract_derivative_features(self, sample_result: XASSampleResult) -> Dict[str, float]:
        """
        Extract derivative-based features with smoothing.
        """
        features = {}

        energy = sample_result.energy
        mu = self._smooth(sample_result.normalized_mu)

        dmu_de = np.gradient(mu, energy)
        d2mu_de2 = np.gradient(dmu_de, energy)

        max_idx = int(np.argmax(dmu_de))
        features['first_derivative_max'] = float(energy[max_idx])

        if hasattr(sample_result, 'e0') and sample_result.e0 is not None:
            target = float(sample_result.e0)
        else:
            target = float(energy[max_idx])
        zero_crossing = self._find_zero_crossing_near(energy, d2mu_de2, target)
        features['second_derivative_zero'] = zero_crossing

        return features

    def _extract_statistical_features(self, sample_result: XASSampleResult) -> Dict[str, float]:
        """
        Extract statistical features from spectrum (XANES region preferred).
        """
        energy = sample_result.energy
        mu = sample_result.normalized_mu

        if hasattr(sample_result, 'e0') and sample_result.e0 is not None:
            e0 = float(sample_result.e0)
        else:
            e0 = self._find_edge_inflection(energy, mu)

        _, xanes_range, _, _ = self._get_xanes_ranges()
        mask = (energy >= (e0 + xanes_range[0])) & (energy <= (e0 + xanes_range[1]))
        if np.sum(mask) >= 5:
            mu_use = mu[mask]
        else:
            mu_use = mu

        features = {
            'spectral_mean': float(np.mean(mu_use)),
            'spectral_variance': float(np.var(mu_use)),
            'spectral_skewness': float(stats.skew(mu_use)),
            'spectral_kurtosis': float(stats.kurtosis(mu_use))
        }

        return features

    def _get_pre_post_ranges(self) -> tuple[list, list]:
        cfg = self.feature_config or {}
        pre_edge_range = cfg.get('pre_edge_range')
        post_edge_range = cfg.get('post_edge_range')
        if not pre_edge_range:
            pre_edge_range = cfg.get('edge', {}).get('pre_edge_range')
        if not post_edge_range:
            post_edge_range = cfg.get('edge', {}).get('post_edge_range')
        if not pre_edge_range:
            pre_edge_range = [-150, -30]
        if not post_edge_range:
            post_edge_range = [50, 300]
        return list(pre_edge_range), list(post_edge_range)

    def _get_xanes_ranges(self) -> tuple[list, list, list, float]:
        cfg = self.feature_config or {}
        xanes_cfg = cfg.get('xanes', {}) if isinstance(cfg.get('xanes', {}), dict) else {}
        wl_range = xanes_cfg.get('white_line_search_range', [0, 50])
        xanes_range = xanes_cfg.get('xanes_integration_range', [0, 50])
        centroid_range = xanes_cfg.get('centroid_range', [0, 50])
        wl_threshold = float(xanes_cfg.get('white_line_threshold', 0.1) or 0.1)

        xanes_region = cfg.get('xanes_region', {}) if isinstance(cfg.get('xanes_region', {}), dict) else {}
        if 'start_offset' in xanes_region or 'end_offset' in xanes_region:
            start = xanes_region.get('start_offset', 0)
            end = xanes_region.get('end_offset', 30)
            wl_range = [start, end]
            xanes_range = [start, end]
            centroid_range = [start, end]

        return list(wl_range), list(xanes_range), list(centroid_range), wl_threshold

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        cfg = self.feature_config or {}
        der_cfg = cfg.get('derivatives', {}) if isinstance(cfg.get('derivatives', {}), dict) else {}
        window = int(der_cfg.get('smoothing_window', 5) or 5)
        polyorder = int(der_cfg.get('polyorder', 3) or 3)
        if window < 3 or len(y) < window:
            return y
        if window % 2 == 0:
            window += 1
        if polyorder >= window:
            polyorder = max(1, window - 2)
        try:
            return signal.savgol_filter(y, window_length=window, polyorder=polyorder)
        except Exception:
            return y

    def _find_zero_crossing_near(self, x: np.ndarray, y: np.ndarray, target: float) -> float:
        sign_changes = np.where(np.diff(np.sign(y)))[0]
        if len(sign_changes) == 0:
            return float(x[len(x) // 2])
        crossings = []
        for idx in sign_changes:
            x0, x1 = x[idx], x[idx + 1]
            y0, y1 = y[idx], y[idx + 1]
            if abs(y1 - y0) > 1e-10:
                zx = x0 - y0 * (x1 - x0) / (y1 - y0)
            else:
                zx = (x0 + x1) / 2
            crossings.append(zx)
        crossings = np.array(crossings, dtype=float)
        return float(crossings[np.argmin(np.abs(crossings - target))])

    # =========================================================================
    # Helper methods for feature calculations
    # =========================================================================
    
    def _find_edge_inflection(self, energy: np.ndarray, mu: np.ndarray) -> float:
        """Find absorption edge as inflection point (max of first derivative)."""
        dmu_de = np.gradient(mu, energy)
        edge_idx = np.argmax(dmu_de)
        return float(energy[edge_idx])
    
    def _calculate_edge_step(self, energy: np.ndarray, mu: np.ndarray, e0: float,
                             pre_edge_range: list, post_edge_range: list) -> float:
        """Calculate edge step using pre/post linear fits at E0."""
        pre_mask = (energy >= (e0 + pre_edge_range[0])) & (energy <= (e0 + pre_edge_range[1]))
        post_mask = (energy >= (e0 + post_edge_range[0])) & (energy <= (e0 + post_edge_range[1]))

        if np.sum(pre_mask) < 2 or np.sum(post_mask) < 2:
            return float('nan')

        pre_coeff = np.polyfit(energy[pre_mask], mu[pre_mask], 1)
        post_coeff = np.polyfit(energy[post_mask], mu[post_mask], 1)
        pre_line = np.poly1d(pre_coeff)
        post_line = np.poly1d(post_coeff)

        edge_step = float(post_line(e0) - pre_line(e0))
        return edge_step
    
    def _calculate_edge_slope(self, energy: np.ndarray, mu: np.ndarray, e0: float) -> float:
        """Calculate slope at the edge (steepness)."""
        # Get points around e0 (±2 eV)
        edge_mask = (energy >= (e0 - 2)) & (energy <= (e0 + 2))
        
        if np.sum(edge_mask) < 2:
            return 0.0
        
        edge_energy = energy[edge_mask]
        edge_mu = mu[edge_mask]
        
        # Linear fit to get slope
        coeffs = np.polyfit(edge_energy, edge_mu, 1)
        slope = coeffs[0]
        
        return float(slope)
    
    def _calculate_pre_edge_area(self, energy: np.ndarray, mu: np.ndarray, e0: float, pre_edge_range: list) -> float:
        """Calculate baseline-corrected area under pre-edge region."""
        pre_start = e0 + pre_edge_range[0]
        pre_end = e0 + pre_edge_range[1]
        pre_mask = (energy >= pre_start) & (energy <= pre_end)
        if np.sum(pre_mask) < 2:
            return float('nan')
        pre_energy = energy[pre_mask]
        pre_mu = mu[pre_mask]

        coeffs = np.polyfit(pre_energy, pre_mu, 1)
        baseline = np.poly1d(coeffs)(pre_energy)
        area = integrate.simpson(pre_mu - baseline, pre_energy)
        return float(area)
    
    def _calculate_fwhm(self, x: np.ndarray, y: np.ndarray, peak_x: float, baseline: float = 0.0) -> float:
        """Calculate full width at half maximum."""
        # Find peak value
        peak_idx = np.argmin(np.abs(x - peak_x))
        peak_val = y[peak_idx]
        
        # Half maximum
        half_max = baseline + (peak_val - baseline) / 2.0
        
        # Find points where y crosses half_max
        above_half = y > half_max
        
        if not np.any(above_half):
            return 0.0
        
        # Find left and right edges
        indices = np.where(above_half)[0]
        
        if len(indices) < 2:
            return 0.0
        
        left_idx = indices[0]
        right_idx = indices[-1]
        
        fwhm = x[right_idx] - x[left_idx]
        
        return float(fwhm)
    
    def _find_zero_crossing(self, x: np.ndarray, y: np.ndarray) -> float:
        """Find first zero crossing of array y."""
        # Look for sign changes
        sign_changes = np.where(np.diff(np.sign(y)))[0]
        
        if len(sign_changes) == 0:
            # No zero crossing found, return middle point
            return float(x[len(x) // 2])
        
        # Return energy at first zero crossing
        idx = sign_changes[0]
        
        # Linear interpolation for better accuracy
        if idx < len(x) - 1:
            x0, x1 = x[idx], x[idx + 1]
            y0, y1 = y[idx], y[idx + 1]
            
            # Interpolate to find exact zero
            if abs(y1 - y0) > 1e-10:
                zero_x = x0 - y0 * (x1 - x0) / (y1 - y0)
            else:
                zero_x = (x0 + x1) / 2
            
            return float(zero_x)
        
        return float(x[idx])
    
    def extract_features_batch(self, sample_results: List[XASSampleResult]) -> List[XASFeatures]:
        """
        Extract features from multiple samples (batch processing).
        
        Args:
            sample_results: List of XASSampleResult objects
            
        Returns:
            List of XASFeatures objects
        """
        features_list = []
        
        for i, sample_result in enumerate(sample_results):
            try:
                features = self.extract_features(sample_result)
                features_list.append(features)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Extracted features from {i + 1}/{len(sample_results)} samples")
                    
            except Exception as e:
                logger.error(f"Failed to extract features from {sample_result.sample_name}: {e}")
                # Create placeholder features with None values
                features_list.append(XASFeatures(sample_name=sample_result.sample_name, e0=float('nan'), edge_step=float('nan')))
        
        logger.info(f"Batch feature extraction complete: {len(features_list)}/{len(sample_results)} successful")
        return features_list

    def extract_features_batch_with_plots(
        self,
        sample_results: List[XASSampleResult],
        output_dir: Optional[Path] = None
    ) -> List[XASFeatures]:
        """
        Batch feature extraction + feature comparison plots.
        """
        features_list = self.extract_features_batch(sample_results)

        if not HAS_FEATURE_PLOTTER or not features_list:
            return features_list

        if output_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent / "project_root"
            output_dir = project_root / "xas_results" / "03_feature_extraction" / "features_plots"

        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            features_dict = {
                feat.sample_name: {'features': feat.model_dump()}
                for feat in features_list
                if getattr(feat, 'sample_name', None)
            }
            create_feature_comparison_plots(
                features_dict,
                output_dir=output_dir,
                export_csv=False
            )
            logger.info(f"Feature plots saved to: {output_dir}")
        except Exception as e:
            logger.warning(f"Feature plot generation failed: {e}")

        return features_list


# =============================================================================
# Standalone utility functions
# =============================================================================

def extract_features_from_sample(
    sample_result: XASSampleResult,
    config_path: Optional[Path] = None
) -> XASFeatures:
    """
    Convenience function for extracting features from a single sample.
    
    Args:
        sample_result: XASSampleResult object
        config_path: Optional path to config file
        
    Returns:
        XASFeatures object
    """
    extractor = XASFeatureExtractor(config_path)
    return extractor.extract_features(sample_result)


def extract_features_from_batch(
    sample_results: List[XASSampleResult],
    config_path: Optional[Path] = None,
    plot: bool = True,
    plots_dir: Optional[Path] = None
) -> List[XASFeatures]:
    """
    Convenience function for batch feature extraction.
    
    Args:
        sample_results: List of XASSampleResult objects
        config_path: Optional path to config file
        
    Returns:
        List of XASFeatures objects
    """
    extractor = XASFeatureExtractor(config_path)
    if plot:
        return extractor.extract_features_batch_with_plots(sample_results, output_dir=plots_dir)
    return extractor.extract_features_batch(sample_results)


def generate_feature_comparison_plots_from_dicts(
    features_list: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Generate feature comparison plots from a list of feature dicts.
    """
    if not HAS_FEATURE_PLOTTER or not features_list:
        return None

    if output_dir is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent / "project_root"
        output_dir = project_root / "xas_results" / "03_feature_extraction" / "features_plots"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dict = {
        feat['sample_name']: {'features': feat}
        for feat in features_list
        if isinstance(feat, dict) and feat.get('sample_name')
    }
    create_feature_comparison_plots(
        features_dict,
        output_dir=output_dir,
        export_csv=False
    )
    return output_dir
