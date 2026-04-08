#!/usr/bin/env python3
"""
Stage 2 Detailed Spectrum Quality Assessment Module

Performs comprehensive quality analysis on imported XPS spectra AFTER successful import.
Generates detailed reports and diagnostic plots in 01_converted_csv directory.

Stage 2 checks (post-import, detailed):
- XPS-specific SNR calculation (peak vs baseline)
- Spectral resolution validation
- Energy axis shift detection
- Peak quality assessment
- Baseline stability
- Noise characteristics

Outputs:
- CSV quality report per region
- Diagnostic plots showing flagged issues
- Warning flags for low-quality data (data kept but flagged)
- example: A spectrum is only "high quality" if it has high SNR, a narrow FWHM, and a flat baseline.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from matplotlib.mlab import detrend
import numpy as np
import csv
from enum import Enum
import sys

# Import background correction utilities
sys.path.append(str(Path(__file__).parent.parent))
from XPS_utils.background_correction import baseline_shirley, subtract_background


class SpectrumQualityFlag(Enum):
    """Quality classification for individual spectra."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    SUSPICIOUS = "suspicious"


class QualityMetric(Enum):
    """Quality metrics tracked."""
    SNR_XPS = "snr_xps_specific"
    RESOLUTION = "resolution"
    DATA_POINTS = "data_points"
    ENERGY_SHIFT = "energy_shift"
    PEAK_QUALITY = "peak_quality"
    BASELINE_STABILITY = "baseline_stability"
    NOISE_LEVEL = "noise_level"


@dataclass
class SpectrumQualityMetrics:
    """Detailed quality metrics for a single spectrum."""
    
    # Identification
    sample_id: str
    region: str
    file_path: str
    
    # SNR metrics
    snr_basic: float = 0.0  # mean/std
    snr_xps: float = 0.0  # (peak_max - baseline_mean) / baseline_std
    
    # Resolution metrics
    points_per_ev: float = 0.0
    energy_step_ev: float = 0.0
    is_hr_scan: bool = False
    
    # Energy axis metrics
    energy_min: float = 0.0
    energy_max: float = 0.0
    energy_range: float = 0.0
    suspected_shift: bool = False
    shift_estimate_ev: float = 0.0
    
    # Peak quality metrics
    peak_height: float = 0.0
    peak_width_fwhm: float = 0.0   
    peak_to_baseline_ratio: float = 0.0
    dynamic_range: float = 0.0
    
    # Baseline metrics
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    baseline_slope: float = 0.0
    baseline_curvature: float = 0.0
    
    # Noise metrics: this subtle distinguishes noise from baseline std, useful for complex spectra analysis like valence bands with spikes
    noise_rms: float = 0.0
    noise_peak_to_peak: float = 0.0
    relative_noise: float = 0.0
    
    # Overall assessment
    quality_flag: SpectrumQualityFlag = SpectrumQualityFlag.ACCEPTABLE
    warnings: List[str] = field(default_factory=list)
    suitable_for_fitting: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            'sample_id': self.sample_id,
            'region': self.region,
            'quality_flag': self.quality_flag.value,
            'snr_xps': f"{self.snr_xps:.2f}",
            'resolution_pts_per_ev': f"{self.points_per_ev:.2f}",
            'energy_step_ev': f"{self.energy_step_ev:.4f}",
            'is_hr_scan': self.is_hr_scan,
            'energy_range_ev': f"{self.energy_range:.2f}",
            'shift_estimate_ev': f"{self.shift_estimate_ev:+.3f}",
            'suspected_shift': self.suspected_shift,
            'peak_height': f"{self.peak_height:.1f}",
            'peak_width_fwhm': f"{self.peak_width_fwhm:.2f}",
            'peak_to_baseline': f"{self.peak_to_baseline_ratio:.2f}",
            'baseline_std': f"{self.baseline_std:.2f}",
            'relative_noise': f"{self.relative_noise:.3f}",
            'suitable_for_fitting': self.suitable_for_fitting,
            'warnings': '; '.join(self.warnings) if self.warnings else 'None'
        }


class SpectrumQualityAnalyzer:
    """
    Stage 2 detailed quality analyzer for XPS spectra.
    
    Performs comprehensive analysis after successful import, generating
    detailed reports and diagnostic plots.
    """
    
    def __init__(self, config: Dict = None, debug: bool = False):
        """
        Initialize analyzer.
        
        Args:
            config: Configuration dict with thresholds
            debug: Enable debug output
        """
        self.debug = debug
        self.config = config or self._default_config()
        self.quality_reports: List[SpectrumQualityMetrics] = []
    
    def analyze_spectrum(self, spectrum, sample_id: str, region: str, 
                        file_path: str = "") -> SpectrumQualityMetrics:
        """
        Perform detailed quality analysis on a single spectrum.
        
        Args:
            spectra: The list of CALIBRATED spectra.
            cal_result: The result object from the EnergyCalibrator.
            spectrum: Spectrum object with .energy and .intensity attributes
            sample_id: Sample identifier
            region: Region name (e.g., 'C1s', 'O1s')
            file_path: Original file path
            
        Returns:
            SpectrumQualityMetrics with comprehensive analysis
        """
        energy = np.array(spectrum.energy)
        intensity = np.array(spectrum.intensity)
        
        # Initialize metrics object
        metrics = SpectrumQualityMetrics(
            sample_id=sample_id,
            region=region,
            file_path=file_path
        )
        
      
        # 1. XPS-specific SNR (peak vs baseline)
        metrics.snr_xps, metrics.baseline_mean, metrics.baseline_std = \
            self._calculate_snr_xps(energy, intensity)

        # 2. Resolution analysis
        metrics.points_per_ev, metrics.energy_step_ev = \
            self._calculate_resolution(energy)
        metrics.is_hr_scan = metrics.points_per_ev >= self.config['hr_resolution_threshold']

        # 3. Energy axis analysis (verify calibrated position)
        metrics.energy_min = float(np.min(energy))
        metrics.energy_max = float(np.max(energy))
        metrics.energy_range = metrics.energy_max - metrics.energy_min
        metrics.suspected_shift, metrics.shift_estimate_ev = \
            self._detect_energy_shift(energy, intensity, region)

        # 4. Peak quality analysis, e.g. FWHM = 0.9 eV perfect for identifying chemical states vs FWHM = 4.5 eV very broad, smeared-out peak.
        metrics.peak_height, metrics.peak_width_fwhm, metrics.peak_to_baseline_ratio, \
        metrics.dynamic_range = self._analyze_peak_quality(energy, intensity)

        # 5. Baseline analysis
        metrics.baseline_slope, metrics.baseline_curvature = \
            self._analyze_baseline(energy, intensity)

        # 6. Noise analysis
        metrics.noise_rms, metrics.noise_peak_to_peak, metrics.relative_noise = \
            self._analyze_noise(energy, intensity)

        # 7. Overall quality assessment
        metrics.quality_flag, metrics.warnings, metrics.suitable_for_fitting = \
            self._assess_overall_quality(metrics)
        
        # Store for batch reporting
        self.quality_reports.append(metrics)
        
        if self.debug:
            print(f"[SpectrumQuality] {sample_id} - {region}: {metrics.quality_flag.value} "
                  f"(SNR_XPS={metrics.snr_xps:.2f}, Res={metrics.points_per_ev:.1f} pts/eV)")
        
        return metrics
    
    def analyze_batch(self, spectra_dict: Dict, region: str,
                     output_dir: Path, plots_dir: Optional[Path] = None) -> List[SpectrumQualityMetrics]:
        """
        Analyze multiple spectra and generate reports.
        
        Args:
            spectra_dict: Dict mapping sample_id -> spectrum object
            region: Region name
            output_dir: Output directory (01_converted_csv)
            
        Returns:
            List of quality metrics for all spectra
        """
        batch_metrics = []
        
        for dict_key, spectrum in spectra_dict.items():
            # Get spectrum name and metadata
            spectrum_name = getattr(spectrum, 'name', dict_key)
            region_from_metadata = spectrum.metadata.get('region', region or '')
            
            # Debug: Print all metadata keys
            if self.debug:
                print(f"\n[DEBUG] Spectrum metadata keys: {list(spectrum.metadata.keys())}")
                print(f"[DEBUG] spectrum.name = '{spectrum_name}'")
                print(f"[DEBUG] region from metadata = '{region_from_metadata}'")
                print(f"[DEBUG] dict_key = '{dict_key}'")
            
            # Extract base filename (sample ID) from spectrum name
            # Spectrum name formats:
            # 1. From region_extraction: "filename_regionname" (e.g., "PS_GenF_Cyc_2_surface_Si2p")
            # 2. From SOFH parser: "filename_regionname" (e.g., "PS_GenF_Cyc_2_surface_Si2p")
            # 3. From metadata original_spectrum: base filename
            
            # PRIORITY 1: Check metadata['original_spectrum'] for base filename
            original_name = spectrum.metadata.get('original_spectrum', '')
            
            if self.debug:
                print(f"[DEBUG] original_spectrum from metadata = '{original_name}'")
            
            # Extract sample_id: remove region suffix from original_spectrum or spectrum_name
            # Expected formats:
            # - spectrum_name: "E1_C1s", "E10_P2p", "filename_regionname"
            # - original_spectrum: "E1", "E10", "filename"
            # Goal: Extract just the base filename without region suffix
            
            if original_name:
                # If original_spectrum exists, use it as base (it should already be clean)
                sample_id = original_name
            else:
                # Parse spectrum_name to extract base filename
                # Remove region suffix (last component after underscore)
                if region_from_metadata and spectrum_name.endswith(f"_{region_from_metadata}"):
                    # Remove "_regionname" suffix (e.g., "E1_C1s" -> "E1")
                    sample_id = spectrum_name[:-len(f"_{region_from_metadata}")]
                elif '_' in spectrum_name and not spectrum_name.startswith('_'):
                    # Generic fallback: remove last underscore component if it looks like a region
                    parts = spectrum_name.rsplit('_', 1)
                    last_part = parts[1] if len(parts) > 1 else ''
                    # Check if last part looks like a region name (short, has letters)
                    if last_part and len(last_part) <= 6 and any(c.isalpha() for c in last_part):
                        sample_id = parts[0] if parts[0] else spectrum_name
                    else:
                        sample_id = spectrum_name
                else:
                    sample_id = spectrum_name
            
            # Additional cleaning: use the dict_key (which comes from reader_main.py name cleaning)
            # The dict_key should already be cleaned by reader_main.py
            if dict_key and dict_key != spectrum_name:
                # Use the cleaned dict_key if it's different from spectrum_name
                sample_id = dict_key
            
            # Get source file from metadata if available
            source_file = getattr(spectrum, 'source_file', '')
            if not source_file:
                source_file = spectrum.metadata.get('source_file', '')

            # Normalize sample_id using source filename when available
            is_depth_profile = bool(spectrum.metadata.get('depth_profile', False))
            clean_source = self._clean_sample_name(source_file) if source_file else ''
            if clean_source and not is_depth_profile:
                sample_id = clean_source
            else:
                sample_id = self._clean_sample_name(sample_id)
            
            if self.debug:
                print(f"[SpectrumQuality] Processing: {sample_id} (spectrum.name={spectrum_name}, original={original_name})")
            
            metrics = self.analyze_spectrum(
                spectrum=spectrum,
                sample_id=sample_id,
                region=region,
                file_path=str(source_file)
            )
            batch_metrics.append(metrics)
        
        # Generate CSV report
        self._export_quality_report(batch_metrics, region, output_dir)
        
        # Generate diagnostic plots if matplotlib available
        try:
            self._generate_quality_plots(batch_metrics, region, output_dir, spectra_dict, plots_dir=plots_dir)
        except ImportError:
            if self.debug:
                print("[SpectrumQuality] Matplotlib not available, skipping plots")
        
        return batch_metrics
    
    def _analyze_xps_region_quality(self, energy: np.ndarray, intensity: np.ndarray) -> Dict:
        """
        Comprehensive quality analysis combining SNR, baseline, and noise metrics.
        
        Calculates:
        1. Signal-to-Noise Ratio (SNR), robust to baseline drift
        2. Baseline drift metrics (slope and curvature)
        3. Noise characteristics (RMS, peak-to-peak)
        
        Returns:
            Dict with keys: snr_xps, baseline_mean, baseline_std, baseline_slope,
                           baseline_curvature, noise_rms, noise_pk_pk, relative_noise
        """
        if len(energy) < 10:
            return {
                'snr_xps': 0.0, 'baseline_mean': 0.0, 'baseline_std': 0.0,
                'baseline_slope': 0.0, 'baseline_curvature': 0.0,
                'noise_rms': 0.0, 'noise_pk_pk': 0.0, 'relative_noise': 0.0
            }
        
        # Sort data by energy
        sort_indices = np.argsort(energy)
        energy_sorted = energy[sort_indices]
        intensity_sorted = intensity[sort_indices]
        
        # --- Define baseline region at high-BE end (25% of scan width) ---
        baseline_width_ev = (energy_sorted[-1] - energy_sorted[0]) * 0.25
        high_be_cutoff = energy_sorted[-1] - baseline_width_ev
        baseline_mask = energy_sorted >= high_be_cutoff
        
        if np.sum(baseline_mask) < 5:
            # Fallback: use entire spectrum if baseline region too small
            baseline_energy = energy_sorted
            baseline_intensity = intensity_sorted
        else:
            baseline_energy = energy_sorted[baseline_mask]
            baseline_intensity = intensity_sorted[baseline_mask]
        
        # --- 1. SNR Calculation ---
        baseline_mean = np.mean(baseline_intensity)
        detrended_baseline = detrend(baseline_intensity)
        baseline_std = np.std(detrended_baseline)
        
        peak_max = np.max(intensity_sorted)
        signal = peak_max - baseline_mean
        snr_xps = signal / (baseline_std + 1e-12)
        
        # --- 2. Baseline Drift Analysis ---
        if len(baseline_energy) >= 3:
            # Linear fit for slope
            slope_coeffs = np.polyfit(baseline_energy, baseline_intensity, 1)
            baseline_slope = slope_coeffs[0]  # counts per eV
            
            # Quadratic fit for curvature
            if len(baseline_energy) >= 5:
                curv_coeffs = np.polyfit(baseline_energy, baseline_intensity, 2)
                baseline_curvature = abs(curv_coeffs[0])
            else:
                baseline_curvature = 0.0
        else:
            baseline_slope = 0.0
            baseline_curvature = 0.0
        
        # --- 3. Noise Analysis (using detrended baseline) ---
        noise_rms = baseline_std  # Already calculated from detrended baseline
        noise_pk_pk = np.max(detrended_baseline) - np.min(detrended_baseline)
        
        mean_intensity = np.mean(intensity_sorted)
        relative_noise = noise_rms / (mean_intensity + 1e-10)
        
        return {
            'snr_xps': float(snr_xps),
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'baseline_slope': float(baseline_slope),
            'baseline_curvature': float(baseline_curvature),
            'noise_rms': float(noise_rms),
            'noise_pk_pk': float(noise_pk_pk),
            'relative_noise': float(relative_noise)
        }
    
    def _calculate_snr_xps(self, energy: np.ndarray, intensity: np.ndarray) -> Tuple[float, float, float]:
        """Legacy wrapper for backward compatibility."""
        result = self._analyze_xps_region_quality(energy, intensity)
        return result['snr_xps'], result['baseline_mean'], result['baseline_std']
      
    
    def _calculate_resolution(self, energy: np.ndarray) -> Tuple[float, float]:
        """
        Calculate spectral resolution.
        
        Returns:
            (points_per_ev, energy_step_ev)
        """
        energy_range = np.max(energy) - np.min(energy)
        n_points = len(energy)
        points_per_ev = n_points / energy_range if energy_range > 0 else 0.0
        
        # Calculate median energy step for robustness
        energy_diffs = np.abs(np.diff(energy))
        energy_step = np.median(energy_diffs) if len(energy_diffs) > 0 else 0.0
        
        return float(points_per_ev), float(energy_step)
    
    def _detect_energy_shift(self, energy: np.ndarray, intensity: np.ndarray, region: str) -> Tuple[bool, float]:
        """
        Verify calibrated energy position against expected values.
        
        After calibration, the peak maximum should be at the expected binding energy
        (e.g., C1s at 284.8 eV). This function finds the peak and checks if it's
        within the allowed tolerance from project_setting.yaml.
        
        Args:
            energy: Calibrated energy axis
            intensity: Intensity values
            region: Region name (e.g., 'C1s', 'O1s')
            
        Returns:
            (suspected_shift, shift_estimate_ev)
        """
        # Get expected positions from config (from project_setting.yaml)
        # Fallback to typical values if not in config
        default_positions = {
            'C1s': 284.8,  # Adventitious carbon reference
            'O1s': 532.0,
            'N1s': 400.0,
            'Si2p': 103.0,
            'Au4f': 84.0,
            'F1s': 688.0,
            'S2p': 164.0,
            'P2p': 130.0,
        }
        
        # Extract element from region name (e.g., 'C1s_HR' -> 'C1s')
        element = region.split('_')[0] if '_' in region else region
        
        # Get expected position from config or use default
        region_config = self.config.get('region_definitions', {}).get(element, {})
        expected_peak = region_config.get('typical_be', default_positions.get(element))
        
        if expected_peak is None:
            # Unknown element, cannot verify
            return False, 0.0
        
        # Find actual peak position (use argmax for simplicity)
        peak_idx = np.argmax(intensity)
        actual_peak = energy[peak_idx]
        
        # Calculate shift from expected position
        shift = actual_peak - expected_peak
        
        # Get tolerance from config (max_allowed_shift_ev per region or global)
        max_shift = region_config.get('max_allowed_shift_ev', 
                                     self.config.get('max_allowed_shift_ev', 5.0))
        
        # Flag if shift exceeds tolerance
        if abs(shift) > max_shift:
            return True, float(shift)
        
        return False, float(shift)
    
    def _analyze_peak_quality(self, energy: np.ndarray, intensity: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Analyzes peak characteristics using a robust Shirley background model.
        
        Returns:
            (peak_height, peak_to_baseline_ratio, fwhm, dynamic_range)
        """
        # 1. Calculate the Shirley background across the region
        shirley_bg = baseline_shirley(energy, intensity)
        
        # 2. Subtract the background to get a corrected intensity profile
        intensity_corrected = intensity - shirley_bg
        
        # 3. Find the peak height from the corrected data
        # The baseline is now effectively zero, so the peak height is just the max intensity.
        peak_height = np.max(intensity_corrected)
        peak_idx = np.argmax(intensity_corrected)
        
        # --- define the P/B Ratio ---
        # The "baseline" is the value of the Shirley background directly under the peak.
        baseline_value_at_peak = shirley_bg[peak_idx]
        
        # 4. Calculate the robust Peak-to-Baseline ratio
        # Avoid division by zero if the baseline is near zero for some reason
        peak_to_baseline_ratio = peak_height / (baseline_value_at_peak + 1e-10)
        
        # 5. Calculate FWHM on corrected data
        half_max = peak_height / 2.0
        above_half = intensity_corrected >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm = abs(energy[indices[-1]] - energy[indices[0]])
        else:
            fwhm = 0.0
        
        # 6. Calculate dynamic range
        dynamic_range = peak_height / (np.std(intensity_corrected) + 1e-10)
        
        # Return in the order expected by caller: (peak_height, fwhm, peak_to_baseline_ratio, dynamic_range)
        return float(peak_height), float(fwhm), float(peak_to_baseline_ratio), float(dynamic_range)

    
    def _analyze_baseline(self, energy: np.ndarray, intensity: np.ndarray) -> Tuple[float, float]:
        """Legacy wrapper for backward compatibility."""
        result = self._analyze_xps_region_quality(energy, intensity)
        return result['baseline_slope'], result['baseline_curvature']
    
    def _analyze_noise(self, energy: np.ndarray, intensity: np.ndarray) -> Tuple[float, float, float]:
        """Legacy wrapper for backward compatibility."""
        result = self._analyze_xps_region_quality(energy, intensity)
        return result['noise_rms'], result['noise_pk_pk'], result['relative_noise']

  
    def _assess_overall_quality(self, metrics: SpectrumQualityMetrics) -> Tuple[SpectrumQualityFlag, List[str], bool]:
        """
        Assess overall quality and generate warnings.
        
        Returns:
            (quality_flag, warnings, suitable_for_fitting)
        """
        warnings = []
        
        # SNR checks
        if metrics.snr_xps < self.config['min_snr_xps_poor']:
            warnings.append(f"Very low XPS SNR ({metrics.snr_xps:.2f})")
        elif metrics.snr_xps < self.config['min_snr_xps_acceptable']:
            warnings.append(f"Low XPS SNR ({metrics.snr_xps:.2f})")
        
        # Resolution checks
        if metrics.points_per_ev < self.config['min_resolution_for_fitting']:
            warnings.append(f"Low resolution ({metrics.points_per_ev:.1f} pts/eV)")
        
        # Energy shift check
        if metrics.suspected_shift:
            warnings.append(f"Suspected energy shift ({metrics.shift_estimate_ev:+.1f} eV)")
        
        # Peak quality checks
        if metrics.peak_to_baseline_ratio < self.config['min_peak_to_baseline']:
            warnings.append(f"Weak peak (ratio={metrics.peak_to_baseline_ratio:.2f})")
        
        # Noise checks
        if metrics.relative_noise > self.config['max_relative_noise']:
            warnings.append(f"High noise level ({metrics.relative_noise:.2%})")
        
        # Determine quality flag
        if metrics.snr_xps >= self.config['min_snr_xps_excellent'] and \
           metrics.is_hr_scan and len(warnings) == 0:
            quality_flag = SpectrumQualityFlag.EXCELLENT
        elif metrics.snr_xps >= self.config['min_snr_xps_good'] and \
             metrics.points_per_ev >= self.config['hr_resolution_threshold']:
            quality_flag = SpectrumQualityFlag.GOOD
        elif metrics.snr_xps >= self.config['min_snr_xps_acceptable']:
            quality_flag = SpectrumQualityFlag.ACCEPTABLE
        elif metrics.snr_xps >= self.config['min_snr_xps_poor']:
            quality_flag = SpectrumQualityFlag.POOR
        else:
            quality_flag = SpectrumQualityFlag.SUSPICIOUS
        
        # Fitting suitability
        suitable_for_fitting = (
            metrics.is_hr_scan and
            metrics.snr_xps >= self.config['min_snr_xps_acceptable'] and
            metrics.peak_to_baseline_ratio >= self.config['min_peak_to_baseline']
        )
        
        return quality_flag, warnings, suitable_for_fitting
    
    def _export_quality_report(self, metrics_list: List[SpectrumQualityMetrics], 
                               region: str, output_dir: Path):
        """
        Export quality metrics to CSV.
        
        Args:
            metrics_list: List of quality metrics
            region: Region name
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / f"{region}_quality_report.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if not metrics_list:
                return
            
            fieldnames = list(metrics_list[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for metrics in metrics_list:
                writer.writerow(metrics.to_dict())
        
        if self.debug:
            print(f"[SpectrumQuality] Quality report saved: {csv_path}")
    
    def _generate_quality_plots(self, metrics_list: List[SpectrumQualityMetrics],
                               region: str, output_dir: Path, spectra_dict: Dict,
                               plots_dir: Optional[Path] = None):
        """
        Generate diagnostic plots using dedicated plotting module.
        
        Args:
            metrics_list: List of quality metrics
            region: Region name
            output_dir: Output directory
            spectra_dict: Dict of {key: Spectrum} for raw data overlay
        """
        try:
            # Import plotting utilities
            from pathlib import Path as P
            import sys
            plotter_path = P(__file__).resolve().parents[1] / "XPS_Plotter" / "plot_modules" / "data_quality"
            if str(plotter_path) not in sys.path:
                sys.path.insert(0, str(plotter_path))
            
            from quality_report_plots import plot_quality_report_diagnostics, plot_sample_quality_details
            
            # Generate main diagnostic plot
            plot_path = plot_quality_report_diagnostics(
                metrics_list=metrics_list,
                region=region,
                output_dir=output_dir,
                config=self.config,
                spectra_dict=spectra_dict,
                plots_dir=plots_dir,
            )
            
            if plot_path and self.debug:
                print(f"[SpectrumQuality] Quality diagnostics saved: {plot_path}")
            
            # Generate detailed per-sample plots if many samples
            if len(metrics_list) > 4:
                detail_paths = plot_sample_quality_details(
                    metrics_list=metrics_list,
                    region=region,
                    output_dir=output_dir,
                    config=self.config,
                    max_samples_per_plot=20,
                    plots_dir=plots_dir,
                )
                if detail_paths and self.debug:
                    print(f"[SpectrumQuality] Detailed plots saved: {len(detail_paths)} files")
                    
        except ImportError as e:
            if self.debug:
                print(f"[SpectrumQuality] Could not generate plots: {e}")
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            # SNR thresholds (XPS-specific)
            'min_snr_xps_poor': 2.0,
            'min_snr_xps_acceptable': 3.0,
            'min_snr_xps_good': 10.0,
            'min_snr_xps_excellent': 20.0,
            
            # Resolution thresholds
            'hr_resolution_threshold': 5.0,  # pts/eV
            'min_resolution_for_fitting': 3.0,
            
            # Peak quality thresholds
            'min_peak_to_baseline': 0.2,
            
            # Noise thresholds
            'max_relative_noise': 0.2,  # 20%
            
            # Baseline width for SNR calculation
            'baseline_width_ev': 5.0,  # Width of baseline region at high BE end
            
            # Calibration verification (max allowed deviation from expected position)
            'max_allowed_shift_ev': 5.0,  # Global tolerance, can be overridden per region
            
            # Region-specific config can be added via project_setting.yaml:
            # 'region_definitions': {
            #     'C1s': {'typical_be': 284.8, 'max_allowed_shift_ev': 5.0},
            #     'O1s': {'typical_be': 532.0, 'max_allowed_shift_ev': 3.0},
            # }
        }

    @staticmethod
    def _clean_sample_name(name: str) -> str:
        """Strip paths and common data file extensions from sample names."""
        if not name:
            return name
        clean_name = Path(str(name)).name
        suffix = Path(clean_name).suffix
        known_suffixes = {
            ".spe", ".vgd", ".npl", ".xy", ".txt", ".asc", ".dat", ".csv", ".vms", ".vamas", ".pro"
        }
        if suffix.lower() in known_suffixes:
            clean_name = clean_name[: -len(suffix)]
        return clean_name or name
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics across all analyzed spectra.
        
        Returns:
            Dictionary with summary stats
        """
        if not self.quality_reports:
            return {}
        
        snr_values = [m.snr_xps for m in self.quality_reports]
        resolution_values = [m.points_per_ev for m in self.quality_reports]
        
        quality_counts = {}
        for m in self.quality_reports:
            flag = m.quality_flag.value
            quality_counts[flag] = quality_counts.get(flag, 0) + 1
        
        total = len(self.quality_reports)
        suitable_count = sum(1 for m in self.quality_reports if m.suitable_for_fitting)
        
        return {
            'total_spectra': total,
            'mean_snr_xps': np.mean(snr_values),
            'median_snr_xps': np.median(snr_values),
            'mean_resolution': np.mean(resolution_values),
            'quality_distribution': quality_counts,
            'suitable_for_fitting': suitable_count,
            'fitting_success_rate': suitable_count / total if total > 0 else 0.0
        }


if __name__ == "__main__":
    print("Stage 2 Spectrum Quality Analyzer")
    print("=" * 60)
    print("\nPurpose: Detailed post-import quality assessment")
    print("\nMetrics:")
    print("  - XPS-specific SNR (peak vs baseline)")
    print("  - Spectral resolution")
    print("  - Energy axis shift detection")
    print("  - Peak quality assessment")
    print("  - Baseline stability")
    print("  - Noise characteristics")
    print("\nOutputs:")
    print("  - CSV quality report per region")
    print("  - Diagnostic plots")
    print("  - Warning flags for low-quality data")
