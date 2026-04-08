"""
XAS Spectrum Quality Check Module

Detects globally poor-quality spectra based on saturation, low edge jump, excessive noise.
Provides detailed quality metrics and generates comprehensive reports.

Outputs:
- quality classification ('usable', 'usable_with_warning', 'invalid')
- quality confidence
- flags
- detailed metrics for reporting
- CSV quality reports
- diagnostic plots
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import yaml


class XASQualityFlag(Enum):
    """Quality classification for XAS spectra."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class XASSpectrumQualityMetrics:
    """Detailed quality metrics for a single XAS spectrum."""

    # Identification
    sample_id: str
    file_path: str

    # Basic quality assessment (from original checker)
    classification: str = "usable"
    confidence: float = 1.0
    flags: List[str] = field(default_factory=list)

    # Data integrity metrics
    data_points: int = 0
    energy_range: float = 0.0
    energy_min: float = 0.0
    energy_max: float = 0.0

    # Edge jump metrics
    edge_jump: float = 0.0
    edge_position: float = 0.0  # Estimated E0

    # Noise metrics
    noise_level: float = 0.0
    signal_to_noise: float = 0.0

    # Saturation metrics
    max_intensity: float = 0.0
    saturation_ratio: float = 0.0

    # Processing quality metrics
    deglitching_points_removed: int = 0
    normalization_quality: float = 0.0  # Based on pre/post edge stability

    # Spectral features
    pre_edge_slope: float = 0.0
    post_edge_slope: float = 0.0
    white_line_intensity: float = 0.0

    # Overall assessment
    quality_flag: XASQualityFlag = XASQualityFlag.ACCEPTABLE
    warnings: List[str] = field(default_factory=list)
    suitable_for_analysis: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "file_path": self.file_path,
            "classification": self.classification,
            "confidence": self.confidence,
            "flags": self.flags,
            "data_points": self.data_points,
            "energy_range": self.energy_range,
            "energy_min": self.energy_min,
            "energy_max": self.energy_max,
            "edge_jump": self.edge_jump,
            "edge_position": self.edge_position,
            "noise_level": self.noise_level,
            "signal_to_noise": self.signal_to_noise,
            "max_intensity": self.max_intensity,
            "saturation_ratio": self.saturation_ratio,
            "deglitching_points_removed": self.deglitching_points_removed,
            "normalization_quality": self.normalization_quality,
            "pre_edge_slope": self.pre_edge_slope,
            "post_edge_slope": self.post_edge_slope,
            "white_line_intensity": self.white_line_intensity,
            "quality_flag": self.quality_flag.value,
            "warnings": self.warnings,
            "suitable_for_analysis": self.suitable_for_analysis
        }


class XASSpectrumQualityChecker:
    """
    Enhanced quality assessment for XAS spectra with detailed metrics.
    """

    def __init__(self,
                 quality_thresholds: Optional[Dict] = None):
        """
        Initialize quality checker.

        Parameters
        ----------
        quality_thresholds : dict, optional
            Thresholds for quality checks
        """
        self.thresholds = quality_thresholds or {
            "min_edge_jump": 0.1,      # Minimum edge step
            "max_noise_level": 0.05,   # Maximum noise fraction
            "saturation_threshold": 0.95,  # Saturation as fraction of max
            "min_data_points": 50,     # Minimum data points
            "min_snr": 10.0,           # Minimum signal-to-noise ratio
            "min_snr_invalid": 5.0,    # Severe SNR cutoff for invalid
            "max_pre_edge_slope": 0.01, # Maximum pre-edge slope (stability)
            "min_edge_jump_invalid": 0.05, # Severe edge jump cutoff
            "excellent_edge_jump": 0.5, # Threshold for excellent quality
            "good_edge_jump": 0.3,     # Threshold for good quality
        }

    def check_spectrum_quality(self,
                              energy: np.ndarray,
                              mu: np.ndarray,
                              normalized_mu: Optional[np.ndarray] = None,
                              sample_id: str = "",
                              file_path: str = "") -> XASSpectrumQualityMetrics:
        """
        Assess overall quality of XAS spectrum with detailed metrics.

        Parameters
        ----------
        energy : np.ndarray
            Energy values
        mu : np.ndarray
            Absorption coefficient
        normalized_mu : np.ndarray, optional
            Normalized absorption
        sample_id : str
            Sample identifier
        file_path : str
            File path

        Returns
        -------
        metrics : XASSpectrumQualityMetrics
            Detailed quality assessment
        """
        metrics = XASSpectrumQualityMetrics(
            sample_id=sample_id,
            file_path=file_path
        )

        # Basic data integrity checks
        metrics.data_points = len(energy)
        metrics.energy_min = float(np.min(energy))
        metrics.energy_max = float(np.max(energy))
        metrics.energy_range = metrics.energy_max - metrics.energy_min
        metrics.max_intensity = float(np.max(mu))

        flags = []
        confidence = 1.0
        classification = "usable"

        # Check data integrity
        if len(energy) < self.thresholds["min_data_points"]:
            flags.append("insufficient_data_points")
            classification = "invalid"
            confidence = 0.0

        # Check for NaN or infinite values
        if np.any(~np.isfinite(mu)):
            flags.append("invalid_intensity_values")
            classification = "invalid"
            confidence = 0.0

        if classification == "invalid":
            metrics.classification = classification
            metrics.confidence = confidence
            metrics.flags = flags
            metrics.quality_flag = XASQualityFlag.INVALID
            metrics.suitable_for_analysis = False
            return metrics

        # Estimate edge jump and position
        edge_jump, edge_position = self._estimate_edge_jump(energy, mu, normalized_mu)
        metrics.edge_jump = edge_jump
        metrics.edge_position = edge_position

        if edge_jump < self.thresholds["min_edge_jump"]:
            flags.append("low_edge_jump")
            classification = "usable_with_warning"
            confidence *= 0.8

        # Check for saturation
        saturation_ratio = self._check_saturation(mu)
        metrics.saturation_ratio = saturation_ratio
        if saturation_ratio > self.thresholds["saturation_threshold"]:
            flags.append("possible_saturation")
            classification = "usable_with_warning"
            confidence *= 0.7

        # Estimate noise level and SNR
        noise_level, snr = self._estimate_noise_and_snr(energy, mu)
        metrics.noise_level = noise_level
        metrics.signal_to_noise = snr

        if snr < self.thresholds["min_snr"]:
            flags.append("low_signal_to_noise")
            if classification == "usable":
                classification = "usable_with_warning"
            confidence *= 0.6

        # Severe issues make spectrum invalid
        if (edge_jump < self.thresholds.get("min_edge_jump_invalid", 0.05) or
            snr < self.thresholds.get("min_snr_invalid", 5.0)):
            classification = "invalid"
            confidence *= 0.3

        # Additional detailed metrics
        if normalized_mu is not None:
            pre_edge_slope, post_edge_slope = self._analyze_edge_regions(
                energy, normalized_mu, edge_position
            )
            metrics.pre_edge_slope = pre_edge_slope
            metrics.post_edge_slope = post_edge_slope

            if abs(pre_edge_slope) > self.thresholds["max_pre_edge_slope"]:
                flags.append("unstable_pre_edge")
                confidence *= 0.9

            white_line_intensity = self._estimate_white_line(normalized_mu)
            metrics.white_line_intensity = white_line_intensity

        # Overall quality flag
        quality_flag = self._determine_quality_flag(metrics, flags)
        warnings = self._generate_warnings(flags)

        # Update metrics
        metrics.classification = classification
        metrics.confidence = confidence
        metrics.flags = flags
        metrics.quality_flag = quality_flag
        metrics.warnings = warnings
        metrics.suitable_for_analysis = classification != "invalid"

        return metrics

    def _estimate_edge_jump(self, energy: np.ndarray, mu: np.ndarray,
                           normalized_mu: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Estimate edge jump and position."""
        if normalized_mu is not None:
            edge_jump = np.max(normalized_mu) - np.min(normalized_mu)
            edge_position = energy[np.argmax(normalized_mu)]
        else:
            # Estimate from raw data around median energy
            median_energy = np.median(energy)
            edge_region = np.where((energy > median_energy - 50) &
                                 (energy < median_energy + 50))[0]
            if len(edge_region) > 0:
                edge_jump = np.max(mu[edge_region]) - np.min(mu[edge_region])
                edge_position = energy[edge_region[np.argmax(mu[edge_region])]]
            else:
                edge_jump = 0
                edge_position = median_energy

        return float(edge_jump), float(edge_position)

    def _check_saturation(self, mu: np.ndarray) -> float:
        """Check for saturation as ratio of max to mean."""
        max_mu = np.max(mu)
        mean_mu = np.mean(mu)
        return max_mu / mean_mu if mean_mu > 0 else 1.0

    def _estimate_noise_and_snr(self, energy: np.ndarray, mu: np.ndarray) -> Tuple[float, float]:
        """Estimate noise level and signal-to-noise ratio."""
        # Use derivative noise as proxy for high-frequency noise
        noise_level = np.std(np.diff(mu)) / np.mean(np.abs(mu))
        signal_level = np.max(mu) - np.min(mu)
        snr = signal_level / (np.std(mu) + 1e-10)

        return float(noise_level), float(snr)

    def _analyze_edge_regions(self, energy: np.ndarray, normalized_mu: np.ndarray,
                            edge_position: float) -> Tuple[float, float]:
        """Analyze pre-edge and post-edge region stability."""
        # Pre-edge region: 50-20 eV before edge
        pre_edge_mask = (energy >= edge_position - 50) & (energy <= edge_position - 20)
        pre_edge_slope = 0.0
        if np.sum(pre_edge_mask) > 5:
            pre_edge_data = normalized_mu[pre_edge_mask]
            pre_edge_energy = energy[pre_edge_mask]
            coeffs = np.polyfit(pre_edge_energy, pre_edge_data, 1)
            pre_edge_slope = coeffs[0]

        # Post-edge region: 20-100 eV after edge
        post_edge_mask = (energy >= edge_position + 20) & (energy <= edge_position + 100)
        post_edge_slope = 0.0
        if np.sum(post_edge_mask) > 5:
            post_edge_data = normalized_mu[post_edge_mask]
            post_edge_energy = energy[post_edge_mask]
            coeffs = np.polyfit(post_edge_energy, post_edge_data, 1)
            post_edge_slope = coeffs[0]

        return float(pre_edge_slope), float(post_edge_slope)

    def _estimate_white_line(self, normalized_mu: np.ndarray) -> float:
        """Estimate white line intensity."""
        return float(np.max(normalized_mu))

    def _determine_quality_flag(self, metrics: XASSpectrumQualityMetrics,
                               flags: List[str]) -> XASQualityFlag:
        """Determine overall quality flag based on metrics."""
        # Excellent: High edge jump, good SNR, stable edges
        if (metrics.edge_jump > self.thresholds["excellent_edge_jump"] and
            metrics.signal_to_noise > 50 and
            len(flags) == 0):
            return XASQualityFlag.EXCELLENT

        # Good: Decent edge jump, acceptable SNR
        if (metrics.edge_jump > self.thresholds["good_edge_jump"] and
            metrics.signal_to_noise > 20 and
            len([f for f in flags if f not in ["possible_saturation"]]) <= 1):
            return XASQualityFlag.GOOD

        # Acceptable: Basic requirements met
        if (metrics.classification == "usable" and
            metrics.signal_to_noise > self.thresholds["min_snr"]):
            return XASQualityFlag.ACCEPTABLE

        # Poor: Issues present but still usable
        if metrics.classification == "usable_with_warning":
            return XASQualityFlag.POOR

        # Invalid: Severe issues
        return XASQualityFlag.INVALID

    def _generate_warnings(self, flags: List[str]) -> List[str]:
        """Generate human-readable warnings from flags."""
        warnings = []
        flag_descriptions = {
            "insufficient_data_points": "Spectrum has too few data points",
            "invalid_intensity_values": "Spectrum contains invalid intensity values",
            "low_edge_jump": "Edge jump is below recommended threshold",
            "possible_saturation": "Possible detector saturation detected",
            "excessive_noise": "Spectrum has excessive noise",
            "low_signal_to_noise": "Signal-to-noise ratio is too low",
            "unstable_pre_edge": "Pre-edge region shows instability"
        }

        for flag in flags:
            if flag in flag_descriptions:
                warnings.append(flag_descriptions[flag])

        return warnings


class XASQualityReportGenerator:
    """
    Generate CSV reports and diagnostic plots for XAS quality assessment.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize report generator."""
        self.config = config or {}
        self.quality_reports: List[XASSpectrumQualityMetrics] = []

    def add_quality_metrics(self, metrics: XASSpectrumQualityMetrics):
        """Add quality metrics to the report."""
        self.quality_reports.append(metrics)

    def generate_csv_report(self, output_dir: Path, batch_name: str = "xas_quality_report"):
        """
        Generate CSV quality report.

        Parameters
        ----------
        output_dir : Path
            Output directory
        batch_name : str
            Name for the report file
        """
        if not self.quality_reports:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        data = []
        for metrics in self.quality_reports:
            row = metrics.to_dict()
            data.append(row)

        df = pd.DataFrame(data)

        # Save CSV
        csv_path = output_dir / f"{batch_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Quality report saved to: {csv_path}")

        # Generate summary statistics
        self._generate_summary_stats(df, output_dir, batch_name)

    def generate_quality_plots(self, output_dir: Path, batch_name: str = "xas_quality_report"):
        """
        Generate diagnostic quality plots.

        Parameters
        ----------
        output_dir : Path
            Output directory
        batch_name : str
            Name for the plot files
        """
        if not self.quality_reports:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data for plotting
        classifications = [m.classification for m in self.quality_reports]
        confidences = [m.confidence for m in self.quality_reports]
        edge_jumps = [m.edge_jump for m in self.quality_reports]
        snrs = [m.signal_to_noise for m in self.quality_reports]
        sample_ids = [m.sample_id for m in self.quality_reports]

        # Create quality overview plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("XAS Quality Assessment Overview", fontsize=16)

        # Classification distribution
        unique_classes, counts = np.unique(classifications, return_counts=True)
        axes[0, 0].bar(unique_classes, counts, color='skyblue')
        axes[0, 0].set_title('Quality Classification Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Confidence distribution
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Count')

        # Edge jump vs SNR
        scatter = axes[1, 0].scatter(edge_jumps, snrs, c=confidences, cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('Edge Jump vs Signal-to-Noise Ratio')
        axes[1, 0].set_xlabel('Edge Jump')
        axes[1, 0].set_ylabel('SNR')
        plt.colorbar(scatter, ax=axes[1, 0], label='Confidence')

        # Quality flags summary
        quality_flags = [m.quality_flag.value for m in self.quality_reports]
        unique_flags, flag_counts = np.unique(quality_flags, return_counts=True)
        axes[1, 1].pie(flag_counts, labels=unique_flags, autopct='%1.1f%%')
        axes[1, 1].set_title('Quality Flag Distribution')

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"{batch_name}_overview.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Quality overview plot saved to: {plot_path}")

    def _generate_summary_stats(self, df: pd.DataFrame, output_dir: Path, batch_name: str):
        """Generate summary statistics."""
        summary = {
            "total_spectra": len(df),
            "quality_distribution": df['classification'].value_counts().to_dict(),
            "average_confidence": df['confidence'].mean(),
            "average_edge_jump": df['edge_jump'].mean(),
            "average_snr": df['signal_to_noise'].mean(),
            "quality_flag_distribution": df['quality_flag'].value_counts().to_dict(),
            "invalid_spectra": len(df[df['classification'] == 'invalid']),
            "usable_spectra": len(df[df['classification'] == 'usable'])
        }

        # Save summary
        summary_path = output_dir / f"{batch_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("XAS Quality Assessment Summary\n")
            f.write("=" * 40 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        print(f"Quality summary saved to: {summary_path}")


def check_xas_spectrum_quality(energy: np.ndarray,
                              mu: np.ndarray,
                              normalized_mu: Optional[np.ndarray] = None,
                              quality_thresholds: Optional[Dict] = None,
                              sample_id: str = "",
                              file_path: str = "") -> XASSpectrumQualityMetrics:
    """
    Convenience function for XAS spectrum quality check with detailed metrics.

    Parameters
    ----------
    energy : np.ndarray
        Energy values
    mu : np.ndarray
        Absorption coefficient
    normalized_mu : np.ndarray, optional
        Normalized absorption
    quality_thresholds : dict, optional
        Quality thresholds
    sample_id : str
        Sample identifier
    file_path : str
        File path

    Returns
    -------
    metrics : XASSpectrumQualityMetrics
        Detailed quality assessment
    """
    checker = XASSpectrumQualityChecker(quality_thresholds)
    return checker.check_spectrum_quality(energy, mu, normalized_mu, sample_id, file_path)


# Legacy function for backward compatibility
def check_xas_spectrum_quality_basic(energy: np.ndarray,
                                   mu: np.ndarray,
                                   normalized_mu: Optional[np.ndarray] = None,
                                   quality_thresholds: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Legacy function returning basic quality assessment dict.
    """
    metrics = check_xas_spectrum_quality(energy, mu, normalized_mu, quality_thresholds)
    return {
        "classification": metrics.classification,
        "confidence": metrics.confidence,
        "flags": metrics.flags
    }