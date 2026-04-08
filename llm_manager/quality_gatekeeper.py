#!/usr/bin/env python3
"""
Unified Quality Gatekeeper Module

Hierarchical quality validation system for XPS data with modality-specific checks.
Designed as a mandatory start node in LangGraph workflow.

Architecture:
- Level 1: Universal fast checks (all data types)
- Level 2: Modality-specific advanced checks (spectrum vs map)
- Integration with AgentState for persistent quality tracking
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# ============================================================================
# Quality Flags and Enums
# ============================================================================

class QualityFlag(Enum):
    """Overall quality assessment flags."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"
    CRITICAL = "critical"  # Requires immediate user intervention


class DataModality(Enum):
    """XPS data modality types."""
    SINGLE_SPECTRUM = "single_spectrum"
    MULTI_SPECTRUM = "multi_spectrum"
    MAP_2D = "map_2d"
    MAP_HYPERSPECTRAL = "map_hyperspectral"
    UNKNOWN = "unknown"


class IssueLevel(Enum):
    """Severity levels for detected issues."""
    CRITICAL = "critical"  # Data cannot be processed
    WARNING = "warning"    # Data can be processed with caution
    INFO = "info"          # Informational note


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QualityIssue:
    """Individual quality issue with severity and description."""
    level: IssueLevel
    category: str  # e.g., "SNR", "Resolution", "Spatial", "Energy"
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    
    def __str__(self) -> str:
        if self.value is not None and self.threshold is not None:
            return f"[{self.level.value.upper()}] {self.category}: {self.message} (value={self.value:.2f}, threshold={self.threshold:.2f})"
        return f"[{self.level.value.upper()}] {self.category}: {self.message}"


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    modality: DataModality
    quality_flag: QualityFlag
    issues: List[QualityIssue] = field(default_factory=list)
    
    # Universal metrics (Level 1)
    snr: float = 0.0
    data_points: int = 0
    energy_range_ev: float = 0.0
    
    # Spectrum-specific metrics (Level 2)
    resolution_pts_per_ev: float = 0.0
    is_hr_scan: bool = False
    suitable_for_fitting: bool = False
    
    # Map-specific metrics (Level 2)
    spatial_dims: Optional[Tuple[int, int]] = None
    dead_pixels: int = 0
    outlier_pixels: int = 0
    spatial_continuity_score: float = 1.0
    
    # Processing recommendations
    recommended_workflow: str = "standard"
    requires_user_attention: bool = False
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return any(issue.level == IssueLevel.CRITICAL for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if report contains warnings."""
        return any(issue.level == IssueLevel.WARNING for issue in self.issues)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for state storage."""
        return {
            'modality': self.modality.value,
            'quality_flag': self.quality_flag.value,
            'snr': self.snr,
            'data_points': self.data_points,
            'energy_range_ev': self.energy_range_ev,
            'resolution_pts_per_ev': self.resolution_pts_per_ev,
            'is_hr_scan': self.is_hr_scan,
            'suitable_for_fitting': self.suitable_for_fitting,
            'spatial_dims': self.spatial_dims,
            'dead_pixels': self.dead_pixels,
            'outlier_pixels': self.outlier_pixels,
            'spatial_continuity_score': self.spatial_continuity_score,
            'recommended_workflow': self.recommended_workflow,
            'requires_user_attention': self.requires_user_attention,
            'critical_issues': [str(issue) for issue in self.issues if issue.level == IssueLevel.CRITICAL],
            'warnings': [str(issue) for issue in self.issues if issue.level == IssueLevel.WARNING],
            'all_issues': [str(issue) for issue in self.issues]
        }
    
    def get_user_alert(self) -> str:
        """Generate user-facing alert message."""
        if self.has_critical_issues:
            critical = [issue for issue in self.issues if issue.level == IssueLevel.CRITICAL]
            return f"🚨 Quality Gate: CRITICAL issues detected ({len(critical)} issues). Data may be corrupted. " + \
                   f"Primary: {critical[0].message}"
        
        if self.quality_flag == QualityFlag.FAILED:
            return f"❌ Quality Gate: FAILED. Data quality too low for reliable processing (SNR: {self.snr:.2f})"
        
        if self.quality_flag == QualityFlag.POOR:
            return f"⚠️ Quality Gate: POOR. Proceed with caution (SNR: {self.snr:.2f}). {len(self.issues)} issues detected."
        
        if self.has_warnings:
            return f"⚠️ Quality Gate: {self.quality_flag.value.upper()}. {len(self.issues)} warnings (SNR: {self.snr:.2f})"
        
        return f"✅ Quality Gate: {self.quality_flag.value.upper()} (SNR: {self.snr:.2f})"


# ============================================================================
# Abstract Validator Base
# ============================================================================

class QualityValidator(ABC):
    """Abstract base for quality validators."""
    
    @abstractmethod
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        """Run validation and return list of issues."""
        pass


# ============================================================================
# Level 0: File-Level Validators (Pre-Import Checks)
# ============================================================================

class FileSizeValidator(QualityValidator):
    """Check file size before import (Stage 1 pre-import check)."""
    
    def validate(self, data: Union[Path, str], config: Dict) -> List[QualityIssue]:
        """
        Validate file size.
        
        Args:
            data: File path (Path or str)
            config: Configuration dict with 'min_file_size_bytes' and 'max_file_size_bytes'
        
        Returns:
            List of QualityIssue
        """
        issues = []
        file_path = Path(data) if isinstance(data, str) else data
        
        min_size = config.get('min_file_size_bytes', 100)  # 100 bytes minimum
        max_size = config.get('max_file_size_bytes', 500_000_000)  # 500 MB max
        
        try:
            if not file_path.exists():
                issues.append(QualityIssue(
                    level=IssueLevel.CRITICAL,
                    category="FileAccess",
                    message=f"File does not exist: {file_path}"
                ))
                return issues
            
            file_size = file_path.stat().st_size
            
            if file_size == 0:
                issues.append(QualityIssue(
                    level=IssueLevel.CRITICAL,
                    category="FileSize",
                    message="Empty file (0 bytes)"
                ))
            elif file_size < min_size:
                issues.append(QualityIssue(
                    level=IssueLevel.CRITICAL,
                    category="FileSize",
                    message=f"File too small ({file_size} bytes)",
                    value=float(file_size),
                    threshold=float(min_size)
                ))
            elif file_size > max_size:
                issues.append(QualityIssue(
                    level=IssueLevel.WARNING,
                    category="FileSize",
                    message=f"File unusually large ({file_size / 1e6:.1f} MB)",
                    value=float(file_size),
                    threshold=float(max_size)
                ))
        except Exception as e:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="FileAccess",
                message=f"Cannot access file: {str(e)}"
            ))
        
        return issues


class FileReadabilityValidator(QualityValidator):
    """Check if file can be opened and read (Stage 1 pre-import check)."""
    
    def validate(self, data: Union[Path, str], config: Dict) -> List[QualityIssue]:
        """
        Validate file readability.
        
        Args:
            data: File path (Path or str)
            config: Configuration dict
        
        Returns:
            List of QualityIssue
        """
        issues = []
        file_path = Path(data) if isinstance(data, str) else data
        
        try:
            # Try binary read first
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
                
                if len(header) == 0:
                    issues.append(QualityIssue(
                        level=IssueLevel.CRITICAL,
                        category="FileCorruption",
                        message="Cannot read file content"
                    ))
        except PermissionError:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="FileAccess",
                message="Permission denied - cannot read file"
            ))
        except Exception as e:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="FileCorruption",
                message=f"File read error: {str(e)}"
            ))
        
        return issues


class FileFormatValidator(QualityValidator):
    """Check file format markers (Stage 1 pre-import check)."""
    
    def validate(self, data: Union[Path, str], config: Dict) -> List[QualityIssue]:
        """
        Validate file format by inspecting header.
        
        Args:
            data: File path (Path or str)
            config: Configuration dict with 'expected_formats' (list of extensions)
        
        Returns:
            List of QualityIssue
        """
        issues = []
        file_path = Path(data) if isinstance(data, str) else data
        
        expected_formats = config.get('expected_formats', ['.spe', '.vgd', '.npl', '.csv', '.xy', '.vms'])
        
        # Check extension
        if file_path.suffix.lower() not in expected_formats:
            issues.append(QualityIssue(
                level=IssueLevel.WARNING,
                category="FileFormat",
                message=f"Unexpected file extension: {file_path.suffix}"
            ))
        
        try:
            # Check for binary corruption indicators
            with open(file_path, 'rb') as f:
                header = f.read(512)
                
                # Check for null-byte corruption (entire header is zeros)
                if len(header) > 0 and header.count(b'\x00') == len(header):
                    issues.append(QualityIssue(
                        level=IssueLevel.CRITICAL,
                        category="FileCorruption",
                        message="File appears corrupted (all null bytes)"
                    ))
                
                # For text formats (.csv, .xy), check if decodable
                if file_path.suffix.lower() in ['.csv', '.xy', '.txt']:
                    try:
                        header.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            header.decode('latin-1')
                        except UnicodeDecodeError:
                            issues.append(QualityIssue(
                                level=IssueLevel.WARNING,
                                category="FileEncoding",
                                message="Text file has unusual encoding"
                            ))
        except Exception as e:
            issues.append(QualityIssue(
                level=IssueLevel.WARNING,
                category="FileFormat",
                message=f"Cannot inspect file format: {str(e)}"
            ))
        
        return issues


# ============================================================================
# Level 1: Universal Validators (Post-Import Fast Checks)
# ============================================================================

class EmptyDataValidator(QualityValidator):
    """Check if data is empty or has insufficient points."""
    
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        issues = []
        min_points = config.get('min_data_points', 10)
        
        if hasattr(data, 'intensity'):
            n_points = len(data.intensity) if hasattr(data.intensity, '__len__') else 0
        elif hasattr(data, 'shape'):
            n_points = np.prod(data.shape)
        else:
            n_points = 0
        
        if n_points == 0:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="Data",
                message="Empty dataset - no data points",
                value=0,
                threshold=min_points
            ))
        elif n_points < min_points:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="Data",
                message=f"Insufficient data points",
                value=n_points,
                threshold=min_points
            ))
        
        return issues


class EnergyAxisValidator(QualityValidator):
    """Validate energy axis integrity."""
    
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        issues = []
        min_range = config.get('min_energy_range', 5.0)
        
        if not hasattr(data, 'energy'):
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="Energy",
                message="No energy axis found"
            ))
            return issues
        
        energy = data.energy
        if len(energy) == 0:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="Energy",
                message="Empty energy axis"
            ))
            return issues
        
        # Check range
        energy_range = float(np.max(energy) - np.min(energy))
        if energy_range < min_range:
            issues.append(QualityIssue(
                level=IssueLevel.WARNING,
                category="Energy",
                message="Narrow energy range",
                value=energy_range,
                threshold=min_range
            ))
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(energy)):
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="Energy",
                message="Energy axis contains NaN or Inf values"
            ))
        
        return issues


class SNRValidator(QualityValidator):
    """Universal SNR check."""
    
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        issues = []
        
        # Extract intensity data
        if hasattr(data, 'intensity'):
            intensity = data.intensity
        elif hasattr(data, 'mean_spectrum'):
            intensity = data.mean_spectrum
        else:
            return [QualityIssue(
                level=IssueLevel.CRITICAL,
                category="SNR",
                message="Cannot extract intensity data for SNR calculation"
            )]
        
        # Calculate SNR
        mean_intensity = float(np.mean(intensity))
        std_intensity = float(np.std(intensity))
        snr = mean_intensity / (std_intensity + 1e-10) if std_intensity > 0 else 0.0
        
        # Check thresholds
        critical_snr = config.get('critical_snr_threshold', 1.0)
        acceptable_snr = config.get('min_snr_acceptable', 3.0)
        
        if snr < critical_snr:
            issues.append(QualityIssue(
                level=IssueLevel.CRITICAL,
                category="SNR",
                message="Critical low SNR - data unusable",
                value=snr,
                threshold=critical_snr
            ))
        elif snr < acceptable_snr:
            issues.append(QualityIssue(
                level=IssueLevel.WARNING,
                category="SNR",
                message="Low SNR - proceed with caution",
                value=snr,
                threshold=acceptable_snr
            ))
        
        return issues


# ============================================================================
# Level 2: Spectrum-Specific Validators
# ============================================================================

class ResolutionValidator(QualityValidator):
    """Check spectrum resolution."""
    
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        issues = []
        hr_threshold = config.get('hr_resolution_threshold', 5.0)
        
        if not hasattr(data, 'energy') or not hasattr(data, 'intensity'):
            return issues
        
        energy_range = float(np.max(data.energy) - np.min(data.energy))
        n_points = len(data.energy)
        resolution = n_points / energy_range if energy_range > 0 else 0.0
        
        if resolution < hr_threshold:
            issues.append(QualityIssue(
                level=IssueLevel.INFO,
                category="Resolution",
                message="Low resolution - survey scan",
                value=resolution,
                threshold=hr_threshold
            ))
        
        return issues


class PeakQualityValidator(QualityValidator):
    """Assess peak quality for fitting suitability."""
    
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        issues = []
        
        if not hasattr(data, 'intensity'):
            return issues
        
        # Check for flat baseline
        intensity = data.intensity
        dynamic_range = float(np.max(intensity) - np.min(intensity))
        mean_intensity = float(np.mean(intensity))
        
        if dynamic_range / (mean_intensity + 1e-10) < 0.1:
            issues.append(QualityIssue(
                level=IssueLevel.WARNING,
                category="Peak",
                message="Flat spectrum - no clear peaks detected"
            ))
        
        return issues


# ============================================================================
# Level 2: Map-Specific Validators
# ============================================================================

class SpatialDimensionValidator(QualityValidator):
    """Validate map spatial dimensions."""
    
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        issues = []
        
        if not hasattr(data, 'nx') or not hasattr(data, 'ny'):
            return issues
        
        nx, ny = data.nx, data.ny
        min_dim = config.get('min_map_dimension', 2)
        
        if nx < min_dim or ny < min_dim:
            issues.append(QualityIssue(
                level=IssueLevel.WARNING,
                category="Spatial",
                message=f"Small map dimensions: {nx}x{ny}",
                value=min(nx, ny),
                threshold=min_dim
            ))
        
        return issues


class DeadPixelValidator(QualityValidator):
    """Detect dead/missing pixels in map."""
    
    def validate(self, data: Any, config: Dict) -> List[QualityIssue]:
        issues = []
        max_dead_fraction = config.get('max_dead_pixel_fraction', 0.05)
        
        if not hasattr(data, 'intensity_map'):
            return issues
        
        intensity_map = data.intensity_map
        dead_pixels = np.sum(intensity_map == 0) + np.sum(~np.isfinite(intensity_map))
        total_pixels = intensity_map.size
        dead_fraction = dead_pixels / total_pixels if total_pixels > 0 else 0
        
        if dead_fraction > max_dead_fraction:
            issues.append(QualityIssue(
                level=IssueLevel.WARNING,
                category="Spatial",
                message=f"High dead pixel count: {dead_fraction:.1%}",
                value=dead_fraction,
                threshold=max_dead_fraction
            ))
        
        return issues


# ============================================================================
# Unified Quality Gatekeeper
# ============================================================================

class UnifiedQualityGatekeeper:
    """
    Hierarchical quality gatekeeper for XPS data.
    
    Implements two-level validation:
    1. Universal fast checks (all data types)
    2. Modality-specific advanced checks
    
    Designed for use as mandatory start node in LangGraph workflow.
    """
    
    def __init__(self, config: Dict = None, debug: bool = False):
        """
        Initialize gatekeeper with configuration.
        
        Args:
            config: Configuration dictionary with thresholds
            debug: Enable debug output
        """
        self.debug = debug
        self.config = config or self._default_config()
        
        # Level 1: Universal validators (fast, apply to all)
        self.universal_validators = [
            EmptyDataValidator(),
            EnergyAxisValidator(),
            SNRValidator(),
        ]
        
        # Level 2: Modality-specific validators
        self.spectrum_validators = [
            ResolutionValidator(),
            PeakQualityValidator(),
        ]
        
        self.map_validators = [
            SpatialDimensionValidator(),
            DeadPixelValidator(),
        ]
        
        # Level 0: File-level validators (pre-import)
        self.file_validators = [
            FileSizeValidator(),
            FileReadabilityValidator(),
            FileFormatValidator(),
        ]
    
    def validate_file(self, file_path: Union[Path, str]) -> Tuple[bool, List[QualityIssue]]:
        """
        Stage 1: Lightweight file-level validation BEFORE import.
        
        Fast checks that don't require parsing the full file:
        - File exists and is readable
        - File size is reasonable
        - File format markers are valid
        - No obvious corruption
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Tuple of (should_import: bool, issues: List[QualityIssue])
        """
        issues = []
        
        # Run all file-level validators
        for validator in self.file_validators:
            issues.extend(validator.validate(file_path, self.config))
        
        # Check for critical issues
        has_critical = any(issue.level == IssueLevel.CRITICAL for issue in issues)
        
        if self.debug and issues:
            print(f"[QualityGate] File validation for {file_path}:")
            for issue in issues:
                print(f"  {issue}")
        
        return (not has_critical, issues)
    
    def validate(self, data: Any, modality: DataModality = None) -> QualityReport:
        """
        Validate data and generate comprehensive quality report.
        
        Args:
            data: XPS data object (spectrum or map)
            modality: Data modality (auto-detected if None)
            
        Returns:
            QualityReport with all validation results
        """
        # Detect modality if not provided
        if modality is None:
            modality = self._detect_modality(data)
        
        # Level 1: Universal checks (all data types)
        issues = []
        for validator in self.universal_validators:
            issues.extend(validator.validate(data, self.config))
        
        # Early termination if critical issues found
        has_critical = any(issue.level == IssueLevel.CRITICAL for issue in issues)
        if has_critical:
            return self._create_critical_report(data, modality, issues)
        
        # Level 2: Modality-specific checks
        if modality in [DataModality.SINGLE_SPECTRUM, DataModality.MULTI_SPECTRUM]:
            for validator in self.spectrum_validators:
                issues.extend(validator.validate(data, self.config))
        elif modality in [DataModality.MAP_2D, DataModality.MAP_HYPERSPECTRAL]:
            for validator in self.map_validators:
                issues.extend(validator.validate(data, self.config))
        
        # Generate comprehensive report
        return self._create_report(data, modality, issues)
    
    def _detect_modality(self, data: Any) -> DataModality:
        """Auto-detect data modality."""
        if hasattr(data, 'nx') and hasattr(data, 'ny'):
            if hasattr(data, 'energy_points') and data.energy_points > 10:
                return DataModality.MAP_HYPERSPECTRAL
            return DataModality.MAP_2D
        
        if hasattr(data, 'intensity'):
            return DataModality.SINGLE_SPECTRUM
        
        return DataModality.UNKNOWN
    
    def _create_critical_report(self, data: Any, modality: DataModality, 
                               issues: List[QualityIssue]) -> QualityReport:
        """Create report for critical failures."""
        return QualityReport(
            modality=modality,
            quality_flag=QualityFlag.CRITICAL,
            issues=issues,
            recommended_workflow="manual_inspection",
            requires_user_attention=True
        )
    
    def _create_report(self, data: Any, modality: DataModality, 
                      issues: List[QualityIssue]) -> QualityReport:
        """Create comprehensive quality report."""
        # Calculate universal metrics
        snr = self._calculate_snr(data)
        data_points = self._get_data_points(data)
        energy_range = self._get_energy_range(data)
        
        # Determine overall quality flag
        quality_flag = self._determine_quality_flag(snr, issues)
        
        # Spectrum-specific metrics
        resolution = 0.0
        is_hr = False
        suitable_for_fitting = False
        
        if modality in [DataModality.SINGLE_SPECTRUM, DataModality.MULTI_SPECTRUM]:
            resolution = self._calculate_resolution(data)
            is_hr = resolution >= self.config.get('hr_resolution_threshold', 5.0)
            suitable_for_fitting = is_hr and quality_flag not in [QualityFlag.FAILED, QualityFlag.POOR]
        
        # Map-specific metrics
        spatial_dims = None
        dead_pixels = 0
        outlier_pixels = 0
        spatial_continuity = 1.0
        
        if modality in [DataModality.MAP_2D, DataModality.MAP_HYPERSPECTRAL]:
            spatial_dims = (getattr(data, 'nx', 0), getattr(data, 'ny', 0))
            if hasattr(data, 'intensity_map'):
                dead_pixels = int(np.sum(data.intensity_map == 0))
        
        # Determine workflow
        if modality in [DataModality.MAP_2D, DataModality.MAP_HYPERSPECTRAL]:
            recommended_workflow = "map"
        else:
            recommended_workflow = "standard"
        
        return QualityReport(
            modality=modality,
            quality_flag=quality_flag,
            issues=issues,
            snr=snr,
            data_points=data_points,
            energy_range_ev=energy_range,
            resolution_pts_per_ev=resolution,
            is_hr_scan=is_hr,
            suitable_for_fitting=suitable_for_fitting,
            spatial_dims=spatial_dims,
            dead_pixels=dead_pixels,
            outlier_pixels=outlier_pixels,
            spatial_continuity_score=spatial_continuity,
            recommended_workflow=recommended_workflow,
            requires_user_attention=quality_flag in [QualityFlag.FAILED, QualityFlag.POOR]
        )
    
    def _calculate_snr(self, data: Any) -> float:
        """Calculate signal-to-noise ratio."""
        if hasattr(data, 'intensity'):
            intensity = data.intensity
        elif hasattr(data, 'mean_spectrum'):
            intensity = data.mean_spectrum
        else:
            return 0.0
        
        mean = float(np.mean(intensity))
        std = float(np.std(intensity))
        return mean / (std + 1e-10) if std > 0 else 0.0
    
    def _get_data_points(self, data: Any) -> int:
        """Get total data points."""
        if hasattr(data, 'intensity'):
            return len(data.intensity)
        elif hasattr(data, 'shape'):
            return int(np.prod(data.shape))
        return 0
    
    def _get_energy_range(self, data: Any) -> float:
        """Get energy axis range."""
        if hasattr(data, 'energy'):
            return float(np.max(data.energy) - np.min(data.energy))
        return 0.0
    
    def _calculate_resolution(self, data: Any) -> float:
        """Calculate spectral resolution."""
        if not hasattr(data, 'energy') or not hasattr(data, 'intensity'):
            return 0.0
        
        energy_range = float(np.max(data.energy) - np.min(data.energy))
        n_points = len(data.energy)
        return n_points / energy_range if energy_range > 0 else 0.0
    
    def _determine_quality_flag(self, snr: float, issues: List[QualityIssue]) -> QualityFlag:
        """Determine overall quality flag based on SNR and issues."""
        # Critical issues override everything
        if any(issue.level == IssueLevel.CRITICAL for issue in issues):
            return QualityFlag.CRITICAL
        
        # SNR-based classification
        excellent_snr = self.config.get('min_snr_excellent', 10.0)
        good_snr = self.config.get('min_snr_good', 5.0)
        acceptable_snr = self.config.get('min_snr_acceptable', 3.0)
        poor_snr = self.config.get('min_snr_poor', 1.0)
        
        warnings = [issue for issue in issues if issue.level == IssueLevel.WARNING]
        
        if snr >= excellent_snr and not warnings:
            return QualityFlag.EXCELLENT
        elif snr >= good_snr and len(warnings) <= 1:
            return QualityFlag.GOOD
        elif snr >= acceptable_snr and len(warnings) <= 2:
            return QualityFlag.ACCEPTABLE
        elif snr >= poor_snr:
            return QualityFlag.POOR
        else:
            return QualityFlag.FAILED
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            # File-level thresholds (Stage 1 pre-import)
            'min_file_size_bytes': 100,
            'max_file_size_bytes': 500_000_000,  # 500 MB
            'expected_formats': ['.spe', '.vgd', '.npl', '.csv', '.xy', '.vms'],
            
            # Universal thresholds
            'min_data_points': 10,
            'min_energy_range': 5.0,
            'critical_snr_threshold': 1.0,
            'min_snr_acceptable': 3.0,
            'min_snr_good': 5.0,
            'min_snr_excellent': 10.0,
            'min_snr_poor': 1.0,
            
            # Spectrum thresholds
            'hr_resolution_threshold': 5.0,
            'min_points': 20,
            'max_relative_noise': 0.3,
            
            # Map thresholds
            'min_map_dimension': 2,
            'max_dead_pixel_fraction': 0.05,
        }


# ============================================================================
# LangGraph Integration Functions
# ============================================================================

def quality_gate_node(state: Dict) -> Dict:
    """
    LangGraph node function for quality gatekeeper.
    
    Args:
        state: AgentState dictionary
        
    Returns:
        Updated state with quality report
    """
    from pathlib import Path
    
    # Get current file from state
    current_file = state.get('current_file') or state.get('file_path')
    if not current_file:
        state['user_alerts'] = state.get('user_alerts', [])
        state['user_alerts'].append("⚠️ Quality Gate: No file specified for validation")
        return state
    
    # Import data (simplified - actual implementation would load spectrum/map)
    # For now, just update state with placeholder
    gatekeeper = UnifiedQualityGatekeeper()
    
    # Placeholder validation (actual implementation needs data loading)
    state['quality_gate_passed'] = True
    state['quality_report'] = {
        'status': 'pending',
        'message': 'Quality validation ready'
    }
    
    return state


def should_process_data(state: Dict) -> str:
    """
    LangGraph conditional edge based on quality gate.
    
    Args:
        state: AgentState dictionary
        
    Returns:
        "process" if passed, "reject" if failed
    """
    if state.get('quality_gate_passed', False):
        return "process"
    return "reject"


if __name__ == "__main__":
    print("Unified Quality Gatekeeper Module")
    print("=" * 60)
    print("\nArchitecture:")
    print("  Level 1: Universal validators (fast, all data)")
    print("  Level 2: Modality-specific validators")
    print("\nThresholds:")
    gatekeeper = UnifiedQualityGatekeeper()
    for key, value in gatekeeper.config.items():
        print(f"  {key}: {value}")
