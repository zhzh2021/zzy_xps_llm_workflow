"""
Region extraction module for XPS data processing.

This module handles extraction of specific binding energy regions from 
full XPS spectra with automatic scan type classification and YAML-driven configuration.
"""

from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
import numpy as np

from core.data_structures import Spectrum


class ScanType(Enum):
    """Types of XPS scans."""
    SURVEY = "survey"
    NARROW = "narrow"
    AUTO = "auto"


class RegionExtractor:
    """Extract XPS regions from full spectra with YAML-driven configuration."""

    def __init__(self, region_definitions: Dict, config: Dict):
        """
        Initialize with region definitions and YAML configuration.

        Args:
            region_definitions: Dictionary of region definitions
            config: Full YAML configuration dictionary
        """
        self.region_defs = region_definitions or {}
        self.config = config or {}

        # Load processing settings from YAML
        proc = self.config.get('processing', {})
        self.min_region_intensity = float(proc.get('min_region_intensity', 10))
        self.energy_padding = float(proc.get('energy_padding', 2.0))
        self.auto_detect_regions_flag = bool(
            proc.get('auto_detect_regions', True))
        self.extract_all_regions_flag = bool(
            proc.get('extract_all_regions', True))
        self.specific_regions = proc.get('specific_regions', None)
        self.common_grid_step = float(proc.get('common_grid_step', 0.05))

        # Load scan classification thresholds (can be added to YAML if needed)
        scan_class = proc.get('scan_classification', {})
        self.survey_energy_threshold = float(
            scan_class.get('survey_energy_threshold', 100))
        self.narrow_energy_threshold = float(
            scan_class.get('narrow_energy_threshold', 50))
        self.narrow_resolution_threshold = float(
            scan_class.get('narrow_resolution_threshold', 4))
        self.hr_resolution_threshold = float(
            scan_class.get('hr_resolution_threshold', 5))

        # Store output flags
        out = self.config.get('output', {})
        self.aggregate_per_region = bool(out.get('aggregate_per_region', True))

    def classify_scan_type(self, spectrum: Spectrum) -> ScanType:
        """
        Classify spectrum as survey or narrow scan using YAML-configured thresholds.

        Survey scans typically:
        - Cover wide energy range (> survey_energy_threshold eV)
        - Have lower resolution (< hr_resolution_threshold points/eV)

        Narrow scans typically:
        - Cover small energy range (< narrow_energy_threshold eV)
        - Have higher resolution (> narrow_resolution_threshold points/eV)
        """
        energy_range = spectrum.energy.max() - spectrum.energy.min()
        n_points = len(spectrum.energy)
        resolution = n_points / energy_range if energy_range > 0 else 0

        if energy_range > self.survey_energy_threshold:
            scan_type = ScanType.SURVEY
            print(f"   📊 Classified as SURVEY scan: "
                  f"range={energy_range:.1f} eV, resolution={resolution:.2f} pts/eV")
        elif energy_range < self.narrow_energy_threshold and resolution > self.narrow_resolution_threshold:
            scan_type = ScanType.NARROW
            print(f"   🔬 Classified as NARROW scan: "
                  f"range={energy_range:.1f} eV, resolution={resolution:.2f} pts/eV")
        else:
            scan_type = ScanType.AUTO
            print(f"   ⚠️ Ambiguous scan type: "
                  f"range={energy_range:.1f} eV, resolution={resolution:.2f} pts/eV")

        return scan_type

    def detect_regions(self,
                       spectrum: Spectrum,
                       min_intensity: Optional[float] = None,
                       scan_type: ScanType = ScanType.AUTO) -> List[str]:
        """
        Detect which regions are present in spectrum.

        Args:
            spectrum: Input spectrum
            min_intensity: Minimum intensity threshold (uses YAML config if None)
            scan_type: Type of scan (auto-detected if AUTO)
        """
        detected = []

        # Use YAML config value if not provided
        min_intensity = self.min_region_intensity if min_intensity is None else float(
            min_intensity)

        # Auto-classify if needed
        if scan_type == ScanType.AUTO:
            scan_type = self.classify_scan_type(spectrum)

        all_zero = np.all(spectrum.intensity == 0)

        for region_name, region_info in self.region_defs.items():
            e_min, e_max = region_info["energy_range"]

            mask = (spectrum.energy >= e_min) & (spectrum.energy <= e_max)

            if not np.any(mask):
                continue

            region_intensity = spectrum.intensity[mask]

            if len(region_intensity) == 0:
                continue

            max_intensity = np.max(region_intensity)
            n_points = np.sum(mask)

            # Different criteria for survey vs narrow scans
            if scan_type == ScanType.SURVEY:
                # For survey: just check if region exists with any signal
                if all_zero:
                    if np.any(mask):
                        detected.append(region_name)
                        print(f"   ✓ Detected {region_name} in survey "
                              f"(⚠️ zero intensity)")
                elif max_intensity > min_intensity:
                    detected.append(region_name)
                    print(f"   ✓ Detected {region_name} in survey: "
                          f"max={max_intensity:.1f}, pts={n_points}")

            elif scan_type == ScanType.NARROW:
                # For narrow: check resolution and intensity
                region_range = e_max - e_min
                resolution = n_points / region_range if region_range > 0 else 0
                
                # Debug output
                print(f"   🔍 Checking {region_name}: range={region_range:.1f}eV, "
                      f"pts={n_points}, res={resolution:.1f}, max_int={max_intensity:.1f}, "
                      f"threshold={min_intensity:.1f}")

                if resolution > self.hr_resolution_threshold and max_intensity > min_intensity:
                    detected.append(region_name)
                    print(f"   ✓ Detected {region_name} (HR): "
                          f"max={max_intensity:.1f}, res={resolution:.1f} pts/eV")
                elif resolution <= self.hr_resolution_threshold:
                    print(f"   ⚠️ {region_name} has low resolution ({resolution:.1f} pts/eV) "
                          f"- may be from survey scan")
                elif max_intensity <= min_intensity:
                    print(f"   ⚠️ {region_name} has low intensity ({max_intensity:.1f} < {min_intensity:.1f})")

        return detected

    def extract_region(self,
                       spectrum: Spectrum,
                       region_name: str,
                       padding: Optional[float] = None,
                       scan_type: ScanType = ScanType.AUTO) -> Optional[Spectrum]:
        """
        Extract a specific region from spectrum.

        Args:
            spectrum: Input spectrum
            region_name: Name of region to extract
            padding: Energy padding in eV (uses YAML config if None)
            scan_type: Type of scan (auto-detected if AUTO)
        """
        if region_name not in self.region_defs:
            print(f"   ⚠️ Region {region_name} not defined")
            return None

        # Use YAML config value if not provided
        padding = self.energy_padding if padding is None else float(padding)

        # Auto-classify if needed
        if scan_type == ScanType.AUTO:
            scan_type = self.classify_scan_type(spectrum)

        region_info = self.region_defs[region_name]
        e_min, e_max = region_info["energy_range"]

        # Adjust padding based on scan type
        if scan_type == ScanType.SURVEY:
            padding = 0.0  # No padding for survey - use exact range

        mask = ((spectrum.energy >= e_min - padding) &
                (spectrum.energy <= e_max + padding))

        region_energy = spectrum.energy[mask]
        region_intensity = spectrum.intensity[mask]

        if len(region_energy) < 5:
            print(f"   ⚠️ Insufficient data for {region_name}")
            return None

        # Validate for narrow scans
        if scan_type == ScanType.NARROW:
            energy_range = region_energy.max() - region_energy.min()
            resolution = len(region_energy) / \
                energy_range if energy_range > 0 else 0

            if resolution < self.hr_resolution_threshold:
                print(
                    f"   ⚠️ {region_name} has low resolution ({resolution:.1f} pts/eV)")
                print(f"      This may be survey data, not suitable for peak fitting")

        try:
            region_spectrum = Spectrum(
                name=f"{spectrum.name}_{region_name}",
                energy=region_energy,
                intensity=region_intensity,
                source_format=spectrum.source_format,
                metadata={
                    **spectrum.metadata,
                    'region': region_name,
                    'original_spectrum': spectrum.name,
                    'scan_type': scan_type.value,
                    'resolution': len(region_energy) / (region_energy.max() - region_energy.min()),
                    'energy_padding_ev': padding
                }
            )
            return region_spectrum
        except Exception as e:
            print(f"   ❌ Failed to create region spectrum: {e}")
            return None

    def extract_all_regions(self,
                            spectrum: Spectrum,
                            auto_detect: Optional[bool] = None,
                            specific_regions: Optional[List[str]] = None,
                            scan_type: ScanType = ScanType.AUTO,
                            purpose: str = "fitting") -> Dict[str, Spectrum]:
        """
       Extract all regions from spectrum.

        Args:
            spectrum: Input spectrum
            auto_detect: Auto-detect regions (uses YAML config if None)
            specific_regions: List of specific regions to extract (uses YAML config if None)
            scan_type: Type of scan (survey/narrow/auto)
            purpose: "fitting" (requires HR) or "quantification" (survey OK)
        """
        extracted = {}

        # Use YAML config values if not provided
        auto_detect = self.auto_detect_regions_flag if auto_detect is None else bool(
            auto_detect)
        specific_regions = self.specific_regions if specific_regions is None else specific_regions

        # Auto-classify scan type
        if scan_type == ScanType.AUTO:
            scan_type = self.classify_scan_type(spectrum)

        # Warn if using survey for fitting
        if purpose == "fitting" and scan_type == ScanType.SURVEY:
            print(f"   ⚠️ WARNING: Survey scan detected but purpose is 'fitting'")
            print(
                f"      Survey scans have low resolution and are NOT suitable for peak fitting")
            print(f"      Use high-resolution narrow scans for fitting")
            print(f"      Survey scans are only suitable for atomic quantification")

        if auto_detect:
            regions_to_extract = self.detect_regions(
                spectrum, scan_type=scan_type)
        else:
            regions_to_extract = specific_regions or list(
                self.region_defs.keys())

        for region_name in regions_to_extract:
            region_spectrum = self.extract_region(
                spectrum, region_name, scan_type=scan_type)
            if region_spectrum:
                extracted[region_name] = region_spectrum

                # Additional info for narrow scans
                if scan_type == ScanType.NARROW:
                    resolution = region_spectrum.metadata.get('resolution', 0)
                    print(f"   📊 Extracted {region_name} (HR): "
                          f"{len(region_spectrum.energy)} points, "
                          f"BE: {region_spectrum.energy.min():.2f}-{region_spectrum.energy.max():.2f} eV, "
                          f"res: {resolution:.1f} pts/eV")
                else:
                    print(f"   📊 Extracted {region_name} (survey): "
                          f"{len(region_spectrum.energy)} points, "
                          f"BE: {region_spectrum.energy.min():.2f}-{region_spectrum.energy.max():.2f} eV")

        return extracted

    def filter_for_fitting(self,
                           extracted_regions: Dict[str, Spectrum],
                           min_resolution: Optional[float] = None) -> Dict[str, Spectrum]:
        """
        Filter extracted regions to only include those suitable for fitting.

        Args:
            extracted_regions: Dictionary of extracted regions
            min_resolution: Minimum resolution (points/eV) required (uses YAML config if None)

        Returns:
            Filtered dictionary with only high-resolution regions
        """
        # Use YAML config value if not provided
        min_resolution = self.narrow_resolution_threshold if min_resolution is None else float(
            min_resolution)

        filtered = {}

        for region_name, spectrum in extracted_regions.items():
            resolution = spectrum.metadata.get('resolution', 0)
            scan_type = spectrum.metadata.get('scan_type', 'unknown')

            if resolution >= min_resolution:
                filtered[region_name] = spectrum
                print(
                    f"   ✓ {region_name} suitable for fitting (res={resolution:.1f} pts/eV)")
            else:
                print(f"   ✗ {region_name} NOT suitable for fitting "
                      f"(res={resolution:.1f} pts/eV, type={scan_type})")

        return filtered