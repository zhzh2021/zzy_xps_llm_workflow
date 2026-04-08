"""
XPS Raw Data Converter - V5 (Modular)
Converts raw XPS instrument files (.spe, .vgd, .npl, .csv, .vms etc.) to standardized CSV files.

Features:
- Multi-format support (PHI SPE, Thermo Fisher VGD, Kratos NPL, CASA VMS, ASCII)
- Automatic Multi-format detection: Automatically detects multi-region vs single-region files
- Batch processing, calibration, and export support with modular architecture.

"""

# Fix Windows console encoding for emoji support
import sys
import io
import os
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        pass  

# Core imports
from parsers import AVAILABLE_PARSERS, BaseParser
from core.data_structures import Spectrum, XPSMetadata

# Modular components
from calibration import EnergyCalibrator, CalibrationResult
from region_extraction import RegionExtractor, ScanType
from spectrum_import import SpectrumImporter, FormatDetector, FileFormat
from csv_export import CSVExporter
from spectrum_smooth import load_smoothing_settings, smooth_spectra
from spectrum_quality import SpectrumQualityAnalyzer  # Stage 2 quality assessment

# Quality plot utilities
try:
    from XPS_Plotter.plot_modules.data_quality.quality_plots import (
        plot_all_extracted_regions_quality,
    )
except ImportError:
    plot_all_extracted_regions_quality = None

# Depth profile visualization (3D waterfall)
try:
    from XPS_Plotter.plot_modules.quantification.depth_3d_waterfall import (
        load_depth_profile_csv,
        plot_depth_profile_3d_waterfall,
    )
except ImportError:
    load_depth_profile_csv = None
    plot_depth_profile_3d_waterfall = None

# Import unified modules from llm_manager (shared across tools)
import sys
from pathlib import Path
_llm_manager_path = Path(__file__).resolve().parents[2] / "llm_manager"
if str(_llm_manager_path) not in sys.path:
    sys.path.insert(0, str(_llm_manager_path))

try:
    from quality_gatekeeper import UnifiedQualityGatekeeper, QualityFlag, DataModality
    from enhanced_triage_fixed import EnhancedXPSDataTriage, XPSDataType
    # Alias for backward compatibility
    QualityGatekeeper = UnifiedQualityGatekeeper
except ImportError as e:
    print(f"⚠️ Warning: Could not import from llm_manager: {e}")
    # Fallback to local version if unified not available
    from quality_gatekeeper import QualityGatekeeper, QualityFlag
    EnhancedXPSDataTriage = None
    XPSDataType = None

# Standard library imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import re
import traceback
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.signal import find_peaks, savgol_filter
from collections import defaultdict
import warnings
import csv
from datetime import datetime
from types import SimpleNamespace

# Add Tools directory to path for utils import
tools_dir = Path(__file__).resolve().parents[1]
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

try:
    from tool_utils import Timer, TimingStats, load_yaml_settings, log, read_txt_file, list_files, clean_data, DATA_PATH, RESULTS_PATH
except ImportError:
    print("⚠️  Could not import from tool_utils, using fallback implementations")
    # Fallback implementations
    class Timer:
        def __init__(self, name): 
            self.name = name
        def __enter__(self): 
            return self
        def __exit__(self, *args): 
            pass
    
    class TimingStats:
        def __init__(self, name): 
            pass
        def time(self): 
            return Timer("op")
        def print_summary(self): 
            pass
    
    def load_yaml_settings(path): 
        import yaml
        with open(path, 'r') as f: 
            return yaml.safe_load(f)
    def log(msg): 
        print(f"[LOG] {msg}")
    def read_txt_file(path): 
        with open(path, 'r') as f: 
            return f.readlines()
    def list_files(dir, ext): 
        return list(Path(dir).glob(f"*{ext}"))
    def clean_data(df): 
        return df.dropna().reset_index(drop=True)
    DATA_PATH = "data"
    RESULTS_PATH = "results"

# Ensure project imports resolve when running this file directly
CURRENT_DIR = Path(__file__).resolve()
MODULE_ROOT = CURRENT_DIR.parent
PACKAGE_ROOT = CURRENT_DIR.parents[2]
PROJECT_ROOT = CURRENT_DIR.parents[3]
for extra_path in (PROJECT_ROOT, PACKAGE_ROOT, MODULE_ROOT):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

# ========== PROCESSING LOGGER ==========

class ProcessingLogger:
    """Manages processing log file with detailed operation tracking."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.entries = []
        self.start_time = datetime.now()
        
    def log(self, message: str, level: str = "INFO"):
        """Add log entry with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.entries.append(entry)
        print(message)  # Also print to console
        
    def log_header(self, title: str):
        """Log section header."""
        separator = "=" * 70
        self.entries.append("\n" + separator)
        self.entries.append(title)
        self.entries.append(separator)
        
    def log_summary_table(self, data: List[Dict], headers: List[str]):
        """Log table of summary data."""
        if not data:
            return
        
        # Calculate column widths
        widths = {h: len(h) for h in headers}
        for row in data:
            for h in headers:
                widths[h] = max(widths[h], len(str(row.get(h, ''))))
        
        # Format header
        header_line = " | ".join(h.ljust(widths[h]) for h in headers)
        separator = "-+-".join("-" * widths[h] for h in headers)
        
        self.entries.append("\n" + header_line)
        self.entries.append(separator)
        
        # Format rows
        for row in data:
            row_line = " | ".join(str(row.get(h, '')).ljust(widths[h]) for h in headers)
            self.entries.append(row_line)
    
    def write(self):
        """Write all log entries to file."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Add footer
        self.entries.append("\n" + "=" * 70)
        self.entries.append(f"Processing completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.entries.append(f"Total duration: {duration}")
        self.entries.append("=" * 70)
        
        # Write to file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.entries))
        
        print(f"\n📝 Processing log saved: {self.log_file}")

# ========== XPS WORKFLOW INTEGRATION ==========

class UnifiedParser:
    """Unified parser that wraps all available parsers."""
    
    def __init__(self):
        self.parsers = AVAILABLE_PARSERS
            
    def find_proper_parser(self, file_path: Path) -> Optional[BaseParser]:
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
            
        raise RuntimeError(f"No parser found for file: {file_path}")
    
    def parse_file(self, file_path: Path) -> List[Spectrum]:
        """Parse file using appropriate parser."""
        parser = self.find_proper_parser(file_path)
        
        try:
            return parser.parse(file_path)
        except Exception as e:
            log(f"Error parsing file {file_path} with {parser.format_name}: {e}")
            raise


# ========== CONFIGURATION ==========

def _pick_config_file() -> Path:
    """Choose the best-available config path relative to this file."""
    candidates = [
        CURRENT_DIR.parents[2] / "project_root" / "xps_config" / "project_setting.yaml",
        CURRENT_DIR.parents[2] / "xps_config" / "project_setting.yaml",
        Path("../../project_root/xps_config/project_setting.yaml"),
        Path("../../xps_config/project_setting.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fall back to the first candidate so the error message points to the expected location
    return candidates[0]


# Constants for file paths - can be overridden by YAML config
RAW_DATA_DIR = CURRENT_DIR.parents[2] / "project_root" / "00_raw_data"
OUTPUT_DIR = CURRENT_DIR.parents[2] / "project_root" / "01_converted_csv"
CONFIG_FILE = _pick_config_file()
RAW_FILE_PATTERNS = ["*.spe", "*.vgd", "*.npl", "*.xy", "*.txt", "*.asc", "*.dat", "*.csv"]

def resolve_config_paths(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Resolve configuration file paths relative to base directory."""
    config_dir = CONFIG_FILE.parent
    
    project_config = config_dir / "project_setting.yaml"
    region_defs = config_dir / "region_definitions.yaml" 
    
    return project_config, region_defs

def resolve_region_definitions(config: Dict) -> Dict[str, Dict]:
    """Resolve region definitions from config."""
    region_defs = config.get('regions', {})
    return region_defs

# ========== CONFIGURATION LOADER ==========

def load_xps_config(config_file: Path) -> Tuple[Dict, Dict]:
    """Load YAML configuration and region definitions."""
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError("Configuration file is empty or invalid")
    
    # Load region definitions from config
    region_definitions = resolve_region_definitions(config)
    
    if not region_definitions:
        print("⚠️  No region definitions found in config")
        region_definitions = {}
    
    return config, region_definitions

# ========== BATCH PROCESSING ==========

class BatchConverter:
    """Batch converter with config-driven settings and energy calibration."""

    def __init__(self, region_definitions: Dict, config: Dict, debug: bool = False, logger: ProcessingLogger = None):
        """Initialize converter with config."""
        self.config = config
        self.debug = debug
        self.logger = logger

        # Initialize components with config
        self.importer = SpectrumImporter(debug=debug)
        self.extractor = RegionExtractor(region_definitions, config)
        self.calibrator = EnergyCalibrator(config)
        
        # Stage 1: File-level quality checks (pre-import)
        self.gatekeeper = UnifiedQualityGatekeeper(config=config, debug=debug)
        
        # Stage 2: Detailed spectrum quality checks (post-import)
        quality_config = config.get('quality_assessment', {})
        self.quality_analyzer = SpectrumQualityAnalyzer(config=quality_config, debug=debug)
        
        # Initialize triage for map detection (optional - may be None if not available)
        try:
            self.triage = EnhancedXPSDataTriage()
        except Exception as e:
            if debug:
                print(f"Warning: Could not initialize triage: {e}")
            self.triage = None
        
        output_cfg = config.get('output', {})
        self.exporter = CSVExporter(
            output_dir=Path("temp"),
            include_metadata=output_cfg.get('include_metadata', True),
            decimal_places=output_cfg.get('decimal_places', 3),
        )
        self.region_defs = region_definitions

        # Load settings from config
        self.proc = config.get('processing', {})
        self.output_config = output_cfg
        self.smoothing_settings = load_smoothing_settings(config)
        project_info = config.get('project_info', {})
        configured_project_root = project_info.get('project_root')
        if configured_project_root:
            candidate_root = Path(configured_project_root).expanduser()
            if candidate_root.exists():
                self.project_root = candidate_root
            else:
                self.project_root = PROJECT_ROOT
        else:
            self.project_root = PROJECT_ROOT
        self.generate_quality_plots = bool(self.proc.get('generate_quality_plots', False))

        # Step 1 plot output (timestamped per run)
        self._run_timestamp = None
        self._step1_plots_root = None
        self._step1_quality_plots_dir = None
        self._step1_depth_plots_dir = None

        # Extract specific settings
        self.aggregate_per_region = self.output_config.get('aggregate_per_region', True)
        self.common_grid_step = self.proc.get('common_grid_step', 0.1)

        # File format settings
        file_cfg = config.get('file_formats', {})
        self.supported_extensions = file_cfg.get(
            'supported_extensions',
            ['.spe', '.vgd', '.npl', '.xy', '.txt', '.asc', '.dat', '.csv', '.vms', '.vamas']
        )

        # Calibration settings
        cal_cfg = config.get('energy_calibration', {})
        self.calibration_enabled = cal_cfg.get('enable', True)
        self.map_confidence_threshold = 0.7

    @property
    def hr_resolution_threshold(self):
        """Expose HR resolution threshold from exporter for reporting."""
        return getattr(self.exporter, 'hr_resolution_threshold', None)

    def _import_and_extract_regions(self, raw_file: Path,
                                     auto_detect: bool = True,
                                     specific_regions: List[str] = None) -> Dict[str, List]:
        """Import file and extract regions WITHOUT saving - for aggregation mode.
        
        Note: Caller should filter map files using triage BEFORE calling this method.
        
        Returns:
            Dict[region_name, List[Spectrum]]: Extracted spectra grouped by region
        """
        # Import spectra
        try:
            spectra = self.importer.import_file(raw_file)
            if not spectra:
                print(f"Warning: File rejected: {raw_file.name} (corrupted or incompatible)")
                return {}
        except Exception as e:
            print(f"Import failed: {e}")
            return {}

        # Quality check
        quality_results = {}
        for spec in spectra:
            try:
                report = self.gatekeeper.validate(spec, modality=DataModality.SINGLE_SPECTRUM)
                quality_results[spec.name] = report
            except Exception as e:
                if self.debug:
                    print(f"Quality check failed for {spec.name}: {e}")
        
        poor_quality = [name for name, report in quality_results.items() 
                       if report.quality_flag in [QualityFlag.FAILED, QualityFlag.POOR]]
        if poor_quality:
            print(f"⚠️  Note: {len(poor_quality)} spectrum(a) with quality concerns (continuing as approved)")

        # Apply calibration
        calibration_result = None
        if self.calibration_enabled:
            print("\nApplying energy calibration...")
            spectra, calibration_result = self.calibrator.calibrate_spectra(
                spectra,
                self.region_defs,
                reference_region_name=None  # Uses config default
            )

        spectra = self._apply_optional_smoothing(spectra)

        # Extract regions
        print("\nExtracting regions...")
        all_regions = {}
        for spectrum in spectra:
            scan_type = self.extractor.classify_scan_type(spectrum)
            if scan_type == ScanType.SURVEY:
                metadata = getattr(spectrum, "metadata", {}) or {}
                metadata["survey_file_stem"] = raw_file.stem
                spectrum.metadata = metadata
                all_regions.setdefault("survey", []).append(spectrum)
                continue

            regions = self.extractor.extract_all_regions(
                spectrum, auto_detect, specific_regions, scan_type=scan_type
            )
            for region_name, region_spectrum in regions.items():
                all_regions.setdefault(region_name, []).append(region_spectrum)

        if not all_regions:
            print("No regions extracted")
            return {}

        return all_regions

    def convert_file(self, raw_file: Path, output_dir: Path,
                     auto_detect: bool = None,
                     specific_regions: List[str] = None) -> Dict[str, Path]:
        """Convert a single standard-spectrum file to CSV outputs."""

        if auto_detect is None:
            auto_detect = self.proc.get('auto_detect_regions', True)
        if specific_regions is None:
            specific_regions = self.proc.get('specific_regions', [])

        triage_result = None
        if self.triage:
            try:
                triage_result = self.triage.analyze_file_structure(raw_file)
            except Exception as exc:
                if self.debug:
                    print(f"[AI] Triage analysis failed for {raw_file.name}: {exc}")

        if self._should_skip_map_file(triage_result):
            dtype = triage_result.get('data_type', 'map') if triage_result else 'map'
            dtype_value = getattr(dtype, 'value', str(dtype))
            confidence = triage_result.get('confidence', 0.0) if triage_result else 0.0
            print(f"[AI] Skipping map file: {dtype_value} (confidence: {confidence:.1%})")
            print("   🗺️ Map files should be processed via XPS_Mapper workflow")
            return {}

        return self._convert_standard_file(
            raw_file=raw_file,
            output_dir=output_dir,
            auto_detect=auto_detect,
            specific_regions=specific_regions
        )

    def _should_skip_map_file(self, triage_result: Dict, confidence_threshold: float = None) -> bool:
        """Return True when a triage result indicates that the file is map data."""
        if not triage_result:
            return False
        if triage_result.get('should_route_to_mapper'):
            return True
        confidence_threshold = confidence_threshold or self.map_confidence_threshold
        confidence = triage_result.get('confidence', 0.0)
        if confidence < confidence_threshold:
            return False
        data_type = triage_result.get('data_type')
        dtype_value = getattr(data_type, 'value', data_type)
        map_labels = {"map_2d", "map_hyperspectral"}
        if XPSDataType is not None:
            map_labels.update({XPSDataType.MAP_2D.value, XPSDataType.MAP_HYPERSPECTRAL.value})
        if dtype_value in map_labels:
            return True
        return triage_result.get('recommended_processor') == "XPS_mapper"

    def _apply_optional_smoothing(self, spectra: List[Spectrum]) -> List[Spectrum]:
        """Apply Savitzky-Golay smoothing when enabled in the config."""
        if not spectra or not self.smoothing_settings.enable:
            return spectra

        print("\nApplying smoothing filter...")
        smooth_spectra(spectra, self.smoothing_settings)
        return spectra

    def _generate_quality_plots(self, converted_csv_dir: Path):
        """Generate quality plots for extracted regions if enabled."""
        if not self.generate_quality_plots:
            return
        if plot_all_extracted_regions_quality is None:
            if self.debug:
                print("⚠️  Quality plot module not available")
            return

        try:
            plot_all_extracted_regions_quality(
                converted_csv_dir,
                project_root=self.project_root
            )
        except Exception as exc:
            print(f"⚠️  Quality plot generation failed: {exc}")

    def _init_step1_plot_dirs(self):
        """Initialize timestamped Step 1 plot directories for this run."""
        if self._step1_plots_root is not None:
            return
        def _run_tag(run_id_val: str) -> str:
            run_id_val = (run_id_val or "").strip()
            if run_id_val:
                if "_" in run_id_val:
                    date_part, time_part = run_id_val.split("_", 1)
                    date_part = re.sub(r"\D", "", date_part)
                    time_part = re.sub(r"\D", "", time_part)
                    if len(date_part) >= 8:
                        date_part = date_part[:8]
                    else:
                        date_part = datetime.now().strftime("%Y%m%d")
                    if len(time_part) >= 6:
                        time_part = time_part[:6]
                    else:
                        time_part = datetime.now().strftime("%H%M%S")
                    return f"{date_part}{time_part}"
                digits = re.sub(r"\D", "", run_id_val)
                if len(digits) >= 14:
                    return digits[:14]
                if len(digits) >= 8:
                    return f"{digits[:8]}{datetime.now().strftime('%H%M%S')}"
            return datetime.now().strftime("%Y%m%d%H%M%S")

        run_tag = _run_tag(os.environ.get("XPS_RUN_ID", ""))
        self._run_timestamp = datetime.now()
        self._step1_plots_root = (
            self.project_root
            / "04_plots"
            / "01_converted_csv"
            / run_tag
        )
        self._step1_quality_plots_dir = self._step1_plots_root / "quality"
        self._step1_depth_plots_dir = self._step1_plots_root / "depth_profile_3d"

    def _generate_depth_profile_waterfall(self, csv_path: Path, region_name: str) -> Optional[Path]:
        """Generate 3D waterfall plot for depth profiles when multiple layers exist."""
        if load_depth_profile_csv is None or plot_depth_profile_3d_waterfall is None:
            try:
                plotter_path = Path(__file__).resolve().parents[1] / "XPS_Plotter" / "plot_modules" / "quantification"
                if str(plotter_path) not in sys.path:
                    sys.path.insert(0, str(plotter_path))
                from depth_3d_waterfall import load_depth_profile_csv as _load, plot_depth_profile_3d_waterfall as _plot
                local_load = _load
                local_plot = _plot
            except Exception:
                return None
        else:
            local_load = load_depth_profile_csv
            local_plot = plot_depth_profile_3d_waterfall
        if not csv_path or not csv_path.exists():
            return None

        try:
            layers = local_load(csv_path)
            if not layers or len(layers) < 2:
                return None

            self._init_step1_plot_dirs()
            out_dir = self._step1_depth_plots_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            return local_plot(
                spectra_dict=layers,
                region=region_name,
                out_dir=out_dir,
                config=None,
            )
        except Exception as exc:
            if self.debug:
                print(f"[DepthProfile] Failed to generate 3D waterfall for {region_name}: {exc}")
            return None

    def _generate_depth_profile_waterfalls_from_output(self, output_dir: Path) -> None:
        """Fallback: scan aggregated CSVs and generate 3D waterfalls."""
        if not output_dir.exists():
            return
        for region_dir in output_dir.iterdir():
            if not region_dir.is_dir():
                continue
            for csv_path in region_dir.glob("aggregated_*_allHR.csv"):
                region = region_dir.name
                self._generate_depth_profile_waterfall(csv_path, region)

    def _generate_quality_plots_from_reports(self, output_dir: Path) -> None:
        """Fallback: generate quality plots from saved quality_report CSVs."""
        if not output_dir.exists():
            return

        self._init_step1_plot_dirs()
        plots_dir = self._step1_quality_plots_dir
        if plots_dir is None:
            return

        # Skip if already generated
        if plots_dir.exists() and any(plots_dir.rglob("*.png")):
            return

        try:
            plotter_path = Path(__file__).resolve().parents[1] / "XPS_Plotter" / "plot_modules" / "data_quality"
            if str(plotter_path) not in sys.path:
                sys.path.insert(0, str(plotter_path))
            from quality_report_plots import plot_quality_report_diagnostics, plot_sample_quality_details
        except Exception:
            return

        def _as_bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            text = str(val).strip().lower()
            return text in ("1", "true", "yes", "y")

        for region_dir in output_dir.iterdir():
            if not region_dir.is_dir():
                continue
            report_files = list(region_dir.glob("*_quality_report.csv"))
            for report_path in report_files:
                try:
                    df = pd.read_csv(report_path)
                except Exception:
                    continue
                if df.empty:
                    continue

                metrics = []
                for _, row in df.iterrows():
                    flag_value = str(row.get("quality_flag", "acceptable"))
                    quality_flag = SimpleNamespace(value=flag_value)
                    metrics.append(
                        SimpleNamespace(
                            sample_id=row.get("sample_id", ""),
                            snr_xps=float(row.get("snr_xps", 0) or 0),
                            peak_height=float(row.get("peak_height", 0) or 0),
                            peak_to_baseline_ratio=float(row.get("peak_to_baseline", 0) or 0),
                            peak_width_fwhm=float(row.get("peak_width_fwhm", 0) or 0),
                            points_per_ev=float(row.get("resolution_pts_per_ev", 0) or 0),
                            quality_flag=quality_flag,
                            is_hr_scan=_as_bool(row.get("is_hr_scan", False)),
                            suspected_shift=_as_bool(row.get("suspected_shift", False)),
                            suitable_for_fitting=_as_bool(row.get("suitable_for_fitting", False)),
                            baseline_std=float(row.get("baseline_std", 0) or 0),
                            relative_noise=float(row.get("relative_noise", 0) or 0),
                            warnings=[],
                        )
                    )

                region_name = region_dir.name
                plot_quality_report_diagnostics(
                    metrics_list=metrics,
                    region=region_name,
                    output_dir=region_dir,
                    config=self.quality_analyzer.config,
                    spectra_dict=None,
                    plots_dir=plots_dir,
                )
                if len(metrics) > 4:
                    plot_sample_quality_details(
                        metrics_list=metrics,
                        region=region_name,
                        output_dir=region_dir,
                        config=self.quality_analyzer.config,
                        max_samples_per_plot=20,
                        plots_dir=plots_dir,
                    )

    def _convert_standard_file(self, raw_file: Path, output_dir: Path,
                               auto_detect: bool,
                               specific_regions: List[str]) -> Dict[str, Path]:
        """Run the standard XPS_reader workflow for non-map files."""
        # Note: Triage and quality gate should run BEFORE calling this method
        # via workflow_orchestrator.py for proper architecture
        
        try:
            spectra = self.importer.import_file(raw_file)
            if not spectra:  # Empty list means graceful rejection
                print(f"Warning: File rejected: {raw_file.name} (corrupted or incompatible)")
                return {}
        except Exception as e:
            print(f"Import failed: {e}")
            return {}

        # Optional: Quick quality check (orchestrator already did full validation)
        quality_results = {}
        for spec in spectra:
            try:
                report = self.gatekeeper.validate(spec, modality=DataModality.SINGLE_SPECTRUM)
                quality_results[spec.name] = report
            except Exception as e:
                if self.debug:
                    print(f"Quality check failed for {spec.name}: {e}")
        
        poor_quality = [name for name, report in quality_results.items() 
                       if report.quality_flag in [QualityFlag.FAILED, QualityFlag.POOR]]
        if poor_quality:
            print(f"⚠️  Note: {len(poor_quality)} spectrum(a) with quality concerns (continuing as approved)")

        
        calibration_result = None
        if self.calibration_enabled and spectra:
            print()
            print("Applying energy calibration...")
            spectra, calibration_result = self.calibrator.calibrate_spectra(
                spectra,
                self.region_defs
            )

        spectra = self._apply_optional_smoothing(spectra)

        print()
        print("Extracting regions...")
        all_regions = {}

        for spectrum in spectra:
            scan_type = self.extractor.classify_scan_type(spectrum)
            if scan_type == ScanType.SURVEY:
                metadata = getattr(spectrum, "metadata", {}) or {}
                metadata["survey_file_stem"] = raw_file.stem
                spectrum.metadata = metadata
                all_regions.setdefault("survey", []).append(spectrum)
                continue

            regions = self.extractor.extract_all_regions(
                spectrum, auto_detect, specific_regions, scan_type=scan_type
            )

            for region_name, region_spectrum in regions.items():
                if region_name not in all_regions:
                    all_regions[region_name] = []
                all_regions[region_name].append(region_spectrum)

        if not all_regions:
            print("No regions extracted")
            return {}

        print()
        print("Saving CSV files...")
        saved_files = {}
        base_name = self._sanitize_filename(raw_file.stem)

        for region_name, region_spectra in all_regions.items():
            region_dir = output_dir / region_name
            region_dir.mkdir(parents=True, exist_ok=True)

            # Update exporter output directory
            self.exporter.output_dir = region_dir

            if region_name.lower() == "survey":
                saved_paths = []
                used_names = set()
                for idx, spectrum in enumerate(region_spectra, 1):
                    metadata = getattr(spectrum, "metadata", {}) or {}
                    raw_stem = metadata.get("survey_file_stem") or raw_file.stem
                    stem = Path(str(raw_stem)).stem
                    base_name = f"survey_{stem}"
                    filename = f"{base_name}.csv"
                    if filename in used_names:
                        counter = 2
                        while f"{base_name}_{counter}.csv" in used_names:
                            counter += 1
                        filename = f"{base_name}_{counter}.csv"
                    used_names.add(filename)
                    saved_paths.append(self.exporter.export_spectrum(spectrum, filename=filename))
                if saved_paths:
                    saved_files[region_name] = saved_paths[0]
                    print(f"   {region_name}: {len(saved_paths)} survey spectrum(a) saved")
                continue

            if len(region_spectra) == 1:
                saved_path = self.exporter.export_spectrum(
                    region_spectra[0], base_name)
                saved_files[region_name] = saved_path
                print(f"   {region_name}: {saved_path.name}")
            else:
                saved_paths = self.exporter.export_multiple_spectra(
                    region_spectra, base_name)
                if saved_paths:
                    saved_files[region_name] = saved_paths[0]
                    print(f"   {region_name}: {saved_paths[0].name}")

        return saved_files

    def batch_convert(self, input_dir: Path, output_dir: Path,
                      file_patterns: List[str] = None,
                      auto_detect: bool = None,
                      specific_regions: List[str] = None,
                      aggregate_per_region: bool = None,
                      prefer_hr: bool = True) -> Dict[str, List[Path]]:
        """Batch convert all files in directory using config settings."""

        # Use config defaults if not specified
        if file_patterns is None:
            file_patterns = [f"*{ext}" for ext in self.supported_extensions]
        if auto_detect is None:
            auto_detect = self.proc.get('auto_detect_regions', True)
        if specific_regions is None:
            specific_regions = self.proc.get('specific_regions', [])
        if aggregate_per_region is None:
            aggregate_per_region = self.aggregate_per_region

        # Initialize timestamped Step 1 plot directories (per run)
        self._step1_plots_root = None
        self._step1_quality_plots_dir = None
        self._step1_depth_plots_dir = None
        self._init_step1_plot_dirs()

        # Find files
        raw_files = []
        for pattern in file_patterns:
            raw_files.extend(input_dir.glob(pattern))
        raw_files = sorted(set(raw_files))

        if not raw_files:
            print(f"❌ No files found matching patterns: {file_patterns}")
            return {}

        # Log processing header
        if self.logger:
            self.logger.log_header("🔄 XPS RAW DATA CONVERTER")
            self.logger.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.log(f"Input directory: {input_dir}")
            self.logger.log(f"Output directory: {output_dir}")
            self.logger.log(f"Files detected: {len(raw_files)}")
            self.logger.log(f"File patterns: {', '.join(file_patterns)}")
            self.logger.log(f"Region detection mode: {'AUTO-DETECT' if auto_detect else 'MANUAL'}")
            if not auto_detect and specific_regions:
                self.logger.log(f"Target regions: {', '.join(specific_regions)}")
            self.logger.log(f"Energy calibration: {'enabled' if self.calibration_enabled else 'disabled'}")
            self.logger.log(f"Aggregate per region: {aggregate_per_region}")
            if aggregate_per_region:
                self.logger.log(f"HR scan preference: {prefer_hr}")
        
        # Print header
        print(f"\n{'='*70}")
        print(f"🔄 XPS RAW DATA CONVERTER")
        print(f"{'='*70}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Files: {len(raw_files)}")
        print(f"Mode: {'AUTO-DETECT' if auto_detect else 'MANUAL'}")
        if not auto_detect and specific_regions:
            print(f"Regions: {', '.join(specific_regions)}")
        print(f"Energy calibration: {'enabled' if self.calibration_enabled else 'disabled'}")
        print(f"Aggregate per region: {aggregate_per_region}")
        if aggregate_per_region:
            print(f"HR scan preference: {prefer_hr}")
        print(f"{'='*70}\n")

        all_region_files = {}
        conversion_log = []
        timing_stats = TimingStats("File processing")

        with Timer("Batch conversion"):
            
            if aggregate_per_region:
                # Aggregation mode: collect all spectra first, then export one single file per region
                all_region_spectra = {}  # {region_name: [spectrum1, spectrum2, ...]}
                
                for i, raw_file in enumerate(raw_files, 1):
                    print(f"\n{'='*70}")
                    print(f"[{i}/{len(raw_files)}] {raw_file.name}")
                    print(f"{'='*70}")
                    
                    if self.logger:
                        self.logger.log(f"\nProcessing file {i}/{len(raw_files)}: {raw_file.name}")

                    with timing_stats.time():
                        try:
                            # Stage 1: File-level quality check (pre-import)
                            should_import, file_issues = self.gatekeeper.validate_file(raw_file)
                            if not should_import:
                                print(f"[QualityGate] ❌ File failed Stage 1 checks:")
                                for issue in file_issues:
                                    print(f"   {issue}")
                                if self.logger:
                                    self.logger.log(f"  ❌ Failed Stage 1 quality checks:", "WARN")
                                    for issue in file_issues:
                                        self.logger.log(f"     - {issue}", "WARN")
                                conversion_log.append({
                                    'file': raw_file.name,
                                    'regions': 0,
                                    'region_names': '',
                                    'status': 'Failed Stage 1 quality checks'
                                })
                                continue  # Skip corrupted/invalid file
                            
                            # Check if file should be routed to mapper (skip in aggregation)
                            if self.triage:
                                triage_result = self.triage.analyze_file_structure(raw_file)
                                if self._should_skip_map_file(triage_result):
                                    dtype = triage_result.get('data_type', 'map')
                                    dtype_value = getattr(dtype, 'value', str(dtype))
                                    confidence = triage_result.get('confidence', 0.0)
                                    print(f"[AI] Skipping map file: {dtype_value} "
                                          f"(confidence: {confidence:.1%})")
                                    print("   🗺️ Map files should be processed via XPS_Mapper workflow")
                                    if self.logger:
                                        self.logger.log(f"  🗺️ Skipped (map file): {dtype_value} (confidence: {confidence:.1%})")
                                    conversion_log.append({
                                        'file': raw_file.name,
                                        'regions': 0,
                                        'region_names': '',
                                        'status': 'Skipped (map file)'
                                    })
                                    continue  # Skip this file
                            
                            # Import and extract regions without saving
                            spectra = self._import_and_extract_regions(
                                raw_file, auto_detect, specific_regions
                            )
                            
                            if self.logger:
                                self.logger.log(f"  ✅ Extracted {len(spectra)} region(s): {', '.join(spectra.keys())}")
                            
                            # Group by region
                            for region_name, spectrum_list in spectra.items():
                                all_region_spectra.setdefault(region_name, []).extend(spectrum_list)
                            
                            conversion_log.append({
                                'file': raw_file.name,
                                'regions': len(spectra),
                                'region_names': ', '.join(spectra.keys()),
                                'status': 'Success'
                            })

                        except Exception as e:
                            print(f"\n❌ Error: {e}")
                            if self.debug:
                                traceback.print_exc()
                            conversion_log.append({
                                'file': raw_file.name,
                                'regions': 0,
                                'region_names': '',
                                'status': f'Failed: {str(e)}'
                            })
                
                # Now export aggregated spectra (one CSV per region containing all files)
                print(f"\n{'='*70}")
                print("Aggregating and exporting regions...")
                print(f"{'='*70}\n")
                
                if self.logger:
                    self.logger.log_header("📊 AGGREGATION & EXPORT")
                    self.logger.log(f"Total regions to aggregate: {len(all_region_spectra)}")
                
                for region_name, region_spectra in all_region_spectra.items():
                    region_dir = output_dir / region_name
                    region_dir.mkdir(parents=True, exist_ok=True)
                    self.exporter.output_dir = region_dir
                    
                    if region_name.lower() == "survey":
                        saved_paths = []
                        used_names = set()
                        for spectrum in region_spectra:
                            metadata = getattr(spectrum, "metadata", {}) or {}
                            raw_stem = (
                                metadata.get("survey_file_stem")
                                or metadata.get("source_file")
                                or metadata.get("original_spectrum")
                                or spectrum.name
                            )
                            stem = Path(str(raw_stem)).stem
                            base_name = f"survey_{stem}"
                            filename = f"{base_name}.csv"
                            if filename in used_names:
                                counter = 2
                                while f"{base_name}_{counter}.csv" in used_names:
                                    counter += 1
                                filename = f"{base_name}_{counter}.csv"
                            used_names.add(filename)
                            saved_paths.append(self.exporter.export_spectrum(spectrum, filename=filename))
                    else:
                        # Export all spectra for this region into ONE combined CSV
                        saved_paths = self.exporter.export_multiple_spectra(
                            region_spectra, 
                            filename_prefix=f"aggregated_{region_name}",
                            combine_in_single_file=True  # KEY: Combine all into one file
                        )
                    if saved_paths:
                        all_region_files[region_name] = saved_paths
                        if region_name.lower() == "survey":
                            print(f"   {region_name}: {len(saved_paths)} survey spectrum(a) saved")
                            if self.logger:
                                self.logger.log(f"  Region {region_name}: {len(saved_paths)} survey spectrum(a) saved")
                        else:
                            print(f"   {region_name}: {len(region_spectra)} spectra → {saved_paths[0].name}")
                            if self.logger:
                                self.logger.log(f"  Region {region_name}: {len(region_spectra)} spectra → {saved_paths[0].name}")
                    
                    # Stage 2: Detailed quality assessment for this region
                    print(f"\n🔍 Running Stage 2 quality assessment for {region_name}...")
                    if self.logger:
                        self.logger.log(f"\n  Stage 2 Quality Assessment: {region_name}")
                    
                    try:
                        # Build spectra_dict: use clean sample names as keys (remove region suffix)
                        spectra_dict = {}
                        for spectrum in region_spectra:
                            # Get original spectrum name and clean it for plot labels
                            original_name = getattr(spectrum, 'name', f'spectrum_{len(spectra_dict)}')
                            
                            # Remove region suffix from name for cleaner plot labels
                            # Pattern: "samplename_regionname" → "samplename"
                            clean_name = original_name
                            
                            # More robust region suffix removal
                            if f'_{region_name}' in original_name:
                                clean_name = original_name.replace(f'_{region_name}', '')
                            elif original_name.endswith(f'_{region_name}'):
                                clean_name = original_name[:-len(f'_{region_name}')]
                            
                            # Also handle cases where region appears at the end without underscore
                            if clean_name.endswith(region_name) and clean_name != region_name:
                                # Only remove if it's not the entire name
                                clean_name = clean_name[:-len(region_name)].rstrip('_')
                            
                            # If clean_name is empty or just underscores, use original
                            if not clean_name or clean_name.replace('_', '') == '':
                                clean_name = original_name
                            
                            if self.debug:
                                print(f"[DEBUG] Name cleaning: '{original_name}' → '{clean_name}' (region: {region_name})")
                            
                            # Ensure unique keys in case multiple spectra have same base name
                            spectrum_key = clean_name
                            counter = 1
                            while spectrum_key in spectra_dict:
                                spectrum_key = f"{clean_name}_{counter}"
                                counter += 1
                            
                            spectra_dict[spectrum_key] = spectrum
                            if self.debug:
                                print(f"[DEBUG] Added spectrum: {spectrum_key} (original: {original_name}, metadata: {spectrum.metadata.get('original_spectrum', 'N/A')})")
                        
                        print(f"  → Analyzing {len(spectra_dict)} spectra for {region_name}...")
                        
                        # Run quality analysis and generate reports
                        quality_metrics = self.quality_analyzer.analyze_batch(
                            spectra_dict=spectra_dict,
                            region=region_name,
                            output_dir=region_dir,
                            plots_dir=self._step1_quality_plots_dir,
                        )
                        
                        # Print summary
                        excellent = sum(1 for m in quality_metrics if m.quality_flag.value == 'excellent')
                        good = sum(1 for m in quality_metrics if m.quality_flag.value == 'good')
                        acceptable = sum(1 for m in quality_metrics if m.quality_flag.value == 'acceptable')
                        poor = sum(1 for m in quality_metrics if m.quality_flag.value in ['poor', 'suspicious'])
                        
                        print(f"   Quality: ✅ {excellent} excellent | 👍 {good} good | "
                              f"⚠️  {acceptable} acceptable | ❌ {poor} poor/suspicious")
                        
                        if self.logger:
                            self.logger.log(f"    Quality distribution: Excellent={excellent}, Good={good}, "
                                          f"Acceptable={acceptable}, Poor/Suspicious={poor}")
                        
                        # Flag samples with warnings
                        for m in quality_metrics:
                            if m.warnings:
                                print(f"   ⚠️  {m.sample_id}: {', '.join(m.warnings[:2])}")
                                if self.logger:
                                    self.logger.log(f"    ⚠️  {m.sample_id}: {', '.join(m.warnings[:2])}", "WARN")
                    
                    except Exception as e:
                        print(f"   ⚠️  Quality assessment failed: {e}")
                        if self.debug:
                            import traceback
                            traceback.print_exc()

                    # Depth profile visualization: 3D waterfall when multiple layers exist
                    if saved_paths:
                        depth_plot = self._generate_depth_profile_waterfall(saved_paths[0], region_name)
                        if depth_plot:
                            print(f"   ✅ 3D depth profile waterfall saved: {depth_plot.name}")
                
            else:
                # Non-aggregation mode: save each file separately
                for i, raw_file in enumerate(raw_files, 1):
                    print(f"\n{'='*70}")
                    print(f"[{i}/{len(raw_files)}] {raw_file.name}")
                    print(f"{'='*70}")

                    with timing_stats.time():
                        try:
                            saved_files = self.convert_file(
                                raw_file, output_dir, auto_detect, specific_regions
                            )

                            for region_name, csv_path in saved_files.items():
                                all_region_files.setdefault(region_name, []).append(csv_path)

                            conversion_log.append({
                                'file': raw_file.name,
                                'regions': len(saved_files),
                                'region_names': ', '.join(saved_files.keys()),
                                'status': 'Success'
                            })

                        except Exception as e:
                            print(f"\n❌ Error: {e}")
                            if self.debug:
                                traceback.print_exc()
                            conversion_log.append({
                                'file': raw_file.name,
                                'regions': 0,
                                'region_names': '',
                                'status': f'Failed: {str(e)}'
                            })

        if timing_stats.timings:
            print()
            timing_stats.print_summary()

        self._print_summary(all_region_files, conversion_log, output_dir, aggregate_per_region)
        
        # Log final summary
        if self.logger:
            self.logger.log_header("📋 FINAL SUMMARY")
            if aggregate_per_region:
                self.logger.log(f"Mode: Aggregated (one CSV per region)")
                for region_name, csv_files in sorted(all_region_files.items()):
                    self.logger.log(f"  {region_name}: {len(csv_files)} file(s)")
            else:
                total_files = len(conversion_log)
                successful = sum(1 for log in conversion_log if log['status'] == 'Success')
                self.logger.log(f"Mode: Per-file conversion")
                self.logger.log(f"  Total files: {total_files}")
                self.logger.log(f"  Successful: {successful}")
                self.logger.log(f"  Failed: {total_files - successful}")
            
            # Log conversion details table
            if conversion_log:
                self.logger.log("\nConversion Details:")
                self.logger.log_summary_table(
                    conversion_log,
                    headers=['file', 'regions', 'region_names', 'status']
                )
            
            self.logger.log(f"\nOutput directory: {output_dir}")
            self.logger.write()

        if self.generate_quality_plots:
            self._generate_quality_plots(output_dir)
        # Fallback: build plots from saved reports/CSVs if none created
        self._generate_quality_plots_from_reports(output_dir)
        self._generate_depth_profile_waterfalls_from_output(output_dir)

        return all_region_files

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize filename for safe filesystem use."""
        import re
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    def _print_summary(self, all_region_files: Dict, conversion_log: List[Dict],
                       output_dir: Path, aggregated: bool):
        """Print conversion summary."""
        print(f"\n{'='*70}")
        print(f"📋 CONVERSION SUMMARY")
        print(f"{'='*70}\n")

        if aggregated:
            print(f"Mode: Aggregated (one CSV per region)")
            for region_name, csv_files in sorted(all_region_files.items()):
                print(f"   {region_name}: {len(csv_files)} file(s)")
        else:
            print(f"Mode: Per-file conversion")
            total_files = len(conversion_log)
            successful = sum(1 for log in conversion_log if log['status'] == 'Success')
            print(f"   Total files: {total_files}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {total_files - successful}")

        if self.calibration_enabled:
            cal_cfg = self.config.get('energy_calibration', {})
            ref_region = cal_cfg.get('reference_region', 'C1s')
            target_be = cal_cfg.get('target_binding_energy_ev', 284.8)
            print(f"\n⚡ Energy Calibration Applied:")
            print(f"   Reference: {ref_region} @ {target_be} eV")

        print(f"\n   Output directory: {output_dir}")
        print(f"\n{'='*70}\n")

# ========== MAIN ==========

def main():
    """Main execution with YAML-driven configuration."""
    import sys

    # Check for debug flag
    debug = '--debug' in sys.argv or '-d' in sys.argv
    if debug:
        print("🐛 DEBUG MODE ENABLED\n")
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO)

    # Print header
    print("\n" + "="*70)
    print("XPS DATA PROCESSOR (MODULAR)")
    print("="*70)

    # Load configuration from YAML
    try:
        config, region_defs = load_xps_config(CONFIG_FILE)
        logging.debug("Configuration loaded successfully")
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return

    # Setup paths from constants
    input_dir = Path(RAW_DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    # Apply run tag subfolder when provided (e.g., 20260213150950)
    run_id = os.environ.get("XPS_RUN_ID", "").strip()
    if run_id:
        if "_" in run_id:
            date_part, time_part = run_id.split("_", 1)
            date_part = re.sub(r"\D", "", date_part)
            time_part = re.sub(r"\D", "", time_part)
            if len(date_part) >= 8:
                date_part = date_part[:8]
            else:
                date_part = datetime.now().strftime("%Y%m%d")
            if len(time_part) >= 6:
                time_part = time_part[:6]
            else:
                time_part = datetime.now().strftime("%H%M%S")
            run_tag = f"{date_part}{time_part}"
        else:
            digits = re.sub(r"\D", "", run_id)
            run_tag = digits[:14] if len(digits) >= 14 else ""
        if run_tag:
            output_dir = output_dir / run_tag

    logging.debug(f"Input directory: {input_dir}")
    logging.debug(f"Output directory: {output_dir}")

    if not input_dir.exists():
        raise SystemExit(f"❌ Input directory not found: {input_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processing logger
    log_file = output_dir / "processing_log.txt"
    proc_logger = ProcessingLogger(log_file)
    proc_logger.log_header("XPS DATA PROCESSOR (MODULAR)")
    proc_logger.log(f"Configuration file: {CONFIG_FILE}")
    proc_logger.log(f"Input directory: {input_dir}")
    proc_logger.log(f"Output directory: {output_dir}")

    # Get processing settings from config
    proc = config.get('processing', {})
    auto_detect = proc.get('auto_detect_regions', True)
    extract_all = proc.get('extract_all_regions', True)
    specific_regions = None if extract_all else proc.get('specific_regions', None)

    # Get output settings from config
    output_cfg = config.get('output', {})
    aggregate_per_region = output_cfg.get('aggregate_per_region', True)

    # Get calibration settings from config
    cal_config = config.get('energy_calibration', {})
    calibration_enabled = cal_config.get('enable', True)

    # Get file format settings from config
    file_cfg = config.get('file_formats', {})
    supported_extensions = file_cfg.get('supported_extensions',
                                        ['.spe', '.xy', '.txt', '.asc', '.dat'])

    file_patterns = [f"*{ext}" for ext in supported_extensions]

    # Get scan classification settings
    scan_cfg = proc.get('scan_classification', {})
    hr_threshold = scan_cfg.get('hr_resolution_threshold', 5)
    grid_step = proc.get('common_grid_step', 0.1)

    # Print configuration summary
    print(f"\n📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🔍 File patterns: {', '.join(file_patterns)}")
    print(f"⚙️  Energy calibration: {'enabled' if calibration_enabled else 'disabled'}")
    print(f"📊 Region detection: {'auto-detect' if auto_detect else 'manual'}")
    if not auto_detect and specific_regions:
        print(f"   Target regions: {', '.join(specific_regions)}")
    print(f"💾 Output mode: {'aggregated per region' if aggregate_per_region else 'per file'}")
    if aggregate_per_region:
        print(f"   Grid step: {grid_step} eV")
        print(f"   HR threshold: {hr_threshold} pts/eV")
    print("="*70 + "\n")

    # Initialize batch converter
    converter = BatchConverter(
        region_definitions=region_defs,
        config=config,
        debug=debug,
        logger=proc_logger
    )
    # Ensure plot output roots align with the run-tagged output directory
    # output_dir = project_root/01_converted_csv/<run_tag>
    converter.project_root = output_dir.parent.parent

    # Batch convert files
    converted_files = converter.batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        file_patterns=file_patterns,
        auto_detect=auto_detect,
        specific_regions=specific_regions,
        aggregate_per_region=aggregate_per_region,
        prefer_hr=True
    )

    # Print final summary
    if converted_files:
        print("\n" + "="*70)
        print("✅ PROCESSING COMPLETE")
        print("="*70)

        total_regions = len(converted_files)
        total_files = sum(len(files) for files in converted_files.values())

        print(f"\n📊 Results:")
        print(f"   Regions processed: {total_regions}")
        print(f"   Files generated: {total_files}")
        print(f"   Output location: {output_dir}")

        print("\n🎯 Next Steps:")
        print(f"   1. Review CSVs in: {output_dir}")
        print(f"   2. Run peak fitting: python xps_peak_fitting.py")
        print(f"   3. Run quantification: python xps_quantification.py")

        if calibration_enabled:
            ref_region = cal_config.get('reference_region', 'C1s')
            target_be = cal_config.get('target_binding_energy_ev', 284.8)
            print(f"\n⚡ Energy Calibration:")
            print(f"   Reference: {ref_region} @ {target_be} eV")
            print(f"   Status: Applied to all spectra")

        print("\n" + "="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("⚠️  NO FILES CONVERTED")
        print("="*70)
        print(f"\n🔍 Troubleshooting:")
        print(f"   1. Check input directory: {input_dir}")
        print(f"   2. Verify file patterns: {', '.join(file_patterns)}")
        print(f"   3. Ensure files match supported formats: {', '.join(supported_extensions)}")
        print(f"   4. Run with --debug flag for detailed output")
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
