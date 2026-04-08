
# This Python pipeline is for the automated XPS analysis workflow, a main script would orchestrate this entire workflow, allowing for high-throughput analysis by looping through a directory of data files.
# Users: only need to adjust the user configuration section below, including input/output paths, and fitting templates.
# default fitting functions: Shirely background, pseudo_voigt_mix
"""
Complete XPS Analysis Workflow with YAML Template Support and Raw Data Import
Features:
- Automatic CSV conversion for workflow compatibility
- Multi-layer/file XPS data parsing
- YAML template-based fitting with constraints
- Individual layer plots with residuals
- Stacked comparison plots across layers/files
- Comprehensive curve data export
- Quantitative analysis reports
"""
from pathlib import Path
import re
import yaml
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from numpy import trapz
from lmfit import Minimizer, Parameters, minimize
import traceback
from scipy.optimize import curve_fit
from scipy.special import wofz
from typing import List, Dict, Optional, Tuple, Any

# Import background correction utilities
sys.path.append(str(Path(__file__).parent.parent))
from XPS_utils.background_correction import (
    baseline_shirley,
    shirley_background,
    apply_background_correction
)

# Fix import to use tool_utils from Tools folder
import sys
from pathlib import Path

# Add Tools directory to path for tool_utils import
tools_dir = Path(__file__).resolve().parents[1]
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

try:
    from tool_utils import Timer, TimingStats
except ImportError:
    print("[WARNING]  Could not import from tool_utils, using fallback implementations")
    class Timer:
        def __init__(self, name): self.name = name
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class TimingStats:
        def __init__(self, name): pass
        def time(self): return Timer("op")
        def print_summary(self): pass

# Add XPS_Plotter directory to path
plotter_dir = Path(__file__).resolve().parents[1] / "XPS_Plotter"
if str(plotter_dir) not in sys.path:
    sys.path.insert(0, str(plotter_dir))

try:
    # Import directly from fitting_plots module
    from plot_modules.fitting.fitting_plots import (
        plot_template_fit,
        plot_stacked_layers_comparison,
    )
    print("[OK] Plotter functions imported successfully")
except ImportError as e:
    print(f"[WARNING]  Could not import plotter functions: {e}")
    print("[WARNING]  Plotting features will be disabled")
    def plot_template_fit(*args, **kwargs): return None
    def plot_stacked_layers_comparison(*args, **kwargs): return None

# ========== USER CONFIGURATION ==========
# Process ALL elements in the converted CSV folder and auto-map templates per file.

# ========== XPS WORKFLOW INTEGRATION ==========
try:
    import sys
    from pathlib import Path
    
    # Add Tools directory to path for import
    tools_dir = Path(__file__).resolve().parents[1]
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    from xps_workflow_manager import get_workflow_config, update_module_paths
    WORKFLOW_MANAGER_AVAILABLE = True
except ImportError:
    WORKFLOW_MANAGER_AVAILABLE = False
    print("[WARNING]  Workflow manager not available, using legacy paths")

# Initialize directory paths using workflow manager or legacy fallback
if WORKFLOW_MANAGER_AVAILABLE:
    try:
        workflow_config = get_workflow_config()
        fitter_config = update_module_paths('fitter', workflow_config)
        
        INPUT_DIR = fitter_config['input_dir']
        OUTPUT_DIR = fitter_config['output_dir']
        TEMPLATE_DIR = fitter_config['config_dir'] / "LIB_fit_template"
        _PROJECT_ROOT = workflow_config.project_root
        PLOTS_DIR = workflow_config.plots_output_dir / "02_peak_fitting"
        
    except Exception as e:
        print(f"[WARNING]  Workflow manager error: {e}, falling back to legacy paths")
        WORKFLOW_MANAGER_AVAILABLE = False

# Legacy fallback paths
if not WORKFLOW_MANAGER_AVAILABLE:
    _ZZY_ROOT = Path(__file__).resolve().parents[2]
    _PROJECT_ROOT = _ZZY_ROOT / "project_root"
    INPUT_DIR = _PROJECT_ROOT / "converted_csv"
    TEMPLATE_DIR = _PROJECT_ROOT / "xps_config" / "LIB_fit_template"
    OUTPUT_DIR = _PROJECT_ROOT / "02_fitted_results"
    PLOTS_DIR = _PROJECT_ROOT / "04_plots" / "02_peak_fitting"


def _resolve_input_directory(requested_dir: Path, project_root: Path) -> Path:
    """
    Ensure the converted CSV input directory exists and gracefully fall back to the
    standardized 01_converted_csv folder if needed.
    """
    def _select_run_subdir(base_dir: Path) -> Path:
        run_id = os.environ.get("XPS_RUN_ID", "").strip()
        run_tag = ""
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
                if len(digits) >= 14:
                    run_tag = digits[:14]
        if run_tag:
            candidate = Path(base_dir) / run_tag
            if candidate.exists():
                return candidate
        # Fall back to latest run tag subdir if present
        if Path(base_dir).exists():
            run_dirs = [
                p for p in Path(base_dir).iterdir()
                if p.is_dir() and re.match(r"^\d{14}$", p.name)
            ]
            if run_dirs:
                return sorted(run_dirs, key=lambda p: p.name)[-1]
        return Path(base_dir)

    requested_resolved = requested_dir.resolve()
    candidates = [requested_resolved]
    if project_root:
        candidates.append(project_root / "01_converted_csv")
        candidates.append(project_root / "converted_csv")

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            selected = _select_run_subdir(resolved)
            if selected != requested_resolved:
                print(f"[INFO]  Using input directory: {selected}")
            return selected

    raise FileNotFoundError(
        f"Input directory does not exist: {requested_resolved}\n"
        f"Tried fallbacks: {[str(path) for path in candidates if path is not None]}"
    )


INPUT_DIR = _resolve_input_directory(INPUT_DIR, _PROJECT_ROOT)

def _run_subdir(base_dir: Path, run_id: str) -> Path:
    run_id = (run_id or "").strip()
    run_tag = ""
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
            if len(digits) >= 14:
                run_tag = digits[:14]
            elif len(digits) >= 8:
                run_tag = f"{digits[:8]}{datetime.now().strftime('%H%M%S')}"
    if not run_tag:
        run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(base_dir) / run_tag

RUN_ID = os.environ.get("XPS_RUN_ID", "")
OUTPUT_DIR = _run_subdir(OUTPUT_DIR, RUN_ID)
PLOTS_DIR = _run_subdir(Path(PLOTS_DIR), RUN_ID)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# File pattern for converted spectra - look in subdirectories too
INPUT_GLOB = "**/*.csv"

# Default visualization settings
SAVE_PLOTS = True


# Template auto-selection mapping
TEMPLATE_MAPPING = {
    "F1s": "LIB_SEI_F1s_v1.yaml",
    "O1s": "LIB_SEI_O1s_v1.yaml", 
    "C1s": "LIB_SEI_C1s_v1.yaml",
    "Li1s": "LIB_SEI_Li1s_v1.yaml",
    "N1s": "LIB_SEI_N1s_v1.yaml",
    "S2p": "LIB_SEI_S2p_v1.yaml",
    "Si2p": "LIB_SEI_Si2p_v1.yaml",
}



def sanitize_filename(s: str) -> str:
    """Keep alphanumerics, _, ., -, replace others with underscore."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))
    
# ========== STEP 1: DATA INGESTION & TEMPLATE LOADING ==========

def load_yaml_template(template_path: Path) -> Dict:
    """Load and validate YAML template file with UTF-8 encoding."""
    try:
        # Try UTF-8 first
        with open(template_path, 'r', encoding='utf-8') as file:
            template = yaml.safe_load(file)
        print(f"[OK] Loaded template: {template.get('template_name', 'Unknown')}")
        return template
    except UnicodeDecodeError:
        try:
            # Fallback to utf-16 if utf-8 fails
            with open(template_path, 'r', encoding='utf-16') as file:
                template = yaml.safe_load(file)
            print(f"[OK] Loaded template: {template.get('template_name', 'Unknown')} (UTF-16)")
            return template
        except Exception as e:
            raise ValueError(f"Failed to load template {template_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load template {template_path}: {e}")

def auto_select_template(filename: str, template_dir: Path) -> Optional[Path]:
    """Auto-select template based on filename pattern using dynamic template discovery."""
    filename_upper = filename.upper()
    
    # Get all available templates
    template_files = list(template_dir.glob("*.yaml")) + list(template_dir.glob("*.yml"))
    
    # Extract region names from available templates
    template_mapping = {}
    for template_file in template_files:
        template_name = template_file.stem
        # Extract region from template name (e.g., "LIB_SEI_P2p_v1" -> "P2P")
        for part in template_name.split('_'):
            if any(char.isdigit() or char in ['s', 'p', 'd', 'f'] for char in part[-2:]):
                # This part looks like a region (ends with orbital notation)
                region = part.upper()
                template_mapping[region] = template_file.name
                break
    
    # Look for matching region in filename
    for region, template_file in template_mapping.items():
        if region in filename_upper:
            template_path = template_dir / template_file
            if template_path.exists():
                print(f"[TARGET] Auto-selected template: {template_file} for {filename}")
                return template_path
            else:
                print(f"[WARNING]  Template {template_file} not found for region {region}")
    
    print(f"[ERROR] No template found for {filename}")
    return None

def parse_template_to_regions(template: Dict) -> List[Dict]:
    """Convert YAML template to internal region format expected by Section 5."""
    regions = []

    def canon_param_dict(d: dict, defaults: dict = None) -> dict:
        """Canonicalize per-parameter dict to use initial_guess/min_bound/max_bound."""
        if not isinstance(d, dict):
            d = {}
        out = {}

        # initial
        if 'initial_guess' in d:
            out['initial_guess'] = d['initial_guess']
        elif 'initial' in d:
            out['initial_guess'] = d['initial']

        # min
        if 'min_bound' in d:
            out['min_bound'] = d['min_bound']
        elif 'min' in d:
            out['min_bound'] = d['min']

        # max
        if 'max_bound' in d:
            out['max_bound'] = d['max_bound']
        elif 'max' in d:
            out['max_bound'] = d['max']

        # pass-through flags if present
        for k in ('is_constrained', 'constraint_to'):
            if k in d:
                out[k] = d[k]

        # apply defaults for any missing fields
        if isinstance(defaults, dict):
            for k, v in defaults.items():
                out.setdefault(k, v)

        return out

    for region in template.get('regions', []):
        # Region-level fields
        name = region.get('element', region.get('name', 'Region'))
        energy_range = region.get('fit_range_ev', region.get('energy_range', None))
        if energy_range is None or len(energy_range) != 2:
            raise ValueError(f"Region {name}: missing or invalid energy range (fit_range_ev)")

        background_type = str(region.get('background_type', 'shirley')).lower().strip()

        # fwhm_constraint mapping (optional)
        fwhm_constraint_in = region.get('fwhm_constraint', {})
        fwhm_constraint = {}
        if isinstance(fwhm_constraint_in, dict):
            ftype = fwhm_constraint_in.get('type', None)
            if isinstance(ftype, str):
                ftype = ftype.lower().strip()
            fwhm_constraint['type'] = ftype
            if 'tol' in fwhm_constraint_in:
                try:
                    fwhm_constraint['tol'] = float(fwhm_constraint_in['tol'])
                except Exception:
                    pass  # ignore non-float tol

        region_config = {
            "name": name,
            "energy_range": (float(min(energy_range)), float(max(energy_range))),
            "background_type": background_type,
            "fwhm_constraint": fwhm_constraint,
            "components": []
        }

        # Components
        for comp in region.get('components', []):
            comp_name = comp.get('name', 'component')

            amp_cfg  = canon_param_dict(comp.get('amplitude', {}),
                                        defaults={'min_bound': 0.0, 'max_bound': 1e9})
            be_cfg   = canon_param_dict(comp.get('binding_energy', {}))
            fwhm_cfg = canon_param_dict(comp.get('fwhm', {}),
                                        defaults={'min_bound': 0.9, 'max_bound': 2.2})
            # Supply sensible defaults for eta if YAML omits them
            mix_cfg  = canon_param_dict(comp.get('pseudo_voigt_mix', {}),
                                        defaults={'initial_guess': 0.30,
                                                'min_bound': 0.25,
                                                'max_bound': 0.40})

            component = {
                "name": comp_name,
                "amplitude": amp_cfg,
                "binding_energy": be_cfg,
                "fwhm": fwhm_cfg,
                "pseudo_voigt_mix": mix_cfg
            }
            region_config["components"].append(component)

        region_config["n_peaks"] = len(region_config["components"])
        regions.append(region_config)

    return regions

def parse_region_header(file: Path) -> Tuple[Optional[str], List[str]]:
    """Parse header lines in an aggregated region CSV. Returns (region_name, original_sample_names)."""
    region_name = None
    sample_names = []
    try:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if line.startswith('# Region:'):
                    m = re.search(r'#\s*Region:\s*(.+)', line)
                    if m:
                        region_name = m.group(1).strip()
                elif line.startswith('# '):
                    m = re.match(r'#\s*(.+?):\s*points=\s*\d+\s*range=', line)
                    if m:
                        full_name = m.group(1).strip()
                        # Remove region suffix if present (e.g., "E1_C1s" -> "E1")
                        if region_name and full_name.endswith(f"_{region_name}"):
                            sample_id = full_name[:-len(f"_{region_name}")]
                        elif '_' in full_name:
                            # Generic fallback: remove last underscore component if it looks like a region
                            parts = full_name.rsplit('_', 1)
                            last_part = parts[1] if len(parts) > 1 else ''
                            if last_part and len(last_part) <= 6 and any(c.isalpha() for c in last_part):
                                sample_id = parts[0]
                            else:
                                sample_id = full_name
                        else:
                            sample_id = full_name
                        sample_names.append(sample_id)
    except Exception:
        pass
    return region_name, sample_names


def parse_multilayer_file(
    file: Path,
    allow_aggregated: bool = True,
    enforce_ascending_energy: bool = True,
    drop_nan_per_sample: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse multi-layer or multiple XPS files.

    Supports:
      1) Legacy alternating energy/intensity rows (2-line blocks).
      2) Aggregated per-region CSV with common energy column + multiple Intensity_* columns.
      3) Single-layer files (two-column format: energy, intensity).

    Returns list of (E, I) pairs for each layer/sample.
    """
    # First, try aggregated per-region CSV if enabled
    if allow_aggregated:
        try:
            # Lightweight detection: look for header and/or Intensity_* columns
            # Use pandas to read while skipping '#' header lines
            df = pd.read_csv(file, sep=None, engine='python', comment='#')
            cols = list(df.columns)

            # Detect energy and intensity columns
            energy_col = None
            for col in cols:
                if str(col).strip().lower() == 'binding_energy_ev':
                    energy_col = col
                    break
            if energy_col is None:
                energy_col = cols[0]  # fallback to first column

            intensity_cols = [c for c in cols if str(c).startswith('Intensity_')]
            
            # If no Intensity_* columns, try new format with _cps suffix
            if not intensity_cols:
                intensity_cols = [c for c in cols if str(c).endswith('_cps') and c != energy_col]
            
            # Check for single-layer two-column format
            if not intensity_cols and len(cols) == 2:
                # Assume second column is intensity
                intensity_cols = [cols[1]]
            
            if intensity_cols:
                # Aggregated format detected
                E = df[energy_col].to_numpy(dtype=float)
                if enforce_ascending_energy and len(E) >= 2 and E[0] > E[-1]:
                    df = df.iloc[::-1].reset_index(drop=True)
                    E = df[energy_col].to_numpy(dtype=float)

                pairs: List[Tuple[np.ndarray, np.ndarray]] = []
                for col in intensity_cols:
                    I = df[col].to_numpy(dtype=float)
                    if drop_nan_per_sample:
                        mask = np.isfinite(I)
                        Ej = E[mask]
                        Ij = I[mask]
                    else:
                        Ej = E.copy()
                        Ij = I
                    # Ensure minimum points
                    if len(Ej) >= 5:
                        pairs.append((Ej, Ij))
                # If we successfully parsed aggregated, return
                if pairs:
                    return pairs
        except Exception:
            # Fall back to legacy parsing if anything goes wrong
            pass

    # Legacy alternating-lines parsing (original behavior)
    lines = file.read_text(errors="ignore").splitlines()
    numeric_lines: List[List[float]] = []

    for s in lines:
        floats = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
        if len(floats) >= 3:
            numeric_lines.append([float(x) for x in floats])

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    i = 0
    while i + 1 < len(numeric_lines):
        E = np.array(numeric_lines[i], dtype=float)
        I = np.array(numeric_lines[i + 1], dtype=float)

        # Optional: enforce ascending energy for legacy format
        if enforce_ascending_energy and len(E) >= 2 and E[0] > E[-1]:
            E = E[::-1]
            I = I[::-1]

        if len(E) == len(I) and len(E) >= 5:
            pairs.append((E, I))
            i += 2
        else:
            i += 1

    return pairs


def get_aggregated_layer_labels(file: Path) -> List[str]:
    """
    Best-effort recovery of sample/layer labels from an aggregated per-region CSV.
    Returns original sample names when available, else column suffixes.
    """
    # Try header (original names)
    region_name, original_sample_names = parse_region_header(file)
    if original_sample_names:
        return original_sample_names

    # Fallback: use Intensity_* or *_cps column suffixes
    try:
        df = pd.read_csv(file, sep=None, engine='python', comment='#')
        intensity_cols = [c for c in df.columns if str(c).startswith('Intensity_')]
        
        # If no Intensity_* columns, try _cps format
        if not intensity_cols:
            intensity_cols = [c for c in df.columns if str(c).endswith('_cps')]
            # Extract labels by removing _cps suffix
            labels = [str(c).replace('_cps', '') for c in intensity_cols]
        else:
            # Return labels with Intensity_ prefix removed
            labels = [str(c).replace('Intensity_', '', 1) for c in intensity_cols]
        
        # Strip region suffix if present (e.g., "E10_F1s" -> "E10")
        cleaned_labels = []
        for label in labels:
            if '_' in label:
                parts = label.rsplit('_', 1)
                last_part = parts[1] if len(parts) > 1 else ''
                # Check if last part looks like a region name (e.g., C1s, F1s, O1s)
                if last_part and len(last_part) <= 6 and any(c.isalpha() for c in last_part):
                    cleaned_labels.append(parts[0])
                else:
                    cleaned_labels.append(label)
            else:
                cleaned_labels.append(label)
        
        return cleaned_labels
    except Exception:
        return []

def ensure_ascending_be(E: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort data by ascending binding energy."""
    idx = np.argsort(E)
    return E[idx], I[idx]

def slice_roi(E: np.ndarray, I: np.ndarray, erange: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract region of interest from spectrum."""
    e_min, e_max = float(min(erange)), float(max(erange))
    mask = (E >= e_min) & (E <= e_max)
    return E[mask], I[mask]

# ========== STEP 2: BACKGROUND CORRECTION & PEAK DETECTION ==========

# Background correction functions  imported from XPS_utils.background_correction

# ========== STEP 3: TEMPLATE-BASED PEAK FITTING ==========

def pseudo_voigt(x, center, amplitude, fwhm, eta):
    """Pseudo-Voigt profile (height-normalized L and G; amplitude = peak height)."""
    fwhm  = max(float(fwhm), 1e-6)
    eta   = float(np.clip(eta, 0.0, 1.0))
    gamma = fwhm / 2.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    t = (x - center)
    lorentz = 1.0 / (1.0 + (t / gamma)**2)
    gauss   = np.exp(-0.5 * (t / sigma)**2)

    return amplitude * (eta * lorentz + (1.0 - eta) * gauss)


def fit_region_with_template(energy, intensity, region_config):
    """
    Fit an XPS region using template-defined constraints.
    Returns a dict of fit results or None on failure.

    Notes on FWHM handling:
    - 'uniform': one shared FWHM across ALL components, bounded by the intersection
      of all components' FWHM bounds; if infeasible, falls back to per-peak.
    - 'tolerance': grouped FWHM using fwhm.is_constrained/constraint_to flags.
        * One base FWHM per group (bounded by the intersection of that group's min/max).
        * Per-peak deltas with bounds chosen so that, for any allowed base and delta,
          the effective FWHM remains within the per-peak min/max, and within ±tol
          if possible (may be tightened further to respect the template bounds).
        * If a group's intersection is infeasible, that group falls back to per-peak.
    - any other type or missing: per-peak FWHM (independent, each with its own bounds).
    """
    # 1) Turn inputs into flat numpy arrays
    x = np.asarray(energy, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()

    # 1b) Apply fit range if provided
    fit_range = region_config.get('fit_range_ev', None)
    if fit_range and len(fit_range) == 2:
        lo, hi = sorted(map(float, fit_range))
        m = (x >= lo) & (x <= hi)
        if m.sum() >= 10:
            x, y = x[m], y[m]
        else:
            print("      [WARNING]  Fit range yielded too few points; using full range")

    # 2) Quick sanity checks
    if x.size < 10:
        print(f"      [ERROR] Too few data points: {x.size}")
        return None
    print(f"      [DATA] Data range: {x.min():.1f}-{x.max():.1f} eV, "
          f"I: {y.min():.1f}-{y.max():.1f}")

    # 3) Background subtraction (with fallback)
    try:
        bg_type = region_config.get('background_type', 'shirley')
        bg_type = str(bg_type).lower().strip()
        
        # Get Tougaard parameters if using tougaard background
        tb = region_config.get('tougaard_B', 2866)
        tc = region_config.get('tougaard_C', 1643)
        
        bg = apply_background_correction(x, y, bg_type, tb, tc)
        y_corr = np.maximum(y - bg, 0.0)
        if y_corr.max() < 1e-10:
            print("      [ERROR] No signal after background correction")
            return None
    except Exception as e:
        print(f"      [ERROR] Background correction failed ({e}); falling back to linear")
        bg = np.polyval(np.polyfit(x, y, 1), x)
        y_corr = np.maximum(y - bg, 0.0)

    # 4) Grab component templates
    comps = region_config.get('components', [])
    if not comps:
        print("      [ERROR] No components defined in template")
        return None
    n_comp = len(comps)

    # ----------------------------------------------------------------
    # 5) Build up p0, lower, upper AND an index-lookup for each comp's params
# ----------------------------------------------------------------
    p0 = []
    lower = []
    upper = []

    # index lookup: param_idx['C=O']['fwhm'] -> index (or tuple) into `p0`
    param_idx = {c['name']: {} for c in comps}
    comp_names = [c['name'] for c in comps]

    Imax = float(max(y_corr.max(), 1e-6))
    x_min, x_max = float(x.min()), float(x.max())
    
    # Check if amplitude auto-scaling will be needed
    template_max_amps = [float(c.get('amplitude', {}).get('max_bound', 1000)) for c in comps]
    avg_template_max = np.mean(template_max_amps) if template_max_amps else 1000
    if Imax > avg_template_max * 1.5:
        scale_factor = Imax / avg_template_max
        print(f"      [DATA] Auto-scaling amplitude bounds: data intensity {Imax:.1f} >> template max {avg_template_max:.1f} (scale factor: {scale_factor:.2f}x)")
    
    # Auto-detect peaks in the data to provide better initial guesses
    try:
        # Smooth data for peak detection
        from scipy.signal import savgol_filter
        y_smooth = savgol_filter(y_corr, window_length=min(11, len(y_corr)//2*2+1), polyorder=2)
        
        # Find peaks with prominence threshold
        peak_threshold = 0.1 * y_corr.max()  # 10% of max intensity
        peak_indices, peak_props = find_peaks(y_smooth, prominence=peak_threshold, width=1)
        
        if len(peak_indices) > 0:
            peak_positions = x[peak_indices]
            peak_heights = y_corr[peak_indices]
            # Only print if helpful (not too many peaks cluttering output)
            if len(peak_indices) <= n_comp + 2:
                print(f"      [DATA] Peak detection: {len(peak_indices)} peaks found to guide fitting")
        else:
            peak_positions = []
            peak_heights = []
    except Exception as e:
        peak_positions = []
        peak_heights = []

    # ========================================================================
    # Pass 1: amplitude and center per component
    # ========================================================================
    for comp in comps:
        name = comp['name']

        # ===== AMPLITUDE =====
        A_cfg = comp.get('amplitude', {})
        if not A_cfg:
            raise ValueError(f"Component '{name}' missing 'amplitude' configuration in YAML")
        
        if 'min_bound' not in A_cfg or 'max_bound' not in A_cfg:
            raise ValueError(f"Component '{name}' missing amplitude min_bound or max_bound in YAML")
        
        # Read template bounds as baseline reference
        Amin_template = float(A_cfg['min_bound'])
        Amax_template = float(A_cfg['max_bound'])
        
        # Get component's expected binding energy for matching with detected peaks
        be_cfg = comp.get('binding_energy', {})
        be_guess = float(be_cfg.get('initial_guess', (be_cfg.get('min_bound', 0) + be_cfg.get('max_bound', 0)) / 2))
        
        # Try to match this component with a detected peak
        matched_peak_height = None
        if len(peak_positions) > 0:
            # Find closest detected peak to this component's expected position
            be_diffs = np.abs(peak_positions - be_guess)
            closest_idx = np.argmin(be_diffs)
            if be_diffs[closest_idx] < 3.0:  # Within 3 eV tolerance
                matched_peak_height = peak_heights[closest_idx]
        
        # Auto-scale bounds based on actual data intensity
        data_scale = max(Imax, 1.0)
        template_scale = max(Amax_template, 1.0)
        scale_factor = data_scale / template_scale if template_scale > 0 else 1.0
        
        # Only scale up if data is significantly larger than template expectation
        if scale_factor > 1.5:
            Amin = Amin_template
            Amax = Amax_template * scale_factor * 3.0  # 3x safety margin
        else:
            Amin = Amin_template
            Amax = Amax_template
        
        # Initial guess priority:
        # 1. Use matched detected peak height if available
        # 2. Use YAML initial_guess scaled by data
        # 3. Fallback to data-driven estimate
        if matched_peak_height is not None:
            A0 = matched_peak_height * 0.8  # Use 80% of peak height as initial guess
        else:
            A0_guess = float(A_cfg.get('initial_guess', Imax / max(1, n_comp)))
            # Scale initial guess with data
            A0 = A0_guess * scale_factor if scale_factor > 1.5 else A0_guess
        
        A0 = float(np.clip(A0, Amin, Amax))
        
        idxA = len(p0)
        p0.append(A0)
        lower.append(Amin)
        upper.append(Amax)
        param_idx[name]['amplitude'] = idxA

        # ===== CENTER (Binding Energy) =====
        be = comp.get('binding_energy', {})
        if not be:
            raise ValueError(f"Component '{name}' missing 'binding_energy' configuration in YAML")
        
        if 'min_bound' not in be or 'max_bound' not in be:
            raise ValueError(f"Component '{name}' missing binding_energy min_bound or max_bound in YAML")
        
        cmin = float(be['min_bound'])
        cmax = float(be['max_bound'])
        
        if cmin > cmax:
            cmin, cmax = cmax, cmin
            print(f"      [WARNING]  Warning: BE bounds inverted for '{name}', swapping to [{cmin:.2f}, {cmax:.2f}]")
        
        # Initial guess: use YAML value or fallback to midpoint
        c0 = float(be.get('initial_guess', (cmin + cmax) / 2.0))
        c0 = float(np.clip(c0, cmin, cmax))
        
        idxC = len(p0)
        p0.append(c0)
        lower.append(cmin)
        upper.append(cmax)
        param_idx[name]['center'] = idxC

    # ========================================================================
    # Pass 2: FWHM constraints (uniform, grouped tolerance, or per-peak)
    # ========================================================================
    fwhm_cfg = region_config.get('fwhm_constraint', {}) or {}
    fwhm_type = str(fwhm_cfg.get('type', '')).lower().strip()
    tol = float(fwhm_cfg.get('tol', 0.0))
    tol = max(0.0, min(tol, 0.95))  # guard against pathological tolerances

    # Collect per-component FWHM hints from template
    percomp_w = {}
    for comp in comps:
        name = comp['name']
        fw = comp.get('fwhm', {})
        
        if not fw:
            raise ValueError(f"Component '{name}' missing 'fwhm' configuration in YAML")
        
        if 'min_bound' not in fw or 'max_bound' not in fw:
            raise ValueError(f"Component '{name}' missing fwhm min_bound or max_bound in YAML")
        
        wmin = float(fw['min_bound'])
        wmax = float(fw['max_bound'])
        
        if wmin <= 0:
            print(f"      [WARNING]  Warning: FWHM min_bound ≤ 0 for '{name}', setting to 0.05")
            wmin = 0.05
        
        if wmin > wmax:
            wmin, wmax = wmax, wmin
            print(f"      [WARNING]  Warning: FWHM bounds inverted for '{name}', swapping to [{wmin:.2f}, {wmax:.2f}]")
        
        # Initial guess: use YAML value or fallback to midpoint
        w0 = float(fw.get('initial_guess', (wmin + wmax) / 2.0))
        w0 = float(np.clip(w0, wmin, wmax))
        
        percomp_w[name] = dict(wmin=wmin, wmax=wmax, w0=w0, cfg=fw)

    # Helper: build FWHM groups from per-component flags (constraint_to)
    def build_groups_from_flags():
        leader_for = {}
        for comp in comps:
            name = comp['name']
            fw = comp.get('fwhm', {}) or {}
            if bool(fw.get('is_constrained', False)) and isinstance(fw.get('constraint_to'), str):
                master = fw['constraint_to']
                if master in comp_names and master != name:
                    leader_for[name] = master
                else:
                    leader_for[name] = name
            else:
                leader_for[name] = name
        groups = {}
        for comp in comps:
            name = comp['name']
            gkey = leader_for[name]
            groups.setdefault(gkey, []).append(name)
        return groups

    # Case 1: region-wide uniform (single shared FWHM across ALL components)
    if fwhm_type == 'uniform':
        all_wmins = [percomp_w[n]['wmin'] for n in comp_names]
        all_wmaxs = [percomp_w[n]['wmax'] for n in comp_names]
        all_w0s = [percomp_w[n]['w0'] for n in comp_names]
        lo_inter = float(max(all_wmins))
        hi_inter = float(min(all_wmaxs))
        if lo_inter < hi_inter:
            idxW = len(p0)
            p0.append(float(np.clip(np.mean(all_w0s), lo_inter, hi_inter)))
            lower.append(lo_inter)
            upper.append(hi_inter)
            for name in comp_names:
                param_idx[name]['fwhm'] = idxW
        else:
            # Intersection empty -> fallback to per-peak
            print(f"      [WARNING]  Warning: Uniform FWHM bounds have no intersection, using per-peak")
            for name in comp_names:
                wmin = percomp_w[name]['wmin']
                wmax = percomp_w[name]['wmax']
                w0 = percomp_w[name]['w0']
                idxW = len(p0)
                p0.append(w0)
                lower.append(wmin)
                upper.append(wmax)
                param_idx[name]['fwhm'] = idxW

    # Case 2: grouped tolerance ±tol within per-peak ranges
    elif fwhm_type == 'tolerance':
        EPS = 1e-8  # tiny epsilon for strict inequalities
        groups = build_groups_from_flags()
        fwhm_base_indices = set()

        for gkey, members in groups.items():
            # Group intersection for base bounds
            wmins = [percomp_w[n]['wmin'] for n in members]
            wmaxs = [percomp_w[n]['wmax'] for n in members]
            w0s = [percomp_w[n]['w0'] for n in members]
            B_lo = float(max(wmins))
            B_hi = float(min(wmaxs))

            # If intersection is empty or nearly degenerate, fallback per-peak for this group
            if not (B_lo + EPS < B_hi):
                print(f"      [WARNING]  Warning: FWHM group '{gkey}' has no valid intersection, using per-peak")
                for name in members:
                    wmin = percomp_w[name]['wmin']
                    wmax = percomp_w[name]['wmax']
                    w0 = percomp_w[name]['w0']
                    idxW = len(p0)
                    p0.append(w0)
                    lower.append(wmin)
                    upper.append(wmax)
                    param_idx[name]['fwhm'] = idxW
                continue

            # Create shared base within intersection
            idx_base = len(p0)
            w0_master = float(np.clip(np.mean(w0s), B_lo, B_hi))
            p0.append(w0_master)
            lower.append(B_lo)
            upper.append(B_hi)
            fwhm_base_indices.add(idx_base)

            # For each member, create a delta only if it has a non-trivial feasible interval
            for name in members:
                wmin_i = percomp_w[name]['wmin']
                wmax_i = percomp_w[name]['wmax']

                d_lo_safe = (wmin_i / B_hi) - 1.0  # Note: use B_hi for lower delta bound
                d_hi_safe = (wmax_i / B_lo) - 1.0  # Note: use B_lo for upper delta bound
                d_lo = max(-tol, d_lo_safe, -0.95)
                d_hi = min(+tol, d_hi_safe)

                # If delta interval collapses or is too narrow, share base only (no delta param)
                if d_hi - d_lo < 1e-6:
                    param_idx[name]['fwhm'] = idx_base
                else:
                    idx_delta = len(p0)
                    p0.append(0.0)
                    lower.append(d_lo)
                    upper.append(d_hi)
                    param_idx[name]['fwhm'] = (idx_base, idx_delta)

    # Default: per-peak FWHM (unconstrained)
    else:
        for name in comp_names:
            wmin = percomp_w[name]['wmin']
            wmax = percomp_w[name]['wmax']
            w0 = percomp_w[name]['w0']
            idxW = len(p0)
            p0.append(w0)
            lower.append(wmin)
            upper.append(wmax)
            param_idx[name]['fwhm'] = idxW

    # ========================================================================
    # Pass 3: pseudo-Voigt mix (eta) per component, with optional constraint_to
    # ========================================================================
    eta_group_index = {}  # maps group key -> (eta_idx, eta_min, eta_max, eta0)

    for comp in comps:
        name = comp['name']
        mix = comp.get('pseudo_voigt_mix', {})
        
        if not mix:
            raise ValueError(f"Component '{name}' missing 'pseudo_voigt_mix' configuration in YAML")
        
        if 'min_bound' not in mix or 'max_bound' not in mix:
            raise ValueError(f"Component '{name}' missing eta min_bound or max_bound in YAML")
        
        is_constrained = bool(mix.get('is_constrained', False))
        constraint_to = mix.get('constraint_to', None)
        
        # Determine group key
        if is_constrained and isinstance(constraint_to, str) and constraint_to in comp_names and constraint_to != name:
            group_key = constraint_to
        else:
            group_key = name
        
        # Read eta bounds from YAML
        eta_min = float(mix['min_bound'])
        eta_max = float(mix['max_bound'])
        
        if eta_min > eta_max:
            eta_min, eta_max = eta_max, eta_min
            print(f"      [WARNING]  Warning: eta bounds inverted for '{name}', swapping to [{eta_min:.3f}, {eta_max:.3f}]")
        
        # Initial guess: use YAML value or fallback to midpoint
        eta0 = float(mix.get('initial_guess', (eta_min + eta_max) / 2.0))
        eta0 = float(np.clip(eta0, eta_min, eta_max))
        
        if group_key in eta_group_index:
            # Group already exists - use intersection of bounds
            existing_idx, existing_min, existing_max, existing_eta0 = eta_group_index[group_key]
            
            # Take intersection of bounds (most restrictive)
            new_min = max(existing_min, eta_min)
            new_max = min(existing_max, eta_max)
            
            if new_min > new_max:
                raise ValueError(
                    f"[ERROR] Eta bounds conflict for constrained group '{group_key}':\n"
                    f"   Master component bounds: [{existing_min:.3f}, {existing_max:.3f}]\n"
                    f"   Component '{name}' bounds: [{eta_min:.3f}, {eta_max:.3f}]\n"
                    f"   No valid intersection! Please fix YAML to use consistent bounds."
                )
            
            # Warn if bounds were tightened
            if new_min > existing_min or new_max < existing_max:
                print(f"      [RULE] Tightening eta bounds for group '{group_key}': "
                    f"[{existing_min:.3f}, {existing_max:.3f}] -> [{new_min:.3f}, {new_max:.3f}]")
            
            # Update bounds in p0 arrays
            lower[existing_idx] = new_min
            upper[existing_idx] = new_max
            p0[existing_idx] = float(np.clip(p0[existing_idx], new_min, new_max))
            
            # Update stored bounds
            eta_group_index[group_key] = (existing_idx, new_min, new_max, p0[existing_idx])
            param_idx[name]['eta'] = existing_idx
        else:
            # Create new eta parameter for this group
            idxE = len(p0)
            p0.append(eta0)
            lower.append(eta_min)
            upper.append(eta_max)
            eta_group_index[group_key] = (idxE, eta_min, eta_max, eta0)
            param_idx[name]['eta'] = idxE

    # ========================================================================
    # Sanity checks for param_idx mapping
    # ========================================================================
    for comp in comps:
        name = comp['name']
        for key in ['amplitude', 'center', 'eta']:
            ix = param_idx[name][key]
            assert isinstance(ix, int), f"{name}.{key} index is not int: {ix}"
            assert 0 <= ix < len(p0), f"{name}.{key} index {ix} out of range for p0 len {len(p0)}"
        w_ref = param_idx[name]['fwhm']
        if isinstance(w_ref, tuple):
            base_idx, delta_idx = w_ref
            assert isinstance(base_idx, int) and isinstance(delta_idx, int), f"{name}.fwhm tuple not ints: {w_ref}"
            assert 0 <= base_idx < len(p0), f"{name}.fwhm base idx {base_idx} out of range"
            assert 0 <= delta_idx < len(p0), f"{name}.fwhm delta idx {delta_idx} out of range"
        else:
            assert isinstance(w_ref, int), f"{name}.fwhm index not int: {w_ref}"
            assert 0 <= w_ref < len(p0), f"{name}.fwhm index {w_ref} out of range"

    # ========================================================================
    # Debug printout
    # ========================================================================
    print(f"      [TARGET] Fitting {n_comp} components with {len(p0)} free params")
    for comp in comps:
        name = comp['name']
        A0 = p0[param_idx[name]['amplitude']]
        c0 = p0[param_idx[name]['center']]
        w_ref = param_idx[name]['fwhm']
        eta_idx = param_idx[name]['eta']
        eta0 = p0[eta_idx]
        eta_lo = lower[eta_idx]
        eta_hi = upper[eta_idx]
        
        if isinstance(w_ref, tuple):
            base_idx, delta_idx = w_ref
            b0 = p0[base_idx]
            d0 = p0[delta_idx]
            b_lo, b_hi = lower[base_idx], upper[base_idx]
            d_lo, d_hi = lower[delta_idx], upper[delta_idx]
            w0 = b0 * (1.0 + d0)
            w_lo = b_lo * (1.0 + d_lo)
            w_hi = b_hi * (1.0 + d_hi)
            constr = f"base idx {base_idx} + delta idx {delta_idx}"
        else:
            w_idx = w_ref
            w0 = p0[w_idx]
            w_lo = lower[w_idx]
            w_hi = upper[w_idx]
            if fwhm_type == 'uniform':
                constr = f"shared idx {w_idx}"
            elif fwhm_type == 'tolerance' and 'fwhm_base_indices' in locals() and w_idx in fwhm_base_indices:
                constr = "shared base (no delta)"
            else:
                constr = "None"
        
        print(f"         {name:12s}: A={A0:7.1f}, BE={c0:6.2f}, "
            f"FWHM={w0:.2f} [{w_lo:.2f},{w_hi:.2f}] ({constr}), "
            f"eta={eta0:.3f} [{eta_lo:.3f},{eta_hi:.3f}]")

  
    # 6) Define the composite model (captures param_idx & comps by closure)
    # ----------------------------------------------------------------
    def simple_multi_peak(x_data, *params):
        """
        Composite pseudo-Voigt model for multiple components.
        Reconstructs parameters using param_idx, supporting:
          - shared FWHM (uniform)
          - base FWHM + per-peak delta (tolerance, grouped by flags)
          - per-peak FWHM (unconstrained)
        """
        y_sum = np.zeros_like(x_data, dtype=float)

        for comp in comps:
            name = comp['name']

            # amplitude (non-negative)
            A_idx = param_idx[name]['amplitude']
            A     = max(float(params[A_idx]), 0.0)

            # center (binding energy)
            C_idx = param_idx[name]['center']
            cen   = float(params[C_idx])

            # FWHM reconstruction
            w_ref = param_idx[name]['fwhm']
            if isinstance(w_ref, tuple):
                base_idx, delta_idx = w_ref
                base  = max(float(params[base_idx]), 0.05)
                delta = float(params[delta_idx])      # bounds enforce valid range
                w     = max(base * (1.0 + delta), 0.05)
            else:
                w = max(float(params[w_ref]), 0.05)

            # pseudo-Voigt mixing (eta in [0,1])
            E_idx = param_idx[name]['eta']
            eta   = float(np.clip(params[E_idx], 0.0, 1.0))

            # accumulate component
            y_sum += pseudo_voigt(x_data, cen, A, w, eta)

        return y_sum

    # ----------------------------------------------------------------
    # 7) Try fitting
    # ----------------------------------------------------------------
    # Add bounds validation before fitting
    try:
        # Verify all bounds are valid
        for i, (lo, hi) in enumerate(zip(lower, upper)):
            if lo >= hi:
                print(f"      [WARNING]  Invalid bounds at parameter {i}: [{lo}, {hi}]")
                # Fix by adjusting bounds slightly
                mean = (lo + hi) / 2
                span = abs(hi - lo) or 0.1  # Use 0.1 if bounds are equal
                lower[i] = mean - span/2
                upper[i] = mean + span/2
                print(f"      [NOTE] Adjusted to: [{lower[i]}, {upper[i]}]")
        
        # Verify initial values are within bounds
        for i, (p, lo, hi) in enumerate(zip(p0, lower, upper)):
            if not (lo <= p <= hi):
                print(f"      [WARNING]  Initial value {p} outside bounds [{lo}, {hi}]")
                p0[i] = (lo + hi) / 2
                print(f"      [NOTE] Adjusted to: {p0[i]}")

    except Exception as e:
        print(f"      [ERROR] Error validating bounds: {str(e)}")
        return None

    # Continue with fitting attempt
    
    popt = None
    pcov = None

    attempts = [
        dict(method='trf',    maxfev=10000),
        dict(method='dogbox', maxfev= 8000),
    ]
    for kw in attempts:
        try:
            popt, pcov = curve_fit(
                simple_multi_peak, x, y_corr,
                p0=p0, bounds=(lower, upper),
                **kw
            )
            print(f"      [OK] Fit succeeded ({kw['method']})")
            break
        except Exception as e:
            print(f"      [WARNING]  {kw['method']} failed ({str(e)[:80]}...)")

    if popt is None:
        print("      [ERROR] All fitting attempts failed")
        return None
    # function should respect bounds, not due to numerical precision or convergence tolerance
    # Clip all fitted parameters to bounds (curve_fit can return values slightly outside due to numerical precision)
    popt = np.array([np.clip(p, lo, hi) for p, lo, hi in zip(popt, lower, upper)])

    # 8) Compute quality & extract
    fitted = simple_multi_peak(x, *popt)
    ss_res = np.sum((y_corr - fitted)**2)
    ss_tot = np.sum((y_corr - y_corr.mean())**2)
    r2     = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

    components = {}
    peaks = []
    temp_peaks = []
    total_area = 0.0

    for comp in comps:
        name = comp['name']

        # amplitude & center
        A   = max(popt[param_idx[name]['amplitude']], 0.0)
        cen = popt[param_idx[name]['center']]

        # FWHM (shared/tol/per-peak)
        w_ix = param_idx[name]['fwhm']
        if isinstance(w_ix, tuple):
            base_idx, delta_idx = w_ix
            fwhm = max(popt[base_idx] * (1.0 + popt[delta_idx]), 0.1)
        else:
            fwhm = max(popt[w_ix], 0.1)

        # mixing
        eta_ix = param_idx[name]['eta']
        eta    = float(np.clip(popt[eta_ix], 0.0, 1.0))

        # curve & area
        curve = pseudo_voigt(x, cen, A, fwhm, eta)
        area  = float(np.abs(np.trapezoid(curve, x)))
        total_area += area
        temp_peaks.append((name, curve, A, cen, fwhm, eta, area))

    # Filter only >=2%-area peaks
    components = {}
    peaks      = []
    for name, curve, A, cen, fwhm, eta, area in temp_peaks:
        pct = 100.0 * area / total_area if total_area > 0 else 0
        if pct >= 2.0:
            components[name] = curve
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            gamma = fwhm / 2.0
            peaks.append({
                'name':         name,
                'amplitude':    A,
                'center':       cen,
                'fwhm':         fwhm,
                'eta':          eta,
                'area':         area,
                'area_percent': pct,
                'sigma':        sigma,
                'gamma':        gamma
            })

    if not peaks:
        print("      [WARNING]  All components have area < 2%")
        return None

    print(f"      [OK] Fit complete: R^2 = {r2:.3f}, {len(peaks)} peaks >=2% area")
    return {
        'x':            x,
        'raw':          y,
        'baseline':     bg,
        'corrected':    y_corr,
        'fit':          simple_multi_peak(x, *popt),
        'components':   components,
        'peaks':        peaks,
        'params':       popt,
        'r2':           r2,
        'template_used': region_config.get('template_name', region_config.get('name', 'Unknown'))
    }

# ========== STEP 4: REPORTING ==========

def process_file_with_template(file: Path, template_path: Path) -> Tuple[pd.DataFrame, List, Dict]:
    """Process XPS file using YAML template."""
    print(f"  [FILE] Loading template: {template_path.name}")
    template = load_yaml_template(template_path)
    viz_root = template.get('visualization', {}) or {}
    vis_settings = viz_root.get('colormap', {}) or {}
    use_save_plots = bool(viz_root.get('SAVE_PLOTS', SAVE_PLOTS))
    # Only use template figsize if explicitly set, otherwise None lets plotter use plot_settings.yaml
    fig_size = tuple(viz_root['FIGSIZE']) if 'FIGSIZE' in viz_root else None
    fig_dpi = int(viz_root.get('DPI', 300))
    regions = parse_template_to_regions(template)

    layers = parse_multilayer_file(file)
    if not layers:
        print(f"  [WARNING]  No layers found in {file.name}")
        return pd.DataFrame(), [], None

    print(f"  [DATA] Found {len(layers)} layers in {file.name}")

    # Try to recover per-layer labels for aggregated CSVs
    labels = get_aggregated_layer_labels(file)  # [] if not aggregated

    # Determine if this is a true multi-layer depth profile or a standard single-spectrum file
    is_depth_profile = len(layers) > 1

    all_rows = []
    plots = []

    # For stacked comparison plots: {region_name: {label: fit_result}}
    sample_layer_fits: Dict[str, Dict[str, Dict]] = {}
    # For curve export
    fit_results_data = {
        'sample_name': file.stem,
        'layers_by_region': {},
        'layer_labels': {}
    }

    for layer_idx, (E, I) in enumerate(layers, start=1):
        # Determine label: use recovered sample name or fallback
        if labels and 0 <= (layer_idx - 1) < len(labels):
            sample_label = sanitize_filename(labels[layer_idx - 1])
        else:
            sample_label = f"{sanitize_filename(file.stem)}_L{layer_idx}"

        E, I = ensure_ascending_be(E, I)
        print(f"    [SEARCH] Processing layer {layer_idx} ({sample_label}): {len(E)} data points")

        for reg in regions:
            Ex, Ix = slice_roi(E, I, reg["energy_range"])
            if len(Ex) < 10:
                print(f"      [WARNING]  Too few points in {reg['name']} region")
                continue

            print(f"      [GEAR]  Fitting {reg['name']} region ({len(Ex)} points)")
            fit = fit_region_with_template(Ex, Ix, reg)
            if fit is None:
                continue

            # Store fit for stacked plots keyed by sample label
            sample_layer_fits.setdefault(reg["name"], {})[sample_label] = fit
            # Store fit for curve export keyed by layer_idx
            fit_results_data['layers_by_region'].setdefault(reg["name"], {})[layer_idx] = fit
            fit_results_data['layer_labels'][layer_idx] = sample_label

            # Quantitative results
            total_area = sum(p["area"] for p in fit["peaks"] if np.isfinite(p["area"]))
            for p in fit["peaks"]:
                all_rows.append({
                    "File": file.name,
                    "Sample": sample_label,
                    "Layer": layer_idx if is_depth_profile else 1,  # Layer=1 for standard files, layer_idx for depth profiles
                    "Region": reg["name"],
                    "Component": p["name"],
                    "Center_eV": round(p["center"], 2),
                    "FWHM_eV": round(p["fwhm"], 2),
                    "Eta_mix": round(p["eta"], 2),
                    "Total_area": round(total_area, 1),
                    "Area_counts": round(p["area"], 1),
                    "Area_percent": round(100 * p["area"] / total_area, 1) if total_area > 0 else 0,
                    "R_squared": round(fit["r2"], 3),
                    "Template": fit["template_used"]
                })

            # Individual layer plot
            if use_save_plots:
                out_plot_dir = Path(PLOTS_DIR) / "individual_layers"
                out_plot_dir.mkdir(parents=True, exist_ok=True)
                
                # Only pass figsize if template explicitly sets it, otherwise use plot_settings.yaml
                kwargs = {'outdir': out_plot_dir, 'dpi': fig_dpi}
                if fig_size is not None:
                    kwargs['figsize'] = fig_size
                
                plot_path = plot_template_fit(
                    sample_label,
                    fit,
                    reg["name"],
                    vis_settings,
                    **kwargs
                )
                if plot_path is not None:
                    plots.append((file.name, reg["name"], sample_label, plot_path))

            print(f"      [OK] {sample_label}: R² = {fit['r2']:.3f}")

    # Stacked comparison plots per region
    if use_save_plots and sample_layer_fits:
        comparison_plot_dir = Path(PLOTS_DIR) / "stacked_comparison"
        comparison_plot_dir.mkdir(parents=True, exist_ok=True)
        for region_name, layer_fits in sample_layer_fits.items():
            # Changed condition: create stacked plot for 1 or more layers
            if len(layer_fits) >= 1:  # Changed from > 1 to >= 1
                try:
                    # Only pass figsize if template explicitly sets it, otherwise use plot_settings.yaml
                    kwargs = {
                        'outdir': comparison_plot_dir,
                        'dpi': fig_dpi,
                    }
                    if fig_size is not None:
                        kwargs['figsize'] = fig_size
                    
                    stacked_plot_path = plot_stacked_layers_comparison(
                        file.stem,
                        layer_fits,
                        region_name,
                        vis_settings,
                        template_path,
                        **kwargs
                    )
                    # Add stacked plot to plots list with special marker
                    if stacked_plot_path is not None:
                        plots.append((file.name, region_name, "STACKED", stacked_plot_path))
                        print(f"      [BOOK] Stacked plot created: {stacked_plot_path.name}")
                except Exception as e:
                    print(f"      [WARNING]  Failed to create stacked plot for {region_name}: {e}")

    return pd.DataFrame(all_rows), plots, fit_results_data


def discover_available_regions_and_templates(input_dir: Path, template_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Dynamically discover available regions from converted CSV folder and match with templates.
    Returns dict: {region_name: template_path or None}
    """
    region_template_map = {}
    
    # Get all region folders from input directory
    region_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"[LIST] Found {len(region_dirs)} region directories: {[d.name for d in region_dirs]}")
    
    # Get all available templates
    template_files = list(template_dir.glob("*.yaml")) + list(template_dir.glob("*.yml"))
    print(f"[LIST] Found {len(template_files)} template files: {[t.name for t in template_files]}")
    
    # Match regions with templates
    for region_dir in region_dirs:
        region_name = region_dir.name
        template_path = None
        
        # Look for matching template
        for template_file in template_files:
            template_stem = template_file.stem.upper()
            region_upper = region_name.upper()
            
            # Check if region name is in template name
            if region_upper in template_stem:
                template_path = template_file
                break
        
        region_template_map[region_name] = template_path
        
        if template_path:
            print(f"  [OK] {region_name} -> {template_path.name}")
        else:
            print(f"  [ERROR] {region_name} -> No matching template found")
    
    return region_template_map


def main():
    """Main execution with dynamic region discovery and template matching."""
    print("[SCIENCE] Starting Dynamic Template-Based XPS Analysis...")
    
    # Setup directories
    in_dir = Path(INPUT_DIR)
    template_dir = Path(TEMPLATE_DIR)
    out_dir = Path(OUTPUT_DIR)
    
    print(f"[DIR] Input directory: {in_dir}")
    print(f"[DIR] Template directory: {template_dir}")
    print(f"[DIR] Output directory: {out_dir}")
    
    # Check if directories exist
    if not in_dir.exists():
        raise SystemExit(f"[ERROR] Input directory does not exist: {in_dir}")
    if not template_dir.exists():
        raise SystemExit(f"[ERROR] Template directory does not exist: {template_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover available regions and templates
    region_template_map = discover_available_regions_and_templates(in_dir, template_dir)

    # Exclude non-fitting folders (survey and other helper folders)
    excluded_regions = {"survey", "quality", "debug", "plots"}
    candidate_regions = [
        region for region in region_template_map.keys()
        if region.lower() not in excluded_regions
    ]
    
    # Find files to process based on discovered regions
    files_to_process = []
    regions_with_templates = [
        region for region, template in region_template_map.items()
        if region in candidate_regions and template is not None
    ]
    regions_without_templates = [
        region for region, template in region_template_map.items()
        if region in candidate_regions and template is None
    ]

    if not candidate_regions:
        print("[WARNING] No narrow-scan region folders found (only survey or helper folders).")
        print("[INFO] Skipping peak fitting step.")
        raise SystemExit(0)
    
    print(f"\n[TARGET] Regions with templates: {regions_with_templates}")
    if regions_without_templates:
        print(f"[WARNING]  Regions without templates: {regions_without_templates}")
    
    # Collect files from regions that have templates
    for region in regions_with_templates:
        region_dir = in_dir / region
        region_files = list(region_dir.glob("*.csv"))
        for file in region_files:
            files_to_process.append((file, region_template_map[region]))
    
    if not files_to_process:
        if regions_without_templates:
            print("\n[ERROR] No files to process. Missing templates for regions:")
            for region in regions_without_templates:
                print(f"   - {region}: Please add matching template in {template_dir}")
        raise SystemExit("[ERROR] No files to process - no matching region/template pairs found.")
    
    print(f"\n[FOLDER] Found {len(files_to_process)} files to process across {len(regions_with_templates)} regions")

    all_results = []
    all_plots = []
    all_fit_data = []
    timing_stats = TimingStats("File processing")

    # Process each file with its matched template
    for i, (file, template_path) in enumerate(files_to_process, 1):
        print(f"\n[GEAR]  Processing file {i}/{len(files_to_process)}: {file.name} with {template_path.name}")

        with timing_stats.time():
            try:
                df, plots, fit_data = process_file_with_template(file, template_path)
                if not df.empty:
                    all_results.append(df)
                    all_plots.extend(plots)
                    if fit_data:
                        all_fit_data.append(fit_data)
                    print(f"  [OK] Processed {len(df)} components")
            except Exception as e:
                print(f"  [ERROR] Error processing {file.name}: {str(e)}")
                continue

    # Generate final reports
    if not all_results:
        if timing_stats.timings:
            print()
            timing_stats.print_summary()
        raise SystemExit("[ERROR] No successful fits produced.")

    if timing_stats.timings:
        print()
        timing_stats.print_summary()

    df_final = pd.concat(all_results, ignore_index=True)

    # Save results by region and sample
    for (sample, region), group in df_final.groupby(['Sample', 'Region']):
        sample_label = sanitize_filename(sample)

        # Create region-specific subfolder
        region_folder = out_dir / region
        region_folder.mkdir(parents=True, exist_ok=True)

        peaks_csv = region_folder / f"{sample_label}_{region}_analysis_results.csv"
        peaks_json = region_folder / f"{sample_label}_{region}_analysis_results.json"

        group.to_csv(peaks_csv, index=False)
        group.to_json(peaks_json, orient='records', indent=2)

    def pad_arrays_to_common_length(curves_data):
                """Pad all arrays in curves_data to the length of the longest array."""
                # Find maximum length
                max_len = max(len(arr) for arr in curves_data.values())
                
                # Pad each array with NaN to match max_len
                padded_data = {}
                for key, arr in curves_data.items():
                    if len(arr) < max_len:
                        padded = np.pad(arr, 
                                    (0, max_len - len(arr)), 
                                    mode='constant', 
                                    constant_values=np.nan)
                        padded_data[key] = padded
                    else:
                        padded_data[key] = arr
                        
                return padded_data
    # Save curve data
    for fit_data in all_fit_data:
        default_sample = sanitize_filename(fit_data.get('sample_name', 'sample'))
        layer_labels_map = fit_data.get('layer_labels', {})  # {layer_idx: sample_label}

        for region_name, layer_fits in fit_data.get('layers_by_region', {}).items():
            if not layer_fits:
                continue

            for layer_idx, fit_result in layer_fits.items():
                if not fit_result:
                    continue

                # Build curves_data...
                curves_data = {}
                x = fit_result['x']
                curves_data['BE_L1'] = x
                curves_data['Raw_L1'] = fit_result['raw']
                curves_data['Baseline_L1'] = fit_result['baseline']
                curves_data['Fit_L1'] = fit_result['fit']
                for comp_name, comp_curve in fit_result['components'].items():
                    curves_data[f'{comp_name}_L1'] = comp_curve

                if curves_data:
                    padded_data = pad_arrays_to_common_length(curves_data)
                    curves_df = pd.DataFrame(padded_data)

                    # Create region-specific subfolder for curve data
                    region_folder = out_dir / region_name
                    region_folder.mkdir(parents=True, exist_ok=True)

                    # Use per-layer sample label if available
                    sample_label = layer_labels_map.get(layer_idx, default_sample)
                    curves_csv = region_folder / f"{sample_label}_{region_name}_fitted_curves.csv"
                    curves_df.to_csv(curves_csv, index=False)

    
    # Save plot index
    if all_plots:
        # plots entries: (File, Region, Sample, PlotPath)
        plot_df = pd.DataFrame(all_plots, columns=["File", "Region", "Sample", "PlotPath"])
        plot_index_path = out_dir / "Plot_Index.csv"
        plot_df.to_csv(plot_index_path, index=False)


    # Print summary
    print(f"\n[OK] Template-based analysis complete!")
    print(f"[DATA] Results saved in: {out_dir}")
    print(f"[PLOT] Plots generated: {len(all_plots)}")
    print(f"   [LIST] Individual layer plots: {len([p for p in all_plots if 'individual_layers' in str(p[3])])}")
    print(f"   [BOOK] Stacked comparison plots: {len([p for p in all_plots if 'stacked_comparison' in str(p[3])])}")
    
    # Show sample results
    if not df_final.empty:
        print("\n[LIST] Sample Results:")
        print(df_final.head(10).to_string(index=False))

if __name__ == "__main__":
    with Timer("XPS peak fitting pipeline"):
        main()
