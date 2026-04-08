"""
INTEGRATED XPS WORKFLOW with YAML Configuration
Step 1: Load fitting parameters from JSON files(total area needed)
Step 2: Calculate the atomic concentration from fitted total area of each element(default); or if user provided, read from the XPS text files (RSF-corrected)
Step 3: Calculate component atomic % based on area % and elemental atomic %
Step 4: Generate plots for both atomic concentration and component chemistry
"""

from __future__ import annotations
import re
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json
import argparse
from typing import List, Dict, Optional, Tuple
import yaml
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
# Add XPS_Plotter directory to path
import sys
from pathlib import Path

tools_dir = Path(__file__).resolve().parents[1]
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

try:
    from XPS_Plotter.plot_modules.quantification.quantification_plots import (
        plot_atomic_concentration_per_sample,
        plot_atomic_concentration_layer_comparison,
        generate_component_chemistry_plots,
        generate_component_heatmap
    )
except ImportError as e:
    print(f"⚠️  Could not import quantification plotting functions: {e}")
    def plot_atomic_concentration_per_sample(*args, **kwargs): 
        return None
    def plot_atomic_concentration_layer_comparison(*args, **kwargs): 
        return None
    def generate_component_chemistry_plots(*args, **kwargs):
        return []
    def generate_component_heatmap(*args, **kwargs):
        return None

# Simple config replacement for missing config_loader
@dataclass
class MockProject:
    name: str = "XPS Quantification Analysis"
    description: str = "Automated XPS elemental quantification workflow"

@dataclass  
class MockPlot:
    style: str = "default"
    figsize: tuple = (10, 6)  # Add default figsize for plotting
    dpi: int = 300  # Add default dpi
    save_format: str = "png"  # Add default format

@dataclass
class MockParsing:
    include_std_rows: bool = False
    skip_rsf_rows: bool = True
    enable_txt_import: bool = False

@dataclass
class MockPaths:
    step1_raw_dir: Path = Path("00_raw_data")
    step1_pattern: str = "*.txt"
    output_folder: str = "03_quantified_data"
    step2_converted_base_dir: Path = Path("02_fitted_results") 
    step2_fit_folder_name: str = "02_fitted_results"
    step2_param_file_pattern: str = "*_params.json"
    plots_dir: str = "plots"

@dataclass
class MockElements:
    selected: List[str] = None
    region_map: Dict[str, str] = None
    
    def __post_init__(self):
        if self.selected is None:
            self.selected = ["C", "O", "F", "Li", "Si", "P"]
        if self.region_map is None:
            self.region_map = {
                "C": "C1s", "O": "O1s", "F": "F1s", 
                "Li": "Li1s", "Si": "Si2p", "P": "P2p"
            }

@dataclass
class MockSearch:
    recursive: bool = True

@dataclass
class MockSampleId:
    json_pattern: str = r"_[A-Z][a-z0-9]*[sp]*_L\d+$"  # Strip _Region_Layer suffix (e.g., _C1s_L1)
    pro_pattern: str = r"_[A-Z][a-z0-9]*[sp]*$"        # Strip _Region suffix (e.g., _C1s)@dataclass
class MockSanityCheck:
    tolerance: float = 1.0

@dataclass
class MockChemistry:
    groups: Dict[str, object] = None
    
    def __post_init__(self):
        if self.groups is None:
            # Create mock chemistry groups with region_prefix
            from types import SimpleNamespace
            self.groups = {
                "carbon_chemistry": SimpleNamespace(
                    region_prefix="C",
                    components=["C-C", "C-O", "C=O", "C-O=C", "CO3"]
                ),
                "oxygen_chemistry": SimpleNamespace(
                    region_prefix="O", 
                    components=["O-C", "O=C", "O-H", "O-Si"]
                ),
                "fluorine_chemistry": SimpleNamespace(
                    region_prefix="F",
                    components=["F-C", "F-Li", "F-P"]
                ),
                "lithium_chemistry": SimpleNamespace(
                    region_prefix="Li",
                    components=["Li-C", "Li-O", "Li-F"]
                ),
                "phosphorus_chemistry": SimpleNamespace(
                    region_prefix="P",
                    components=["Li3PO4", "LixPOyFz", "PO4", "P-O", "P-F"]
                )
            }

@dataclass
class AppConfig:
    project_root: Path = Path.cwd()
    project: MockProject = None
    plot: MockPlot = None
    parsing: MockParsing = None
    paths: MockPaths = None
    elements: MockElements = None
    search: MockSearch = None
    sample_id: MockSampleId = None
    sanity_check: MockSanityCheck = None
    chemistry: MockChemistry = None
    
    def __post_init__(self):
        if self.project is None:
            self.project = MockProject()
        if self.plot is None:
            self.plot = MockPlot()
        if self.parsing is None:
            self.parsing = MockParsing()
        if self.paths is None:
            self.paths = MockPaths()
        if self.elements is None:
            self.elements = MockElements()
        if self.search is None:
            self.search = MockSearch()
        if self.sample_id is None:
            self.sample_id = MockSampleId()
        if self.sanity_check is None:
            self.sanity_check = MockSanityCheck()
        if self.chemistry is None:
            self.chemistry = MockChemistry()

# Alias for compatibility
XPSConfig = AppConfig
    
def load_config(config_path=None) -> AppConfig:
    """Simple config loader replacement"""
    # Try to get project root from workflow manager if available
    try:
        # Add Tools directory to path for import
        tools_dir = Path(__file__).resolve().parents[1]
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
            
        from xps_workflow_manager import get_workflow_config
        workflow_config = get_workflow_config()
        project_root = workflow_config.project_root
    except Exception:
        project_root = Path.cwd()
    
    return AppConfig(project_root=project_root)

# ========= XPS WORKFLOW INTEGRATION ==========
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
    print("⚠️  Workflow manager not available, using legacy paths")

def resolve_quant_paths(config: XPSConfig):
    """
    Resolve quantification paths using workflow manager for unified structure.
    """
    run_id = os.environ.get("XPS_RUN_ID", "").strip()

    def _run_subdir(base_dir: Path, run_id_val: str) -> Path:
        run_id_val = (run_id_val or "").strip()
        run_tag = ""
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
                run_tag = f"{date_part}{time_part}"
            else:
                digits = re.sub(r"\D", "", run_id_val)
                if len(digits) >= 14:
                    run_tag = digits[:14]
                elif len(digits) >= 8:
                    run_tag = f"{digits[:8]}{datetime.now().strftime('%H%M%S')}"
        if not run_tag:
            run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
        return Path(base_dir) / run_tag

    def _latest_run_dir(base_dir: Path) -> Optional[Path]:
        candidates = [
            p for p in base_dir.iterdir()
            if p.is_dir() and re.match(r"^\d{14}$", p.name)
        ] if base_dir.exists() else []
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: p.name)[-1]

    def _resolve_run_dir(base_dir: Path, create: bool) -> Path:
        if run_id:
            return _run_subdir(base_dir, run_id)
        if create:
            return _run_subdir(base_dir, "")
        latest = _latest_run_dir(base_dir)
        return latest or base_dir

    if WORKFLOW_MANAGER_AVAILABLE:
        try:
            workflow_config = get_workflow_config(str(config.project_root))
            quantifier_config = update_module_paths('quantifier', workflow_config)

            return {
                "input_dir": _resolve_run_dir(Path(quantifier_config['input_dir']), create=False),
                "output_dir": _resolve_run_dir(Path(quantifier_config['output_dir']), create=True),
                "plots_dir": _resolve_run_dir(workflow_config.plots_output_dir / "03_quantification", create=True)
            }
        except Exception as e:
            print(f"⚠️  Workflow manager error: {e}, falling back to legacy paths")
    
    # Legacy fallback
    root = Path(config.project_root)
    input_dir = _resolve_run_dir(root / "02_fitted_results", create=False)
    output_dir = _resolve_run_dir(root / "03_quantified_data", create=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = _resolve_run_dir(root / "04_plots" / "03_quantification", create=True)

    return {"input_dir": input_dir, "output_dir": output_dir, "plots_dir": plots_dir}

# ========= TXT PARSER IMPORT =========
try:
    from .txt_ac_parser import parse_atomic_concentration_rows as _txt_parse_rows
    TXT_PARSER_AVAILABLE = True
except ImportError:
    print("⚠️  TXT parser module not available")
    TXT_PARSER_AVAILABLE = False

# ========= STEP 1: LOAD RSF configure file and FITTING PARAMETERS =========

def _load_rsf_corrected_map() -> Dict[str, float]:
    """Load corrected RSF values from the default YAML.

    Returns a dict mapping region name (e.g., 'C1s', 'O1s') to corrected_rsf.
    """
    # Prefer repo-relative path to ensure stability
    try_paths = [
        Path(__file__).resolve().parents[2]
        / "project_root" / "xps_config" / "RSF_Al source.yaml",
        # Fallback to current working directory if running standalone
        Path.cwd() / "zzy_llm" / "project_root" / "xps_config" / "RSF_Al source.yaml",
    ]

    rsf_path = next((p for p in try_paths if p.exists()), None)
    if rsf_path is None:
        raise FileNotFoundError(
            "RSF YAML not found. Expected at 'zzy_llm/project_root/xps_config/RSF_Al source.yaml'"
        )

    try:
        with open(rsf_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read RSF YAML file: {e}")

    if not isinstance(data, dict):
        raise ValueError("RSF YAML must contain a dictionary")

    rsf_map: Dict[str, float] = {}
    rsf_values = data.get("rsf_values")
    if not rsf_values:
        raise ValueError("No 'rsf_values' section found in RSF YAML")

    for region, vals in rsf_values.items():
        if isinstance(vals, dict) and "corrected_rsf" in vals:
            try:
                rsf_map[region] = float(vals["corrected_rsf"])
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid RSF value for {region}: {vals['corrected_rsf']}")
                continue
    
    if not rsf_map:
        raise ValueError("No valid corrected_rsf entries found in RSF YAML")

    return rsf_map

def load_all_step2_json(cfg: AppConfig) -> Optional[pd.DataFrame]:
    """Load all JSON analysis result files directly from region subfolders."""
    print("\n" + "="*70)
    print("STEP 2: Loading fitting parameters from JSON analysis results")
    print("="*70 + "\n")

    try:
        # Use workflow manager to get fitted results directory
        if WORKFLOW_MANAGER_AVAILABLE:
            try:
                workflow_config = get_workflow_config(str(cfg.project_root))
                fitter_config = update_module_paths('fitter', workflow_config)
                base_dir = Path(fitter_config['output_dir'])
            except Exception as e:
                print(f"⚠️  Workflow manager error: {e}, using fallback path")
                base_dir = cfg.project_root / "02_fitted_results"
        else:
            base_dir = cfg.project_root / "02_fitted_results"

        # Apply run subdir if requested, otherwise use latest run folder if available
        run_id = os.environ.get("XPS_RUN_ID", "").strip()
        run_dir = None
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
                candidate = Path(base_dir) / run_tag
                if candidate.exists():
                    run_dir = candidate
        if run_dir is None:
            candidates = [
                p for p in Path(base_dir).iterdir()
                if p.is_dir() and re.match(r"^\d{14}$", p.name)
            ] if Path(base_dir).exists() else []
            if candidates:
                run_dir = sorted(candidates, key=lambda p: p.name)[-1]
        if run_dir is not None:
            base_dir = run_dir
        
        if not base_dir.exists():
            print(f"⚠️  Fitted results directory not found: {base_dir}")
            return None

        print(f"📂 Scanning for analysis results in: {base_dir}")
        
        # Look for region subfolders and JSON analysis results
        all_data = []
        regions_found = []
        
        # Scan for region subfolders (C1s, F1s, O1s, etc.)
        for region_folder in base_dir.iterdir():
            if not region_folder.is_dir() or region_folder.name in ['plots']:
                continue
                
            region_name = region_folder.name
            json_pattern = "*_analysis_results.json"
            json_files = list(region_folder.glob(json_pattern))
            
            if json_files:
                regions_found.append(region_name)
                print(f"\n  Region: {region_name}")
                
                for json_file in json_files:
                    try:
                        if not json_file.exists():
                            print(f"    ✗ File not found: {json_file.name}")
                            continue
                            
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if not isinstance(data, list):
                            print(f"    ✗ Invalid JSON format in {json_file.name}: expected list")
                            continue
                        
                        if not data:
                            print(f"    ⚠️  Empty data in {json_file.name}")
                            continue
                        
                        # Convert to DataFrame and validate
                        df = pd.DataFrame(data)
                        
                        # Validate required columns for quantification
                        required_cols = ['Sample', 'Region', 'Total_area']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            print(f"    ✗ Missing columns in {json_file.name}: {missing_cols}")
                            continue
                        
                        # Ensure Layer column exists (default to 1 if missing)
                        if 'Layer' not in df.columns:
                            df['Layer'] = 1
                        
                        all_data.append(df)
                        print(f"    ✓ {json_file.name}: {len(df)} components, Total_area available")
                        
                    except json.JSONDecodeError as e:
                        print(f"    ✗ JSON decode error in {json_file.name}: {e}")
                    except Exception as e:
                        print(f"    ✗ Error loading {json_file.name}: {e}")

        if not all_data:
            print("⚠️  No valid analysis result JSON files found in region subfolders")
            return None

        # Combine all data
        try:
            combined = pd.concat(all_data, ignore_index=True)
        except Exception as e:
            print(f"⚠️  Error combining DataFrames: {e}")
            return None
        
        if combined.empty:
            print("⚠️  Combined DataFrame is empty")
            return None

        # Create BaseID for matching with quantifier
        if 'Sample' not in combined.columns:
            print("⚠️  'Sample' column missing from combined data")
            return None

        try:
            def _extract_base_id(sample_val: object) -> str:
                if pd.isna(sample_val):
                    return ''
                s = str(sample_val)
                # Prefer full pattern with region + layer suffix
                cleaned = re.sub(cfg.sample_id.json_pattern, '', s)
                if cleaned != s:
                    return cleaned
                # Fallback: strip layer suffix only (e.g., _L1)
                cleaned = re.sub(r"_L\d+$", "", s)
                if cleaned != s:
                    return cleaned
                # Fallback: strip region-only suffix
                cleaned = re.sub(cfg.sample_id.pro_pattern, '', s)
                return cleaned

            combined['BaseID'] = combined['Sample'].apply(_extract_base_id)
        except Exception as e:
            print(f"⚠️  Error creating BaseID: {e}")
            combined['BaseID'] = combined['Sample']  # Fallback

        print(f"\n✓ Total components loaded: {len(combined)}")
        print(f"✓ Regions found: {regions_found}")
        print(f"✓ Unique samples: {combined['Sample'].nunique()}")
        
        return combined
        
    except Exception as e:
        print(f"⚠️  Error in JSON loading process: {e}")
        return None

# ========= STEP 2: COMPUTE ATOMIC CONCENTRATION ========= 
def _compute_atomic_concentration_from_json(cfg: AppConfig) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """Compute elemental atomic % using JSON fit outputs and corrected RSFs.

    - Aggregates total area per (Sample, Layer, Region) from JSON files
    - Uses corrected RSF from RSF_Al source.yaml
    - Applies: Atomic%(X) = (Area_X/RSF_X) / sum_i(Area_i/RSF_i) * 100 per (Sample, Layer)

    Returns (df, header) with columns ['Sample', 'Layer'] + sorted region list, or None if JSON missing.
    """
    try:
        df_step2 = load_all_step2_json(cfg)
        if df_step2 is None or df_step2.empty:
            # Try a broader search for typical fitter outputs
            print("ℹ️  No JSON via configured paths; scanning common '02_fitted_results' locations for analysis JSON...")
            candidates = [
                Path(__file__).resolve().parents[2] / "02_fitted_results",
                Path(__file__).resolve().parents[2] / "project_root" / "02_fitted_results",
                cfg.paths.step2_converted_base_dir,
                Path.cwd() / "02_fitted_results",
            ]
            json_files: List[Path] = []
            for base in candidates:
                if not base or not Path(base).exists():
                    continue
                try:
                    json_files.extend(Path(base).rglob("*_analysis_results.json"))
                    json_files.extend(Path(base).rglob("*_params.json"))
                except Exception as e:
                    print(f"⚠️  Error scanning {base}: {e}")
                    continue

            if not json_files:
                return None

            frames: List[pd.DataFrame] = []
            for jf in sorted(set(json_files)):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list) and data:
                        frames.append(pd.DataFrame(data))
                except Exception as e:
                    print(f"⚠️  Failed reading JSON {jf.name}: {e}")
            if not frames:
                return None
            df_step2 = pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"⚠️  Error in JSON loading process: {e}")
        return None

    # Ensure required columns exist
    required = {"Sample", "Layer", "Region", "Total_area"}
    if not required.issubset(df_step2.columns):
        print(f"⚠️  Missing required columns in analysis results: {required - set(df_step2.columns)}")
        return None

    # FIX: Extract base sample name for depth profiles
    # Sample names may include region suffix (e.g., "15Si_depth_C1s_L1")
    # We need to remove "_{Region}_L{Layer}" pattern to get base name "15Si_depth"
    def extract_base_sample_name(sample: str, region: str, layer: int) -> str:
        """Remove region and layer suffix from sample name if present."""
        # Try to remove pattern: _{region}_L{layer}
        pattern_with_layer = f"_{region}_L{layer}"
        if sample.endswith(pattern_with_layer):
            return sample[:-len(pattern_with_layer)]
        
        # Try to remove just region suffix: _{region}
        pattern_region = f"_{region}"
        if sample.endswith(pattern_region):
            return sample[:-len(pattern_region)]
        
        # Return as-is if no pattern found (standard non-depth samples)
        return sample
    
    df_step2["BaseSample"] = df_step2.apply(
        lambda row: extract_base_sample_name(row["Sample"], row["Region"], row["Layer"]),
        axis=1
    )

    # Use Total_area directly from analysis results
    # Group by (BaseSample, Layer, Region) to get unique entries - Total_area should be same for all components in a region
    df_tmp = (
        df_step2.groupby(["BaseSample", "Layer", "Region"], as_index=False)
        ["Total_area"].first()
    )
    # Rename BaseSample back to Sample for consistency with rest of code
    df_tmp = df_tmp.rename(columns={"BaseSample": "Sample"})
    area_agg_col = "Total_area"

    # Load corrected RSFs
    try:
        rsf_map = _load_rsf_corrected_map()
    except Exception as e:
        print(f"⚠️  Error loading RSF values: {e}")
        return None

    # Map RSF to regions; drop regions without RSF mapping
    df_tmp["_RSF_"] = df_tmp["Region"].map(rsf_map)
    missing = df_tmp["_RSF_"].isna()
    if missing.any():
        missing_regions = sorted(df_tmp.loc[missing, "Region"].unique().tolist())
        print(f"⚠️  Missing RSF for regions (will be excluded): {missing_regions}")
        df_tmp = df_tmp.loc[~missing].copy()

    if df_tmp.empty:
        print("⚠️  No regions left after RSF mapping; cannot compute atomic %")
        return None

    try:
        # Compute normalized contribution per region
        df_tmp["_Norm_"] = df_tmp[area_agg_col] / df_tmp["_RSF_"]

        # Normalize within each (Sample, Layer) to get atomic percent per Region
        sums = df_tmp.groupby(["Sample", "Layer"], as_index=False)["_Norm_"].sum()
        sums = sums.rename(columns={"_Norm_": "_NormSum_"})
        df_norm = df_tmp.merge(sums, on=["Sample", "Layer"], how="left")
        
        # Avoid division by zero
        df_norm["Atomic_percent"] = np.where(
            df_norm["_NormSum_"] > 0,
            (df_norm["_Norm_"] / df_norm["_NormSum_"]) * 100.0,
            0.0
        )
    except Exception as e:
        print(f"⚠️  Error in atomic % calculations: {e}")
        return None

    try:
        # Pivot to wide format: columns are Regions (e.g., C1s, O1s)
        df_wide = df_norm.pivot_table(
            index=["Sample", "Layer"],
            columns="Region",
            values="Atomic_percent",
            aggfunc="first"
        ).reset_index()

        # Determine header order: use region_map selection order if available, else sorted columns
        present_regions = [c for c in df_wide.columns if c not in ("Sample", "Layer")]
        header: List[str] = []
        if cfg and cfg.elements and cfg.elements.region_map:
            # flatten region_map order based on selected elements
            ordered = []
            for el in (cfg.elements.selected or []):
                v = cfg.elements.region_map.get(el)
                if isinstance(v, str):
                    ordered.append(v)
                elif isinstance(v, (list, tuple)):
                    ordered.extend(list(v))
            header = [r for r in ordered if r in present_regions]
            # append any remaining regions
            header += [r for r in present_regions if r not in header]
        else:
            header = sorted(present_regions)

        # Reindex columns to consistent order
        df_wide = df_wide.reindex(columns=["Sample", "Layer"] + header)
    except Exception as e:
        print(f"⚠️  Error in pivot table creation: {e}")
        return None

    # Save intermediate CSV using run-tagged output directory
    try:
        resolved = resolve_quant_paths(cfg)
        output_path = Path(resolved["output_dir"])
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "atomic_concentration_raw.csv"
        df_wide.to_csv(csv_path, index=False)
        print(f"\nAtomic % computed from JSON using RSFs -> {csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        # Continue without saving

    return df_wide, header


def _txt_enabled(cfg: AppConfig) -> bool:
    flag = getattr(cfg.parsing, 'enable_txt_import', False)
    if flag and not TXT_PARSER_AVAILABLE:
        print("⚠️  TXT parser module not available; skipping TXT import")
        return False
    return bool(flag and TXT_PARSER_AVAILABLE)


def build_atomic_concentration_dataframe(cfg: AppConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Step 1: Build atomic concentration dataframe from raw XPS text files.
    Returns: (DataFrame, master_header)
    """
    print("\n" + "="*70)
    print("STEP 1: Extracting atomic concentration from raw XPS files")
    print("="*70 + "\n")

    try:
        input_dir = cfg.paths.step1_raw_dir
        input_pattern = cfg.paths.step1_pattern

        use_txt = _txt_enabled(cfg)
        files = sorted(input_dir.glob(input_pattern)) if use_txt and input_dir.exists() else []
        
        if use_txt:
            print(f"Found {len(files)} text files in {input_dir}")
        else:
            print("TXT import disabled (default). Using JSON/RSF workflow computation.")

        all_rows = []
        master_header = None

        for f in files:
            if not f.exists():
                print(f"  ⚠ File not found: {f}")
                continue
                
            parsed = _txt_parse_rows(f, cfg)
            if parsed is None:
                print(f"  ⚠ Skipping {f.name} (no atomic concentration data)")
                continue

            if master_header is None:
                master_header = parsed["header"]
            elif parsed["header"] != master_header:
                print(
                    f"  ⚠ Header differs in {f.name}. Aligning by element names.")
                for i, row_vals in enumerate(parsed["rows"]):
                    name_to_val = dict(zip(parsed["header"], row_vals))
                    parsed["rows"][i] = [name_to_val.get(
                        h, np.nan) for h in master_header]

            # Append rows with Sample and Layer (RowIndex)
            for idx, vals in enumerate(parsed["rows"], start=1):
                row = {"Sample": parsed["sample"], "Layer": idx}
                for name, val in zip(master_header, vals):
                    row[name] = val
                all_rows.append(row)

            print(f"  ✓ {f.name}: {len(parsed['rows'])} layers")

        if not all_rows:
            print("⚠️  No TXT atomic concentration data used. Trying JSON/RSF-based computation...")
            fallback = _compute_atomic_concentration_from_json(cfg)
            if fallback is None:
                raise RuntimeError(
                    "No valid TXT files and could not compute from JSON/RSF.")
            # Fallback returns a complete dataframe and header
            return fallback

        df = pd.DataFrame(all_rows)
        df = df.reindex(columns=(["Sample", "Layer"] + master_header))

        # Save intermediate CSV using run-tagged output directory
        resolved = resolve_quant_paths(cfg)
        output_path = Path(resolved["output_dir"])
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "atomic_concentration_raw.csv"
        df.to_csv(csv_path, index=False)

        print(f"\n✓ Extracted {len(df)} rows from {len(files)} files")
        print(f"✓ Saved to: {csv_path}")
        print(f"✓ Elements found: {master_header}")

        return df, master_header
        
    except Exception as e:
        print(f"⚠️  Error in atomic concentration extraction: {e}")
        # Try fallback method
        try:
            fallback = _compute_atomic_concentration_from_json(cfg)
            if fallback is not None:
                return fallback
        except Exception as fallback_error:
            print(f"⚠️  Fallback method also failed: {fallback_error}")
        
        # Return empty result if both methods fail
        return pd.DataFrame(), []

# ========= ATOMIC CONCENTRATION PLOTTING =========


def sanitize_filename(s: str) -> str:
    """Replace problematic characters for Windows filenames."""
    return re.sub(r'[^\w\-.]+', '_', s)


def resolve_selected_elements(master_header: List[str], selected_symbols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Map user-selected element symbols to header column names.
    Returns: (plot_columns, plot_labels)
    """
    plot_columns = []
    plot_labels = []
    for sym in selected_symbols:
        matches = [
            col for col in master_header if col.lower().startswith(sym.lower())]
        if not matches:
            print(f"  ⚠ No column found for element '{sym}'. Skipping.")
            continue
        plot_columns.append(matches[0])
        plot_labels.append(sym)

    if not plot_columns:
        plot_columns = master_header[:]
        plot_labels = master_header[:]
        print("  ⚠ No valid selected elements matched; plotting all columns.")
    else:
        mapping_str = ", ".join(
            f"{sym}->{col}" for sym, col in zip(plot_labels, plot_columns))
        print(f"  Element mapping: {mapping_str}")

    return plot_columns, plot_labels


def generate_atomic_concentration_plots(df: pd.DataFrame, master_header: List[str], plots_dir: Path, cfg: AppConfig):
    """Generate atomic concentration plots for selected elements."""
    print("\nGenerating atomic concentration plots...")

    # Resolve selected elements
    plot_columns, plot_labels = resolve_selected_elements(
        master_header, cfg.elements.selected)

    # NEW: Plot each element concentration across all samples in single plots
    print("  📊 Creating element concentration cross-sample plots...")
    for element_col, element_label in zip(plot_columns, plot_labels):
        create_element_cross_sample_plot(df, element_col, element_label, plots_dir, cfg)
    
    print(f"  ✓ Generated cross-sample element concentration plots")
    
    # Plot per sample atomic concentrations
    print("  📊 Creating per-sample atomic concentration plots...")
    # Natural sort samples (S1, S2, ..., S10 order, not S1, S10, S2)
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
    samples = sorted(df["Sample"].unique(), key=natural_sort_key)
    for sample in samples:
        plot_atomic_concentration_per_sample(
            df=df, 
            sample=sample, 
            out_dir=plots_dir,
            plot_columns=plot_columns, 
            plot_labels=plot_labels, 
            config=None)  # Let function load plot config from YAML
    print(f"  ✓ Generated {len(samples)} per-sample atomic concentration plots")

    # Plot per layer comparison (keep existing)
    print("  📊 Creating layer comparison plots...")
    for layer in sorted(df["Layer"].unique()):
        if df[df["Layer"] == layer]["Sample"].nunique() >= 2:
            plot_atomic_concentration_layer_comparison(
                df=df, layer=layer, out_dir=plots_dir, 
                plot_columns=plot_columns, plot_labels=plot_labels, config=None)  # Let function load plot config from YAML
    print(f"  ✓ Generated layer comparison plots")


def create_element_cross_sample_plot(df: pd.DataFrame, element_col: str, element_label: str, plots_dir: Path, cfg: AppConfig):
    """Create a single plot showing element concentration across all samples."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load plot configuration
    try:
        from XPS_Plotter.plot_modules.utils.plot_utils import load_plot_config
        plot_config = load_plot_config()
        plot_settings = plot_config['plot_settings']
        font_cfg = plot_settings['fonts']
        export_config = plot_config['export']
    except Exception as e:
        print(f"⚠️  Could not load plot_settings.yaml, using defaults: {e}")
        font_cfg = {'title_size': 14, 'axis_label_size': 12, 'tick_label_size': 10, 'legend_size': 10}
        plot_settings = {'lines': {'grid_alpha': 0.3}}
        export_config = {'default_format': 'png'}
    
    # Filter data for this element (non-null values)
    element_data = df[df[element_col].notna()].copy()
    if element_data.empty:
        print(f"    ⚠️ No data for {element_label}")
        return
    
    # Natural sort for samples (E1, E2, ..., E10 order)
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
    
    # Get unique samples and layers with natural sorting
    samples = sorted(element_data["Sample"].unique(), key=natural_sort_key)
    layers = sorted(element_data["Layer"].unique())
    
    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize))
    
    # Create x positions for samples
    x_pos = np.arange(len(samples))
    width = 0.9/ max(len(layers), 1)  # Width of each bar
    
    # Color map for layers
    colors = plt.cm.Set3(np.linspace(0, 1, len(layers)))
    
    # Plot bars for each layer
    for i, layer in enumerate(layers):
        layer_data = element_data[element_data["Layer"] == layer]
        values = []
        
        for sample in samples:
            sample_data = layer_data[layer_data["Sample"] == sample]
            if not sample_data.empty:
                values.append(sample_data[element_col].iloc[0])
            else:
                values.append(0)  # No data for this sample-layer combination
        
        ax.bar(x_pos + i * width, values, width, 
               label=f'Layer {layer}', color=colors[i], alpha=0.8)
    
    # Formatting with plot_settings.yaml
    ax.set_xlabel('Samples', fontsize=font_cfg['axis_label_size'], fontweight='bold')
    ax.set_ylabel(f'{element_label} Atomic %', fontsize=font_cfg['axis_label_size'], fontweight='bold')
    ax.set_title(f'{element_label} Concentration Across All Samples', fontsize=font_cfg['title_size'], fontweight='bold')
    # Center x-ticks under the grouped bars
    bar_group_center = x_pos + (len(layers) * width) / 2 - width / 2
    ax.set_xticks(bar_group_center)
    ax.set_xticklabels([s.replace('Si_', '') for s in samples], rotation=45, ha='right', fontsize=font_cfg['tick_label_size'])
    #ax.legend(fontsize=font_cfg['legend_size'])
    ax.grid(axis='y', alpha=plot_settings['lines']['grid_alpha'])
    ax.tick_params(axis='both', labelsize=font_cfg['tick_label_size'])
    
    plt.tight_layout()
    
    # Save plot with export config
    filename = f"{sanitize_filename(element_label)}_cross_sample_concentration.{export_config.get('default_format', 'png')}"
    filepath = plots_dir / filename
    fig.savefig(filepath, dpi=cfg.plot.dpi, bbox_inches=export_config.get('bbox_inches', 'tight'))
    plt.close(fig)
    
    print(f"    ✓ {filename}")



# ========= STEP 3: COMPONENT QUANTIFICATION =========


def prepare_step1_data(df_step1: pd.DataFrame, master_header: List[str], cfg: AppConfig) -> pd.DataFrame:
    """Convert Step 1 data to long format for merging."""
    print("\nPreparing Step 1 data for merging...")

    # Extract BaseID
    df_step1['BaseID'] = df_step1['Sample'].apply(
        lambda x: re.sub(cfg.sample_id.pro_pattern, '', x)
    )

    # Melt to long format
    id_cols = ['BaseID', 'Sample', 'Layer']
    element_cols = master_header

    df_long = df_step1.melt(
        id_vars=id_cols,
        value_vars=element_cols,
        var_name='Region',
        value_name='Element_atomic_percent'
    )

    print(f"  ✓ Converted to long format: {len(df_long)} rows")
    print(f"  ✓ Unique samples: {df_long['BaseID'].nunique()}")

    return df_long


def compute_component_atomic_percent(
    df_step1_long: pd.DataFrame,
    df_step2: pd.DataFrame,
    cfg: AppConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Step 3: Calculate component atomic percentages."""
    print("\n" + "="*70)
    print("STEP 3: Calculating component atomic percentages")
    print("="*70 + "\n")

    # Merge data
    merged = df_step2.merge(
        df_step1_long[['BaseID', 'Layer', 'Region', 'Element_atomic_percent']],
        on=['BaseID', 'Layer', 'Region'],
        how='left'
    )

    # Check for unmatched
    unmatched = merged['Element_atomic_percent'].isna()
    if unmatched.sum() > 0:
        print(f"⚠ WARNING: {unmatched.sum()} components could not be matched")

    # Calculate component atomic percentage
    merged['Component_atomic_percent'] = (
        merged['Element_atomic_percent'] * (merged['Area_percent'] / 100.0)
    )

    matched_count = (~merged['Component_atomic_percent'].isna()).sum()
    print(f"✓ Successfully calculated atomic % for {matched_count} components")

    # Sanity check
    sanity = (
        merged[~merged['Component_atomic_percent'].isna()]
        .groupby(['BaseID', 'Sample', 'Layer', 'Region'], as_index=False)
        .agg(
            Element_atomic_percent=('Element_atomic_percent', 'first'),
            Sum_components=('Component_atomic_percent', 'sum'),
            Num_components=('Component', 'count')
        )
    )

    sanity['Diff'] = sanity['Sum_components'] - \
        sanity['Element_atomic_percent']
    sanity['Pass'] = sanity['Diff'].abs() <= cfg.sanity_check.tolerance

    print(f"\nSanity check results:")
    print(f"  ✓ Passed: {sanity['Pass'].sum()}/{len(sanity)}")

    if (~sanity['Pass']).sum() > 0:
        print(f"  ✗ Failed: {(~sanity['Pass']).sum()}")

    return merged, sanity


# ========= MAIN WORKFLOW =========


def main():
    """Execute complete XPS analysis workflow."""
    try:
        parser = argparse.ArgumentParser(
            description="Integrated XPS Analysis Workflow")
        parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                            help="Path to YAML configuration file")
        args = parser.parse_args()

        # Load configuration
        try:
            cfg = load_config(args.config)
        except Exception as e:
            print(f"⚠️  Error loading configuration: {e}")
            print("Using default configuration...")
            cfg = load_config()

        print("\n" + "="*70)
        print(f"{cfg.project.name}")
        if cfg.project.description:
            print(cfg.project.description)
        print("="*70)

        # Set plot style if specified
        if hasattr(cfg.plot, 'style') and cfg.plot.style:
            try:
                plt.style.use(cfg.plot.style)
            except Exception as e:
                print(f"⚠️  Error setting plot style: {e}")

        # Setup output directories using workflow manager paths
        try:
            # Use workflow manager for proper path resolution
            resolved_paths = resolve_quant_paths(cfg)
            output_path = resolved_paths["output_dir"]
            output_path.mkdir(parents=True, exist_ok=True)
            plots_path = resolved_paths.get("plots_dir")
            if plots_path is None:
                plots_path = output_path / cfg.paths.plots_dir
            plots_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Error creating output directories: {e}")
            return

        # Step 1: Extract atomic concentration from raw XPS files
        try:
            df_step1, master_header = build_atomic_concentration_dataframe(cfg)
            if df_step1.empty:
                print("⚠️  No atomic concentration data available. Exiting.")
                return
        except Exception as e:
            print(f"⚠️  Error in Step 1: {e}")
            return

        # Generate atomic concentration plots
        try:
            print("\n" + "="*70)
            print("GENERATING ATOMIC CONCENTRATION PLOTS")
            print("="*70)
            generate_atomic_concentration_plots(
                df_step1, master_header, plots_path, cfg)
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")

        # Prepare Step 1 data for merging
        try:
            df_step1_long = prepare_step1_data(df_step1, master_header, cfg)
        except Exception as e:
            print(f"⚠️  Error preparing Step 1 data: {e}")
            return

        # Step 2: Load fitting parameters (if available)
        try:
            df_step2 = load_all_step2_json(cfg)
        except Exception as e:
            print(f"⚠️  Error in Step 2: {e}")
            df_step2 = None

        if df_step2 is not None:
            try:
                # Step 3: Calculate component atomic percentages
                results, sanity = compute_component_atomic_percent(
                    df_step1_long, df_step2, cfg)

                # Save results
                output_cols = [
                    'File', 'Sample', 'Layer', 'Region', 'Component',
                    'Center_eV', 'FWHM_eV', 'Area_percent',
                    'Element_atomic_percent', 'Component_atomic_percent'
                ]
                output_cols = [c for c in output_cols if c in results.columns]

                # Save combined results
                combined_csv = output_path / 'all_components_with_atomic_percent.csv'
                results[output_cols].to_csv(combined_csv, index=False)
                print(f"\n✓ Saved component results: {combined_csv}")

                # Save sanity check
                sanity_csv = output_path / 'sanity_check.csv'
                sanity.to_csv(sanity_csv, index=False)
                print(f"✓ Saved sanity check: {sanity_csv}")

                # Step 4: Generate component chemistry plots
                try:
                    print("\n" + "="*70)
                    print("STEP 4: Generating component chemistry analysis plots")
                    print("="*70)
                    generate_component_chemistry_plots(results, plots_path, cfg, config=None)

                    # Generate heatmaps for each chemistry group (if configured)
                    if hasattr(cfg, 'chemistry') and hasattr(cfg.chemistry, 'groups'):
                        for chem_name in cfg.chemistry.groups.keys():
                            try:
                                heatmap_result = generate_component_heatmap(results, plots_path, cfg, chem_name, config=None)
                                if heatmap_result:
                                    print(f"✓ Heatmap saved: {heatmap_result}")
                                else:
                                    print(f"⚠️  No data available for {chem_name} heatmap")
                            except Exception as e:
                                print(f"⚠️  Error generating heatmap for {chem_name}: {e}")
                except Exception as e:
                    print(f"⚠️  Error in Step 4 plotting: {e}")

            except Exception as e:
                print(f"⚠️  Error in component analysis: {e}")
        else:
            print("\n⚠ No fitting data found. Only atomic concentration extracted and plotted.")

        print("\n" + "="*70)
        print("WORKFLOW COMPLETE!")
        print(f"Output folder: {output_path}")
        print(f"Plots folder: {plots_path}")
        print("="*70)

    except Exception as e:
        print(f"⚠️  Critical error in main workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
