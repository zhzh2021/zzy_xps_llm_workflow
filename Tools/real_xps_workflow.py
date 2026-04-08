#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real XPS Workflow Runner
========================

This script runs the complete XPS workflow using the existing XPS tools:
1. XPS_Reader (main.py): Raw data → Standardized CSV files
2. XPS_Fitter (XPS_peakfitting_V2.py): CSV files → Peak fitted results  
3. XPS_Quantifier (XPS_Quantifier.py): Fitted results → Quantification data
4. XPS_Plotter (plotter.py): Generate plots and visualizations
5. XPS_Mapper (XPS_map.py): Process hyperspectral map data (if applicable)

This uses the ACTUAL existing tools in the Tools folder, not custom implementations.

Usage:
    python real_xps_workflow.py [project_root]
"""

import sys
import os
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional
import time
import json
from datetime import datetime
from scipy.signal import detrend

# Force UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set matplotlib to non-interactive backend to prevent GUI blocking
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    pass

# Add current directory to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from xps_workflow_manager import get_workflow_config, update_module_paths
    WORKFLOW_MANAGER_AVAILABLE = True
except ImportError:
    print("❌ Workflow manager not available")
    sys.exit(1)

# Import workflow orchestrator (handles triage + quality gate)
try:
    from workflow_orchestrator import orchestrate_workflow
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Workflow orchestrator not available - will skip pre-flight checks")
    ORCHESTRATOR_AVAILABLE = False


class RealXPSWorkflow:
    """Real XPS workflow that uses the existing XPS tools."""

    def __init__(self, project_root: str = None, interactive: bool = True, debug: bool = False):
        """Initialize with project configuration."""
        self.interactive = interactive  # Control whether to prompt user for input
        self.debug = debug
        # Timing and statistics
        self.start_time = time.time()
        self.step_times = {}  # Track time for each step
        self.total_files_processed = 0
        self.total_operations = 0
        self.step_file_counts = {}  # Track files processed in each step

        # Workflow state
        self.config = get_workflow_config(project_root)
        self.steps_completed = []
        self.errors = []

        # Workflow logging
        self.workflow_log = {
            'workflow_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'project_root': str(self.config.project_root),
            'steps': [],
            'summary': {}
        }

        # Define the actual XPS tool paths
        self.tools_dir = Path(__file__).parent
        self.xps_tools = {
            'reader': self.tools_dir / 'XPS_reader' / 'reader_main.py',
            'mapper': self.tools_dir / 'XPS_mapper' / 'XPS_map.py',
            'fitter': self.tools_dir / 'XPS_Fitter' / 'XPS_peakfitting_V2.py',
            'quantifier': self.tools_dir / 'XPS_Quantifier' / 'XPS_Quantifier.py',
            'plotter': self.tools_dir / 'XPS_Plotter' / 'plotter.py'
        }

        # Track workflow routing decision
        self.workflow_route = None  # 'standard' or 'map'

        print(f"📂 Project Root: {self.config.project_root}")
        print(f"🔧 Tools Directory: {self.tools_dir}")

    def _get_step1_plot_root(self) -> Path:
        """Build timestamped Step 1 plot directory path."""
        workflow_id = self.workflow_log.get("workflow_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = ""
        if workflow_id:
            if "_" in workflow_id:
                date_part, time_part = workflow_id.split("_", 1)
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
                digits = re.sub(r"\D", "", workflow_id)
                if len(digits) >= 14:
                    run_tag = digits[:14]
        if not run_tag:
            run_tag = datetime.now().strftime("%Y%m%d%H%M%S")

        return (
            self.config.project_root
            / "04_plots"
            / "01_converted_csv"
            / run_tag
        )

    def _run_tag_from_workflow_id(self) -> str:
        """Return 14-digit run tag (YYYYMMDDHHMMSS) from workflow_id."""
        workflow_id = self.workflow_log.get("workflow_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        if "_" in workflow_id:
            date_part, time_part = workflow_id.split("_", 1)
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
        digits = re.sub(r"\D", "", workflow_id)
        if len(digits) >= 14:
            return digits[:14]
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def step1_depth_profile_plots_from_csv(self) -> bool:
        """Generate depth profile 3D waterfall plots from existing aggregated CSV files."""
        try:
            from XPS_Plotter.plot_modules.quantification.depth_3d_waterfall import (
                load_depth_profile_csv,
                plot_depth_profile_3d_waterfall,
            )
        except Exception as exc:
            print(f"⚠️  Depth profile plot module not available: {exc}")
            return False

        converted_dir = self.config.project_root / "01_converted_csv"
        if not converted_dir.exists():
            print("⚠️  No converted CSV directory found for depth profile plots")
            return False

        plots_root = self._get_step1_plot_root()
        plots_dir = plots_root / "depth_profile_3d"
        plots_dir.mkdir(parents=True, exist_ok=True)

        csv_paths = list(converted_dir.glob("*/aggregated_*_allHR.csv"))
        if not csv_paths:
            print("⚠️  No aggregated CSV files found for depth profile plots")
            return True

        plot_count = 0
        for csv_path in csv_paths:
            layers = load_depth_profile_csv(csv_path)
            if not layers or len(layers) < 2:
                continue

            name = csv_path.stem  # aggregated_<region>_allHR
            region = name
            match = re.match(r"aggregated_(.+)_allHR$", name, re.IGNORECASE)
            if match:
                region = match.group(1)

            plot_path = plot_depth_profile_3d_waterfall(
                spectra_dict=layers,
                region=region,
                out_dir=plots_dir,
                config=None,
            )
            if plot_path:
                plot_count += 1

        if plot_count == 0:
            print("⚠️  No multi-layer depth profiles found for 3D waterfall plots")
        else:
            print(f"✅ Generated {plot_count} depth profile 3D waterfall plot(s)")

        return True

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        print("\n🔍 Checking Dependencies...")

        # Check if all tool files exist
        missing_tools = []
        for tool_name, tool_path in self.xps_tools.items():
            if not tool_path.exists():
                missing_tools.append(f"{tool_name}: {tool_path}")
            else:
                print(f"   ✓ {tool_name}: {tool_path.name}")

        if missing_tools:
            print("❌ Missing XPS tools:")
            for tool in missing_tools:
                print(f"   ✗ {tool}")
            return False

        # Check for required Python packages
        required_packages = ['numpy', 'pandas', 'matplotlib', 'scipy', 'lmfit']

        for package in required_packages:
            try:
                # Test import in subprocess to replicate tool execution environment
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'

                result = subprocess.run(
                    [sys.executable, '-c', f'import {package}; print("OK")'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env,
                    encoding='utf-8',
                    errors='replace'
                )

                if result.returncode == 0:
                    print(f"   ✓ {package} available")
                else:
                    print(
                        f"   ❌ {package} import failed: {result.stderr.strip()}")
                    print(
                        f"   💡 Try: conda install {package} or pip install {package}")
                    return False

            except Exception as e:
                print(f"   ❌ {package} check failed: {e}")
                return False

        return True

    def check_input_data(self) -> Dict:
        """Check what input data is available."""
        print("\n📊 Checking Input Data...")

        raw_files = list(self.config.raw_data_dir.glob("*"))
        # Use run-tag subfolder if available to avoid reusing old runs
        run_tag = self._run_tag_from_workflow_id()
        run_csv_dir = self.config.csv_output_dir / run_tag
        if run_csv_dir.exists():
            csv_files = list(run_csv_dir.rglob("*.csv"))
        else:
            csv_files = []

        # Count total files for tracking
        self.total_files_processed = len([f for f in raw_files if f.is_file()])

        data_info = {
            'raw_files': raw_files,
            'csv_files': csv_files,
            'can_start_from_reader': len(raw_files) > 0,
            'can_start_from_fitter': len(csv_files) > 0
        }

        print(
            f"   📁 Raw data files: {len(raw_files)} in {self.config.raw_data_dir.name}/")
        for f in raw_files[:5]:  # Show first 5
            print(f"     • {f.name}")
        if len(raw_files) > 5:
            print(f"     ... and {len(raw_files) - 5} more")

        print(
            f"   📁 CSV files: {len(csv_files)} in {self.config.csv_output_dir.name}/")
        for f in csv_files[:5]:  # Show first 5
            print(f"     • {f.name}")
        if len(csv_files) > 5:
            print(f"     ... and {len(csv_files) - 5} more")

        return data_info

    def run_xps_tool(self, tool_name: str, tool_path: Path, description: str) -> bool:
        """Run a specific XPS tool."""
        step_start_time = time.time()

        # Initialize step log entry
        step_log = {
            'step_name': tool_name,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'errors': [],
            'files_processed': 0,
            'detail_logs': []  # Will store paths to tool-specific log files
        }

        # Map tool names to their expected log file locations
        log_file_map = {
            'XPS_Reader': [
                '01_converted_csv/processing_log.txt'
            ],
            'XPS_Fitter': [
                '02_fitted_results/fitting_log.txt',
                '02_fitted_results/*/fitting_summary.json'  # Per-region summaries
            ],
            'XPS_Quantifier': [
                '03_quantified_data/quantification_log.txt',
                '03_quantified_data/quantification_summary.json'
            ],
            'XPS_Plotter': [
                '04_plots/plotting_log.txt'
            ],
            'XPS_Mapper': [
                '05_map_data/mapper_log.txt',
                '05_map_data/mcr_processing_log.txt'
            ]
        }

        # Add expected log paths for this tool
        if tool_name in log_file_map:
            step_log['detail_logs'] = log_file_map[tool_name]

        print(f"\n🔄 Running {tool_name}: {description}")
        print(f"   📄 Script: {tool_path}")

        if not tool_path.exists():
            error_msg = f"{tool_name} script not found: {tool_path}"
            print(f"❌ {error_msg}")
            self.errors.append(error_msg)
            step_log['status'] = 'failed'
            step_log['errors'].append(error_msg)
            step_log['end_time'] = datetime.now().isoformat()
            step_log['duration_seconds'] = time.time() - step_start_time
            self.workflow_log['steps'].append(step_log)
            return False

        try:
            # Change to the project root directory for execution
            original_cwd = os.getcwd()
            os.chdir(self.config.project_root)

            # Run the tool using subprocess with proper environment
            cmd = [sys.executable, str(tool_path)]

            print(f"   🚀 Executing: {' '.join(cmd)}")
            print(f"   📁 Working directory: {os.getcwd()}")

            # Ensure subprocess uses the same environment with UTF-8 encoding
            env = os.environ.copy()
            # Add tools directory to Python path
            env['PYTHONPATH'] = str(self.tools_dir)
            env['PYTHONIOENCODING'] = 'utf-8'  # Fix encoding issues
            # For Windows compatibility
            env['PYTHONLEGACYWINDOWSFSENCODING'] = 'utf-8'
            env['XPS_RUN_ID'] = self.workflow_log.get('workflow_id', datetime.now().strftime('%Y%m%d_%H%M%S'))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
                encoding='utf-8',  # Ensure UTF-8 encoding
                errors='replace'  # Replace problematic characters instead of failing
            )

            # Restore original directory
            os.chdir(original_cwd)

            # Check result
            step_duration = time.time() - step_start_time
            self.step_times[tool_name] = step_duration
            self.total_operations += 1

            if result.returncode == 0:
                print(f"✅ {tool_name} completed successfully")
                if result.stdout.strip():
                    # Show last few lines of output
                    output_lines = result.stdout.strip().split('\n')
                    print(f"   📤 Output (last 3 lines):")
                    for line in output_lines[-3:]:
                        print(f"      {line}")

                print(
                    f"✅ Step {len(self.steps_completed)+1}: {tool_name} completed in {step_duration:.1f}s")
                self.steps_completed.append(tool_name)

                # Update step log
                step_log['status'] = 'completed'
                step_log['end_time'] = datetime.now().isoformat()
                step_log['duration_seconds'] = step_duration
                step_log['exit_code'] = 0
                self.workflow_log['steps'].append(step_log)

                return True
            else:
                error_msg = f"{tool_name} failed with exit code {result.returncode}"
                print(f"❌ {error_msg}")

                if result.stderr.strip():
                    print(f"   📤 Error output:")
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-5:]:  # Show last 5 error lines
                        print(f"      {line}")
                    step_log['errors'].extend(error_lines[-5:])

                if result.stdout.strip():
                    print(f"   📤 Standard output:")
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines[-5:]:  # Show last 5 output lines
                        print(f"      {line}")

                self.errors.append(error_msg)
                print(
                    f"❌ Step {len(self.steps_completed)+1}: {tool_name} FAILED after {step_duration:.1f}s")

                # Update step log
                step_log['status'] = 'failed'
                step_log['end_time'] = datetime.now().isoformat()
                step_log['duration_seconds'] = step_duration
                step_log['exit_code'] = result.returncode
                step_log['errors'].append(error_msg)
                self.workflow_log['steps'].append(step_log)

                return False

        except subprocess.TimeoutExpired:
            step_duration = time.time() - step_start_time
            self.step_times[tool_name] = step_duration
            error_msg = f"{tool_name} timed out after 5 minutes"
            print(f"❌ {error_msg}")
            print(
                f"❌ Step {len(self.steps_completed)+1}: {tool_name} FAILED after {step_duration:.1f}s")
            self.errors.append(error_msg)
            os.chdir(original_cwd)  # Restore directory

            # Update step log
            step_log['status'] = 'timeout'
            step_log['end_time'] = datetime.now().isoformat()
            step_log['duration_seconds'] = step_duration
            step_log['errors'].append(error_msg)
            self.workflow_log['steps'].append(step_log)

            return False

        except Exception as e:
            step_duration = time.time() - step_start_time
            self.step_times[tool_name] = step_duration
            error_msg = f"{tool_name} execution error: {e}"
            print(f"❌ {error_msg}")
            print(
                f"❌ Step {len(self.steps_completed)+1}: {tool_name} FAILED after {step_duration:.1f}s")
            self.errors.append(error_msg)

            # Update step log
            step_log['status'] = 'error'
            step_log['end_time'] = datetime.now().isoformat()
            step_log['duration_seconds'] = step_duration
            step_log['errors'].append(error_msg)
            self.workflow_log['steps'].append(step_log)
            os.chdir(original_cwd)  # Restore directory
            return False

    def step0_triage_and_quality(self) -> bool:
        """Step 0: Triage and quality validation using unified orchestrator."""
        if not ORCHESTRATOR_AVAILABLE:
            print("⚠️  Orchestrator not available - skipping pre-flight checks")
            self.workflow_route = 'standard'
            return True

        print("📋 Running Unified Triage & Quality Gate...")
        print("   Using: llm_manager/triage_router.py + quality_gatekeeper.py")
        print()

        # Get raw files only from 00_raw_data (not recursive)
        raw_data_dir = self.config.project_root / "00_raw_data"
        if not raw_data_dir.exists():
            print("⚠️  No 00_raw_data folder found")
            self.workflow_route = 'standard'
            return True

        raw_files = []
        extensions = [
            ".spe", ".vgd", ".npl", ".xy", ".txt", ".asc", ".dat",
            ".csv", ".vms", ".vamas", ".pro"
        ]
        for ext in extensions:
            raw_files.extend(raw_data_dir.glob(f"*{ext}"))

        # Filter out non-data files
        raw_files = [f for f in raw_files if f.is_file()
                     and not f.name.endswith('.py')]

        if not raw_files:
            print("⚠️  No raw files found in 00_raw_data/")
            self.workflow_route = 'standard'
            return True

        # Detect data types across all files (silent mode)
        print(f"🔍 Scanning {len(raw_files)} files to determine workflow...")

        # Use lightweight triage directly (bypass orchestrator verbosity)
        import sys
        from pathlib import Path as PathLib
        _llm_path = PathLib(__file__).resolve().parents[1] / "llm_manager"
        if str(_llm_path) not in sys.path:
            sys.path.insert(0, str(_llm_path))

        from enhanced_triage_fixed import EnhancedXPSDataTriage
        triage = EnhancedXPSDataTriage(debug=False)

        map_files = []
        standard_files = []
        unknown_files = []

        for raw_file in raw_files:
            try:
                result = triage.analyze_file_structure(raw_file)
                data_type = result['data_type'].value
                confidence = result['confidence']

                if confidence < 0.5:
                    unknown_files.append(raw_file.name)
                elif data_type in ['map_2d', 'map_hyperspectral']:
                    map_files.append(raw_file.name)
                elif data_type == 'standard_spectra':
                    standard_files.append(raw_file.name)
                else:
                    unknown_files.append(raw_file.name)

            except Exception as e:
                if self.debug:
                    print(f"⚠️  Triage error for {raw_file.name}: {e}")
                unknown_files.append(raw_file.name)

        # Summary
        print("📊 Triage Summary:")
        if map_files:
            print(f"   🗺️  Map files: {len(map_files)}")
            for f in map_files[:3]:
                print(f"      • {f}")
            if len(map_files) > 3:
                print(f"      ... and {len(map_files) - 3} more")

        if standard_files:
            print(f"   📈 Standard files: {len(standard_files)}")
            for f in standard_files[:3]:
                print(f"      • {f}")
            if len(standard_files) > 3:
                print(f"      ... and {len(standard_files) - 3} more")

        if unknown_files:
            print(f"   ❓ Unknown files: {len(unknown_files)}")
            for f in unknown_files[:3]:
                print(f"      • {f}")

        print()

        # Log triage results
        triage_log = {
            'step_name': 'Triage_QualityGate',
            'description': 'File classification and quality gate',
            'start_time': datetime.now().isoformat(),
            'status': 'completed',
            'errors': [],
            'triage_results': {
                'total_files': len(raw_files),
                'map_files': len(map_files),
                'standard_files': len(standard_files),
                'unknown_files': len(unknown_files),
                'map_file_list': map_files,
                'standard_file_list': standard_files,
                'unknown_file_list': unknown_files
            }
        }

        # Determine workflow routing
        if map_files and not standard_files:
            self.workflow_route = 'map'
            print("✅ Routing decision: MAP WORKFLOW (XPS_mapper)")
            triage_log['routing_decision'] = 'map'
            triage_log['workflow_type'] = 'XPS_mapper'
        elif standard_files and not map_files:
            self.workflow_route = 'standard'
            print("✅ Routing decision: STANDARD WORKFLOW (reader → fitter → quantifier)")
            triage_log['routing_decision'] = 'standard'
            triage_log['workflow_type'] = 'reader → fitter → quantifier'
        elif map_files and standard_files:
            print("⚠️  Mixed data types detected!")
            print(f"   • {len(map_files)} map files")
            print(f"   • {len(standard_files)} standard files")
            print()
            print("✅ Automatic handling: Will run BOTH workflows sequentially")
            print("   1. Standard workflow (process spectra)")
            print("   2. Map workflow (process maps)")

            # Set to 'both' mode - we'll handle this in run_complete_workflow
            self.workflow_route = 'both'
            triage_log['routing_decision'] = 'both'
            triage_log['workflow_type'] = 'mixed (standard + map)'
        elif unknown_files and not map_files and not standard_files:
            # All files are unknown - check if we have converted CSVs already
            csv_dir = self.config.project_root / "01_converted_csv"
            has_csv_files = csv_dir.exists() and any(csv_dir.rglob("*.csv"))

            if has_csv_files:
                print("⚠️  Raw file types uncertain, but converted CSVs exist")
                print("✅ SKIPPING Step 1 (Reader) - proceeding with existing CSV files")
                print("   Starting from Step 2: Peak Fitting")
                self.workflow_route = 'standard_skip_reader'
                triage_log['routing_decision'] = 'standard_skip_reader'
                triage_log['workflow_type'] = 'skip reader (CSVs exist)'
            else:
                # No CSVs, assume standard workflow
                print("⚠️  File types could not be determined with confidence")
                print("✅ Defaulting to STANDARD WORKFLOW (most common for XPS data)")
                print("   If files are actually maps, please use XPS_mapper directly")
                self.workflow_route = 'standard'
                triage_log['routing_decision'] = 'standard'
                triage_log['workflow_type'] = 'default standard (uncertain file types)'
        else:
            print("❌ No files found for processing")
            triage_log['status'] = 'failed'
            triage_log['errors'] = ['No files found for processing']
            triage_log['end_time'] = datetime.now().isoformat()
            triage_log['duration_seconds'] = 0
            self.workflow_log['steps'].append(triage_log)
            return False

        # Finalize triage log
        triage_log['end_time'] = datetime.now().isoformat()
        triage_log['duration_seconds'] = 0.1  # Triage is fast
        self.workflow_log['steps'].append(triage_log)

        return True

    def step1_xps_reader(self) -> bool:
        """Step 1: Run XPS_Reader (includes triage and quality validation)."""
        return self.run_xps_tool(
            "XPS_Reader",
            self.xps_tools['reader'],
            "Triage + Quality Gate + Convert raw XPS files to standardized CSV format"
        )

    def step1_xps_mapper(self) -> bool:
        """Step 1 (Alt): Run XPS_Mapper for map data."""
        return self.run_xps_tool(
            "XPS_Mapper",
            self.xps_tools['mapper'],
            "Process hyperspectral map data: clustering, MCR, visualization"
        )

    def step2_xps_fitter(self) -> bool:
        """Step 2: Run XPS_Fitter."""
        return self.run_xps_tool(
            "XPS_Fitter",
            self.xps_tools['fitter'],
            "Fit peaks to XPS spectra using template-based fitting"
        )

    def step3_xps_quantifier(self) -> bool:
        """Step 3: Run XPS_Quantifier."""
        return self.run_xps_tool(
            "XPS_Quantifier",
            self.xps_tools['quantifier'],
            "Quantify elemental composition from fitted results"
        )

    def step4_xps_plotter(self) -> bool:
        """Step 4: Run XPS_Plotter."""
        return self.run_xps_tool(
            "XPS_Plotter",
            self.xps_tools['plotter'],
            "Generate plots and visualizations"
        )

    def _execute_workflow_step(self, step_name: str, step_func, mandatory: bool = False):
        """Run individual workflow steps with consistent logging and control flow."""
        print(f"\n{'='*60}")
        print(f"\U0001F504 {step_name}")
        print(f"{'='*60}")
        step_start = time.time()
        success = step_func()
        step_time = time.time() - step_start

        if success:
            print(f"\u2705 {step_name} completed in {step_time:.1f}s")
            return True, True

        print(f"\u274c {step_name} FAILED after {step_time:.1f}s")

        if mandatory:
            return False, False

        if self.interactive:
            print("\nOptions:")
            print("1. Continue to next step (may work if this step was optional)")
            print("2. Stop workflow here")

            try:
                response = input("Continue? (y/n): ").lower()
                if response != 'y':
                    print("\U0001F6D1 Workflow stopped by user")
                    return False, False
            except KeyboardInterrupt:
                print("\n\U0001F6D1 Workflow interrupted by user")
                return False, False
        else:
            print("\u26a0\ufe0f Continuing to next step (non-interactive mode)")
            print("   Previous step failure may affect subsequent steps")

        return False, True

    def run_complete_workflow(self) -> bool:
        """Run the complete XPS workflow using existing tools."""
        start_time_str = time.strftime(
            "%H:%M:%S", time.localtime(self.start_time))
        print("\U0001F680 STARTING REAL XPS WORKFLOW")
        print("=" * 80)
        print("Using existing XPS tools from the Tools folder")
        print(f"\U0001F550 Started at: {start_time_str}")
        print()

        # Check dependencies
        if not self.check_dependencies():
            print("\u274c Cannot proceed without required tools")
            return False

        # Check input data
        data_info = self.check_input_data()

        if not data_info['can_start_from_reader']:
            print("\u274c No raw data files found to process")
            return False

        all_success = True
        start_time = time.time()

        # Step 0 is mandatory and determines routing
        step0_success, _ = self._execute_workflow_step(
            "Step 0: Triage & Quality Gate (Mandatory)",
            self.step0_triage_and_quality,
            mandatory=True
        )

        if not step0_success:
            total_time = time.time() - start_time
            self.print_workflow_summary(total_time)
            return False

        if not self.workflow_route:
            self.workflow_route = 'standard'

        if self.workflow_route == 'map':
            print("\n\U0001F500 Workflow routing: MAP ONLY")
            workflow_steps = [
                ("Step 1: XPS Mapper (Map Processing)", self.step1_xps_mapper),
                ("Step 2: XPS Plotter (Map Visualization)", self.step4_xps_plotter)
            ]
        elif self.workflow_route == 'standard_skip_reader':
            print("\n🔀 Workflow routing: STANDARD (starting from CSV files)")
            print("   Skipping Step 1 (Reader) - CSV files already exist")
            workflow_steps = [
                ("Step 1b: Depth Profile Waterfall (from CSV)", self.step1_depth_profile_plots_from_csv),
                ("Step 2: XPS Fitter", self.step2_xps_fitter),
                ("Step 3: XPS Quantifier", self.step3_xps_quantifier),
                ("Step 4: XPS Plotter", self.step4_xps_plotter)
            ]
        elif self.workflow_route == 'both':
            print("\n\U0001F500 Workflow routing: MIXED DATA (standard \u279c map)")
            print(
                "   Will execute the standard workflow first, followed by the map workflow")
            workflow_steps = [
                ("Step 1: XPS Reader (Standard Files)", self.step1_xps_reader),
                ("Step 2: XPS Fitter", self.step2_xps_fitter),
                ("Step 3: XPS Quantifier", self.step3_xps_quantifier),
                ("Step 4: XPS Plotter (Standard)", self.step4_xps_plotter),
                ("Step 5: XPS Mapper (Map Files)", self.step1_xps_mapper),
                ("Step 6: XPS Plotter (Maps)", self.step4_xps_plotter)
            ]
        else:
            print("\n\U0001F500 Workflow routing: STANDARD SPECTRA")
            workflow_steps = [
                ("Step 1: XPS Reader + Data Validation", self.step1_xps_reader),
                ("Step 2: XPS Fitter", self.step2_xps_fitter),
                ("Step 3: XPS Quantifier", self.step3_xps_quantifier),
                ("Step 4: XPS Plotter", self.step4_xps_plotter)
            ]

        for step_name, step_func in workflow_steps:
            success, should_continue = self._execute_workflow_step(
                step_name, step_func)

            if not success:
                all_success = False

            if not should_continue:
                break

        total_time = time.time() - start_time
        self.print_workflow_summary(total_time)

        return all_success

    def print_workflow_summary(self, total_time: float):
        """Print final workflow summary."""
        print("\n" + "=" * 80)
        print("📋 REAL XPS WORKFLOW SUMMARY")
        print("=" * 80)

        print(f"⏱️  Total Workflow Time: {total_time:.1f} seconds")
        print(f"🔧 Total Operations: {self.total_operations}")
        print(f"� Input Files Processed: {self.total_files_processed}")
        print(f"�📂 Project: {self.config.project_root}")

        # Step timing breakdown
        if self.step_times:
            print(f"\n⏱️  Step Timing Breakdown:")
            for i, step_name in enumerate(["XPS_Reader", "XPS_Fitter", "XPS_Quantifier", "XPS_Plotter"], 1):
                if step_name in self.step_times:
                    duration = self.step_times[step_name]
                    percentage = (duration / total_time) * 100
                    status = "✅" if step_name in self.steps_completed else "❌"
                    print(
                        f"   {status} Step {i} ({step_name}): {duration:.1f}s ({percentage:.1f}%)")
                else:
                    print(f"   ⏭️  Step {i} ({step_name}): Not executed")

        print(f"\n✅ Completed Steps: {len(self.steps_completed)}/4")

        step_names = ["XPS_Reader", "XPS_Fitter",
                      "XPS_Quantifier", "XPS_Plotter"]
        for i, step in enumerate(step_names, 1):
            status = "✅" if step in self.steps_completed else "❌"
            print(f"   {status} Step {i}: {step}")

        if self.errors:
            print(f"\n❌ Errors Encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"   ✗ {error}")

        # Show generated files
        print("\n📁 Generated Files:")

        output_dirs = [
            ("Raw Data", self.config.raw_data_dir),
            ("Converted CSV", self.config.csv_output_dir),
            ("Fitted Results", self.config.fits_output_dir),
            ("Quantification", self.config.quant_output_dir),
            ("Plots", self.config.plots_output_dir)
        ]

        for name, directory in output_dirs:
            if directory.exists():
                files = list(directory.glob("*"))
                print(f"   📊 {name}: {len(files)} files in {directory.name}/")

                # Show a few example files
                for file in files[:3]:
                    if file.is_file():
                        print(f"     • {file.name}")
                if len(files) > 3:
                    print(f"     ... and {len(files) - 3} more files")
            else:
                print(f"   📂 {name}: Directory not found")

        # Performance statistics
        if self.step_times:
            total_step_time = sum(self.step_times.values())
            overhead_time = max(0, total_time - total_step_time)

            print(f"\n📈 Performance Statistics:")
            print(f"   ⏱️  Active Processing Time: {total_step_time:.1f}s")
            print(f"   ⏱️  Overhead Time: {overhead_time:.1f}s")
            if self.total_operations > 0:
                avg_time_per_op = total_step_time / self.total_operations
                print(
                    f"   ⚡ Average Time per Operation: {avg_time_per_op:.1f}s")

            # Show fastest and slowest steps
            if len(self.step_times) > 1:
                fastest_step = min(self.step_times.items(), key=lambda x: x[1])
                slowest_step = max(self.step_times.items(), key=lambda x: x[1])
                print(
                    f"   🏃 Fastest Step: {fastest_step[0]} ({fastest_step[1]:.1f}s)")
                print(
                    f"   🐌 Slowest Step: {slowest_step[0]} ({slowest_step[1]:.1f}s)")

        success_rate = len(self.steps_completed) / 4 * 100
        print(f"\n🎯 Success Rate: {success_rate:.0f}%")

        if len(self.steps_completed) == 4:
            print("🎉 COMPLETE XPS WORKFLOW SUCCESSFUL!")
        elif len(self.steps_completed) > 0:
            print("⚠️  PARTIAL XPS WORKFLOW COMPLETED")
        else:
            print("❌ XPS WORKFLOW FAILED")

        print("=" * 80)

        # Save workflow log
        self._save_workflow_log()

    def _save_workflow_log(self):
        """Save workflow execution log to JSON file."""
        # Finalize log
        self.workflow_log['end_time'] = datetime.now().isoformat()
        self.workflow_log['total_duration_seconds'] = time.time() - \
            self.start_time
        self.workflow_log['summary'] = {
            'total_steps': len(self.workflow_log['steps']),
            'completed_steps': len(self.steps_completed),
            'failed_steps': len([s for s in self.workflow_log['steps'] if s['status'] in ['failed', 'error', 'timeout']]),
            'success_rate_percent': len(self.steps_completed) / 4 * 100 if len(self.workflow_log['steps']) > 0 else 0,
            'total_files_processed': self.total_files_processed,
            'total_errors': len(self.errors),
            'errors': self.errors
        }

        # Save to _logs directory
        logs_dir = self.config.project_root / "_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON log (AI-readable)
        json_log_path = logs_dir / \
            f"workflow_log_{self.workflow_log['workflow_id']}.json"
        with open(json_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.workflow_log, f, indent=2, ensure_ascii=False)

        print(f"\n📝 Workflow log saved: {json_log_path}")

        # Also save human-readable text summary
        txt_log_path = logs_dir / \
            f"workflow_summary_{self.workflow_log['workflow_id']}.txt"
        with open(txt_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("XPS WORKFLOW EXECUTION LOG\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Workflow ID: {self.workflow_log['workflow_id']}\n")
            f.write(f"Project Root: {self.workflow_log['project_root']}\n")
            f.write(f"Start Time: {self.workflow_log['start_time']}\n")
            f.write(f"End Time: {self.workflow_log['end_time']}\n")
            f.write(
                f"Total Duration: {self.workflow_log['total_duration_seconds']:.1f}s\n\n")

            f.write("STEPS EXECUTED:\n")
            f.write("-" * 80 + "\n")
            for step in self.workflow_log['steps']:
                f.write(f"\nStep: {step['step_name']}\n")
                f.write(f"  Description: {step['description']}\n")
                f.write(f"  Status: {step['status'].upper()}\n")
                f.write(f"  Duration: {step['duration_seconds']:.1f}s\n")
                if step['errors']:
                    f.write(f"  Errors:\n")
                    for error in step['errors']:
                        f.write(f"    - {error}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY:\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"Success Rate: {self.workflow_log['summary']['success_rate_percent']:.0f}%\n")
            f.write(
                f"Steps Completed: {self.workflow_log['summary']['completed_steps']}/{self.workflow_log['summary']['total_steps']}\n")
            f.write(
                f"Failed Steps: {self.workflow_log['summary']['failed_steps']}\n")
            f.write(
                f"Total Errors: {self.workflow_log['summary']['total_errors']}\n")

        print(f"📝 Workflow summary saved: {txt_log_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run complete XPS workflow')
    parser.add_argument('project_root', nargs='?',
                        help='Project root directory')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run in non-interactive mode (no user prompts)')

    args = parser.parse_args()
    project_root = args.project_root
    interactive = not args.non_interactive

    try:
        workflow = RealXPSWorkflow(project_root, interactive=interactive)
        success = workflow.run_complete_workflow()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
