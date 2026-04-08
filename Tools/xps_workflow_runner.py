#!/usr/bin/env python3
"""
XPS Workflow Runner - Complete Processing Pipeline
=================================================

This script runs the complete XPS workflow pipeline:
1. Check and organize input data
2. Run XPS_Fitter (CSV -> Fitted results)
3. Run XPS_Quantifier (Fitted -> Quantified data)
4. Run XPS_Plotter (Create visualizations)
5. Run XPS_Correlator (ML analysis)

Usage:
    python xps_workflow_runner.py [project_root]

Features:
- Automatic dependency checking
- Step-by-step progress tracking
- Error handling and recovery
- Results summary
"""

import sys
import os
from pathlib import Path
import subprocess
import time
from typing import List, Dict, Optional

# Add current directory to Python path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from xps_workflow_manager import (
        get_workflow_config, 
        update_module_paths,
        XPSWorkflowConfig
    )
    WORKFLOW_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"ERROR Workflow manager not available: {e}")
    sys.exit(1)


class XPSWorkflowRunner:
    """Complete XPS workflow execution manager."""
    
    def __init__(self, project_root: str = None):
        """Initialize workflow runner with project configuration."""
        self.config = get_workflow_config(project_root)
        self.steps_completed = []
        self.errors = []
        
    def check_dependencies(self) -> bool:
        """Check if required Python packages are available."""
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'scipy', 'lmfit'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"ERROR Missing required packages: {', '.join(missing_packages)}")
            print("   Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("OK All required packages available")
        return True
    
    def check_input_data(self) -> Dict[str, List[Path]]:
        """Check what input data is available and organize by processing step."""
        raw_data = list(self.config.raw_data_dir.glob("*"))
        csv_data = list(self.config.csv_output_dir.glob("*.csv"))
        
        data_summary = {
            'raw_files': raw_data,
            'csv_files': csv_data,
            'can_start_fitting': len(csv_data) > 0,
            'can_start_reading': len(raw_data) > 0
        }
        
        print(f"\nData inventory:")
        print(f"   Raw files: {len(raw_data)} in {self.config.raw_data_dir}")
        print(f"   CSV files: {len(csv_data)} in {self.config.csv_output_dir}")
        
        if data_summary['can_start_fitting']:
            print("   Ready to start XPS fitting workflow")
        elif data_summary['can_start_reading']:
            print("   Ready to start XPS reading workflow")
        else:
            print("   No input data found")
        
        return data_summary
    
    def run_step(
        self,
        step_name: str,
        script_path: Path,
        description: str,
        extra_args: Optional[List[str]] = None,
    ) -> bool:
        """Run a single workflow step."""
        print(f"\nRunning {step_name}: {description}")
        print(f"   Script: {script_path}")
        
        if not script_path.exists():
            error_msg = f"Script not found: {script_path}"
            print(f"ERROR {error_msg}")
            self.errors.append(error_msg)
            return False
        
        try:
            cmd = [sys.executable, str(script_path)]
            if extra_args:
                cmd.extend(extra_args)
            
            result = subprocess.run(
                cmd,
                cwd=str(self.config.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"OK {step_name} completed successfully")
                self.steps_completed.append(step_name)
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
                return True
            else:
                error_msg = f"{step_name} failed (exit code {result.returncode})"
                print(f"ERROR {error_msg}")
                if result.stderr.strip():
                    print(f"   Error: {result.stderr.strip()}")
                self.errors.append(error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = f"{step_name} timed out"
            print(f"ERROR {error_msg}")
            self.errors.append(error_msg)
            return False
        except Exception as e:
            error_msg = f"{step_name} error: {e}"
            print(f"ERROR {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def copy_csv_files_to_workflow(self, data_summary: Dict) -> bool:
        """Copy CSV files from raw_data to converted_csv if needed."""
        if not data_summary['csv_files'] and data_summary['raw_files']:
            # Check if raw files are actually CSV files
            raw_csv_files = [f for f in data_summary['raw_files'] if f.suffix.lower() == '.csv']
            
            if raw_csv_files:
                print(f"\nSummary Moving CSV files to workflow directory...")
                for csv_file in raw_csv_files:
                    dest_file = self.config.csv_output_dir / csv_file.name
                    try:
                        # Copy the file
                        import shutil
                        shutil.copy2(csv_file, dest_file)
                        print(f"   OK Copied: {csv_file.name}")
                    except Exception as e:
                        print(f"   ERROR Failed to copy {csv_file.name}: {e}")
                        return False
                
                print(f"OK Moved {len(raw_csv_files)} CSV files to workflow")
                return True
        
        return True
    
    def run_complete_workflow(self) -> bool:
        """Run the complete XPS workflow from start to finish."""
        
        print("Starting XPS complete workflow...")
        print("=" * 60)
        print(f"Project: {self.config.project_root}")
        
        # Check dependencies
        if not self.check_dependencies():
            print("\nERROR Cannot proceed without required dependencies")
            return False
        
        # Check input data
        data_summary = self.check_input_data()
        
        # Move CSV files if they're in raw_data
        if not self.copy_csv_files_to_workflow(data_summary):
            return False
        
        # Re-check after moving files
        data_summary = self.check_input_data()
        
        # Always delegate to the consolidated real workflow script (live under Tools)
        real_workflow_script = SCRIPT_DIR / "real_xps_workflow.py"
        if not real_workflow_script.exists():
            raise FileNotFoundError(f"real_xps_workflow.py not found at {real_workflow_script}")

        print("\nExecuting real_xps_workflow end-to-end run...")
        success = self.run_step(
            'Real_XPS_Workflow',
            real_workflow_script,
            'Run the full reader -> fitter -> quantifier -> plotter pipeline',
            extra_args=[str(self.config.project_root)],
        )
        
        # Print summary
        self.print_workflow_summary()
        return success
    
    def print_workflow_summary(self):
        """Print a summary of the workflow execution."""
        print("\n" + "=" * 60)
        print("WORKFLOW EXECUTION SUMMARY")
        print("=" * 60)
        
        print(f"Project: {self.config.project_root}")
        print(f"Completed steps: {len(self.steps_completed)}")
        for step in self.steps_completed:
            print(f"   - {step}")
        
        if self.errors:
            print(f"Errors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"   - {error}")
        else:
            print("No errors encountered")
        
        # Check output directories
        print("\nOutput directories:")
        dirs_to_check = [
            ('Converted CSV', self.config.csv_output_dir),
            ('Fitted Results', self.config.fits_output_dir),
            ('Quantified Data', self.config.quant_output_dir),
            ('Plots', self.config.plots_output_dir),
        ]
        
        for name, directory in dirs_to_check:
            if directory.exists():
                file_count = len(list(directory.glob("*")))
                print(f"   {name}: {file_count} items in {directory}")
            else:
                print(f"   {name}: Directory not created")
        
        print("\nWorkflow execution complete.\n")


def main():
    """Main entry point for workflow runner."""
    
    # Check for custom project root
    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        # Create and run workflow
        runner = XPSWorkflowRunner(project_root)
        success = runner.run_complete_workflow()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nWorkflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
