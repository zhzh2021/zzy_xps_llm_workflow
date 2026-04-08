"""
Comparison Tool: Unconstrained Fit vs Your Existing Template-Based Results

This script:
1. Loads your existing template-based fitted results from 02_fitted_results
2. Runs unconstrained fitting on the same raw data
3. Generates side-by-side comparison plots showing:
   - Fit quality (R², residuals)
   - Parameter values (FWHM, positions, areas)
   - Physical plausibility of results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from lmfit.models import PseudoVoigtModel, LinearModel

# Add parent directories to path
tools_dir = Path(__file__).resolve().parents[1]
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from XPS_utils.background_correction import baseline_shirley

# ========== CONFIGURATION ==========
# Resolve project root relative to this file so the script works on any machine.
# Override by setting the ZZY_LLM_HOME environment variable to your project root.
import os as _os
PROJECT_ROOT = Path(
    _os.environ.get("ZZY_LLM_HOME") or Path(__file__).resolve().parents[3] / "project_root"
)
DATA_PATH = PROJECT_ROOT / "01_converted_csv" / "Li1s"
FITTED_RESULTS_PATH = PROJECT_ROOT / "02_fitted_results" / "Li1s"
OUTPUT_DIR = PROJECT_ROOT / "04_plots" / "fitter_comparison"

# Auto-detect available test files from fitted results
def get_available_samples():
    """Auto-detect samples from fitted results directory."""
    csv_files = list(FITTED_RESULTS_PATH.glob("*_analysis_results.csv"))
    samples = []
    for csv_file in csv_files:
        # Extract sample name (e.g., "5scans" from "5scans_Li1s_analysis_results.csv")
        sample_name = csv_file.stem.replace("_Li1s_analysis_results", "")
        samples.append(sample_name)
    return sorted(samples)


class UnconstrainedFitter:
    """Unconstrained peak fitting with minimal constraints."""
    
    def __init__(self, n_peaks=2):
        self.n_peaks = n_peaks
        self.result = None
        
    def fit(self, x, y):
        """Fit with unconstrained parameters."""
        # Build model
        models = []
        for i in range(self.n_peaks):
            prefix = f'p{i+1}_'
            models.append(PseudoVoigtModel(prefix=prefix))
        
        background = LinearModel()
        # Build composite model by adding each peak model to the background.
        # Avoid `sum(models)` because Python's sum starts from 0 and cannot
        # be used to add model objects directly.
        model = background
        for m in models:
            model = model + m
        
        # Set up unconstrained parameters
        params = model.make_params()
        
        # Very wide bounds, minimal constraints
        x_range = x.max() - x.min()
        x_mid = (x.max() + x.min()) / 2
        
        for i in range(self.n_peaks):
            prefix = f'p{i+1}_'
            params[f'{prefix}center'].set(value=x_mid + (i - self.n_peaks/2) * 2, 
                                         min=x.min(), max=x.max())
            params[f'{prefix}sigma'].set(value=1.0, min=0.1, max=x_range/2)
            params[f'{prefix}amplitude'].set(value=y.max() / self.n_peaks, min=0)
            params[f'{prefix}fraction'].set(value=0.5, min=0, max=1)
        
        params['slope'].set(value=0, min=-100, max=100)
        params['intercept'].set(value=np.mean(y), min=0)
        
        # Fit
        self.result = model.fit(y, params, x=x)
        
        return self.result
    
    def get_metrics(self):
        """Extract fit metrics."""
        if self.result is None:
            return {}
        
        metrics = {
            'r2': 1 - self.result.residual.var() / np.var(self.result.data),
            'redchi': self.result.redchi,
            'peaks': []
        }
        
        for i in range(self.n_peaks):
            prefix = f'p{i+1}_'
            center = self.result.params[f'{prefix}center'].value
            sigma = self.result.params[f'{prefix}sigma'].value
            fwhm = sigma * 2.355
            amplitude = self.result.params[f'{prefix}amplitude'].value
            
            metrics['peaks'].append({
                'name': f'Peak {i+1}',
                'center': center,
                'fwhm': fwhm,
                'amplitude': amplitude
            })
        
        return metrics


class ExistingFitLoader:
    """Load your existing template-based fit results."""
    
    def __init__(self, sample_name, region='Li1s'):
        self.sample_name = sample_name
        self.region = region
        self.results_df = None
        self.results_json = None
        self.x = None
        self.y = None
        self.y_corrected = None
        self.bg = None
        
    def load_results(self):
        """Load CSV and JSON results."""
        csv_file = FITTED_RESULTS_PATH / f"{self.sample_name}_{self.region}_analysis_results.csv"
        json_file = FITTED_RESULTS_PATH / f"{self.sample_name}_{self.region}_analysis_results.json"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Results not found: {csv_file}")
        
        self.results_df = pd.read_csv(csv_file)
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.results_json = json.load(f)
        
        return self.results_df
    
    def load_raw_data(self):
        """Load the original raw data that was fitted."""
        # Use aggregated file (standard location for converted data)
        data_file = DATA_PATH / f"aggregated_{self.region}_allHR.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Aggregated data file not found: {data_file}")
        
        # Load CSV with pandas to handle headers and comments
        df = pd.read_csv(data_file, comment='#')
        
        # Extract specific sample column
        sample_col = f"{self.sample_name}_{self.region}_cps"
        if sample_col not in df.columns:
            # Try without region suffix
            sample_col = f"{self.sample_name}_cps"
            if sample_col not in df.columns:
                available_cols = [c for c in df.columns if c != 'Binding_Energy_eV']
                raise ValueError(
                    f"Column '{self.sample_name}_{self.region}_cps' not found. "
                    f"Available columns: {', '.join(available_cols)}"
                )
        
        self.x = df['Binding_Energy_eV'].values
        self.y = df[sample_col].values
        
        # Apply background correction like the fitter does
        self.bg = baseline_shirley(self.x, self.y)
        # Clamp negative values to 0 (matches XPS_peakfitting_V2.py line 590)
        self.y_corrected = np.maximum(self.y - self.bg, 0.0)
        
        return self.x, self.y
    
    def get_metrics(self):
        """Extract metrics from loaded results."""
        if self.results_df is None:
            return {}
        
        # Get R² from first row (same for all components)
        r2 = self.results_df.iloc[0]['R_squared']
        
        metrics = {
            'r2': r2,
            'redchi': np.nan,  # Not stored in CSV
            'peaks': []
        }
        
        # Extract each component
        for _, row in self.results_df.iterrows():
            metrics['peaks'].append({
                'name': row['Component'],
                'center': row['Center_eV'],
                'fwhm': row['FWHM_eV'],
                'area_percent': row['Area_percent']
            })
        
        return metrics
    
    def reconstruct_fit(self):
        """Reconstruct fitted curve from components using pseudo-Voigt."""
        if self.x is None or self.results_df is None:
            return None
        
        model = np.zeros_like(self.x)
        
        for _, row in self.results_df.iterrows():
            center = row['Center_eV']
            fwhm = row['FWHM_eV']
            sigma = fwhm / 2.355
            # Use area_percent as relative amplitude
            amplitude = row['Area_percent'] / 100.0 * self.y_corrected.max()
            eta = row['Eta_mix']
            
            # Pseudo-Voigt approximation
            gaussian = np.exp(-((self.x - center) ** 2) / (2 * sigma ** 2))
            lorentzian = 1 / (1 + ((self.x - center) / sigma) ** 2)
            component = amplitude * ((1 - eta) * gaussian + eta * lorentzian)
            model += component
        
        return model


def compare_fits(sample_name, output_dir):
    """Compare unconstrained vs existing template-based results."""
    
    print(f"\n{'='*70}")
    print(f"Comparing fits for: {sample_name}")
    print(f"{'='*70}")
    
    # Load existing template-based results
    print("\n[1/2] Loading existing template-based fit results...")
    existing_fit = ExistingFitLoader(sample_name)
    existing_fit.load_results()
    existing_fit.load_raw_data()
    
    # Debug: Check data ranges
    print(f"      [DEBUG] Raw data range: {existing_fit.y.min():.1f} - {existing_fit.y.max():.1f}")
    print(f"      [DEBUG] Background range: {existing_fit.bg.min():.1f} - {existing_fit.bg.max():.1f}")
    print(f"      [DEBUG] Corrected range: {existing_fit.y_corrected.min():.1f} - {existing_fit.y_corrected.max():.1f}")
    
    metrics_template = existing_fit.get_metrics()
    
    # Run unconstrained fit on same background-corrected data
    print("[2/2] Running unconstrained fit on background-corrected data...")
    n_peaks = len(existing_fit.results_df)
    unconstrained = UnconstrainedFitter(n_peaks=n_peaks)
    unconstrained.fit(existing_fit.x, existing_fit.y_corrected)
    metrics_unc = unconstrained.get_metrics()
    
    # Print comparison
    print(f"\n{'='*70}")
    print("FIT QUALITY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'R²':<12}")
    print(f"{'-'*70}")
    print(f"{'Unconstrained':<25} {metrics_unc['r2']:<12.4f}")
    print(f"{'Template-Based (Yours)':<25} {metrics_template['r2']:<12.4f}")
    
    print(f"\n{'='*70}")
    print("PEAK PARAMETERS")
    print(f"{'='*70}")
    print("\nUnconstrained Fit:")
    for peak in metrics_unc['peaks']:
        print(f"  {peak['name']}: Center={peak['center']:.2f} eV, FWHM={peak['fwhm']:.2f} eV")
    
    print("\nYour Template-Based Fit:")
    for peak in metrics_template['peaks']:
        print(f"  {peak['name']}: Center={peak['center']:.2f} eV, FWHM={peak['fwhm']:.2f} eV, Area={peak['area_percent']:.1f}%")
    
    # Generate comparison plot
    plot_comparison(existing_fit, unconstrained, sample_name, output_dir)
    
    return metrics_unc, metrics_template


def plot_comparison(existing_fit, unc_fitter, sample_name, output_dir):
    """Create side-by-side comparison plot."""
    x = existing_fit.x
    y = existing_fit.y
    y_corr = existing_fit.y_corrected
    bg = existing_fit.bg
    
    # Debug: verify data ranges at plotting time
    print(f"      [PLOT DEBUG] y (raw) range: {y.min():.1f} - {y.max():.1f}")
    print(f"      [PLOT DEBUG] bg range: {bg.min():.1f} - {bg.max():.1f}")
    print(f"      [PLOT DEBUG] y_corr range: {y_corr.min():.1f} - {y_corr.max():.1f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Unconstrained fit (left column)
    ax1, ax2 = axes[:, 0]
    
    # Main plot - shift fit to align with raw data by adding background offset
    ax1.plot(x, y, 'o', color='gray', alpha=0.6, markersize=4, label='Raw Data')
    ax1.plot(x, bg, '-', color='orange', linewidth=1.5, label='Background')
    ax1.plot(x, unc_fitter.result.best_fit + bg, '-', color='red', linewidth=2.5, label='Total Fit')
    
    # Components (fitted on corrected data, shifted up by background)
    components = unc_fitter.result.eval_components(x=x)
    for i in range(unc_fitter.n_peaks):
        prefix = f'p{i+1}_'
        ax1.plot(x, components[prefix] + bg, '--', linewidth=2, label=f'Peak {i+1}')
    
    ax1.set_title('Unconstrained Fit', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Intensity', fontsize=16, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=15)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(np.arange(0, y.max() + 200, 200))
    ax1.invert_xaxis()
    
    # Residuals (corrected data minus fit)
    residuals = y_corr - unc_fitter.result.best_fit
    ax2.plot(x, residuals, 'o-', color='red', markersize=3, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Binding Energy (eV)', fontsize=16, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=15)
    ax2.set_ylabel('Residuals', fontsize=16, fontweight='bold')
    # Set y-ticks based on actual residuals range
    res_min, res_max = residuals.min(), residuals.max()
    ytick_min = np.floor(res_min / 50) * 50
    ytick_max = np.ceil(res_max / 50) * 50
    ax2.set_yticks(np.arange(ytick_min, ytick_max + 50, 100))
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Your template-based fit (right column)
    ax3, ax4 = axes[:, 1]
    
    # Main plot - shift fit to align with raw data by adding background offset
    y_corr = existing_fit.y_corrected
    bg = existing_fit.bg
    
    ax3.plot(x, y, 'o', color='gray', alpha=0.6, markersize=4, label='Raw Data')
    ax3.plot(x, bg, '-', color='orange', linewidth=1.5, label='Background')
    
    # Reconstruct model from existing fit
    model = existing_fit.reconstruct_fit()
    
    # Plot individual components shifted up by background
    for _, row in existing_fit.results_df.iterrows():
        center = row['Center_eV']
        fwhm = row['FWHM_eV']
        sigma = fwhm / 2.355
        amplitude = row['Area_percent'] / 100.0 * y_corr.max()
        eta = row['Eta_mix']
        
        # Pseudo-Voigt approximation
        gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        lorentzian = 1 / (1 + ((x - center) / sigma) ** 2)
        component = amplitude * ((1 - eta) * gaussian + eta * lorentzian)
        
        # Shift component up by background
        ax3.plot(x, component + bg, '--', linewidth=2, label=row['Component'])
    
    # Plot total fit shifted up by background
    if model is not None:
        ax3.plot(x, model + bg, '-', color='red', linewidth=2.5, label='Total Fit', alpha=0.8)

    ax3.set_title('Template-Based Fit', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Intensity', fontsize=16, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=15)
    ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=15)
    ax3.set_yticks(np.arange(0, y.max() + 200, 200))
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Residuals
    if model is not None:
        residuals_temp = existing_fit.y_corrected - model
        ax4.plot(x, residuals_temp, 'o-', color='blue', markersize=3, alpha=0.6)
        # Set y-ticks based on actual residuals range
        res_min, res_max = residuals_temp.min(), residuals_temp.max()
        ytick_min = np.floor(res_min / 50) * 50
        ytick_max = np.ceil(res_max / 50) * 50
        ax4.set_yticks(np.arange(ytick_min, ytick_max + 50, 50))
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Binding Energy (eV)', fontsize=16, fontweight='bold')
    ax4.set_ylabel('Residuals', fontsize=16, fontweight='bold')
    ax4.set_xticklabels(ax4.get_xticklabels(), fontsize=15) 
    ax4.set_yticklabels(ax4.get_yticklabels(), fontsize=15)
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    # Overall title
    fig.suptitle(f'Fit Comparison: {sample_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_name}_Li1s_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[✓] Comparison plot saved: {output_path}")


def main():
    """Run comparison for all available samples."""
    print("\n" + "="*70)
    print("XPS FITTER COMPARISON TOOL")
    print("Unconstrained vs Existing Template-Based Results")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect available samples
    available_samples = get_available_samples()
    print(f"\n[INFO] Found {len(available_samples)} samples in 02_fitted_results:")
    for sample in available_samples:
        print(f"  - {sample}")
    
    results = []
    for sample_name in available_samples:
        try:
            metrics = compare_fits(sample_name, OUTPUT_DIR)
            results.append({
                'sample': sample_name,
                'unconstrained': metrics[0],
                'template': metrics[1]
            })
        except Exception as e:
            print(f"\n[ERROR] Failed to process {sample_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary report
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"{'Sample':<20} {'Method':<25} {'R²':<10}")
    print(f"{'-'*70}")
    
    for result in results:
        sample_name = result['sample']
        unc = result['unconstrained']
        temp = result['template']
        
        print(f"{sample_name:<20} {'Unconstrained':<25} {unc['r2']:<10.4f}")
        print(f"{'':20} {'Template (Yours)':<25} {temp['r2']:<10.4f}")
        
        # Highlight if unconstrained is worse
        r2_diff = temp['r2'] - unc['r2']
        if r2_diff > 0.05:
            print(f"{'':20} {'→ Template wins by':<25} {r2_diff:<10.4f}")
        elif r2_diff < -0.05:
            print(f"{'':20} {'→ Unconstrained wins by':<25} {-r2_diff:<10.4f}")
        print()
    
    print(f"\n[✓] All comparison plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
