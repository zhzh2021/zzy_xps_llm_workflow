"""
XAS Whole-Spectrum PCA Module

Principal Component Analysis on full XAS spectra (not extracted features).
Captures oxidation state evolution, coordination changes, reaction pathways,
and hidden spectral variations without predefined features.

Workflow:
<<<<<<< Updated upstream
    1. use Normalized spectra (pre-edge normalization, edge step)
=======
    1. Normalize spectra (pre-edge normalization, edge step)
>>>>>>> Stashed changes
    2. Interpolate onto common energy grid
    3. Assemble spectrum matrix (rows=spectra, columns=energy points)
    4. Run PCA
       - Scores → clustering/trajectories
       - Loadings → spectral interpretation
       - Variance ratio → component importance

Author: ZZY Lab
Date: March 5, 2026
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import warnings

# Try to import xarray
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    warnings.warn("xarray not available. Install with: pip install xarray")

# Try to import sklearn
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class SpectrumPCAResult:
    """
    Results from whole-spectrum PCA analysis.
    
    Attributes:
        n_components: Number of principal components
        n_spectra: Number of spectra analyzed
        n_energy_points: Number of energy points in common grid
        
        energy_grid: Common energy grid (eV)
        spectra_matrix: Normalized spectra matrix (n_spectra × n_energy_points)
        
        explained_variance: Variance explained by each component
        cumulative_variance: Cumulative variance explained
        variance_ratio: Variance ratio for each component
        
        scores: PCA scores (n_spectra × n_components)
        loadings: PCA loadings/components (n_components × n_energy_points)
        
        sample_names: List of sample names
        normalization_method: Method used for normalization
        energy_range: (E_min, E_max) tuple
        
        confidence: Overall confidence score (0-1)
        flags: Warning/info flags
    """
    
    def __init__(
        self,
        n_components: int,
        n_spectra: int,
        n_energy_points: int,
        energy_grid: np.ndarray,
        spectra_matrix: np.ndarray,
        explained_variance: np.ndarray,
        cumulative_variance: np.ndarray,
        variance_ratio: np.ndarray,
        scores: np.ndarray,
        loadings: np.ndarray,
        sample_names: List[str],
        normalization_method: str,
        energy_range: Tuple[float, float],
        confidence: float = 1.0,
        flags: Optional[List[str]] = None
    ):
        self.n_components = n_components
        self.n_spectra = n_spectra
        self.n_energy_points = n_energy_points
        
        self.energy_grid = energy_grid
        self.spectra_matrix = spectra_matrix
        
        self.explained_variance = explained_variance
        self.cumulative_variance = cumulative_variance
        self.variance_ratio = variance_ratio
        
        self.scores = scores
        self.loadings = loadings
        
        self.sample_names = sample_names
        self.normalization_method = normalization_method
        self.energy_range = energy_range
        
        self.confidence = confidence
        self.flags = flags or []
    
    def __repr__(self):
        return (
            f"SpectrumPCAResult("
            f"n_spectra={self.n_spectra}, "
            f"n_components={self.n_components}, "
            f"variance_captured={self.cumulative_variance[-1]:.1%})"
        )
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 80,
            "WHOLE-SPECTRUM PCA ANALYSIS RESULTS",
            "=" * 80,
            f"Number of spectra: {self.n_spectra}",
            f"Energy grid: {self.n_energy_points} points ({self.energy_range[0]:.1f} - {self.energy_range[1]:.1f} eV)",
            f"Normalization: {self.normalization_method}",
            f"",
            f"Principal Components: {self.n_components}",
            f"Total variance captured: {self.cumulative_variance[-1]:.1%}",
            f"",
            "Variance explained by each component:",
        ]
        
        for i in range(self.n_components):
            lines.append(
                f"  PC{i+1}: {self.variance_ratio[i]:.1%} "
                f"(cumulative: {self.cumulative_variance[i]:.1%})"
            )
        
        if self.flags:
            lines.extend([
                "",
                "Flags:",
                *[f"  - {flag}" for flag in self.flags]
            ])
        
        lines.append("=" * 80)
        return "\n".join(lines)


class XASSpectrumPCA:
    """
    Whole-spectrum Principal Component Analysis for XAS data.
    
    This analyzer performs PCA directly on XAS spectra without feature extraction,
    enabling discovery of:
    - Oxidation state evolution
    - Coordination environment changes
    - Reaction pathways
    - Hidden spectral variations
    
    Usage:
        analyzer = XASSpectrumPCA()
        
        # From xarray datasets
        result = analyzer.analyze_datasets(
            datasets=[ds1, ds2, ds3, ...],
            sample_names=['sample1', 'sample2', 'sample3', ...]
        )
        
        # From arrays
        result = analyzer.analyze_spectra(
            energies=[e1, e2, e3, ...],
            spectra=[mu1, mu2, mu3, ...],
            sample_names=['sample1', 'sample2', 'sample3', ...]
        )
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        normalization: str = 'standard',
        energy_range: Optional[Tuple[float, float]] = None,
        n_grid_points: int = 500
    ):
        """
        Initialize whole-spectrum PCA analyzer.
        
        Args:
            n_components: Number of components to keep (None = auto-select)
            variance_threshold: Variance threshold for auto-selecting components
            normalization: Normalization method ('standard', 'minmax', 'none')
            energy_range: Energy range to use (E_min, E_max) or None for auto
            n_grid_points: Number of points in common energy grid
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.normalization = normalization
        self.energy_range = energy_range
        self.n_grid_points = n_grid_points
        
        self.pca_model = None
        self.scaler = None
        
        logger.info(f"XASSpectrumPCA initialized (normalization={normalization})")
    
    def analyze_datasets(
        self,
        datasets: List[xr.Dataset],
        sample_names: Optional[List[str]] = None,
        mu_variable: str = 'mu_trans'
    ) -> SpectrumPCAResult:
        """
        Analyze XAS spectra from xarray datasets.
        
        Args:
            datasets: List of xarray.Dataset objects from APS reader
            sample_names: Optional list of sample names
            mu_variable: Name of mu variable in datasets ('mu_trans', 'fluor_total', etc.)
        
        Returns:
            SpectrumPCAResult object
        """
        if not HAS_XARRAY:
            raise ImportError("xarray required for dataset analysis. Install with: pip install xarray")
        
        # Extract energies and spectra
        energies = []
        spectra = []
        
        for ds in datasets:
            if 'energy' not in ds.coords:
                raise ValueError("Dataset must have 'energy' coordinate")
            if mu_variable not in ds:
                raise ValueError(f"Dataset must have '{mu_variable}' variable")
            
            energies.append(ds['energy'].values)
            spectra.append(ds[mu_variable].values)
        
        # Use dataset filenames if sample_names not provided
        if sample_names is None:
            sample_names = [
                ds.attrs.get('filename', f'sample_{i}')
                for i, ds in enumerate(datasets)
            ]
        
        return self.analyze_spectra(energies, spectra, sample_names)
    
    def analyze_spectra(
        self,
        energies: List[np.ndarray],
        spectra: List[np.ndarray],
        sample_names: List[str]
    ) -> SpectrumPCAResult:
        """
        Analyze XAS spectra from energy/mu arrays.
        
        Args:
            energies: List of energy arrays (one per spectrum)
            spectra: List of mu arrays (one per spectrum)
            sample_names: List of sample names
        
        Returns:
            SpectrumPCAResult object
        """
        if len(energies) != len(spectra) or len(energies) != len(sample_names):
            raise ValueError("energies, spectra, and sample_names must have same length")
        
        if len(energies) < 2:
            raise ValueError("Need at least 2 spectra for PCA")
        
        logger.info(f"Analyzing {len(energies)} spectra")
        
        # Step 1: Determine common energy grid
        energy_grid = self._create_common_energy_grid(energies)
        logger.info(f"Common energy grid: {len(energy_grid)} points "
                   f"({energy_grid[0]:.1f} - {energy_grid[-1]:.1f} eV)")
        
        # Step 2: Interpolate all spectra onto common grid
        spectra_matrix = self._interpolate_spectra(energies, spectra, energy_grid)
        logger.info(f"Spectra matrix shape: {spectra_matrix.shape}")
        
        # Step 3: Normalize spectra
        spectra_normalized, norm_method = self._normalize_spectra(spectra_matrix)
        logger.info(f"Normalization: {norm_method}")
        
        # Step 4: Run PCA
        pca_result = self._run_pca(spectra_normalized)
        
        # Step 5: Package results
        result = SpectrumPCAResult(
            n_components=pca_result['n_components'],
            n_spectra=len(sample_names),
            n_energy_points=len(energy_grid),
            energy_grid=energy_grid,
            spectra_matrix=spectra_normalized,
            explained_variance=pca_result['explained_variance'],
            cumulative_variance=pca_result['cumulative_variance'],
            variance_ratio=pca_result['variance_ratio'],
            scores=pca_result['scores'],
            loadings=pca_result['loadings'],
            sample_names=sample_names,
            normalization_method=norm_method,
            energy_range=(energy_grid[0], energy_grid[-1]),
            confidence=pca_result['confidence'],
            flags=pca_result['flags']
        )
        
        logger.info(f"PCA complete: {result.n_components} components, "
                   f"{result.cumulative_variance[-1]:.1%} variance captured")
        
        return result
    
    def _create_common_energy_grid(
        self,
        energies: List[np.ndarray]
    ) -> np.ndarray:
        """
        Create common energy grid for interpolation.
        
        Args:
            energies: List of energy arrays
        
        Returns:
            Common energy grid
        """
        # Find overlapping energy range
        if self.energy_range is not None:
            e_min, e_max = self.energy_range
        else:
            e_min = max(e[0] for e in energies)
            e_max = min(e[-1] for e in energies)
        
        # Create uniform grid
        energy_grid = np.linspace(e_min, e_max, self.n_grid_points)
        
        return energy_grid
    
    def _interpolate_spectra(
        self,
        energies: List[np.ndarray],
        spectra: List[np.ndarray],
        energy_grid: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate all spectra onto common energy grid.
        
        Args:
            energies: List of energy arrays
            spectra: List of mu arrays
            energy_grid: Common energy grid
        
        Returns:
            Matrix of interpolated spectra (n_spectra × n_energy_points)
        """
        n_spectra = len(spectra)
        n_points = len(energy_grid)
        
        spectra_matrix = np.zeros((n_spectra, n_points))
        
        for i, (energy, mu) in enumerate(zip(energies, spectra)):
            # Linear interpolation
            spectra_matrix[i, :] = np.interp(energy_grid, energy, mu)
        
        return spectra_matrix
    
    def _normalize_spectra(
        self,
        spectra_matrix: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        Normalize spectra for PCA.
        
        Args:
            spectra_matrix: Raw spectra matrix (n_spectra × n_energy_points)
        
        Returns:
            (normalized_matrix, method_description)
        """
        if self.normalization == 'none':
            return spectra_matrix, 'none'
        
        elif self.normalization == 'standard':
            # Standardize: zero mean, unit variance (across spectra for each energy)
            self.scaler = StandardScaler()
            normalized = self.scaler.fit_transform(spectra_matrix.T).T
            return normalized, 'standard (zero mean, unit variance)'
        
        elif self.normalization == 'minmax':
            # Min-max normalize each spectrum to [0, 1]
            normalized = np.zeros_like(spectra_matrix)
            for i in range(spectra_matrix.shape[0]):
                mu_min = spectra_matrix[i, :].min()
                mu_max = spectra_matrix[i, :].max()
                if mu_max > mu_min:
                    normalized[i, :] = (spectra_matrix[i, :] - mu_min) / (mu_max - mu_min)
                else:
                    normalized[i, :] = 0.0
            return normalized, 'minmax [0, 1] per spectrum'
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
    
    def _run_pca(
        self,
        spectra_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run PCA on normalized spectra matrix.
        
        Args:
            spectra_matrix: Normalized spectra (n_spectra × n_energy_points)
        
        Returns:
            Dictionary with PCA results
        """
        n_spectra = spectra_matrix.shape[0]
        
        # Determine number of components
        if self.n_components is not None:
            n_components = min(self.n_components, n_spectra)
        else:
            # Use all components initially to assess variance
            n_components = min(n_spectra, spectra_matrix.shape[1])
        
        # Fit PCA
        self.pca_model = PCA(n_components=n_components)
        scores = self.pca_model.fit_transform(spectra_matrix)
        
        # Get results
        explained_variance = self.pca_model.explained_variance_
        variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        loadings = self.pca_model.components_
        
        # Auto-select components if needed
        flags = []
        if self.n_components is None:
            # Select based on variance threshold
            n_selected = np.searchsorted(cumulative_variance, self.variance_threshold) + 1
            n_selected = max(2, min(n_selected, n_components))  # At least 2 components
            
            # Re-fit with selected components
            self.pca_model = PCA(n_components=n_selected)
            scores = self.pca_model.fit_transform(spectra_matrix)
            
            explained_variance = self.pca_model.explained_variance_
            variance_ratio = self.pca_model.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratio)
            loadings = self.pca_model.components_
            
            flags.append(f"Auto-selected {n_selected} components (>{self.variance_threshold:.0%} variance)")
            n_components = n_selected
        
        # Calculate confidence
        confidence = min(1.0, cumulative_variance[0] / 0.5)  # PC1 should capture significant variance
        
        if n_spectra < 5:
            flags.append("Warning: Small sample size (< 5 spectra)")
            confidence *= 0.8
        
        if cumulative_variance[-1] < 0.8:
            flags.append(f"Warning: Low total variance captured ({cumulative_variance[-1]:.1%})")
            confidence *= 0.9
        
        return {
            'n_components': n_components,
            'explained_variance': explained_variance,
            'variance_ratio': variance_ratio,
            'cumulative_variance': cumulative_variance,
            'scores': scores,
            'loadings': loadings,
            'confidence': confidence,
            'flags': flags
        }
    
    def plot_scree(
        self,
        result: SpectrumPCAResult,
        save_path: Optional[Path] = None
    ):
        """
        Plot scree plot showing variance explained by each component.
        
        Args:
            result: SpectrumPCAResult object
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
<<<<<<< Updated upstream
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # Scree plot
        components = np.arange(1, result.n_components + 1)
        ax1.bar(components, result.variance_ratio * 100, alpha=0.8, color='steelblue', width=0.6)
        ax1.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Variance Explained (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Scree Plot', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.25)
        ax1.tick_params(labelsize=15)
        
        # Cumulative variance
        ax2.plot(components, result.cumulative_variance * 100,
                marker='o', linewidth=2.5, color='steelblue', markersize=7)
        ax2.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
        ax2.set_xlabel('Number of Components', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Cumulative Variance Explained', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=18)
        ax2.grid(True, alpha=0.25)
        ax2.tick_params(labelsize=15)
=======
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Scree plot
        components = np.arange(1, result.n_components + 1)
        ax1.bar(components, result.variance_ratio * 100, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component', fontsize=11)
        ax1.set_ylabel('Variance Explained (%)', fontsize=11)
        ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        ax2.plot(components, result.cumulative_variance * 100, 
                marker='o', linewidth=2, color='steelblue', markersize=6)
        ax2.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
        ax2.set_xlabel('Number of Components', fontsize=11)
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=11)
        ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
>>>>>>> Stashed changes
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scree plot saved to {save_path}")
        
        plt.show()
    
    def plot_scores(
        self,
        result: SpectrumPCAResult,
        pc_x: int = 1,
        pc_y: int = 2,
        color_by: Optional[List[Any]] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot PCA scores (sample trajectories in PC space).
        
        Args:
            result: SpectrumPCAResult object
            pc_x: Principal component for x-axis (1-indexed)
            pc_y: Principal component for y-axis (1-indexed)
            color_by: Optional list of values for color-coding points
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        if pc_x > result.n_components or pc_y > result.n_components:
            raise ValueError(f"PC indices exceed n_components={result.n_components}")
        
<<<<<<< Updated upstream
        fig, ax = plt.subplots(figsize=(11, 9))
=======
        fig, ax = plt.subplots(figsize=(10, 8))
>>>>>>> Stashed changes
        
        # Convert to 0-indexed
        idx_x = pc_x - 1
        idx_y = pc_y - 1
        
        # Plot scores
        if color_by is not None:
            scatter = ax.scatter(
                result.scores[:, idx_x],
                result.scores[:, idx_y],
                c=color_by,
                cmap='viridis',
<<<<<<< Updated upstream
                s=120,
                alpha=0.8,
                edgecolors='black',
                linewidth=1
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Color Value', fontsize=18)
=======
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=1
            )
            plt.colorbar(scatter, ax=ax, label='Color Value')
>>>>>>> Stashed changes
        else:
            ax.scatter(
                result.scores[:, idx_x],
                result.scores[:, idx_y],
<<<<<<< Updated upstream
                s=120,
                alpha=0.8,
=======
                s=100,
                alpha=0.7,
>>>>>>> Stashed changes
                edgecolors='black',
                linewidth=1,
                color='steelblue'
            )
        
        # Add sample labels
        for i, name in enumerate(result.sample_names):
            ax.annotate(
                name,
                (result.scores[i, idx_x], result.scores[i, idx_y]),
<<<<<<< Updated upstream
                fontsize=16,
=======
                fontsize=8,
>>>>>>> Stashed changes
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        var_x = result.variance_ratio[idx_x] * 100
        var_y = result.variance_ratio[idx_y] * 100
        
<<<<<<< Updated upstream
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}%)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}%)', fontsize=14, fontweight='bold')
        ax.set_title('PCA Scores (Sample Trajectories)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=15)
=======
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}%)', fontsize=12)
        ax.set_title('PCA Scores (Sample Trajectories)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
>>>>>>> Stashed changes
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scores plot saved to {save_path}")
        
        plt.show()
    
    def plot_loadings(
        self,
        result: SpectrumPCAResult,
        components: Optional[List[int]] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot PCA loadings (spectral interpretation).
        
        Args:
            result: SpectrumPCAResult object
            components: List of components to plot (1-indexed), or None for all
            save_path: Optional path to save figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        if components is None:
            components = list(range(1, min(5, result.n_components + 1)))  # First 4 components
        
        n_plots = len(components)
<<<<<<< Updated upstream
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots), sharex=True)
=======
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
>>>>>>> Stashed changes
        
        if n_plots == 1:
            axes = [axes]
        
        for i, pc in enumerate(components):
            if pc > result.n_components:
                continue
            
            idx = pc - 1
            loading = result.loadings[idx, :]
            variance = result.variance_ratio[idx] * 100
            
<<<<<<< Updated upstream
            axes[i].plot(result.energy_grid, loading, linewidth=2.2, color='darkblue')
            axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            axes[i].set_ylabel(f'PC{pc} Loading', fontsize=16, fontweight='bold')
            axes[i].set_title(f'PC{pc} ({variance:.1f}% variance)', fontsize=18, fontweight='bold')
            axes[i].grid(True, alpha=0.25)
            axes[i].tick_params(labelsize=15)
        
        axes[-1].set_xlabel('Energy (eV)', fontsize=16, fontweight='bold')
=======
            axes[i].plot(result.energy_grid, loading, linewidth=2, color='darkblue')
            axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            axes[i].set_ylabel(f'PC{pc} Loading', fontsize=11)
            axes[i].set_title(f'PC{pc} ({variance:.1f}% variance)', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Energy (eV)', fontsize=12)
>>>>>>> Stashed changes
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Loadings plot saved to {save_path}")
        
        plt.show()
    
    def export_results(
        self,
        result: SpectrumPCAResult,
        output_dir: Path
    ):
        """
        Export PCA results to files.
        
        Args:
            result: SpectrumPCAResult object
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scores
        scores_file = output_dir / 'pca_scores.csv'
        with open(scores_file, 'w') as f:
            # Header
            header = ['sample'] + [f'PC{i+1}' for i in range(result.n_components)]
            f.write(','.join(header) + '\n')
            
            # Data
            for i, name in enumerate(result.sample_names):
                row = [name] + [f'{result.scores[i, j]:.6f}' for j in range(result.n_components)]
                f.write(','.join(row) + '\n')
        
        logger.info(f"Scores saved to {scores_file}")
        
        # Save loadings
        loadings_file = output_dir / 'pca_loadings.csv'
        with open(loadings_file, 'w') as f:
            # Header
            header = ['energy_eV'] + [f'PC{i+1}' for i in range(result.n_components)]
            f.write(','.join(header) + '\n')
            
            # Data
            for i, energy in enumerate(result.energy_grid):
                row = [f'{energy:.2f}'] + [f'{result.loadings[j, i]:.6f}' for j in range(result.n_components)]
                f.write(','.join(row) + '\n')
        
        logger.info(f"Loadings saved to {loadings_file}")
        
        # Save variance explained
        variance_file = output_dir / 'pca_variance.csv'
        with open(variance_file, 'w') as f:
            f.write('component,variance_ratio,cumulative_variance\n')
            for i in range(result.n_components):
                f.write(f'PC{i+1},{result.variance_ratio[i]:.6f},{result.cumulative_variance[i]:.6f}\n')
        
        logger.info(f"Variance info saved to {variance_file}")
        
        # Save summary
        summary_file = output_dir / 'pca_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(result.summary())
        
        logger.info(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    # Example usage with synthetic data
    print("XAS Whole-Spectrum PCA Module")
    print("=" * 80)
    
    # Create synthetic spectra
    n_spectra = 10
    n_points = 300
    energy_base = np.linspace(7000, 7200, n_points)
    
    spectra_list = []
    energy_list = []
    sample_names = []
    
    for i in range(n_spectra):
        # Add some noise to energy grid
        energy = energy_base + np.random.randn(n_points) * 0.1
        
        # Create synthetic spectrum (edge + oscillations)
        e0 = 7112 + i * 0.5  # Shifting edge
        edge = 1 / (1 + np.exp(-(energy - e0) / 2))
        oscillations = 0.1 * np.sin(2 * np.pi * (energy - e0) / 20)
        noise = 0.02 * np.random.randn(n_points)
        
        mu = edge + oscillations + noise
        
        energy_list.append(energy)
        spectra_list.append(mu)
        sample_names.append(f'Sample_{i+1}')
    
    # Run PCA
    print("\nRunning whole-spectrum PCA on synthetic data...")
    analyzer = XASSpectrumPCA(variance_threshold=0.95, n_grid_points=250)
    
    result = analyzer.analyze_spectra(
        energies=energy_list,
        spectra=spectra_list,
        sample_names=sample_names
    )
    
    print("\n" + result.summary())
    
    print("\nScores (first 3 samples, first 3 PCs):")
    print(result.scores[:3, :3])
    
    print("\nModule demo complete!")
