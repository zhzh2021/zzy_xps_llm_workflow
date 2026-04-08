"""
XAS Experiment Planner using PCA

Uses whole-spectrum PCA to guide experimental design by:
1. Interpreting PCA axes physically (loadings → spectral features)
2. Mapping experimental conditions onto PCA space
3. Identifying unexplored regions (convex hull)
4. Suggesting next experiments (maximize information gain)

Author: ZZY Lab
Date: March 5, 2026
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import warnings

# Try to import scipy for convex hull
try:
    from scipy.spatial import ConvexHull, distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Install with: pip install scipy")

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class PCInterpretation:
    """Physical interpretation of a principal component."""
    
    def __init__(
        self,
        pc_number: int,
        variance_explained: float,
        peak_energies: List[float],
        peak_regions: List[str],
        interpretation: str
    ):
        self.pc_number = pc_number
        self.variance_explained = variance_explained
        self.peak_energies = peak_energies
        self.peak_regions = peak_regions
        self.interpretation = interpretation
    
    def __repr__(self):
        return (
            f"PC{self.pc_number} ({self.variance_explained:.1%}): "
            f"{self.interpretation}"
        )


class ExperimentSuggestion:
    """Suggested experiment based on PCA analysis."""
    
    def __init__(
        self,
        strategy: str,
        predicted_scores: np.ndarray,
        distance_to_nearest: float,
        reason: str,
        suggested_conditions: Optional[Dict[str, Any]] = None,
        priority: float = 1.0
    ):
        self.strategy = strategy
        self.predicted_scores = predicted_scores
        self.distance_to_nearest = distance_to_nearest
        self.reason = reason
        self.suggested_conditions = suggested_conditions or {}
        self.priority = priority
    
    def __repr__(self):
        return (
            f"Experiment Suggestion ({self.strategy}): "
            f"distance={self.distance_to_nearest:.2f}, "
            f"priority={self.priority:.2f}"
        )


class XASExperimentPlanner:
    """
    Use PCA to guide XAS experiment planning.
    
    This class interprets PCA results in chemical terms and suggests
    experiments to maximize information gain.
    
    Workflow:
        1. Analyze existing data with whole-spectrum PCA
        2. Interpret PC axes (loadings → chemical features)
        3. Map experimental conditions to PCA space
        4. Identify unexplored regions
        5. Suggest next experiments
    
    Usage:
        from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA
        
        # Run PCA first
        analyzer = XASSpectrumPCA()
        pca_result = analyzer.analyze_datasets(datasets)
        
        # Plan experiments
        planner = XASExperimentPlanner()
        interpretations = planner.interpret_components(pca_result)
        suggestions = planner.suggest_experiments(pca_result, experimental_params)
    """
    
    def __init__(
        self,
        edge_energy: float = 7112.0,  # Fe K-edge
        xanes_range: Tuple[float, float] = (7100, 7160),
        exafs_range: Tuple[float, float] = (7160, 7500)
    ):
        """
        Initialize experiment planner.
        
        Args:
            edge_energy: Approximate edge energy (eV)
            xanes_range: XANES region energy range
            exafs_range: EXAFS region energy range
        """
        self.edge_energy = edge_energy
        self.xanes_range = xanes_range
        self.exafs_range = exafs_range
        
        logger.info("XASExperimentPlanner initialized")
    
    def interpret_components(
        self,
        pca_result,
        n_components: Optional[int] = None,
        peak_threshold: float = 0.1
    ) -> List[PCInterpretation]:
        """
        Interpret principal components in physical/chemical terms.
        
        Analyzes PC loadings to identify:
        - Peak energies (where loadings are largest)
        - Spectral regions (pre-edge, edge, XANES, EXAFS)
        - Physical interpretation (oxidation state, coordination, etc.)
        
        Args:
            pca_result: SpectrumPCAResult from XASSpectrumPCA
            n_components: Number of components to interpret (None = all)
            peak_threshold: Threshold for peak detection (fraction of max)
        
        Returns:
            List of PCInterpretation objects
        """
        if n_components is None:
            n_components = pca_result.n_components
        else:
            n_components = min(n_components, pca_result.n_components)
        
        interpretations = []
        
        for i in range(n_components):
            loading = pca_result.loadings[i, :]
            energy_grid = pca_result.energy_grid
            variance = pca_result.variance_ratio[i]
            
            # Find peaks in loading
            abs_loading = np.abs(loading)
            threshold = abs_loading.max() * peak_threshold
            peak_indices = np.where(abs_loading > threshold)[0]
            
            if len(peak_indices) == 0:
                peak_indices = [np.argmax(abs_loading)]
            
            peak_energies = energy_grid[peak_indices].tolist()
            
            # Classify regions
            peak_regions = []
            for e in peak_energies:
                if e < self.edge_energy - 10:
                    peak_regions.append('pre-edge')
                elif e < self.edge_energy + 5:
                    peak_regions.append('edge')
                elif e < self.xanes_range[1]:
                    peak_regions.append('XANES')
                else:
                    peak_regions.append('EXAFS')
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                peak_energies, peak_regions, variance
            )
            
            interp = PCInterpretation(
                pc_number=i + 1,
                variance_explained=variance,
                peak_energies=peak_energies,
                peak_regions=peak_regions,
                interpretation=interpretation
            )
            
            interpretations.append(interp)
            
            logger.info(f"PC{i+1}: {interpretation}")
        
        return interpretations
    
    def _generate_interpretation(
        self,
        peak_energies: List[float],
        peak_regions: List[str],
        variance: float
    ) -> str:
        """Generate physical interpretation from peak analysis."""
        
        # Count region occurrences
        region_counts = {}
        for region in peak_regions:
            region_counts[region] = region_counts.get(region, 0) + 1
        
        # Determine dominant region
        dominant_region = max(region_counts, key=region_counts.get)
        
        # Interpretation rules
        interpretations = {
            'pre-edge': 'Pre-edge features / ligand field effects',
            'edge': 'Edge position / oxidation state changes',
            'XANES': 'White line / coordination geometry',
            'EXAFS': 'Bond distances / coordination number'
        }
        
        base_interp = interpretations.get(dominant_region, 'Spectral variation')
        
        # Add variance context
        if variance > 0.5:
            return f"Major {base_interp.lower()}"
        elif variance > 0.2:
            return f"Moderate {base_interp.lower()}"
        else:
            return f"Minor {base_interp.lower()}"
    
    def map_conditions_to_pca(
        self,
        pca_result,
        experimental_params: Dict[str, List[Any]],
        sample_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Map experimental conditions to PCA space.
        
        Args:
            pca_result: SpectrumPCAResult
            experimental_params: Dict of parameter_name -> values
                Example: {'pH': [2.0, 5.1, 5.2], 'temperature': [25, 25, 25]}
            sample_names: Optional sample names to match with params
        
        Returns:
            Dictionary with PCA scores and experimental parameters
        """
        if sample_names is None:
            sample_names = pca_result.sample_names
        
        # Validate
        n_samples = len(sample_names)
        for param, values in experimental_params.items():
            if len(values) != n_samples:
                raise ValueError(
                    f"Parameter '{param}' has {len(values)} values, "
                    f"but {n_samples} samples"
                )
        
        # Combine scores and parameters
        mapping = {
            'scores': pca_result.scores,
            'sample_names': sample_names,
            **experimental_params
        }
        
        return mapping
    
    def identify_explored_region(
        self,
        pca_result,
        pc_x: int = 1,
        pc_y: int = 2
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Identify explored region in PCA space using convex hull.
        
        Args:
            pca_result: SpectrumPCAResult
            pc_x: X-axis PC (1-indexed)
            pc_y: Y-axis PC (1-indexed)
        
        Returns:
            (hull_points, hull_object)
            hull_points: Vertices of convex hull
            hull_object: ConvexHull object (if scipy available)
        """
        if not HAS_SCIPY:
            warnings.warn("scipy required for convex hull. Returning bounding box.")
            # Simple bounding box instead
            idx_x, idx_y = pc_x - 1, pc_y - 1
            scores = pca_result.scores[:, [idx_x, idx_y]]
            min_x, max_x = scores[:, 0].min(), scores[:, 0].max()
            min_y, max_y = scores[:, 1].min(), scores[:, 1].max()
            hull_points = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])
            return hull_points, None
        
        # Extract 2D scores
        idx_x, idx_y = pc_x - 1, pc_y - 1
        scores_2d = pca_result.scores[:, [idx_x, idx_y]]
        
        if len(scores_2d) < 3:
            warnings.warn("Need at least 3 points for convex hull")
            return scores_2d, None
        
        # Compute convex hull
        try:
            hull = ConvexHull(scores_2d)
            hull_points = scores_2d[hull.vertices]
            return hull_points, hull
        except Exception as e:
            warnings.warn(f"Convex hull failed: {e}")
            return scores_2d, None
    
    def suggest_experiments(
        self,
        pca_result,
        experimental_params: Optional[Dict[str, List[Any]]] = None,
        strategy: str = 'maxdist',
        n_suggestions: int = 3,
        pc_x: int = 1,
        pc_y: int = 2
    ) -> List[ExperimentSuggestion]:
        """
        Suggest next experiments based on PCA results.
        
        Strategies:
            'maxdist': Maximize distance from existing points
            'boundary': Sample near cluster boundaries
            'trajectory': Interpolate between clusters
            'hull': Expand convex hull
        
        Args:
            pca_result: SpectrumPCAResult
            experimental_params: Optional dict of experimental conditions
            strategy: Suggestion strategy
            n_suggestions: Number of suggestions to generate
            pc_x: X-axis PC (1-indexed)
            pc_y: Y-axis PC (1-indexed)
        
        Returns:
            List of ExperimentSuggestion objects
        """
        idx_x, idx_y = pc_x - 1, pc_y - 1
        scores_2d = pca_result.scores[:, [idx_x, idx_y]]
        
        suggestions = []
        
        if strategy == 'maxdist':
            suggestions = self._suggest_maxdist(
                scores_2d, n_suggestions, pca_result
            )
        
        elif strategy == 'boundary':
            suggestions = self._suggest_boundary(
                scores_2d, n_suggestions, pca_result
            )
        
        elif strategy == 'trajectory':
            suggestions = self._suggest_trajectory(
                scores_2d, experimental_params, n_suggestions, pca_result
            )
        
        elif strategy == 'hull':
            suggestions = self._suggest_hull_expansion(
                scores_2d, n_suggestions, pca_result, pc_x, pc_y
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Sort by priority
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return suggestions[:n_suggestions]
    
    def _suggest_maxdist(
        self,
        scores_2d: np.ndarray,
        n_suggestions: int,
        pca_result
    ) -> List[ExperimentSuggestion]:
        """Suggest experiments maximizing distance from existing points."""
        
        suggestions = []
        
        # Define search grid
        min_pc1, max_pc1 = scores_2d[:, 0].min(), scores_2d[:, 0].max()
        min_pc2, max_pc2 = scores_2d[:, 1].min(), scores_2d[:, 1].max()
        
        # Expand by 20%
        range_pc1 = max_pc1 - min_pc1
        range_pc2 = max_pc2 - min_pc2
        min_pc1 -= 0.2 * range_pc1
        max_pc1 += 0.2 * range_pc1
        min_pc2 -= 0.2 * range_pc2
        max_pc2 += 0.2 * range_pc2
        
        # Grid search
        grid_size = 20
        pc1_grid = np.linspace(min_pc1, max_pc1, grid_size)
        pc2_grid = np.linspace(min_pc2, max_pc2, grid_size)
        
        best_points = []
        
        for pc1 in pc1_grid:
            for pc2 in pc2_grid:
                point = np.array([pc1, pc2])
                
                # Distance to nearest existing point
                if HAS_SCIPY:
                    dist = distance.cdist([point], scores_2d).min()
                else:
                    dist = np.sqrt(((scores_2d - point) ** 2).sum(axis=1)).min()
                
                best_points.append((point, dist))
        
        # Sort by distance
        best_points.sort(key=lambda x: x[1], reverse=True)
        
        # Create suggestions
        for i in range(min(n_suggestions, len(best_points))):
            point, dist = best_points[i]
            
            # Full PC scores (zeros for other PCs)
            full_scores = np.zeros(pca_result.n_components)
            full_scores[0] = point[0]
            full_scores[1] = point[1]
            
            suggestion = ExperimentSuggestion(
                strategy='maxdist',
                predicted_scores=full_scores,
                distance_to_nearest=dist,
                reason=f"Maximizes distance from existing data (d={dist:.2f})",
                priority=dist / (max_pc1 - min_pc1)  # Normalize priority
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_boundary(
        self,
        scores_2d: np.ndarray,
        n_suggestions: int,
        pca_result
    ) -> List[ExperimentSuggestion]:
        """Suggest experiments at cluster boundaries."""
        
        suggestions = []
        
        # Simple approach: midpoints between distant pairs
        n_samples = len(scores_2d)
        
        midpoints = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                p1, p2 = scores_2d[i], scores_2d[j]
                
                # Distance between points
                if HAS_SCIPY:
                    dist = distance.euclidean(p1, p2)
                else:
                    dist = np.sqrt(((p1 - p2) ** 2).sum())
                
                # Only consider distant pairs (potential different clusters)
                if dist > np.percentile(distance.pdist(scores_2d) if HAS_SCIPY else [0], 75):
                    midpoint = (p1 + p2) / 2
                    midpoints.append((midpoint, dist))
        
        # Sort by distance (farther pairs = more interesting boundaries)
        midpoints.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(min(n_suggestions, len(midpoints))):
            point, dist = midpoints[i]
            
            # Distance to nearest existing point
            if HAS_SCIPY:
                nearest_dist = distance.cdist([point], scores_2d).min()
            else:
                nearest_dist = np.sqrt(((scores_2d - point) ** 2).sum(axis=1)).min()
            
            full_scores = np.zeros(pca_result.n_components)
            full_scores[0] = point[0]
            full_scores[1] = point[1]
            
            suggestion = ExperimentSuggestion(
                strategy='boundary',
                predicted_scores=full_scores,
                distance_to_nearest=nearest_dist,
                reason=f"Explores boundary between clusters (cluster dist={dist:.2f})",
                priority=dist * 0.8  # Slightly lower priority than maxdist
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_trajectory(
        self,
        scores_2d: np.ndarray,
        experimental_params: Optional[Dict[str, List[Any]]],
        n_suggestions: int,
        pca_result
    ) -> List[ExperimentSuggestion]:
        """Suggest experiments along reaction trajectories."""
        
        suggestions = []
        
        if experimental_params is None:
            # No time/sequence info, can't suggest trajectory
            return suggestions
        
        # Look for time/sequence parameter
        time_param = None
        for param in ['time', 'step', 'sequence', 'iteration']:
            if param in experimental_params:
                time_param = param
                break
        
        if time_param is None:
            return suggestions
        
        # Sort by time
        time_values = np.array(experimental_params[time_param])
        sort_idx = np.argsort(time_values)
        trajectory = scores_2d[sort_idx]
        
        # Find largest gaps in trajectory
        gaps = []
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            if HAS_SCIPY:
                gap_dist = distance.euclidean(p1, p2)
            else:
                gap_dist = np.sqrt(((p1 - p2) ** 2).sum())
            
            # Midpoint
            midpoint = (p1 + p2) / 2
            gaps.append((midpoint, gap_dist, i))
        
        # Sort by gap size
        gaps.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(min(n_suggestions, len(gaps))):
            point, gap_dist, idx = gaps[i]
            
            # Distance to nearest
            if HAS_SCIPY:
                nearest_dist = distance.cdist([point], scores_2d).min()
            else:
                nearest_dist = np.sqrt(((scores_2d - point) ** 2).sum(axis=1)).min()
            
            full_scores = np.zeros(pca_result.n_components)
            full_scores[0] = point[0]
            full_scores[1] = point[1]
            
            # Suggest intermediate time
            t1 = time_values[sort_idx[idx]]
            t2 = time_values[sort_idx[idx + 1]]
            t_mid = (t1 + t2) / 2
            
            suggestion = ExperimentSuggestion(
                strategy='trajectory',
                predicted_scores=full_scores,
                distance_to_nearest=nearest_dist,
                reason=f"Fills gap in trajectory (between t={t1} and t={t2})",
                suggested_conditions={time_param: t_mid},
                priority=gap_dist * 0.9
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_hull_expansion(
        self,
        scores_2d: np.ndarray,
        n_suggestions: int,
        pca_result,
        pc_x: int,
        pc_y: int
    ) -> List[ExperimentSuggestion]:
        """Suggest experiments expanding convex hull."""
        
        hull_points, hull = self.identify_explored_region(pca_result, pc_x, pc_y)
        
        if hull is None:
            # Use maxdist instead
            return self._suggest_maxdist(scores_2d, n_suggestions, pca_result)
        
        suggestions = []
        
        # For each hull edge, suggest point extending outward
        n_vertices = len(hull_points)
        
        for i in range(n_vertices):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n_vertices]
            
            # Midpoint of edge
            midpoint = (p1 + p2) / 2
            
            # Centroid of hull
            centroid = hull_points.mean(axis=0)
            
            # Direction outward from centroid
            direction = midpoint - centroid
            direction = direction / np.linalg.norm(direction)
            
            # Extend by 20% of hull size
            hull_size = np.sqrt(((hull_points - centroid) ** 2).sum(axis=1)).max()
            extension = midpoint + 0.2 * hull_size * direction
            
            # Distance to nearest
            if HAS_SCIPY:
                nearest_dist = distance.cdist([extension], scores_2d).min()
            else:
                nearest_dist = np.sqrt(((scores_2d - extension) ** 2).sum(axis=1)).min()
            
            full_scores = np.zeros(pca_result.n_components)
            full_scores[0] = extension[0]
            full_scores[1] = extension[1]
            
            suggestion = ExperimentSuggestion(
                strategy='hull',
                predicted_scores=full_scores,
                distance_to_nearest=nearest_dist,
                reason=f"Expands explored region (hull expansion)",
                priority=nearest_dist / hull_size
            )
            suggestions.append(suggestion)
        
        # Sort and return top suggestions
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        return suggestions[:n_suggestions]
    
<<<<<<< Updated upstream
    def plot_conditions_overlay(
        self,
        pca_result,
        experimental_params: Dict[str, List[Any]],
        pc_x: int = 1,
        pc_y: int = 2,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[Path] = None
    ):
        """
        Create multi-panel overlay plot showing how experimental conditions map to PCA space.
        
        Each panel shows the same PCA space colored by a different experimental variable
        (e.g., pH, ligand type, concentration, temperature, etc.).
        
        Args:
            pca_result: SpectrumPCAResult from XASSpectrumPCA
            experimental_params: Dict of parameter_name -> values
                Example: {
                    'pH': [2.0, 5.1, 5.2, 5.3],
                    'ligand': ['malic', 'malic', 'tartaric', 'tartaric'],
                    'concentration': [0.01, 0.01, 0.05, 0.05]
                }
            pc_x: X-axis PC (1-indexed)
            pc_y: Y-axis PC (1-indexed)
            figsize: Figure size (width, height)
            save_path: Optional save path
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        idx_x, idx_y = pc_x - 1, pc_y - 1
        scores = pca_result.scores
        
        # Determine layout
        n_params = len(experimental_params)
        if n_params == 0:
            raise ValueError("No experimental parameters provided")
        
        # Calculate grid layout (try to make roughly square)
        n_cols = int(np.ceil(np.sqrt(n_params)))
        n_rows = int(np.ceil(n_params / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each experimental parameter
        param_names = list(experimental_params.keys())
        
        for i, param_name in enumerate(param_names):
            ax = axes[i]
            values = experimental_params[param_name]
            
            # Check if numeric or categorical (handle None values safely)
            try:
                numeric_values = np.array(
                    [v if v is not None else np.nan for v in values],
                    dtype=float
                )
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False
                numeric_values = None
            
            # Plot
            if is_numeric:
                finite_vals = numeric_values[np.isfinite(numeric_values)]
                is_numeric = finite_vals.size > 0

            if is_numeric and len(set(finite_vals.tolist())) > 5:
                # Continuous colormap
                scatter = ax.scatter(
                    scores[:, idx_x], scores[:, idx_y],
                    c=numeric_values, cmap='viridis', s=200,
                    alpha=0.8, edgecolors='black', linewidth=2
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(param_name, fontsize=18)
            else:
                # Categorical colors
                safe_values = ['Unknown' if v is None else v for v in values]
                unique_vals = sorted(set(safe_values), key=lambda x: str(x))
                n_unique = len(unique_vals)
                colors = plt.cm.tab10(np.linspace(0, 1, n_unique))
                
                for j, val in enumerate(unique_vals):
                    mask = np.array([(v if v is not None else 'Unknown') == val for v in values])
                    ax.scatter(
                        scores[mask, idx_x], scores[mask, idx_y],
                        c=[colors[j]], s=200, alpha=0.8,
                        edgecolors='black', linewidth=2,
                        label=str(val)
                    )
                
                ax.legend(title=param_name, loc='best', fontsize=18)
            
            # Add sample labels
            for k, name in enumerate(pca_result.sample_names):
                ax.annotate(
                    name, (scores[k, idx_x], scores[k, idx_y]),
                    fontsize=16, alpha=0.6, xytext=(5, 5),
                    textcoords='offset points'
                )
            
            var_x = pca_result.variance_ratio[idx_x] * 100
            var_y = pca_result.variance_ratio[idx_y] * 100
            
            ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}%)', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}%)', fontsize=14, fontweight='bold')
            ax.set_title(f'Colored by: {param_name}', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=15)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(
            'Experimental Conditions Overlay on PCA Space',
            fontsize=20, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Conditions overlay plot saved to {save_path}")
        
        plt.show()
    
=======
>>>>>>> Stashed changes
    def plot_experiment_planning(
        self,
        pca_result,
        experimental_params: Optional[Dict[str, List[Any]]] = None,
        suggestions: Optional[List[ExperimentSuggestion]] = None,
        pc_x: int = 1,
        pc_y: int = 2,
        color_by: Optional[str] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot PCA space with experimental conditions and suggestions.
        
        Args:
            pca_result: SpectrumPCAResult
            experimental_params: Dict of experimental parameters
            suggestions: List of ExperimentSuggestion objects
            pc_x: X-axis PC (1-indexed)
            pc_y: Y-axis PC (1-indexed)
            color_by: Parameter name to color points by
            save_path: Optional save path
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        idx_x, idx_y = pc_x - 1, pc_y - 1
        scores = pca_result.scores
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot explored region (convex hull)
        hull_points, hull = self.identify_explored_region(pca_result, pc_x, pc_y)
        if hull_points is not None and len(hull_points) > 2:
            hull_polygon = Polygon(
                hull_points, fill=True, alpha=0.1,
                facecolor='lightblue', edgecolor='blue',
                linewidth=2, linestyle='--', label='Explored region'
            )
            ax.add_patch(hull_polygon)
        
        # Plot existing experiments
        if color_by and experimental_params and color_by in experimental_params:
            color_values = experimental_params[color_by]
            scatter = ax.scatter(
                scores[:, idx_x], scores[:, idx_y],
                c=color_values, cmap='viridis', s=150,
                alpha=0.8, edgecolors='black', linewidth=2,
                label='Existing experiments', zorder=5
            )
            cbar = plt.colorbar(scatter, ax=ax)
<<<<<<< Updated upstream
            cbar.set_label(color_by, fontsize=18)
=======
            cbar.set_label(color_by, fontsize=12)
>>>>>>> Stashed changes
        else:
            ax.scatter(
                scores[:, idx_x], scores[:, idx_y],
                s=150, alpha=0.8, edgecolors='black',
                linewidth=2, color='steelblue',
                label='Existing experiments', zorder=5
            )
        
        # Add sample labels
        for i, name in enumerate(pca_result.sample_names):
            ax.annotate(
                name, (scores[i, idx_x], scores[i, idx_y]),
<<<<<<< Updated upstream
                fontsize=16, alpha=0.7, xytext=(5, 5),
=======
                fontsize=8, alpha=0.7, xytext=(5, 5),
>>>>>>> Stashed changes
                textcoords='offset points'
            )
        
        # Plot suggested experiments
        if suggestions:
            for i, sug in enumerate(suggestions):
                sx = sug.predicted_scores[idx_x]
                sy = sug.predicted_scores[idx_y]
                
<<<<<<< Updated upstream
                # Priority determines size and color (clamp for safe plotting)
                size = 100 + 200 * max(0.0, float(sug.priority))
                size = min(size, 800)
                alpha = 0.5 + 0.3 * max(0.0, float(sug.priority))
                alpha = max(0.2, min(alpha, 1.0))
=======
                # Priority determines size and color
                size = 100 + 200 * sug.priority
                alpha = 0.5 + 0.3 * sug.priority
>>>>>>> Stashed changes
                
                ax.scatter(
                    sx, sy, s=size, alpha=alpha,
                    marker='*', edgecolors='red', linewidth=2,
                    color='yellow', zorder=10
                )
                
                # Label
                label = f"Sug {i+1}"
                ax.annotate(
                    label, (sx, sy),
                    fontsize=10, fontweight='bold', color='red',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
        
        var_x = pca_result.variance_ratio[idx_x] * 100
        var_y = pca_result.variance_ratio[idx_y] * 100
        
<<<<<<< Updated upstream
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}%)', fontsize=15, fontweight='bold')
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}%)', fontsize=15, fontweight='bold')
        ax.set_title('XAS Experiment Planning: PCA Space', fontsize=17, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=15)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.legend(loc='best', fontsize=18)
=======
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}%)', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}%)', fontsize=13, fontweight='bold')
        ax.set_title('XAS Experiment Planning: PCA Space', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.legend(loc='best', fontsize=10)
>>>>>>> Stashed changes
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Experiment planning plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Demo with synthetic data
    print("XAS Experiment Planner Demo")
    print("=" * 80)
    
    # Create synthetic PCA result
    from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA, SpectrumPCAResult
    
    # Synthetic scores (4 samples, 2 PCs)
    scores = np.array([
        [-11.07, -0.00],
        [3.69, -0.02],
        [3.70, -0.02],
        [3.68, 0.04]
    ])
    
    # Create minimal result object
    energy_grid = np.linspace(7000, 7200, 100)
    loadings = np.random.randn(2, 100) * 0.1
    
    result = SpectrumPCAResult(
        n_components=2,
        n_spectra=4,
        n_energy_points=100,
        energy_grid=energy_grid,
        spectra_matrix=np.zeros((4, 100)),
        explained_variance=np.array([0.95, 0.05]),
        cumulative_variance=np.array([0.95, 1.0]),
        variance_ratio=np.array([0.95, 0.05]),
        scores=scores,
        loadings=loadings,
        sample_names=['pH2.2', 'pH5.1', 'pH5.2', 'pH5.3'],
        normalization_method='standard',
        energy_range=(7000, 7200)
    )
    
    # Initialize planner
    planner = XASExperimentPlanner(edge_energy=7112.0)
    
    # Interpret components
    print("\n1. Interpreting PCA components...")
    interpretations = planner.interpret_components(result)
    for interp in interpretations:
        print(f"  {interp}")
    
    # Suggest experiments
    print("\n2. Suggesting next experiments...")
<<<<<<< Updated upstream
    experimental_params = {
        'pH': [2.2, 5.1, 5.2, 5.3],
        'ligand': ['malic', 'malic', 'tartaric', 'tartaric'],
        'iron_source': ['FeCl2', 'FeSO4', 'FeSO4', 'FeCl2']
    }
=======
    experimental_params = {'pH': [2.2, 5.1, 5.2, 5.3]}
>>>>>>> Stashed changes
    
    suggestions = planner.suggest_experiments(
        result,
        experimental_params=experimental_params,
        strategy='maxdist',
        n_suggestions=3
    )
    
    print(f"\n  Found {len(suggestions)} suggestions:")
    for i, sug in enumerate(suggestions):
        print(f"\n  Suggestion {i+1}:")
        print(f"    Strategy: {sug.strategy}")
        print(f"    Predicted PC1: {sug.predicted_scores[0]:.2f}")
        print(f"    Predicted PC2: {sug.predicted_scores[1]:.2f}")
        print(f"    Distance to nearest: {sug.distance_to_nearest:.2f}")
        print(f"    Reason: {sug.reason}")
        print(f"    Priority: {sug.priority:.2f}")
    
<<<<<<< Updated upstream
    # Plot conditions overlay
    print("\n3. Creating conditions overlay plot...")
    if HAS_MATPLOTLIB:
        planner.plot_conditions_overlay(
            result,
            experimental_params=experimental_params,
            pc_x=1, pc_y=2
        )
        print("  Overlay plot displayed!")
    
=======
>>>>>>> Stashed changes
    print("\n" + "=" * 80)
    print("Demo complete!")
