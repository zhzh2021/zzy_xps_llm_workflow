"""
XAS PCA Analyzer Module

Principal Component Analysis for XAS feature datasets.
Dimensionality reduction, variance analysis, feature importance.

Author: XAS ML Integration Team
Date: 2026-03-03
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Try to import sklearn
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. PCA analysis will not work.")

# Local imports
try:
    from xas_analyzer.xas_models import XASDataset, PCAAnalysisResult
    from xas_ml_modules.config_utils import ConfigLoader
except ImportError:
    from ..xas_analyzer.xas_models import XASDataset, PCAAnalysisResult
    from .config_utils import ConfigLoader


logger = logging.getLogger(__name__)


class XASPCAAnalyzer:
    """
    Principal Component Analysis for XAS datasets.
    
    This class performs PCA with:
    - Automatic component selection (variance threshold or Kaiser criterion)
    - Feature importance analysis (loadings)
    - Validation metrics (stability, explained variance)
    - Scree plot data generation
    
    Usage:
        analyzer = XASPCAAnalyzer()
        pca_result = analyzer.analyze(dataset)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize PCA analyzer.
        
        Args:
            config_path: Path to YAML config (optional, auto-detected if None)
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for PCA analysis. Install with: pip install scikit-learn")
        
        self.config = ConfigLoader(config_path)
        self.pca_config = self.config.get_section('pca')
        self.analysis_limits = self.config.get_section('analysis_limits')
        
        self.scaler = None
        self.pca_model = None
        
        logger.info("XASPCAAnalyzer initialized")
    
    def analyze(
        self,
        dataset: XASDataset,
        n_components: Optional[int] = None
    ) -> PCAAnalysisResult:
        """
        Perform PCA analysis on dataset.
        
        Main entry point for PCA analysis.
        
        Args:
            dataset: XASDataset with feature matrix
            n_components: Number of components (optional, auto-determined if None)
            
        Returns:
            PCAAnalysisResult with loadings, scores, variance analysis
            
        Raises:
            ValueError: If dataset is invalid or has insufficient samples
        """
        if dataset.feature_matrix is None:
            raise ValueError("Dataset has no feature matrix")
        
        if dataset.n_samples < 2:
            raise ValueError("Need at least 2 samples for PCA")
        
        logger.info(f"Running PCA on dataset: {dataset.n_samples} samples × {dataset.n_features} features")
        
        # Step 1: Standardize features
        X_scaled = self._standardize_features(dataset.feature_matrix)
        
        # Step 2: Determine number of components
        if n_components is None:
            n_components = self._determine_n_components(X_scaled)
        
        logger.info(f"Using {n_components} principal components")
        
        # Step 3: Fit PCA
        pca_model, scores = self._fit_pca(X_scaled, n_components)
        self.pca_model = pca_model
        
        # Step 4: Analyze variance
        variance_metrics = self._analyze_variance(pca_model)
        
        # Step 5: Extract feature loadings
        loadings = self._get_loadings(pca_model)
        
        # Step 6: Identify important features
        feature_importance = self._compute_feature_importance(
            loadings, 
            dataset.feature_names
        )
        
        # Step 7: Validation metrics
        kaiser_criterion = self._kaiser_criterion(pca_model)
        stability_score = self._estimate_stability(X_scaled, n_components)
        
        # Step 8: Quality assessment
        confidence, flags = self._assess_quality(
            pca_model, 
            dataset.n_samples, 
            n_components,
            variance_metrics
        )
        
        # Step 9: Create result object
        result = PCAAnalysisResult(
            n_components=n_components,
            explained_variance=variance_metrics['explained_variance'],
            cumulative_variance=variance_metrics['cumulative_variance'],
            variance_captured=variance_metrics['total_variance_captured'],
            loadings=loadings,
            scores=scores,
            feature_importance=feature_importance,
            kaiser_criterion=kaiser_criterion,
            stability_score=stability_score,
            confidence=confidence,
            flags=flags
        )
        
        logger.info(
            f"PCA complete: {n_components} components, "
            f"{variance_metrics['total_variance_captured']:.1%} variance captured"
        )
        
        return result
    
    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.
        
        Args:
            X: Feature matrix (n_samples × n_features)
            
        Returns:
            Standardized feature matrix
        """
        standardize = self.pca_config.get('standardize_features', True)
        
        if not standardize:
            logger.info("Feature standardization disabled")
            return X
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info("Features standardized")
        return X_scaled
    
    def _determine_n_components(self, X: np.ndarray) -> int:
        """
        Automatically determine number of components.
        
        Uses variance threshold or Kaiser criterion from config.
        
        Args:
            X: Standardized feature matrix
            
        Returns:
            Number of components to use
        """
        method = self.pca_config.get('n_components_method', 'variance_threshold')
        
        # Fit PCA with all components to analyze variance
        max_components = min(X.shape[0], X.shape[1])
        pca_full = PCA(n_components=max_components)
        pca_full.fit(X)
        
        if method == 'variance_threshold':
            # Select components that capture threshold % of variance
            threshold = self.pca_config.get('variance_threshold', 0.95)
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = int(np.argmax(cumsum >= threshold) + 1)
            logger.info(f"Variance threshold method: {n_components} components for {threshold:.1%} variance")
            
        elif method == 'kaiser':
            # Kaiser criterion: eigenvalue > 1
            eigenvalues = pca_full.explained_variance_
            n_components = int(np.sum(eigenvalues > 1.0))
            logger.info(f"Kaiser criterion: {n_components} components with eigenvalue > 1")
            
        elif method == 'fixed':
            # Fixed number from config
            n_components = self.pca_config.get('n_components_fixed', 3)
            n_components = min(n_components, max_components)
            logger.info(f"Fixed method: {n_components} components")
            
        else:
            logger.warning(f"Unknown method '{method}', using variance_threshold")
            threshold = 0.95
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = int(np.argmax(cumsum >= threshold) + 1)
        
        # Enforce minimum and maximum
        min_components = self.pca_config.get('min_components', 2)
        max_components_config = self.pca_config.get('max_components', 10)
        
        n_components = max(min_components, min(n_components, max_components_config, max_components))
        
        return n_components
    
    def _fit_pca(self, X: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
        """
        Fit PCA model and compute scores.
        
        Args:
            X: Standardized feature matrix
            n_components: Number of components
            
        Returns:
            Tuple of (fitted PCA model, sample scores)
        """
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X)
        
        return pca, scores
    
    def _analyze_variance(self, pca: PCA) -> Dict[str, Any]:
        """
        Analyze variance explained by components.
        
        Args:
            pca: Fitted PCA model
            
        Returns:
            Dictionary with variance metrics
        """
        explained_variance = pca.explained_variance_ratio_.tolist()
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_).tolist()
        total_variance_captured = cumulative_variance[-1]
        
        return {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'total_variance_captured': total_variance_captured,
            'eigenvalues': pca.explained_variance_.tolist()
        }
    
    def _get_loadings(self, pca: PCA) -> np.ndarray:
        """
        Get feature loadings (component coefficients).
        
        Args:
            pca: Fitted PCA model
            
        Returns:
            Loading matrix (n_features × n_components)
        """
        loadings = pca.components_.T
        return loadings
    
    def _compute_feature_importance(
        self,
        loadings: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Compute feature importance for each component.
        
        Identifies top contributing features per component.
        
        Args:
            loadings: Loading matrix (n_features × n_components)
            feature_names: Names of features
            
        Returns:
            Dictionary mapping component names to feature importance dicts
        """
        n_features, n_components = loadings.shape
        n_top_features = self.pca_config.get('n_top_features', 5)
        
        feature_importance = {}
        
        for i in range(n_components):
            component_loadings = loadings[:, i]
            
            # Get absolute loadings and sort
            abs_loadings = np.abs(component_loadings)
            top_indices = np.argsort(abs_loadings)[::-1][:n_top_features]
            
            # Store feature names and their loadings
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature': feature_names[idx],
                    'loading': float(component_loadings[idx]),
                    'abs_loading': float(abs_loadings[idx])
                })
            
            feature_importance[f'PC{i+1}'] = top_features
        
        return feature_importance
    
    def _kaiser_criterion(self, pca: PCA) -> int:
        """
        Calculate Kaiser criterion (components with eigenvalue > 1).
        
        Args:
            pca: Fitted PCA model
            
        Returns:
            Number of components with eigenvalue > 1
        """
        eigenvalues = pca.explained_variance_
        n_kaiser = int(np.sum(eigenvalues > 1.0))
        return n_kaiser
    
    def _estimate_stability(self, X: np.ndarray, n_components: int) -> float:
        """
        Estimate PCA stability using bootstrap resampling.
        
        Simple stability metric: correlation of loadings across bootstrap samples.
        
        Args:
            X: Standardized feature matrix
            n_components: Number of components
            
        Returns:
            Stability score (0-1, higher is better)
        """
        # Check if bootstrap is enabled
        if not self.pca_config.get('bootstrap_validation', False):
            return None
        
        n_bootstrap = self.pca_config.get('n_bootstrap', 100)
        n_samples = X.shape[0]
        
        if n_samples < 10:
            logger.warning("Too few samples for bootstrap validation")
            return None
        
        # Fit reference PCA
        pca_ref = PCA(n_components=n_components)
        pca_ref.fit(X)
        loadings_ref = pca_ref.components_
        
        # Bootstrap resampling
        correlations = []
        
        for _ in range(min(n_bootstrap, 20)):  # Limit to 20 for speed
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            
            # Fit PCA on bootstrap sample
            try:
                pca_boot = PCA(n_components=n_components)
                pca_boot.fit(X_boot)
                loadings_boot = pca_boot.components_
                
                # Calculate correlation between loadings
                for i in range(n_components):
                    corr = np.abs(np.corrcoef(loadings_ref[i], loadings_boot[i])[0, 1])
                    correlations.append(corr)
            except:
                continue
        
        if len(correlations) == 0:
            return None
        
        stability = float(np.mean(correlations))
        logger.info(f"Bootstrap stability score: {stability:.3f}")
        
        return stability
    
    def _assess_quality(
        self,
        pca: PCA,
        n_samples: int,
        n_components: int,
        variance_metrics: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Assess quality and confidence of PCA results.
        
        Args:
            pca: Fitted PCA model
            n_samples: Number of samples
            n_components: Number of components
            variance_metrics: Variance analysis results
            
        Returns:
            Tuple of (confidence score, list of warning flags)
        """
        flags = []
        confidence = 1.0
        
        # Check sample size
        min_samples = self.pca_config.get('min_samples_required', 10)
        # Rule of thumb: 5x number of features
        samples_per_feature = self.analysis_limits.get('pca_samples_per_feature', 5) if hasattr(self, 'analysis_limits') else 5
        pca_min_samples = int(samples_per_feature * pca.n_features_in_)
        if n_samples < pca_min_samples:
            flags.append(f"LOW_SAMPLE_SIZE_PCA: {n_samples} < {pca_min_samples} (5x features)")
            confidence *= 0.7
        if n_samples < min_samples:
            flags.append(f"LOW_SAMPLE_SIZE: {n_samples} < {min_samples}")
            confidence *= 0.5
        
        # Check variance captured
        min_variance = self.pca_config.get('min_variance_captured', 0.70)
        variance_captured = variance_metrics['total_variance_captured']
        if variance_captured < min_variance:
            flags.append(f"LOW_VARIANCE_CAPTURED: {variance_captured:.2%} < {min_variance:.0%}")
            confidence *= 0.8
        
        # Check component count
        if n_components >= n_samples:
            flags.append("TOO_MANY_COMPONENTS: n_components >= n_samples")
            confidence *= 0.6
        
        # Check for degenerate eigenvalues
        eigenvalues = pca.explained_variance_
        if len(eigenvalues) > 1:
            smallest_ratio = eigenvalues[-1] / eigenvalues[0]
            if smallest_ratio < 1e-6:
                flags.append("DEGENERATE_EIGENVALUES: Very small eigenvalues detected")
                confidence *= 0.9
        
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, flags
    
    def transform_new_samples(
        self,
        dataset: XASDataset
    ) -> np.ndarray:
        """
        Transform new samples using fitted PCA model.
        
        Args:
            dataset: XASDataset with feature matrix
            
        Returns:
            Sample scores in PC space
            
        Raises:
            RuntimeError: If PCA has not been fitted yet
        """
        if self.pca_model is None:
            raise RuntimeError("PCA model not fitted. Call analyze() first.")
        
        if self.scaler is None:
            X_scaled = dataset.feature_matrix
        else:
            X_scaled = self.scaler.transform(dataset.feature_matrix)
        
        scores = self.pca_model.transform(X_scaled)
        
        return scores
    
    def get_scree_plot_data(self) -> Dict[str, List[float]]:
        """
        Get data for scree plot visualization.
        
        Returns:
            Dictionary with component numbers, eigenvalues, variance explained
            
        Raises:
            RuntimeError: If PCA has not been fitted yet
        """
        if self.pca_model is None:
            raise RuntimeError("PCA model not fitted. Call analyze() first.")
        
        n_components = self.pca_model.n_components_
        
        scree_data = {
            'component_numbers': list(range(1, n_components + 1)),
            'eigenvalues': self.pca_model.explained_variance_.tolist(),
            'variance_explained': self.pca_model.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(self.pca_model.explained_variance_ratio_).tolist()
        }
        
        return scree_data


# =============================================================================
# Standalone utility functions
# =============================================================================

def perform_pca_analysis(
    dataset: XASDataset,
    n_components: Optional[int] = None,
    config_path: Optional[Path] = None
) -> PCAAnalysisResult:
    """
    Convenience function for PCA analysis.
    
    Args:
        dataset: XASDataset with feature matrix
        n_components: Number of components (optional)
        config_path: Optional path to config file
        
    Returns:
        PCAAnalysisResult object
    """
    analyzer = XASPCAAnalyzer(config_path)
    return analyzer.analyze(dataset, n_components)


def get_top_features_per_component(
    pca_result: PCAAnalysisResult,
    n_top: int = 5
) -> Dict[str, List[str]]:
    """
    Extract top contributing features for each component.
    
    Args:
        pca_result: PCAAnalysisResult object
        n_top: Number of top features to return
        
    Returns:
        Dictionary mapping component names to top feature names
    """
    top_features = {}
    
    for component, features in pca_result.feature_importance.items():
        top_names = [f['feature'] for f in features[:n_top]]
        top_features[component] = top_names
    
    return top_features
