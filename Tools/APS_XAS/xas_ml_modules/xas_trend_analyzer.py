"""
XAS Trend Analyzer Module

Correlation analysis and trend detection for XAS datasets.
Identifies relationships between features and metadata.

Author: XAS ML Integration Team
Date: 2026-03-03
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Try to import scipy for statistics
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Statistical tests will be limited.")

# Try to import sklearn for outlier detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Local imports
try:
    from xas_analyzer.xas_models import XASDataset, TrendAnalysisResult, ClusteringResult
    from xas_ml_modules.config_utils import ConfigLoader
except ImportError:
    from ..xas_analyzer.xas_models import XASDataset, TrendAnalysisResult, ClusteringResult
    from .config_utils import ConfigLoader


logger = logging.getLogger(__name__)


class XASTrendAnalyzer:
    """
    Trend and correlation analysis for XAS datasets.
    
    This class:
    - Computes feature-metadata correlations
    - Performs statistical significance tests
    - Identifies outlier samples
    - Analyzes cluster-metadata relationships
    - Generates actionable insights
    
    Usage:
        analyzer = XASTrendAnalyzer()
        result = analyzer.analyze(dataset)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize trend analyzer.
        
        Args:
            config_path: Path to YAML config (optional, auto-detected if None)
        """
        self.config = ConfigLoader(config_path)
        self.trend_config = self.config.get_section('trend_analysis')
        self.analysis_limits = self.config.get_section('analysis_limits')
        
        logger.info("XASTrendAnalyzer initialized")
    
    def analyze(
        self,
        dataset: XASDataset,
        clustering_result: Optional[ClusteringResult] = None
    ) -> TrendAnalysisResult:
        """
        Perform comprehensive trend analysis.
        
        Main entry point for trend analysis.
        
        Args:
            dataset: XASDataset with features and metadata
            clustering_result: Optional clustering results for cluster-metadata analysis
            
        Returns:
            TrendAnalysisResult with correlations, trends, outliers
            
        Raises:
            ValueError: If dataset has no metadata
        """
        if dataset.metadata_dict is None or len(dataset.metadata_dict) == 0:
            logger.warning("Dataset has no metadata, limited analysis possible")
        
        logger.info(f"Analyzing trends for {dataset.n_samples} samples")
        
        # Step 1: Compute feature-metadata correlations
        correlations, p_values = self._compute_correlations(dataset)
        
        # Step 2: Identify significant correlations
        significant_correlations = self._find_significant_correlations(
            correlations,
            p_values,
            dataset.feature_names
        )
        
        # Step 3: Analyze cluster-metadata relationships (if clustering provided)
        cluster_metadata_stats = None
        if clustering_result is not None:
            cluster_metadata_stats = self._analyze_cluster_metadata(
                dataset,
                clustering_result
            )
        
        # Step 4: Detect outliers
        outlier_indices, outlier_scores, outlier_method = self._detect_outliers(dataset)
        
        # Step 5: Quality assessment
        confidence, flags = self._assess_quality(
            correlations,
            p_values,
            significant_correlations,
            dataset.n_samples
        )
        
        # Step 6: Create result object
        result = TrendAnalysisResult(
            correlations=correlations,
            p_values=p_values,
            significant_correlations=significant_correlations,
            cluster_metadata_stats=cluster_metadata_stats,
            outlier_indices=outlier_indices,
            outlier_scores=outlier_scores,
            outlier_method=outlier_method,
            confidence=confidence,
            flags=flags
        )
        
        logger.info(
            f"Trend analysis complete: {len(significant_correlations)} significant correlations, "
            f"{len(outlier_indices)} outliers detected"
        )
        
        return result
    
    def _compute_correlations(
        self,
        dataset: XASDataset
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Compute correlations between features and metadata."""
        correlations = {}
        p_values = {}

        if dataset.metadata_dict is None:
            return correlations, p_values

        X = dataset.feature_matrix
        method = self.trend_config.get('correlation_method', 'pearson')

        for i, feature_name in enumerate(dataset.feature_names):
            feature_values = X[:, i]
            correlations[feature_name] = {}
            p_values[feature_name] = {}

            for meta_key, meta_values in dataset.metadata_dict.items():
                if meta_key == 'sample_name':
                    continue

                meta_arr = []
                feat_arr = []
                for j, val in enumerate(meta_values):
                    if val is None:
                        continue
                    try:
                        v = float(val)
                    except (ValueError, TypeError):
                        continue
                    f = feature_values[j]
                    if not np.isfinite(f) or not np.isfinite(v):
                        continue
                    meta_arr.append(v)
                    feat_arr.append(f)

                if len(meta_arr) < 4:
                    continue

                meta_arr = np.array(meta_arr)
                feat_arr = np.array(feat_arr)

                if np.std(meta_arr) == 0 or np.std(feat_arr) == 0:
                    continue

                if HAS_SCIPY:
                    if method == 'spearman':
                        r, p = stats.spearmanr(feat_arr, meta_arr)
                    else:
                        r, p = stats.pearsonr(feat_arr, meta_arr)
                else:
                    r = np.corrcoef(feat_arr, meta_arr)[0, 1]
                    p = 0.05

                if not np.isnan(r):
                    correlations[feature_name][meta_key] = float(r)
                    p_values[feature_name][meta_key] = float(p)

        return correlations, p_values

    def _find_significant_correlations(
        self,
        correlations: Dict[str, Dict[str, float]],
        p_values: Dict[str, Dict[str, float]],
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify statistically significant correlations."""
        significance_threshold = self.trend_config.get('significance_threshold', 0.05)
        min_correlation = self.trend_config.get('min_correlation', 0.3)
        use_fdr = self.trend_config.get('fdr_correction', True)

        p_list = []
        idx_list = []
        for feature in feature_names:
            for meta_key, p in p_values.get(feature, {}).items():
                p_list.append(p)
                idx_list.append((feature, meta_key))

        p_adj = {}
        if use_fdr and p_list:
            p_sorted = np.argsort(p_list)
            n = len(p_list)
            for rank, idx in enumerate(p_sorted, start=1):
                feature, meta_key = idx_list[idx]
                pval = p_list[idx]
                adj = pval * n / rank
                p_adj[(feature, meta_key)] = min(adj, 1.0)
        else:
            for feature, meta_key in idx_list:
                p_adj[(feature, meta_key)] = p_values.get(feature, {}).get(meta_key, 1.0)

        significant = []

        for feature in feature_names:
            if feature not in correlations:
                continue

            for meta_key, r in correlations[feature].items():
                p = p_values.get(feature, {}).get(meta_key, 1.0)
                p_corr = p_adj.get((feature, meta_key), p)

                if p_corr < significance_threshold and abs(r) >= min_correlation:
                    significant.append({
                        'feature': feature,
                        'metadata': meta_key,
                        'correlation': r,
                        'p_value': p,
                        'p_value_adj': p_corr,
                        'strength': self._classify_correlation_strength(abs(r)),
                        'direction': 'positive' if r > 0 else 'negative'
                    })

        significant.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return significant

    def _classify_correlation_strength(self, abs_r: float) -> str:
        """Classify correlation strength."""
        if abs_r >= 0.7:
            return 'strong'
        elif abs_r >= 0.5:
            return 'moderate'
        elif abs_r >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _analyze_cluster_metadata(
        self,
        dataset: XASDataset,
        clustering_result: ClusteringResult
    ) -> Dict[str, Any]:
        """
        Analyze relationships between clusters and metadata.
        
        Tests if metadata distributions differ across clusters.
        
        Args:
            dataset: XASDataset
            clustering_result: ClusteringResult
            
        Returns:
            Dictionary with cluster-metadata statistics
        """
        if dataset.metadata_dict is None:
            return {}
        
        cluster_stats = {}
        labels = clustering_result.labels
        n_clusters = clustering_result.n_clusters
        
        for meta_key, meta_values in dataset.metadata_dict.items():
            if meta_key == 'sample_name':
                continue
            
            # Try numeric analysis
            try:
                numeric_meta = np.array([float(v) for v in meta_values if v is not None])
                
                if len(numeric_meta) != len(labels):
                    continue
                
                # Compute per-cluster statistics
                cluster_distributions = []
                
                for i in range(n_clusters):
                    mask = labels == i
                    cluster_values = numeric_meta[mask]
                    
                    if len(cluster_values) > 0:
                        cluster_distributions.append({
                            'cluster_id': i,
                            'mean': float(np.mean(cluster_values)),
                            'std': float(np.std(cluster_values)),
                            'median': float(np.median(cluster_values)),
                            'min': float(np.min(cluster_values)),
                            'max': float(np.max(cluster_values)),
                            'n_samples': len(cluster_values)
                        })
                
                # Statistical test: ANOVA (if scipy available)
                if HAS_SCIPY and n_clusters > 1:
                    cluster_groups = [numeric_meta[labels == i] for i in range(n_clusters)]
                    cluster_groups = [g for g in cluster_groups if len(g) > 0]
                    
                    if len(cluster_groups) > 1:
                        try:
                            f_stat, p_val = stats.f_oneway(*cluster_groups)
                            anova_result = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_val),
                                'significant': bool(p_val < 0.05)  # Explicitly convert to Python bool
                            }
                        except:
                            anova_result = None
                    else:
                        anova_result = None
                else:
                    anova_result = None
                
                cluster_stats[meta_key] = {
                    'type': 'numeric',
                    'distributions': cluster_distributions,
                    'anova': anova_result
                }
            
            except (ValueError, TypeError):
                # Categorical metadata
                from collections import Counter
                
                categorical_stats = []
                for i in range(n_clusters):
                    mask = labels == i
                    cluster_values = [meta_values[j] for j in range(len(meta_values)) if mask[j]]
                    
                    if cluster_values:
                        counts = Counter(cluster_values)
                        categorical_stats.append({
                            'cluster_id': i,
                            'value_counts': dict(counts),
                            'most_common': counts.most_common(1)[0][0] if counts else None
                        })
                
                cluster_stats[meta_key] = {
                    'type': 'categorical',
                    'distributions': categorical_stats
                }
        
        return cluster_stats
    
    def _detect_outliers(
        self,
        dataset: XASDataset
    ) -> Tuple[List[int], Optional[np.ndarray], Optional[str]]:
        """
        Detect outlier samples using multiple methods.
        
        Args:
            dataset: XASDataset
            
        Returns:
            Tuple of (outlier_indices, outlier_scores, method_used)
        """
        if not self.trend_config.get('detect_outliers', True):
            return [], None, None
        
        method = self.trend_config.get('outlier_method', 'isolation_forest')
        X = dataset.feature_matrix
        
        if X is None or dataset.n_samples < 5:
            return [], None, None
        
        outlier_indices = []
        outlier_scores = None
        
        try:
            if method == 'isolation_forest' and HAS_SKLEARN:
                contamination = self.trend_config.get('outlier_contamination', 0.1)
                
                clf = IsolationForest(
                    contamination=contamination,
                    random_state=42
                )
                predictions = clf.fit_predict(X)
                outlier_scores = clf.score_samples(X)
                
                outlier_indices = [i for i, pred in enumerate(predictions) if pred == -1]
                
            elif method == 'elliptic_envelope' and HAS_SKLEARN:
                contamination = self.trend_config.get('outlier_contamination', 0.1)
                
                clf = EllipticEnvelope(contamination=contamination, random_state=42)
                predictions = clf.fit_predict(X)
                
                outlier_indices = [i for i, pred in enumerate(predictions) if pred == -1]
                outlier_scores = clf.decision_function(X)
                
            elif method == 'zscore':
                # Z-score based outlier detection
                threshold = self.trend_config.get('zscore_threshold', 3.0)
                
                z_scores = np.abs(stats.zscore(X, axis=0))
                outlier_mask = np.any(z_scores > threshold, axis=1)
                outlier_indices = np.where(outlier_mask)[0].tolist()
                outlier_scores = np.max(z_scores, axis=1)
                
            else:
                logger.warning(f"Unknown outlier method: {method}")
                return [], None, None
        
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return [], None, None
        
        logger.info(f"Detected {len(outlier_indices)} outliers using {method}")
        
        return outlier_indices, outlier_scores, method
    
    def _assess_quality(
        self,
        correlations: Dict[str, Dict[str, float]],
        p_values: Dict[str, Dict[str, float]],
        significant_correlations: List[Dict[str, Any]],
        n_samples: int
    ) -> Tuple[float, List[str]]:
        """
        Assess quality of trend analysis.
        
        Args:
            correlations: Correlation dictionary
            p_values: P-value dictionary
            significant_correlations: List of significant correlations
            n_samples: Number of samples
            
        Returns:
            Tuple of (confidence score, warning flags)
        """
        flags = []
        confidence = 1.0
        
        # Check sample size
        min_samples = self.trend_config.get('min_samples_for_correlation', 5)
        min_samples_trend = self.analysis_limits.get('min_samples_trend', min_samples) if hasattr(self, 'analysis_limits') else min_samples
        if n_samples < min_samples_trend:
            flags.append(f"LOW_SAMPLE_SIZE_TREND: {n_samples} < {min_samples_trend}")
            confidence *= 0.6
        if n_samples < min_samples:
            flags.append(f"LOW_SAMPLE_SIZE: {n_samples} < {min_samples}")
            confidence *= 0.6
        
        # Check if any correlations found
        total_correlations = sum(len(v) for v in correlations.values())
        if total_correlations == 0:
            flags.append("NO_CORRELATIONS: No metadata available for correlation analysis")
            confidence *= 0.5
        
        # Check if significant correlations found
        if total_correlations > 0 and len(significant_correlations) == 0:
            flags.append("NO_SIGNIFICANT_CORRELATIONS: No statistically significant correlations found")
            confidence *= 0.8
        
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, flags
    
    def generate_insights(
        self,
        result: TrendAnalysisResult,
        dataset: XASDataset
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable insights from trend analysis.
        
        Args:
            result: TrendAnalysisResult
            dataset: XASDataset
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        # Insights from significant correlations
        for corr in result.significant_correlations[:5]:  # Top 5
            insights.append({
                'type': 'correlation',
                'priority': 'high' if corr['strength'] in ['strong', 'moderate'] else 'medium',
                'message': (
                    f"{corr['strength'].capitalize()} {corr['direction']} correlation "
                    f"between {corr['feature']} and {corr['metadata']} "
                    f"(r={corr['correlation']:.3f}, p={corr['p_value']:.4f})"
                ),
                'data': corr
            })
        
        # Insights from outliers
        if result.outlier_indices:
            outlier_names = [dataset.sample_names[i] for i in result.outlier_indices]
            insights.append({
                'type': 'outlier',
                'priority': 'high',
                'message': f"Detected {len(result.outlier_indices)} outlier samples: {', '.join(outlier_names[:3])}{'...' if len(outlier_names) > 3 else ''}",
                'data': {
                    'outlier_samples': outlier_names,
                    'method': result.outlier_method
                }
            })
        
        # Insights from cluster-metadata relationships
        if result.cluster_metadata_stats:
            for meta_key, stats in result.cluster_metadata_stats.items():
                if stats['type'] == 'numeric' and stats.get('anova'):
                    anova = stats['anova']
                    if anova.get('significant'):
                        insights.append({
                            'type': 'cluster_metadata',
                            'priority': 'medium',
                            'message': (
                                f"Significant difference in {meta_key} across clusters "
                                f"(ANOVA p={anova['p_value']:.4f})"
                            ),
                            'data': stats
                        })
        
        return insights


# =============================================================================
# Standalone utility functions
# =============================================================================

def perform_trend_analysis(
    dataset: XASDataset,
    clustering_result: Optional[ClusteringResult] = None,
    config_path: Optional[Path] = None
) -> TrendAnalysisResult:
    """
    Convenience function for trend analysis.
    
    Args:
        dataset: XASDataset
        clustering_result: Optional clustering results
        config_path: Path to config file (optional)
        
    Returns:
        TrendAnalysisResult object
    """
    analyzer = XASTrendAnalyzer(config_path)
    return analyzer.analyze(dataset, clustering_result)


def get_top_correlations(
    result: TrendAnalysisResult,
    n_top: int = 10
) -> List[Dict[str, Any]]:
    """
    Get top N significant correlations.
    
    Args:
        result: TrendAnalysisResult
        n_top: Number of top correlations to return
        
    Returns:
        List of top correlation dictionaries
    """
    return result.significant_correlations[:n_top]
