"""
XAS Clusterer Module

Clustering analysis for XAS datasets with validation.
Supports multiple algorithms with XAS-specific validation metrics.

Author: XAS ML Integration Team
Date: 2026-03-03
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Try to import sklearn
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Clustering will not work.")

# Local imports
try:
    from xas_analyzer.xas_models import XASDataset, ClusteringResult
    from xas_ml_modules.config_utils import ConfigLoader
except ImportError:
    from ..xas_analyzer.xas_models import XASDataset, ClusteringResult
    from .config_utils import ConfigLoader


logger = logging.getLogger(__name__)


class XASClusterer:
    """
    Clustering analysis for XAS datasets.
    
    Supports multiple algorithms:
    - K-Means (default)
    - Hierarchical (Agglomerative)
    - DBSCAN (density-based)
    
    Provides validation metrics:
    - Silhouette score
    - Davies-Bouldin index
    - Calinski-Harabasz score
    - XAS-specific spectral similarity metrics
    
    Usage:
        clusterer = XASClusterer()
        result = clusterer.cluster(dataset, n_clusters=3)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize clusterer.
        
        Args:
            config_path: Path to YAML config (optional, auto-detected if None)
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for clustering. Install with: pip install scikit-learn")
        
        self.config = ConfigLoader(config_path)
        self.cluster_config = self.config.get_section('clustering')
        self.analysis_limits = self.config.get_section('analysis_limits')
        
        self.model = None
        self.labels_ = None
        
        logger.info("XASClusterer initialized")
    
    def cluster(
        self,
        dataset: XASDataset,
        n_clusters: Optional[int] = None,
        method: Optional[str] = None,
        use_pca_scores: Optional[np.ndarray] = None
    ) -> ClusteringResult:
        """
        Perform clustering on dataset.
        
        Main entry point for clustering analysis.
        
        Args:
            dataset: XASDataset with feature matrix
            n_clusters: Number of clusters (optional, auto-determined if None)
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            use_pca_scores: Optional PCA scores to cluster instead of raw features
            
        Returns:
            ClusteringResult with labels, validation metrics
            
        Raises:
            ValueError: If dataset is invalid
        """
        if dataset.feature_matrix is None and use_pca_scores is None:
            raise ValueError("Dataset has no feature matrix")
        
        # Use PCA scores if provided, otherwise use feature matrix
        X = use_pca_scores if use_pca_scores is not None else dataset.feature_matrix

        # Impute and scale features if using raw feature matrix
        if use_pca_scores is None:
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        if dataset.n_samples < 2:
            raise ValueError("Need at least 2 samples for clustering")
        
        logger.info(f"Clustering {dataset.n_samples} samples with {X.shape[1]} features")
        
        # Step 1: Determine method and parameters
        if method is None:
            method = self.cluster_config.get('method', 'kmeans')
        
        if n_clusters is None:
            n_clusters = self._determine_n_clusters(X, method)
        
        logger.info(f"Using {method} clustering with {n_clusters} clusters")
        
        # Step 2: Fit clustering model
        labels, cluster_centers = self._fit_clustering(X, n_clusters, method)
        self.labels_ = labels
        
        # Step 3: Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(X, labels)
        
        # Step 4: Analyze cluster characteristics
        cluster_info = self._analyze_clusters(dataset, labels, use_pca_scores)
        
        # Step 5: XAS-specific validation (if raw spectra available)
        spectral_metrics = self._calculate_spectral_similarity(dataset, labels)
        
        # Step 6: Quality assessment
        confidence, flags = self._assess_quality(
            validation_metrics,
            n_clusters,
            dataset.n_samples,
            spectral_metrics
        )
        
        # Step 7: Create result object
        result = ClusteringResult(
            method=method,
            n_clusters=n_clusters,
            labels=labels,
            cluster_centers=cluster_centers,
            cluster_info=cluster_info,
            silhouette_score=validation_metrics['silhouette'],
            davies_bouldin_index=validation_metrics.get('davies_bouldin'),
            calinski_harabasz_score=validation_metrics.get('calinski_harabasz'),
            spectral_similarity_within=spectral_metrics.get('within_cluster', []),
            spectral_separation_between=spectral_metrics.get('between_cluster'),
            confidence=confidence,
            flags=flags
        )
        
        logger.info(
            f"Clustering complete: {n_clusters} clusters, "
            f"silhouette={validation_metrics['silhouette']:.3f}"
        )
        
        return result
    
    def _determine_n_clusters(self, X: np.ndarray, method: str) -> int:
        """
        Automatically determine optimal number of clusters.
        
        Uses elbow method or silhouette analysis.
        
        Args:
            X: Feature matrix
            method: Clustering method
            
        Returns:
            Optimal number of clusters
        """
        auto_method = self.cluster_config.get('n_clusters_method', 'silhouette')
        min_clusters = self.cluster_config.get('min_clusters', 2)
        max_clusters = self.cluster_config.get('max_clusters', 10)
        
        # Limit max_clusters to n_samples - 1
        max_clusters = min(max_clusters, X.shape[0] - 1)
        
        if max_clusters < min_clusters:
            logger.warning(f"Not enough samples, using {min_clusters} clusters")
            return min_clusters
        
        if auto_method == 'silhouette' and method in ['kmeans', 'hierarchical']:
            # Try different cluster counts and pick best silhouette score
            best_n = min_clusters
            best_score = -1
            
            for n in range(min_clusters, max_clusters + 1):
                try:
                    if method == 'kmeans':
                        model = KMeans(n_clusters=n, random_state=42, n_init=10)
                    else:
                        model = AgglomerativeClustering(n_clusters=n)
                    
                    labels = model.fit_predict(X)
                    score = silhouette_score(X, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_n = n
                except:
                    continue
            
            logger.info(f"Silhouette method: optimal {best_n} clusters (score={best_score:.3f})")
            return best_n
        
        elif auto_method == 'fixed':
            n_clusters = self.cluster_config.get('n_clusters_fixed', 3)
            n_clusters = max(min_clusters, min(n_clusters, max_clusters))
            logger.info(f"Fixed method: {n_clusters} clusters")
            return n_clusters
        
        else:
            # Default to middle value
            n_clusters = (min_clusters + max_clusters) // 2
            logger.info(f"Default method: {n_clusters} clusters")
            return n_clusters
    
    def _fit_clustering(
        self,
        X: np.ndarray,
        n_clusters: int,
        method: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit clustering model.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            method: Clustering method
            
        Returns:
            Tuple of (labels, cluster_centers)
        """
        random_state = self.cluster_config.get('random_state', 42)
        
        if method == 'kmeans':
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            labels = self.model.fit_predict(X)
            centers = self.model.cluster_centers_
            
        elif method == 'hierarchical':
            linkage = self.cluster_config.get('linkage', 'ward')
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
            labels = self.model.fit_predict(X)
            # Compute centers manually for hierarchical
            centers = self._compute_cluster_centers(X, labels, n_clusters)
            
        elif method == 'dbscan':
            eps = self.cluster_config.get('dbscan_eps', 0.5)
            min_samples = self.cluster_config.get('dbscan_min_samples', 5)
            
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = self.model.fit_predict(X)
            
            # DBSCAN doesn't have fixed n_clusters
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            logger.info(f"DBSCAN found {n_clusters} clusters")
            
            centers = self._compute_cluster_centers(X, labels, n_clusters)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return labels, centers
    
    def _compute_cluster_centers(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """Compute cluster centers as mean of members."""
        centers = []
        for i in range(n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                center = np.mean(X[mask], axis=0)
                centers.append(center)
            else:
                centers.append(np.zeros(X.shape[1]))
        
        return np.array(centers)
    
    def _calculate_validation_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate clustering validation metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {}
        
        # Check if we have at least 2 clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 2:
            logger.warning("Less than 2 clusters, validation metrics not meaningful")
            return {'silhouette': 0.0}
        
        try:
            # Silhouette score
            metrics['silhouette'] = float(silhouette_score(X, labels))
        except:
            metrics['silhouette'] = 0.0
        
        try:
            # Davies-Bouldin index (lower is better)
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
        except:
            metrics['davies_bouldin'] = None
        
        try:
            # Calinski-Harabasz score (higher is better)
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
        except:
            metrics['calinski_harabasz'] = None
        
        return metrics
    
    def _analyze_clusters(
        self,
        dataset: XASDataset,
        labels: np.ndarray,
        pca_scores: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Analyze characteristics of each cluster.
        
        Args:
            dataset: XASDataset
            labels: Cluster labels
            pca_scores: Optional PCA scores
            
        Returns:
            List of cluster information dictionaries
        """
        cluster_info = []
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        for i in range(n_clusters):
            mask = labels == i
            cluster_samples = [dataset.sample_names[j] for j in range(len(labels)) if mask[j]]
            
            info = {
                'cluster_id': i,
                'size': int(np.sum(mask)),
                'samples': cluster_samples,
                'percentage': float(np.sum(mask) / len(labels) * 100)
            }
            
            # Add metadata statistics if available
            if dataset.metadata_dict is not None:
                metadata_stats = {}
                for key, values in dataset.metadata_dict.items():
                    if key == 'sample_name':
                        continue
                    
                    cluster_values = [values[j] for j in range(len(values)) if mask[j]]
                    
                    # Calculate statistics based on data type
                    if len(cluster_values) > 0:
                        try:
                            # Try numeric statistics
                            numeric_values = [float(v) for v in cluster_values if v is not None]
                            if numeric_values:
                                metadata_stats[key] = {
                                    'mean': float(np.mean(numeric_values)),
                                    'std': float(np.std(numeric_values)),
                                    'min': float(np.min(numeric_values)),
                                    'max': float(np.max(numeric_values))
                                }
                        except:
                            # Categorical data - count occurrences
                            from collections import Counter
                            counts = Counter(cluster_values)
                            metadata_stats[key] = dict(counts.most_common(3))
                
                if metadata_stats:
                    info['metadata_stats'] = metadata_stats
            
            cluster_info.append(info)
        
        return cluster_info
    
    def _calculate_spectral_similarity(
        self,
        dataset: XASDataset,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate XAS-specific spectral similarity metrics.
        
        Measures how similar spectra are within clusters vs between clusters.
        
        Args:
            dataset: XASDataset
            labels: Cluster labels
            
        Returns:
            Dictionary with spectral similarity metrics
        """
        # This requires raw feature matrix (not PCA scores)
        if dataset.feature_matrix is None:
            return {}
        
        X = dataset.feature_matrix
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Within-cluster similarity (higher is better)
        within_similarity = []
        
        for i in range(n_clusters):
            mask = labels == i
            if np.sum(mask) < 2:
                within_similarity.append(0.0)
                continue
            
            cluster_features = X[mask]
            
            # Compute pairwise correlations within cluster
            correlations = []
            n_members = cluster_features.shape[0]
            
            for j in range(n_members):
                for k in range(j + 1, n_members):
                    corr = np.corrcoef(cluster_features[j], cluster_features[k])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                within_similarity.append(float(np.mean(correlations)))
            else:
                within_similarity.append(0.0)
        
        # Between-cluster separation (higher is better)
        between_separation = None
        
        if n_clusters > 1:
            # Compute distance between cluster centers
            centers = []
            for i in range(n_clusters):
                mask = labels == i
                if np.sum(mask) > 0:
                    centers.append(np.mean(X[mask], axis=0))
            
            if len(centers) > 1:
                distances = []
                for j in range(len(centers)):
                    for k in range(j + 1, len(centers)):
                        dist = np.linalg.norm(centers[j] - centers[k])
                        distances.append(dist)
                
                between_separation = float(np.mean(distances))
        
        return {
            'within_cluster': within_similarity,
            'between_cluster': between_separation
        }
    
    def _assess_quality(
        self,
        validation_metrics: Dict[str, float],
        n_clusters: int,
        n_samples: int,
        spectral_metrics: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Assess quality of clustering results.
        
        Args:
            validation_metrics: Validation metric values
            n_clusters: Number of clusters
            n_samples: Number of samples
            spectral_metrics: Spectral similarity metrics
            
        Returns:
            Tuple of (confidence score, warning flags)
        """
        flags = []
        confidence = 1.0
        
        # Check silhouette score
        silhouette = validation_metrics.get('silhouette', 0.0)
        min_silhouette = self.cluster_config.get('min_silhouette_score', 0.3)
        
        if silhouette < min_silhouette:
            flags.append(f"LOW_SILHOUETTE: {silhouette:.3f} < {min_silhouette}")
            confidence *= 0.7
        
        # Check cluster sizes (avoid very small clusters)
        min_cluster_size = self.cluster_config.get('min_cluster_size', 2)
        # Rule of thumb: at least 2x clusters in samples
        samples_per_cluster = self.analysis_limits.get('clustering_samples_per_cluster', 2) if hasattr(self, 'analysis_limits') else 2
        min_samples = int(samples_per_cluster * n_clusters)
        if n_samples < min_samples:
            flags.append(f"LOW_SAMPLE_SIZE_CLUSTER: {n_samples} < {min_samples} (2x clusters)")
            confidence *= 0.8
        avg_cluster_size = n_samples / n_clusters
        
        if avg_cluster_size < min_cluster_size:
            flags.append(f"SMALL_CLUSTERS: avg size {avg_cluster_size:.1f} < {min_cluster_size}")
            confidence *= 0.8
        
        # Check Davies-Bouldin index (lower is better, typically < 1.0 is good)
        db_index = validation_metrics.get('davies_bouldin')
        if db_index is not None and db_index > 2.0:
            flags.append(f"HIGH_DAVIES_BOULDIN: {db_index:.2f} > 2.0")
            confidence *= 0.9
        
        # Check spectral similarity
        within_sim = spectral_metrics.get('within_cluster', [])
        if within_sim and np.mean(within_sim) < 0.5:
            flags.append(f"LOW_SPECTRAL_SIMILARITY: {np.mean(within_sim):.3f} < 0.5")
            confidence *= 0.8
        
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, flags
    
    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new samples.
        
        Args:
            X: Feature matrix for new samples
            
        Returns:
            Predicted cluster labels
            
        Raises:
            RuntimeError: If model has not been fitted
        """
        if self.model is None:
            raise RuntimeError("Clustering model not fitted. Call cluster() first.")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise NotImplementedError("This clustering method does not support prediction")


# =============================================================================
# Standalone utility functions
# =============================================================================

def perform_clustering(
    dataset: XASDataset,
    n_clusters: Optional[int] = None,
    method: Optional[str] = None,
    config_path: Optional[Path] = None,
    use_pca_scores: Optional[np.ndarray] = None
) -> ClusteringResult:
    """
    Convenience function for clustering analysis.
    
    Args:
        dataset: XASDataset
        n_clusters: Number of clusters (optional)
        method: Clustering method (optional)
        config_path: Path to config file (optional)
        use_pca_scores: PCA scores to cluster (optional)
        
    Returns:
        ClusteringResult object
    """
    clusterer = XASClusterer(config_path)
    return clusterer.cluster(dataset, n_clusters, method, use_pca_scores)


def get_cluster_members(
    result: ClusteringResult,
    cluster_id: int
) -> List[str]:
    """
    Get sample names for a specific cluster.
    
    Args:
        result: ClusteringResult object
        cluster_id: Cluster ID
        
    Returns:
        List of sample names in that cluster
    """
    for cluster in result.cluster_info:
        if cluster['cluster_id'] == cluster_id:
            return cluster['samples']
    
    return []
