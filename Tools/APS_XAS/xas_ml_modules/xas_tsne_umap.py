"""
XAS t-SNE / UMAP Module

Provides non-linear dimensionality reduction for XAS feature datasets.
Designed for exploratory visualization (not for quantitative interpretation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Try to import sklearn t-SNE
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

# Try to import UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Try to import sklearn preprocessing
try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Local imports
try:
    from xas_analyzer.xas_models import XASDataset
    from xas_ml_modules.config_utils import ConfigLoader
except ImportError:
    from ..xas_analyzer.xas_models import XASDataset
    from .config_utils import ConfigLoader

logger = logging.getLogger(__name__)


class XASTSNEUMAP:
    """
    Non-linear embedding for XAS datasets (t-SNE / UMAP).

    Notes:
    - These embeddings are for visualization only.
    - Always inspect sensitivity to parameters and random seed.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = ConfigLoader(config_path)
        self.analysis_limits = self.config.get_section('analysis_limits')
        self.tsne_config = self.config.get_section('tsne') if 'tsne' in self.config.get_all() else {}
        self.umap_config = self.config.get_section('umap') if 'umap' in self.config.get_all() else {}

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        if not HAS_SKLEARN:
            return X
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        return scaler.fit_transform(X_imp)

    def fit_tsne(self, dataset: XASDataset) -> Dict[str, Any]:
        if not HAS_TSNE:
            raise ImportError("scikit-learn required for t-SNE")
        X = self._preprocess(dataset.feature_matrix)

        min_samples = self.analysis_limits.get('min_samples_tsne', 30)
        if dataset.n_samples < min_samples:
            logger.warning(f"Small sample size for t-SNE: {dataset.n_samples} < {min_samples}")

        perplexity = self.tsne_config.get('perplexity', 30)
        n_iter = self.tsne_config.get('n_iter', 1000)
        random_state = self.tsne_config.get('random_state', 42)

        try:
            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=random_state,
                init='pca'
            )
        except TypeError:
            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=n_iter,
                random_state=random_state,
                init='pca'
            )
        embedding = model.fit_transform(X)

        return {
            'method': 'tsne',
            'embedding': embedding,
            'perplexity': perplexity,
            'n_iter': n_iter
        }

    def fit_umap(self, dataset: XASDataset) -> Dict[str, Any]:
        if not HAS_UMAP:
            raise ImportError("umap-learn required for UMAP")
        X = self._preprocess(dataset.feature_matrix)

        min_samples = self.analysis_limits.get('min_samples_tsne', 30)
        if dataset.n_samples < min_samples:
            logger.warning(f"Small sample size for UMAP: {dataset.n_samples} < {min_samples}")

        n_neighbors = self.umap_config.get('n_neighbors', 15)
        min_dist = self.umap_config.get('min_dist', 0.1)
        random_state = self.umap_config.get('random_state', 42)

        model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        embedding = model.fit_transform(X)

        return {
            'method': 'umap',
            'embedding': embedding,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }
