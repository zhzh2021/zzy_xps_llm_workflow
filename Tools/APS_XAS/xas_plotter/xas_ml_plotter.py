"""
XAS ML Plotter

Minimal plotting utilities for ML analysis outputs:
- PCA scree + cumulative variance
- PCA score scatter (PC1 vs PC2)
- PCA loadings heatmap
- Cluster scatter in PCA space
- Feature-metadata correlation heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_pca_scree(pca_result, output_path: Path) -> None:
    """Save PCA scree plot with cumulative variance."""
    var = np.array(pca_result.explained_variance)
    cum = np.array(pca_result.cumulative_variance)
    pcs = np.arange(1, len(var) + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(pcs, var * 100, color="#4C78A8", alpha=0.85, label="Variance (%)", width=0.6)
    ax.plot(pcs, cum * 100, color="#F58518", marker="o", linewidth=2.5, label="Cumulative (%)")
    ax.set_xlabel("Principal Component", fontsize=14, fontweight="bold")
    ax.set_ylabel("Variance Explained (%)", fontsize=14, fontweight="bold")
    ax.set_title("PCA Scree Plot", fontsize=16, fontweight="bold")
    ax.set_xticks(pcs)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pca_scores(pca_result, output_path: Path, color_by: Optional[List[Any]] = None, title: str = "PCA Scores (PC1 vs PC2)") -> None:
    """Save PCA scores scatter plot."""
    scores = pca_result.scores
    fig, ax = plt.subplots(figsize=(8, 7))

    if color_by is not None:
        try:
            colors = np.array(color_by, dtype=float)
            sc = ax.scatter(scores[:, 0], scores[:, 1], c=colors, cmap="viridis", s=90, edgecolors="black")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Color", fontsize=12)
        except (ValueError, TypeError):
            unique_vals = sorted(set(color_by))
            cmap = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
            for v, c in zip(unique_vals, cmap):
                mask = np.array([x == v for x in color_by])
                ax.scatter(scores[mask, 0], scores[mask, 1], c=[c], s=90, edgecolors="black", label=str(v))
            ax.legend(title="Group", fontsize=18)
    else:
        ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolors="black")

    ax.set_xlabel("PC1", fontsize=14, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pca_loadings(pca_result, feature_names: List[str], output_path: Path) -> None:
    """Save PCA loadings heatmap."""
    loadings = pca_result.loadings
    pcs = [f"PC{i+1}" for i in range(loadings.shape[1])]

    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(feature_names))))
    if HAS_SEABORN:
        sns.heatmap(loadings, yticklabels=feature_names, xticklabels=pcs, cmap="coolwarm", center=0, ax=ax)
    else:
        im = ax.imshow(loadings, cmap="coolwarm", aspect="auto")
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=11)
        ax.set_xticks(range(len(pcs)))
        ax.set_xticklabels(pcs, fontsize=11)
        fig.colorbar(im, ax=ax)

    ax.set_title("PCA Loadings", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cluster_scatter(pca_result, cluster_labels: np.ndarray, output_path: Path) -> None:
    """Plot clusters in PCA space (PC1 vs PC2)."""
    scores = pca_result.scores
    fig, ax = plt.subplots(figsize=(8, 7))

    unique = sorted(set(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    for label, c in zip(unique, colors):
        mask = cluster_labels == label
        ax.scatter(scores[mask, 0], scores[mask, 1], s=90, edgecolors="black", c=[c], label=f"Cluster {label}")

    ax.set_xlabel("PC1", fontsize=14, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=14, fontweight="bold")
    ax.set_title("Cluster Assignments in PCA Space", fontsize=16, fontweight="bold")
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_correlation_heatmap(trend_results, output_path: Path) -> None:
    """Plot feature-metadata correlations heatmap from TrendAnalysisResult."""
    corr = trend_results.correlations
    if not corr:
        return

    feature_names = sorted(corr.keys())
    meta_keys = sorted({m for feat in corr.values() for m in feat.keys()})

    matrix = np.zeros((len(feature_names), len(meta_keys)))
    for i, f in enumerate(feature_names):
        for j, m in enumerate(meta_keys):
            matrix[i, j] = corr.get(f, {}).get(m, 0.0)

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(meta_keys)), max(5, 0.35 * len(feature_names))))
    if HAS_SEABORN:
        sns.heatmap(matrix, xticklabels=meta_keys, yticklabels=feature_names, cmap="coolwarm", center=0, ax=ax)
    else:
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(meta_keys)))
        ax.set_xticklabels(meta_keys, rotation=45, ha="right", fontsize=15)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=15)
        fig.colorbar(im, ax=ax)

    ax.set_title("Feature–Metadata Correlations", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
