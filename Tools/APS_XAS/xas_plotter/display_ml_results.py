"""
Display ML analysis results from real Excel XAS data.
"""
import json
from pathlib import Path

ml_dir = Path(r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_results\ml_analysis")

# Load all results
dataset = json.load(open(ml_dir / "dataset.json"))
pca = json.load(open(ml_dir / "pca_results.json"))
clustering = json.load(open(ml_dir / "clustering_results.json"))
trends = json.load(open(ml_dir / "trend_results.json"))

print("="*80)
print("ML ANALYSIS RESULTS: Real Excel XAS Data (Malic Acid Samples)")
print("="*80)

print(f"\n📊 DATASET")
print(f"  Samples processed: {dataset['n_samples']}")
print(f"  Features extracted: {dataset['n_features']}")
print(f"  Dataset ID: {dataset['dataset_id']}")
print(f"\n  Sample names:")
for i, name in enumerate(dataset['sample_names'], 1):
    print(f"    {i:2d}. {name}")

print(f"\n🔍 PCA ANALYSIS")
print(f"  Components selected: {pca['n_components']}")
print(f"  Total variance captured: {pca['variance_captured']:.1%}")
print(f"  Kaiser criterion: {pca['kaiser_criterion']} components (eigenvalue > 1)")
print(f"\n  Variance explained per component:")
for i, var in enumerate(pca['explained_variance'], 1):
    cumvar = pca['cumulative_variance'][i-1]
    bar = "█" * int(var * 50)
    print(f"    PC{i}: {var:6.1%} (cumulative: {cumvar:6.1%}) {bar}")

print(f"\n  Top contributing features:")
for i in range(min(3, pca['n_components'])):
    pc_features = pca['feature_importance'][f'PC{i+1}'][:3]
    print(f"    PC{i+1}:", end="")
    for feat in pc_features:
        print(f" {feat['feature']}({feat['abs_loading']:.2f})", end="")
    print()

print(f"\n🎯 CLUSTERING")
print(f"  Method: {clustering['method']}")
print(f"  Clusters found: {clustering['n_clusters']}")
print(f"  Silhouette score: {clustering['silhouette_score']:.3f} ", end="")
if clustering['silhouette_score'] > 0.5:
    print("(excellent)")
elif clustering['silhouette_score'] > 0.3:
    print("(good)")
elif clustering['silhouette_score'] > 0.2:
    print("(acceptable)")
else:
    print("(weak)")

print(f"\n  Cluster sizes:")
cluster_sizes = {}
for label in clustering['labels']:
    cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

for cluster_id in sorted(cluster_sizes.keys()):
    count = cluster_sizes[cluster_id]
    pct = count / dataset['n_samples'] * 100
    bar = "▓" * int(pct / 5)
    print(f"    Cluster {cluster_id}: {count:2d} samples ({pct:5.1f}%) {bar}")

# Map samples to clusters
print(f"\n  Sample assignments:")
for sample_name, label in zip(dataset['sample_names'], clustering['labels']):
    print(f"    {sample_name}: Cluster {label}")

print(f"\n📈 TREND ANALYSIS")
print(f"  Significant correlations: {len(trends['significant_correlations'])}")
print(f"  Outliers detected: {len(trends['outlier_indices'])}")
print(f"  Detection method: {trends.get('outlier_method', 'N/A')}")

if len(trends['significant_correlations']) > 0:
    print(f"\n  Top correlations (p < 0.05):")
    for i, corr in enumerate(trends['significant_correlations'][:5], 1):
        feature = corr['feature']
        metadata = corr['metadata']
        r = corr['correlation']
        p = corr['p_value']
        strength = corr['strength']
        direction = corr['direction']
        print(f"    {i}. {feature} ↔ {metadata}")
        print(f"       r={r:+.3f} (p={p:.2e}) [{strength} {direction}]")

if len(trends['outlier_indices']) > 0:
    print(f"\n  Outlier samples:")
    for idx in trends['outlier_indices']:
        sample_name = dataset['sample_names'][idx]
        print(f"    - {sample_name} (index {idx})")

print(f"\n💾 OUTPUT FILES")
for file in ml_dir.glob("*.json"):
    size_kb = file.stat().st_size / 1024
    print(f"  {file.name}: {size_kb:.1f} KB")

print("\n" + "="*80)
print("✅ Complete ML pipeline successfully processed real XAS data!")
print("="*80)
