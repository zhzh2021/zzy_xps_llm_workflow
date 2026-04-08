"""
Quick test of integrated workflow with just 3 files.
"""
from pathlib import Path
from xas_workflow import run_xas_automated_workflow

# Get first 3 XAS files
# Path is: zz_llm/zzy_llm/Tools/APS_XAS -> zz_llm/zzy_llm/project_root/xas_raw_data
data_dir = Path(__file__).resolve().parents[2] / "project_root" / "xas_raw_data"
all_dat_files = list(data_dir.glob("*.dat"))
all_files = sorted([f for f in all_dat_files if f.is_file()])[:3]

print(f"Testing with {len(all_files)} files:")
for f in all_files:
    print(f"  - {f.name}")

# Create temp directory for test
from tempfile import mkdtemp
import shutil
temp_dir = Path(mkdtemp())
test_data_dir = temp_dir / "test_data"
test_data_dir.mkdir()

# Copy files
for f in all_files:
    shutil.copy(f, test_data_dir / f.name)

print(f"\nRunning workflow...")

try:
    result = run_xas_automated_workflow(
        test_data_dir,
        output_dir=temp_dir / "output",
        create_diagnostic_plots=False,
        enable_ml_analysis=True
    )
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    
    print(f"\nProcessed: {len(result.get('processed_samples', []))} samples")
    
    ml_result = result.get("ml_analysis", {})
    if ml_result.get("success"):
        summary = ml_result["summary"]
        print(f"\nML Analysis:")
        print(f"  Features: {summary['n_features']}")
        print(f"  PCA: {summary['n_components']} components ({summary['variance_captured']})")
        print(f"  Clusters: {summary['n_clusters']} (sil={summary['silhouette_score']})")
        print(f"  Correlations: {summary['n_correlations']}")
        print(f"  Outliers: {summary['n_outliers']}")
    else:
        print(f"\nML Analysis failed: {ml_result.get('error')}")
        
finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temp directory")
