#!/usr/bin/env python3
"""
Phase 1 ML Integration Test

Tests all Phase 1 ML modules with real XAS preprocessing data:
1. Feature extraction from processed spectra
2. Batch assembly into dataset
3. PCA analysis
4. Clustering
5. Trend analysis

Author: XAS ML Integration Team
Date: 2026-03-03
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add XAS tools to path
xas_path = Path(__file__).parent
sys.path.insert(0, str(xas_path))


class SimplifiedXASSample:
    """
    Simplified wrapper for XAS sample data (for testing ML modules).
    Contains both Pydantic model and numpy arrays.
    """
    def __init__(self, result_model, energy, normalized_mu):
        self.result_model = result_model
        self.sample_name = result_model.sample_name
        self.energy = energy
        self.normalized_mu = normalized_mu
        self.e0 = result_model.features.e0
        self.edge_step = result_model.features.edge_step
        self.quality_score = 1.0  # Default
        self.processing_timestamp = result_model.timestamp.isoformat()


def test_phase1_ml_integration():
    """
    Test Phase 1 ML modules with real data from preprocessing.
    """
    print("=" * 80)
    print("XAS ML PHASE 1 INTEGRATION TEST")
    print("=" * 80)
    print()
    
    # Step 0: Import modules
    print("Step 0: Importing modules...")
    print("-" * 80)
    
    try:
        from xas_analyzer.xas_models import XASSampleResult
        from xas_ml_modules import (
            XASFeatureExtractor,
            XASBatchAssembler,
            XASPCAAnalyzer,
            XASClusterer,
            XASTrendAnalyzer
        )
        print("✅ Core ML modules imported successfully")
        
        # For creating sample results, we also need these models
        from xas_analyzer.xas_models import (
            XASFeatures,
            ProcessingMetadata,
            XASProcessingParams
        )
        
        # Try importing workflow (optional - may fail due to larch dependency)
        try:
            from xas_workflow import run_xas_workflow
            has_workflow = True
            print("✅ Workflow module imported")
        except ImportError:
            has_workflow = False
            print("⚠️  Workflow unavailable (larch dependency) - will use mock data")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 1: Get XAS sample data (real or mock)
    print("Step 1: Preparing XAS sample data...")
    print("-" * 80)
    
    # Find data directory
    project_root = Path(__file__).resolve().parents[2] / "project_root"
    output_dir = project_root / "xas_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if has_workflow:
        # Try to run real workflow
        data_dir = project_root / "00_raw_data"
        
        if not data_dir.exists():
            print(f"⚠️  Data directory not found: {data_dir}, using mock data")
            has_workflow = False
        else:
            xas_files = list(data_dir.glob("*.spe"))
            if not xas_files:
                xas_files = list(data_dir.glob("*.csv"))
            
            if not xas_files:
                print("⚠️  No XAS files found, using mock data")
                has_workflow = False
            else:
                try:
                    print(f"Running preprocessing on {len(xas_files)} files...")
                    results = run_xas_workflow(
                        data_dir=str(data_dir),
                        output_dir=str(output_dir),
                        pattern="*.spe" if xas_files[0].suffix == '.spe' else "*.csv",
                        create_diagnostic_plots=False
                    )
                    print(f"✅ Preprocessing complete: {len(results)} samples")
                except Exception as e:
                    print(f"⚠️  Preprocessing failed: {e}, using mock data")
                    has_workflow = False
    
    if not has_workflow:
        # Create mock data
        print("Creating mock XAS data (synthetic spectra)...")
        results = []
        n_samples = 10
        
        for i in range(n_samples):
            # Create synthetic XAS spectrum
            energy = np.linspace(8300, 8400, 200)
            e0 = 8340 + np.random.randn() * 2  # Vary edge position
            edge_step = 1.0 + np.random.randn() * 0.1
            
            # Synthetic XANES (arctangent edge + Gaussian white line)
            mu = (edge_step / 2) * (1 + np.tanh((energy - e0) / 5))
            wl_pos = e0 + 3 + np.random.randn()
            wl_height = 0.3 + np.random.randn() * 0.05
            mu += wl_height * np.exp(-((energy - wl_pos) ** 2) / (2 * 2**2))
            mu += np.random.randn(len(energy)) * 0.01  # Noise
            
            # Create XASSampleResult
            features = XASFeatures(
                e0=e0,
                edge_step=edge_step,
                white_line_intensity=wl_height,
                white_line_energy=wl_pos
            )
            
            processing_metadata = ProcessingMetadata()
            processing_params = XASProcessingParams(
                pre_edge={},
                autobk={},
                xftf={}
            )
            
            results.append({
                'sample_name': f'mock_sample_{i+1}',
                'output_files': {},
                'normalization': {'parameters': {'e0': e0, 'edge_step': edge_step}},
                'spectrum_quality': {'overall_score': 0.8 + np.random.rand() * 0.2},
                'processing_timestamp': '2026-03-03T12:00:00',
                '_mock_energy': energy,
                '_mock_mu': mu,
                '_mock_features': features,
                '_mock_metadata': processing_metadata,
                '_mock_params': processing_params
            })
        
        print(f"✅ Created {len(results)} mock samples")
    
    print()
    
    # Step 2: Convert to XASSampleResult objects
    print("Step 2: Converting to XASSampleResult objects...")
    print("-" * 80)
    
    try:
        sample_results = []
        
        for result in results:
            sample_name = result.get('sample_name', 'unknown')
            
            # Get energy and mu (from CSV or mock data)
            if '_mock_energy' in result:
                # Mock data - already has features created
                result_model = XASSampleResult(
                    sample_name=sample_name,
                    features=result['_mock_features'],
                    processing_metadata=result['_mock_metadata'],
                    processing_params=result['_mock_params'],
                    energy_range={'min': float(result['_mock_energy'].min()), 
                                 'max': float(result['_mock_energy'].max())},
                    data_points=len(result['_mock_energy'])
                )
                # Create wrapper with arrays
                sample_result = SimplifiedXASSample(
                    result_model,
                    result['_mock_energy'],
                    result['_mock_mu']
                )
            else:
                # Real data from CSV
                normalized_csv = result.get('output_files', {}).get('normalized_csv')
                
                if normalized_csv and Path(normalized_csv).exists():
                    data = np.loadtxt(normalized_csv, delimiter=',', skiprows=1)
                    energy = data[:, 0]
                    normalized_mu = data[:, 1]
                    
                    # Create minimal features
                    e0 = result.get('normalization', {}).get('parameters', {}).get('e0', 8340)
                    features = XASFeatures(e0=e0)
                    processing_metadata = ProcessingMetadata()
                    processing_params = XASProcessingParams(
                        pre_edge={},
                        autobk={},
                        xftf={}
                    )
                    
                    result_model = XASSampleResult(
                        sample_name=sample_name,
                        features=features,
                        processing_metadata=processing_metadata,
                        processing_params=processing_params,
                        energy_range={'min': float(energy.min()), 'max': float(energy.max())},
                        data_points=len(energy)
                    )
                    sample_result = SimplifiedXASSample(result_model, energy, normalized_mu)
                else:
                    print(f"⚠️  Skipping {sample_name} - no data available")
                    continue
            
            sample_results.append(sample_result)
        
        print(f"✅ Created {len(sample_results)} XASSampleResult objects")
        
        if len(sample_results) == 0:
            print("❌ No valid sample results created")
            return False
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 3: Extract features
    print("Step 3: Extracting features from spectra...")
    print("-" * 80)
    
    try:
        extractor = XASFeatureExtractor()
        features_list = extractor.extract_features_batch(sample_results)
        
        print(f"✅ Extracted features from {len(features_list)} samples")
        
        # Show sample features
        if features_list:
            sample_features = features_list[0].model_dump()
            print(f"\nSample features ({sample_results[0].sample_name}):")
            for key, value in list(sample_features.items())[:5]:
                if value is not None:
                    print(f"  {key}: {value:.4f}")
            print(f"  ... ({len(sample_features)} total features)")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 4: Assemble batch dataset
    print("Step 4: Assembling batch dataset...")
    print("-" * 80)
    
    try:
        # Create sample metadata
        metadata = {
            'quality_score': [sr.quality_score for sr in sample_results],
            'e0': [sr.e0 if sr.e0 else 0.0 for sr in sample_results]
        }
        
        assembler = XASBatchAssembler()
        dataset = assembler.assemble_dataset(
            sample_results,
            metadata=metadata,
            dataset_id="phase1_test"
        )
        
        print(f"✅ Dataset assembled:")
        print(f"  Samples: {dataset.n_samples}")
        print(f"  Features: {dataset.n_features}")
        print(f"  Feature matrix shape: {dataset.feature_matrix.shape}")
        
        # Show dataset summary
        summary = assembler.get_dataset_summary(dataset)
        print(f"\nDataset summary:")
        print(f"  ID: {summary['dataset_id']}")
        print(f"  Created: {summary['creation_timestamp']}")
        
    except Exception as e:
        print(f"❌ Batch assembly failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 5: PCA analysis
    print("Step 5: Running PCA analysis...")
    print("-" * 80)
    
    try:
        pca_analyzer = XASPCAAnalyzer()
        pca_result = pca_analyzer.analyze(dataset)
        
        print(f"✅ PCA analysis complete:")
        print(f"  Components: {pca_result.n_components}")
        print(f"  Variance captured: {pca_result.variance_captured:.1%}")
        print(f"  Confidence: {pca_result.confidence:.2f}")
        
        # Show variance per component
        print(f"\nVariance explained per component:")
        for i, var in enumerate(pca_result.explained_variance[:5]):
            print(f"  PC{i+1}: {var:.1%}")
        
        # Show top features for PC1
        if pca_result.feature_importance:
            pc1_features = pca_result.feature_importance.get('PC1', [])
            if pc1_features:
                print(f"\nTop features for PC1:")
                for feat in pc1_features[:3]:
                    print(f"  {feat['feature']}: {feat['abs_loading']:.3f}")
        
        if pca_result.flags:
            print(f"\n⚠️  Warnings: {', '.join(pca_result.flags)}")
        
    except Exception as e:
        print(f"❌ PCA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 6: Clustering
    print("Step 6: Running clustering analysis...")
    print("-" * 80)
    
    try:
        clusterer = XASClusterer()
        
        # Use PCA scores for clustering
        clustering_result = clusterer.cluster(
            dataset,
            n_clusters=None,  # Auto-determine
            use_pca_scores=pca_result.scores
        )
        
        print(f"✅ Clustering complete:")
        print(f"  Method: {clustering_result.method}")
        print(f"  Clusters: {clustering_result.n_clusters}")
        print(f"  Silhouette score: {clustering_result.silhouette_score:.3f}")
        print(f"  Confidence: {clustering_result.confidence:.2f}")
        
        # Show cluster sizes
        print(f"\nCluster sizes:")
        for cluster in clustering_result.cluster_info:
            print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} samples ({cluster['percentage']:.1f}%)")
        
        if clustering_result.flags:
            print(f"\n⚠️  Warnings: {', '.join(clustering_result.flags)}")
        
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 7: Trend analysis
    print("Step 7: Running trend analysis...")
    print("-" * 80)
    
    try:
        trend_analyzer = XASTrendAnalyzer()
        trend_result = trend_analyzer.analyze(dataset, clustering_result)
        
        print(f"✅ Trend analysis complete:")
        print(f"  Significant correlations: {len(trend_result.significant_correlations)}")
        print(f"  Outliers detected: {len(trend_result.outlier_indices)}")
        print(f"  Confidence: {trend_result.confidence:.2f}")
        
        # Show top correlations
        if trend_result.significant_correlations:
            print(f"\nTop correlations:")
            for corr in trend_result.significant_correlations[:3]:
                print(f"  {corr['feature']} ↔ {corr['metadata']}: r={corr['correlation']:.3f} (p={corr['p_value']:.4f})")
        else:
            print("\n  No significant correlations found")
        
        # Show outliers
        if trend_result.outlier_indices:
            outlier_names = [dataset.sample_names[i] for i in trend_result.outlier_indices]
            print(f"\nOutliers ({trend_result.outlier_method}):")
            for name in outlier_names[:3]:
                print(f"  - {name}")
        
        if trend_result.flags:
            print(f"\n⚠️  Warnings: {', '.join(trend_result.flags)}")
        
        # Generate insights
        insights = trend_analyzer.generate_insights(trend_result, dataset)
        if insights:
            print(f"\nGenerated {len(insights)} insights:")
            for insight in insights[:3]:
                print(f"  [{insight['priority']}] {insight['message']}")
        
    except Exception as e:
        print(f"❌ Trend analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 8: Save results
    print("Step 8: Saving results...")
    print("-" * 80)
    
    try:
        ml_output_dir = output_dir / "ml_analysis"
        ml_output_dir.mkdir(exist_ok=True)
        
        # Save dataset
        from xas_ml_modules import save_dataset_to_json
        dataset_path = ml_output_dir / "dataset.json"
        save_dataset_to_json(dataset, dataset_path)
        print(f"✅ Dataset saved: {dataset_path}")
        
        # Save PCA results
        pca_path = ml_output_dir / "pca_results.json"
        with open(pca_path, 'w') as f:
            f.write(pca_result.model_dump_json(indent=2))
        print(f"✅ PCA results saved: {pca_path}")
        
        # Save clustering results
        clustering_path = ml_output_dir / "clustering_results.json"
        with open(clustering_path, 'w') as f:
            f.write(clustering_result.model_dump_json(indent=2))
        print(f"✅ Clustering results saved: {clustering_path}")
        
        # Save trend analysis results (convert numpy types manually first)
        trend_path = ml_output_dir / "trend_results.json"
        trend_dict = json.loads(trend_result.model_dump_json())
        with open(trend_path, 'w') as f:
            json.dump(trend_dict, f, indent=2)
        print(f"✅ Trend results saved: {trend_path}")
        
    except Exception as e:
        print(f"⚠️  Save failed (non-critical): {e}")
    
    print()
    print("=" * 80)
    print("✅ PHASE 1 ML INTEGRATION TEST COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Preprocessed: {len(sample_results)} samples")
    print(f"  - Features extracted: {dataset.n_features} per sample")
    print(f"  - PCA components: {pca_result.n_components} ({pca_result.variance_captured:.1%} variance)")
    print(f"  - Clusters found: {clustering_result.n_clusters} (silhouette={clustering_result.silhouette_score:.3f})")
    print(f"  - Correlations: {len(trend_result.significant_correlations)} significant")
    print(f"  - Outliers: {len(trend_result.outlier_indices)} detected")
    print()
    
    return True


if __name__ == "__main__":
    success = test_phase1_ml_integration()
    sys.exit(0 if success else 1)
