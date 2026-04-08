**Stage 06 Deep Learning**
Scope, features, and user guidance for the APS_XAS deep learning module.

**Scope**
This stage trains and evaluates ML/DL models on:
1. Feature datasets from stage 03 (`extracted_features` JSONs).
2. Normalized spectra from stage 02 (`normalized_data` CSVs).

It saves datasets, models, plots, and reports under:
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\06_deep_learning`

**Key Features**
1. Config-driven dataset build, model training, and evaluation.
2. Supports regression and classification.
3. Standardization, optional PCA, and optional target scaling.
4. Class weighting and optional focal loss for imbalanced classification.
5. CNNs with dilation and optional global average pooling.
6. Residual diagnostics for regression.
7. Feature importance for RF regression.
8. Prediction for new samples using either features or spectra inputs.

**Entry Points**
1. Dataset builder: `9.Dataset_standardization.py`
2. MLP regression: `10-1-1Machine_learning_modelingMLP-R.py`
3. MLP classification: `10-1-2Machine_learning_modelingMLP-C.py`
4. CNN regression: `10-2-1Machine_learning_modelingCNN-R.py`
5. CNN classification: `10-2-2Machine_learning_modelingCNN-C.py`
6. RF regression: `10-3-Machine_learning_modelingRF-R.py`
7. Prediction: `11.Prediction.py`
8. Evaluation: `12.Model_performance_evaluation.py`

**Configuration**
All settings live in:
`C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_config\xas_ml_settings.yaml`

Key sections under `deep_learning`:
1. `dataset`
2. `models`

**Dataset Inputs**
1. Features (default):
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\03_feature_extraction\extracted_features`
2. Spectra (default):
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\02_analyzed_data\normalized_data`

**Recommended Usage**
1. Build dataset splits.
```powershell
python C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_deep_learning\9.Dataset_standardization.py
```

2. Train a model.
```powershell
python C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_deep_learning\10-1-1Machine_learning_modelingMLP-R.py
```

3. Run prediction on new data.
```powershell
python C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_deep_learning\11.Prediction.py --model <model_path> --features-dir <features_dir>
```

4. Evaluate predictions (optional).
```powershell
python C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_deep_learning\12.Model_performance_evaluation.py --task regression --y-true <y_true.csv> --y-pred <y_pred.csv>
```

**Outputs**
Datasets:
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\06_deep_learning\datasets`
1. `splits_*.npz`
2. `dataset_summary_*.json`
3. `metadata_*.csv`
4. `label_distribution_*.json`
5. `config_snapshot_*.json`
6. `scaler.pkl` and `pca.pkl` (if enabled)

Models:
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\06_deep_learning\models`

Plots:
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\06_deep_learning\plots`

Reports:
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\06_deep_learning\reports`

Logs:
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\06_deep_learning\logs`

**Model-Specific Guidance**
MLP Regression:
1. Best for feature datasets with nonlinear structure.
2. Enable `dataset.target_standardize: true` for wide‑range targets.

MLP Classification:
1. Use `use_class_weights: true` for imbalanced classes.
2. Optionally enable `use_focal_loss: true` for hard minority classes.

CNN Regression/Classification:
1. Use only with spectra datasets.
2. Consider `dilations` to capture pre‑edge/white‑line shifts.
3. Use `use_global_avg_pool: true` to reduce overfitting.

RF Regression:
1. Strong baseline for feature datasets.
2. Use feature importance to understand drivers.

**Typical Config Edits**
Set dataset source and label:
```yaml
deep_learning:
  dataset:
    source: "features"
    label_source: "metadata"
    label_column: "ligand_type"
```

Enable CNN on spectra:
```yaml
deep_learning:
  dataset:
    source: "spectra"
  models:
    cnn_regression:
      use_global_avg_pool: true
      dilations: [1, 2]
```

Enable focal loss:
```yaml
deep_learning:
  models:
    mlp_classification:
      use_focal_loss: true
      focal_gamma: 2.0
      focal_alpha: 0.25
```

**Troubleshooting**
1. `No split files found`:
Run `9.Dataset_standardization.py` first.
2. `CNN expects spectra`:
Make sure `dataset.source: spectra` and you built splits from spectra.
3. `Too few samples`:
Reduce model size or switch to RF baseline.
4. `TensorFlow missing`:
Install with `pip install tensorflow`.

**Notes**
1. All scripts accept `--config` to point to a custom YAML file.
2. The latest split file is auto-selected if `--splits` is not provided.
