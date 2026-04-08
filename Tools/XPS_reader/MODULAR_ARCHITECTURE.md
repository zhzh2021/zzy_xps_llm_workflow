# XPS_reader Modular Architecture

## Overview
The XPS_reader focuses on the standard spectra pipeline only. Upstream components (workflow_orchestrator.py, LangGraph triage/gate nodes, etc.) decide whether a file belongs in the standard route and ensure quality approvals. Once data reaches reader_main.py, the reader performs four responsibilities:

1. Batch import spectra/regions from supported raw formats.
2. Apply the configured energy calibration.
3. Optionally smooth spectra (disabled by default, controlled via YAML settings).
4. Extract regions and export aggregated CSV files (one combined file per region).

Triage/routing and quality gate logic are documented here for reference but should run before invoking the reader.

## Module Structure

```
XPS_reader/
??? reader_main.py                      # Standard spectra workflow (import?calibrate?smooth?export)
??? calibration.py
??? smooth_spectrum.py           # Optional smoothing utilities (off by default)
??? region_extraction.py         # Region extraction logic
??? spectrum_import.py           # File import/parsing
??? csv_export.py                # CSV export functionality
??? core/
    ??? data_structures.py       # Spectrum, XPSMetadata classes
        ??? parsers/             # Parsers for supported file formats
```

Adjacently maintained modules (invoked before reader):
- `enhanced_triage_fixed.py` ? triage/routing helper used by the orchestrator.
- `quality_gatekeeper.py` ? shared QC logic (LangGraph node + optional reader logging).


### 2. **Quality Gatekeeper Module** (`quality_gatekeeper.py`)

**Purpose**: Validate data quality and flag problematic spectra

**Key Classes**:
- `QualityGatekeeper`: Main validation system
- `QualityMetrics`: Dataclass with all quality metrics
- `QualityFlag`: Enum (EXCELLENT, GOOD, ACCEPTABLE, POOR, FAILED)

**Key Functions**:
```python
validate_spectrum(spectrum) -> QualityMetrics
batch_validate(spectra) -> Dict[str, QualityMetrics]
validate_spectrum_quality(spectrum, config) -> (bool, QualityMetrics)  # Convenience
```

**Quality Thresholds** (configurable in project_setting. yaml):
# this is high level quality check, different from QC step


**Metrics Calculated**:
- XPS SNR (signal-to-noise ratio)
- Intensity statistics (max, mean, std, min)
- Relative noise
- Energy range and resolution
- HR vs survey scan detection
- Suitability for peak fitting

**Integration in reader_main.py**:
```python
# Line 302-311
print("Validating data quality...")
quality_results = self.gatekeeper.batch_validate(spectra)

poor_quality = [name for name, metrics in quality_results.items() 
               if metrics.quality_flag in [QualityFlag.FAILED, QualityFlag.POOR]]
if poor_quality:
    print(f"⚠️  Warning: {len(poor_quality)} spectrum(a) with poor quality:")
    for name in poor_quality:
        metrics = quality_results[name]
        print(f"   - {name}: SNR={metrics.xps_snr:.2f}, {', '.join(metrics.issues[:2])}")
```

**Output Example**:
```
⚠️  Warning: 1 spectrum(a) with poor quality:
   - sample_002_C1s: SNR=2.15, Low SNR: 2.15, High relative noise: 35.2%
```

---

## Workflow Integration

The reader executes a deterministic standard-spectra pipeline. Upstream tooling decides
when to call it (e.g., only after triage routes a file to the standard path).

1. `BatchConverter.convert_file(raw_file)` orchestrates the steps for a single file.
2. `_convert_standard_file()` drives the per-file stages:
   - `SpectrumImporter` loads all spectra for the file.
   - `EnergyCalibrator` applies the configured reference shift.
   - Optional smoothing (controlled via `processing.smoothing` in YAML).
   - `RegionExtractor` slices each requested region.
   - `CSVExporter` writes region CSVs; aggregation mode merges identical regions into one
     combined file.
3. Batch mode simply repeats the same pipeline for every file and aggregates per region.

Map routing and quality gating are no longer decision points inside `reader_main.py`?that logic
sits with the workflow orchestrator. If a map slips through (e.g., manual invocation), the
reader logs the triage recommendation and skips it instead of attempting conversion.

## Import Structure in reader_main.py

```python
# Line 16-20
from enhanced_triage_fixed import (
    EnhancedXPSDataTriage, 
    XPSDataType, 
    should_route_to_mapper
)

from quality_gatekeeper import (
    QualityGatekeeper, 
    QualityFlag, 
    validate_spectrum_quality
)
```

**Clean separation**: Each module is self-contained and can be used independently.

---

## Benefits of Modular Architecture

### 1. **Reusability**
- Triage module can be imported by AI agent router
- Quality gatekeeper can be used in other tools (fitter, quantifier)
- No code duplication

### 2. **Testability**
- Each module has its own test suite
- Mock data easy to create
- Isolated unit tests

### 3. **Maintainability**
- Single responsibility per module
- Easy to update thresholds/logic
- Clear dependencies

### 4. **AI Agent Integration**
- `enhanced_triage_fixed.py` → Used by `llm_manager/triage_router.py`
- `quality_gatekeeper.py` → Used by `llm_manager/Agentstate.py`
- Seamless integration with LangGraph workflow

---

## Configuration

All modules use the same YAML config:

```yaml
# project_setting.yaml

processing:
  auto_detect_regions: true
  common_grid_step: 0.1

energy_calibration:
  enable: true
  reference_region: "C1s"
  target_binding_energy_ev: 284.8

quality_thresholds:
  min_snr_excellent: 10.0
  min_snr_good: 5.0
  min_snr_acceptable: 3.0
  hr_resolution_threshold: 5.0

file_formats:
  supported_extensions:
    - .spe
    - .csv
    - .vms
    - .vgd
```

---

## Testing

### Test Files Created
1. `test_quality_gatekeeper.py` - All 5 tests passing ✅
2. `test_agent_state.py` - All 6 tests passing ✅
3. `test_triage_integration.py` - Integration tests ✅

### Test Coverage
- ✅ Triage detection (95% confidence on C1s maps)
- ✅ Quality validation (SNR thresholds)
- ✅ HR vs survey detection
- ✅ Batch processing
- ✅ CSV export format
- ✅ AI agent state integration

---

## Usage Examples

### Standalone Triage
```python
from enhanced_triage_fixed import should_route_to_mapper

file_path = Path("data/map_file.csv")
should_route, result = should_route_to_mapper(file_path)

if should_route:
    print(f"Route to mapper: {result['data_type'].value}")
    print(f"Confidence: {result['confidence']:.0%}")
```

### Standalone Quality Check
```python
from quality_gatekeeper import validate_spectrum_quality

passes, metrics = validate_spectrum_quality(spectrum)

if not passes:
    print(f"Poor quality: {metrics.quality_flag.value}")
    print(f"Issues: {metrics.issues}")
```

### Full Workflow
```python
# Already integrated in reader_main.py BatchConverter class
converter = BatchConverter(region_defs, config)
results = converter.batch_convert(input_dir, output_dir)
```

---

## Future Enhancements

### Potential Additions
1. **Drift Detector Module** - Monitor energy calibration shifts
2. **Outlier Analyst Module** - Statistical anomaly detection
3. **Peak Quality Scorer** - Assess individual peak quality
4. **Batch QC Reporter** - Generate quality control reports

### Already Planned (in llm_manager/Agentstate.py)
- `detect_drift()` - C1s calibration monitoring
- `flag_outlier()` - Z-score > 3 detection
- `mark_step_complete()` - Pipeline progress tracking

---

## Module Status

| Module | Status | Integration | Tests |
|--------|--------|-------------|-------|
| enhanced_triage_fixed.py | ✅ Complete | reader_main.py, triage_router.py | ✅ Passing |
| quality_gatekeeper.py | ✅ Complete | reader_main.py, Agentstate.py | ✅ Passing |
| calibration.py | ✅ Existing | reader_main.py | ⚠️ Manual |
| region_extraction.py | ✅ Existing | reader_main.py | ⚠️ Manual |
| spectrum_import.py | ✅ Existing | reader_main.py | ⚠️ Manual |
| csv_export.py | ✅ Existing | reader_main.py | ⚠️ Manual |

---

## Summary

- The reader owns the standard spectra pipeline (import ? calibrate ? optional smoothing ? region export).
- Workflow orchestration/triage/quality gates sit outside this module; the reader simply trusts the route decisions.
- Map detections are treated as skips with clear logging so only spectra reach aggregation/export.
- Modular components (importer, calibrator, smoother, extractor, exporter) remain reusable and testable in isolation.
