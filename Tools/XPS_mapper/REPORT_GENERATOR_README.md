# Automated Analysis Report Generator

## Overview

The XPS_map analysis workflow now includes an **automated report generator** that creates comprehensive, publication-ready HTML and PDF reports documenting all analysis results.

## Features

### Report Contents

1. **Dataset Overview**
   - File information and region
   - Map dimensions (nx × ny pixels)
   - Energy range and number of points
   - Preprocessing applied

2. **PCA Analysis**
   - Number of components extracted
   - Explained variance table and scree plot
   - PC loadings (spectral signatures)
   - PC score maps (spatial distribution)

3. **MCR-ALS Component Resolution**
   - Number of pure components resolved
   - Convergence diagnostics (iterations, reconstruction error)
   - Resolved spectral signatures
   - Concentration/abundance maps

4. **Clustering Analysis**
   - Number of clusters and silhouette score
   - Spatial cluster map
   - Mean spectra by cluster
   - Cluster size distribution

5. **Quantification Results (NNLS/LLSF)**
   - Reference components used
   - Average composition across map
   - Concentration maps per component

6. **Pixel-Level Diagnostics**
   - Mean squared error (MSE) map
   - Energy shift map
   - Problem pixel flagging

### Output Formats

- **HTML**: Self-contained, interactive, browser-viewable
- **PDF**: High-resolution, print-ready for publications

## Usage

### Basic Usage

```python
from XPS_map import generate_analysis_report_from_outputs

# After processing your map
report_path = generate_analysis_report_from_outputs(
    parsed=hmap,           # HyperspectralMap object
    outputs=outputs,       # Dict of analysis results
    output_dir=output_dir, # Where to save report
    format="html",         # or "pdf"
    include_diagnostics=True
)
```

### Test Script

Run the test script to see the report generator in action:

```bash
# Generate HTML report with synthetic data
python test_report_generator.py --format html --nx 20 --ny 20

# Generate PDF report (requires weasyprint)
python test_report_generator.py --format pdf --nx 20 --ny 20

# Use real data
python test_report_generator.py --file path/to/map.txt --format html
```

### Command-Line Options

```
--file FILE       Path to real hyperspectral map file (optional)
--format FORMAT   Report format: html or pdf (default: html)
--nx NX           Synthetic map X size (default: 20)
--ny NY           Synthetic map Y size (default: 20)
--output DIR      Output directory
--no-mcr          Skip MCR-ALS (faster)
--verbose         Verbose logging
```

## Integration with XPS_map Workflow

The report generator is automatically available in the XPS_map workflow:

```python
from XPS_map import (
    parse_map_with_config,
    process_hyperspectral,
    generate_analysis_report_from_outputs
)

# Parse and process map
hmap = parse_map_with_config("path/to/map.txt")
outputs = process_hyperspectral(
    hmap=hmap,
    init_peaks=[(284.8, 0.8), (286.5, 0.9)],
    do_pca=True,
    n_pca=3
)

# Generate report
report_path = generate_analysis_report_from_outputs(
    parsed=hmap,
    outputs=outputs,
    output_dir=Path("05_map_data/analysis"),
    format="html"
)

print(f"Report saved to: {report_path}")
```

## Requirements

### HTML Reports (Built-in)
- No additional dependencies
- Works out of the box

### PDF Reports (Optional)
Install WeasyPrint for PDF generation:

```bash
pip install weasyprint
```

**Note**: WeasyPrint requires GTK+ on Windows. If PDF generation fails, use HTML format or install GTK+ from: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases

## Output Structure

```
output_dir/
├── analysis_report.html (or .pdf)
└── plots/
    ├── pca_scree.png
    ├── pca_loadings.png
    ├── pca_score_map_0.png
    ├── pca_score_map_1.png
    ├── mcr_spectra.png
    ├── mcr_concentration_0.png
    ├── cluster_map.png
    ├── cluster_spectra.png
    ├── mse_map.png
    └── shift_map.png
```

## Customization

### Adding Custom Sections

You can extend the report by creating custom sections:

```python
from report_generator import AnalysisReport

report = AnalysisReport(title="Custom XPS Analysis")

# Add dataset overview
section = report.add_section("Dataset Overview")
section.add_text("Region: C1s")
section.add_metric("Total Pixels", 400)

# Add custom table
section.add_table({
    "Component": ["C-C", "C-O", "C=O"],
    "Percentage": [45.2, 30.1, 24.7]
}, caption="Average Composition")

# Add figures
section.add_figure("path/to/plot.png", caption="PCA Score Map")

# Generate report
report.to_html("output/custom_report.html")
```

### Styling

The report uses professional CSS styling with:
- Clean, modern typography
- Color-coded sections
- Responsive images
- Print-friendly layouts (PDF)

Modify `report_generator.py::_get_css()` to customize styles.

## Examples

### Example 1: Quick HTML Report

```python
# Minimal example
from XPS_map import generate_analysis_report_from_outputs

report_path = generate_analysis_report_from_outputs(
    parsed=my_map,
    outputs=my_results,
    output_dir=Path("reports"),
    format="html"
)
```

### Example 2: PDF Report with MCR

```python
# Include MCR results
outputs["mcr"] = {
    "n_components": 3,
    "reconstruction_error": 2.5,
    "n_iterations": 147,
    "converged": True,
    "lack_of_fit": 0.018
}

report_path = generate_analysis_report_from_outputs(
    parsed=my_map,
    outputs=outputs,
    output_dir=Path("reports"),
    format="pdf"
)
```

### Example 3: Batch Processing with Reports

```python
from pathlib import Path

map_files = list(Path("raw_data").glob("*.txt"))

for map_file in map_files:
    hmap = parse_map_with_config(str(map_file))
    outputs = process_hyperspectral(hmap, ...)
    
    # Generate report for each map
    report_dir = Path(f"reports/{map_file.stem}")
    generate_analysis_report_from_outputs(
        parsed=hmap,
        outputs=outputs,
        output_dir=report_dir,
        format="html"
    )
```

## Troubleshooting

### Issue: "cannot import name 'generate_analysis_report_from_outputs'"
**Solution**: Import from `XPS_map`, not `report_generator`:
```python
from XPS_map import generate_analysis_report_from_outputs
```

### Issue: PDF generation fails
**Solution**: 
1. Check WeasyPrint installation: `pip install weasyprint`
2. On Windows, install GTK+: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
3. Alternatively, use HTML format: `format="html"`

### Issue: Missing plots in report
**Solution**: Ensure plots are generated and saved before calling report generator. Check that `plot_paths` dictionary contains valid file paths.

### Issue: Report is blank or incomplete
**Solution**: Verify that `outputs` dictionary contains the expected keys (`pca`, `mcr`, `cluster_labels`, etc.). Enable verbose logging to see warnings.

## API Reference

### `generate_analysis_report_from_outputs()`

Generate comprehensive analysis report from processing outputs.

**Parameters:**
- `parsed` (HyperspectralMap): Input map object
- `outputs` (Dict): Analysis results dictionary
- `output_dir` (Path): Directory to save report
- `format` (str): 'html' or 'pdf' (default: 'html')
- `include_diagnostics` (bool): Include pixel-level diagnostics (default: True)

**Returns:**
- `Path`: Path to generated report file, or None if failed

**Example:**
```python
report_path = generate_analysis_report_from_outputs(
    parsed=hmap,
    outputs=results,
    output_dir=Path("output"),
    format="html",
    include_diagnostics=True
)
```

### `AnalysisReport` Class

Low-level report builder for custom reports.

**Methods:**
- `add_section(title)`: Create new section
- `add_metadata(key, value)`: Add custom metadata
- `to_html(path)`: Generate HTML report
- `to_pdf(path)`: Generate PDF report

**Example:**
```python
from report_generator import AnalysisReport

report = AnalysisReport(title="My Analysis")
section = report.add_section("Results")
section.add_text("Analysis complete")
section.add_metric("SNR", 25.3)
report.to_html("report.html")
```

## Contributing

To add new report sections or metrics:

1. Update `create_*_report_data()` helper functions in `report_generator.py`
2. Add section logic in `generate_comprehensive_report()`
3. Update tests in `test_report_generator.py`

## Version History

- **v1.0**: Initial release with HTML/PDF support, PCA, MCR, clustering, diagnostics

## References

- PCA implementation: scikit-learn
- MCR-ALS: pymcr package
- HTML/PDF rendering: WeasyPrint (optional)

## Support

For issues or questions:
1. Check this README
2. Review `test_report_generator.py` for working examples
3. Enable verbose logging: `setup_logger(verbose=True)`
