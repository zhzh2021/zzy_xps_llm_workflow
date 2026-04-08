"""
Analysis Report Generator for XPS Map Processing

Generates comprehensive HTML and PDF reports documenting:
- Dataset overview and quality metrics
- PCA analysis results
- MCR-ALS component resolution
- Clustering analysis
- Quantification results
- Pixel-level diagnostics
"""

from __future__ import annotations
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import json

try:
    from jinja2 import Template
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False
    Template = None

try:
    from weasyprint import HTML as WeasyHTML
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    # OSError can occur if system libraries (GTK, GObject) are missing
    WEASYPRINT_AVAILABLE = False
    WeasyHTML = None

# Alternative PDF generation (lighter weight, fewer dependencies)
try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except ImportError:
    XHTML2PDF_AVAILABLE = False
    pisa = None


@dataclass
class ReportSection:
    """Container for a report section."""
    title: str
    content: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_text(self, text: str, style: str = "normal"):
        """Add text paragraph."""
        self.content.append({"type": "text", "text": text, "style": style})
    
    def add_table(self, data: Dict[str, List], caption: Optional[str] = None):
        """Add data table."""
        self.content.append({"type": "table", "data": data, "caption": caption})
    
    def add_figure(self, path: Union[str, Path], caption: Optional[str] = None, width: str = "100%"):
        """Add figure from file path."""
        self.content.append({
            "type": "figure",
            "path": str(path),
            "caption": caption,
            "width": width
        })
    
    def add_metric(self, label: str, value: Any, unit: Optional[str] = None):
        """Add key metric (highlighted)."""
        value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
        if unit:
            value_str += f" {unit}"
        self.content.append({"type": "metric", "label": label, "value": value_str})


class AnalysisReport:
    """
    Generate comprehensive analysis reports for XPS map processing.
    
    Attributes:
        title: Report title
        sections: List of report sections
        metadata: Dictionary of metadata (timestamp, software version, etc.)
    """
    
    def __init__(self, title: str):
        self.title = title
        self.sections: List[ReportSection] = []
        self.metadata = {
            "generated": datetime.now().isoformat(),
            "software": "XPS_map v1.0",
            "framework": "SciAgent XPS Workflow Analyzer"
        }
    
    def add_section(self, title: str) -> ReportSection:
        """Create and add a new section."""
        section = ReportSection(title=title)
        self.sections.append(section)
        return section
    
    def add_metadata(self, key: str, value: Any):
        """Add custom metadata."""
        self.metadata[key] = value
    
    def to_html(self, output_path: Union[str, Path]) -> Path:
        """
        Generate HTML report.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self._generate_html()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def to_pdf(self, output_path: Union[str, Path]) -> Path:
        """
        Generate PDF report using available PDF library.
        
        Tries in order:
        1. WeasyPrint (best quality, requires system libraries)
        2. xhtml2pdf (lightweight alternative)
        3. Browser print (fallback - saves HTML with instructions)
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to generated PDF file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try WeasyPrint first (best quality)
        if WEASYPRINT_AVAILABLE:
            try:
                html_content = self._generate_html(for_pdf=True)
                WeasyHTML(string=html_content).write_pdf(output_path)
                return output_path
            except Exception as e:
                # Fall through to next method
                pass
        
        # Try xhtml2pdf as alternative
        if XHTML2PDF_AVAILABLE:
            try:
                html_content = self._generate_html(for_pdf=True)
                with open(output_path, "wb") as pdf_file:
                    pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
                if not pisa_status.err:
                    return output_path
            except Exception as e:
                # Fall through to next method
                pass
        
        # Fallback: Save HTML with print instructions
        raise ImportError(
            "PDF generation requires either:\n"
            "  1. WeasyPrint: pip install weasyprint\n"
            "     (Windows users: Requires GTK3+ runtime: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases)\n"
            "  2. xhtml2pdf: pip install xhtml2pdf\n"
            "  3. Or open the HTML report in a browser and use Print to PDF"
        )
    
    def _generate_html(self, for_pdf: bool = False) -> str:
        """Generate HTML content."""
        
        # CSS styling
        css = self._get_css(for_pdf)
        
        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{self.title}</title>",
            f"<style>{css}</style>",
            "</head>",
            "<body>",
            f"<div class='container'>",
            f"<header>",
            f"<h1>{self.title}</h1>",
            f"<div class='metadata'>",
            f"<p>Generated: {self.metadata['generated']}</p>",
            f"<p>Software: {self.metadata['software']}</p>",
            f"</div>",
            f"</header>",
        ]
        
        # Add sections
        for section in self.sections:
            html_parts.append(f"<section>")
            html_parts.append(f"<h2>{section.title}</h2>")
            
            for item in section.content:
                if item["type"] == "text":
                    style_class = f" class='{item['style']}'" if item['style'] != "normal" else ""
                    html_parts.append(f"<p{style_class}>{item['text']}</p>")
                
                elif item["type"] == "metric":
                    html_parts.append(
                        f"<div class='metric'>"
                        f"<span class='metric-label'>{item['label']}:</span> "
                        f"<span class='metric-value'>{item['value']}</span>"
                        f"</div>"
                    )
                
                elif item["type"] == "table":
                    html_parts.append(self._format_table(item["data"], item.get("caption")))
                
                elif item["type"] == "figure":
                    html_parts.append(self._format_figure(
                        item["path"], 
                        item.get("caption"),
                        item.get("width", "100%"),
                        for_pdf
                    ))
            
            html_parts.append("</section>")
        
        # Footer
        html_parts.extend([
            "<footer>",
            f"<p>Report generated by {self.metadata['framework']}</p>",
            "</footer>",
            "</div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _format_table(self, data: Dict[str, List], caption: Optional[str] = None) -> str:
        """Format data as HTML table."""
        html = ["<div class='table-container'>"]
        
        if caption:
            html.append(f"<p class='table-caption'>{caption}</p>")
        
        html.append("<table>")
        
        # Header
        html.append("<thead><tr>")
        for key in data.keys():
            html.append(f"<th>{key}</th>")
        html.append("</tr></thead>")
        
        # Body
        html.append("<tbody>")
        n_rows = len(next(iter(data.values())))
        for i in range(n_rows):
            html.append("<tr>")
            for key in data.keys():
                value = data[key][i]
                if isinstance(value, float):
                    value = f"{value:.3f}"
                html.append(f"<td>{value}</td>")
            html.append("</tr>")
        html.append("</tbody>")
        
        html.append("</table>")
        html.append("</div>")
        
        return "\n".join(html)
    
    def _format_figure(self, path: str, caption: Optional[str], width: str, for_pdf: bool) -> str:
        """Format figure as HTML."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            return f"<p class='error'>Figure not found: {path}</p>"
        
        html = ["<div class='figure'>"]
        
        # Embed image as base64 for PDF or use file path for HTML
        if for_pdf:
            # Embed as base64 for PDF generation
            with open(path_obj, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            ext = path_obj.suffix.lower()
            mime = 'image/png' if ext == '.png' else 'image/jpeg'
            html.append(f"<img src='data:{mime};base64,{img_data}' style='width:{width};' alt='{caption or 'Figure'}'>")
        else:
            # Use relative or absolute path for HTML
            html.append(f"<img src='file:///{path_obj.absolute()}' style='width:{width};' alt='{caption or 'Figure'}'>")
        
        if caption:
            html.append(f"<p class='figure-caption'>{caption}</p>")
        
        html.append("</div>")
        
        return "\n".join(html)
    
    def _get_css(self, for_pdf: bool = False) -> str:
        """Generate CSS styling."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        header {
            border-bottom: 3px solid #2c3e50;
            margin-bottom: 40px;
            padding-bottom: 20px;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 15px;
        }
        
        h2 {
            color: #34495e;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        
        .metadata {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .metadata p {
            margin: 5px 0;
        }
        
        section {
            margin-bottom: 40px;
        }
        
        p {
            margin-bottom: 15px;
            text-align: justify;
        }
        
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px 25px;
            background-color: #ecf0f1;
            border-left: 4px solid #3498db;
            border-radius: 3px;
        }
        
        .metric-label {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .metric-value {
            color: #3498db;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .table-container {
            margin: 20px 0;
            overflow-x: auto;
        }
        
        .table-caption {
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        
        th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        tr:hover {
            background-color: #ecf0f1;
        }
        
        .figure {
            margin: 30px 0;
            text-align: center;
        }
        
        .figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        
        .figure-caption {
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
            font-size: 0.95em;
        }
        
        .error {
            color: #e74c3c;
            font-weight: bold;
            padding: 10px;
            background-color: #fadbd8;
            border-left: 4px solid #e74c3c;
            margin: 10px 0;
        }
        
        footer {
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        """ + ("""
        /* PDF-specific styles */
        @page {
            size: A4;
            margin: 2cm;
        }
        
        .figure {
            page-break-inside: avoid;
        }
        
        section {
            page-break-inside: avoid;
        }
        """ if for_pdf else "")


def create_pca_report_data(pca_results: Dict) -> Dict[str, Any]:
    """
    Extract PCA data for reporting.
    
    Args:
        pca_results: Dictionary from PCA analysis
        
    Returns:
        Formatted data for report sections
    """
    n_components = len(pca_results.get('explained_variance', []))
    explained = pca_results.get('explained_variance', [])
    cumulative = np.cumsum(explained)
    
    table_data = {
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Variance (%)": [f"{v*100:.1f}" for v in explained],
        "Cumulative (%)": [f"{c*100:.1f}" for c in cumulative]
    }
    
    metrics = {
        "n_components": n_components,
        "total_variance_explained": float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        "table": table_data
    }
    
    return metrics


def create_mcr_report_data(mcr_results: Dict) -> Dict[str, Any]:
    """
    Extract MCR-ALS data for reporting.
    
    Args:
        mcr_results: Dictionary from MCR-ALS analysis
        
    Returns:
        Formatted data for report sections
    """
    return {
        "n_components": mcr_results.get('n_components', 0),
        "reconstruction_error": mcr_results.get('reconstruction_error', 0.0),
        "n_iterations": mcr_results.get('n_iterations', 0),
        "converged": mcr_results.get('converged', False),
        "lack_of_fit": mcr_results.get('lack_of_fit', 0.0)
    }


def create_cluster_report_data(cluster_results: Dict) -> Dict[str, Any]:
    """
    Extract clustering data for reporting.
    
    Args:
        cluster_results: Dictionary from clustering analysis
        
    Returns:
        Formatted data for report sections
    """
    n_clusters = cluster_results.get('n_clusters', 0)
    cluster_info = cluster_results.get('cluster_info', [])
    
    # Use actual cluster_info length if n_clusters is 0 but cluster_info exists
    if n_clusters == 0 and cluster_info:
        n_clusters = len(cluster_info)
    
    # Calculate total size for percentage calculation
    total_size = sum(c.get('size', 0) for c in cluster_info)
    if total_size == 0:
        total_size = 1  # Avoid division by zero
    
    table_data = {
        "Cluster": [f"Cluster {info.get('cluster', i)}" for i, info in enumerate(cluster_info)],
        "Size (pixels)": [info.get('size', 0) for info in cluster_info],
        "Percentage": [f"{info.get('size', 0) / total_size * 100:.1f}%" for info in cluster_info]
    }
    
    return {
        "n_clusters": n_clusters,
        "silhouette_score": cluster_results.get('silhouette_score', 0.0),
        "table": table_data
    }


def generate_comprehensive_report(
    output_dir: Path,
    hmap_metadata: Dict,
    pca_results: Optional[Dict] = None,
    mcr_results: Optional[Dict] = None,
    cluster_results: Optional[Dict] = None,
    nnls_results: Optional[Dict] = None,
    pixel_diagnostics: Optional[Dict] = None,
    plot_paths: Optional[Dict] = None,
    format: str = "html"
) -> Path:
    """
    Generate comprehensive analysis report with all available results.
    
    Args:
        output_dir: Output directory for report
        hmap_metadata: Dictionary with dataset metadata
        pca_results: PCA analysis results
        mcr_results: MCR-ALS results
        cluster_results: Clustering results
        nnls_results: NNLS quantification results
        pixel_diagnostics: Pixel-level fit diagnostics
        plot_paths: Dictionary of plot file paths
        format: 'html' or 'pdf'
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    region = hmap_metadata.get('region', 'Unknown')
    report = AnalysisReport(title=f"XPS Map Analysis Report: {region}")
    
    # Add metadata
    for key, value in hmap_metadata.items():
        report.add_metadata(key, value)
    
    # Section 1: Dataset Overview
    section = report.add_section("Dataset Overview")
    section.add_text(f"<b>Region:</b> {hmap_metadata.get('region', 'N/A')}", style="normal")
    section.add_text(
        f"<b>Dimensions:</b> {hmap_metadata.get('nx', 0)} × {hmap_metadata.get('ny', 0)} pixels "
        f"({hmap_metadata.get('total_pixels', 0)} total)",
        style="normal"
    )
    section.add_text(
        f"<b>Energy Range:</b> {hmap_metadata.get('energy_min', 0):.2f} - "
        f"{hmap_metadata.get('energy_max', 0):.2f} eV "
        f"({hmap_metadata.get('n_energy_points', 0)} points)",
        style="normal"
    )
    if hmap_metadata.get('avg_snr'):
        section.add_metric("Average SNR", hmap_metadata['avg_snr'])
    if hmap_metadata.get('preprocessing'):
        section.add_text(f"<b>Preprocessing:</b> {hmap_metadata['preprocessing']}")
    
    # Section 2: Pattern Recognition Entropy (PRE) Analysis
    if plot_paths and plot_paths.get('pre_map'):
        section = report.add_section("Pattern Recognition Entropy (PRE) Analysis")
        
        section.add_text(
            "<b>What PRE Represents:</b> The PRE map is a quantitative measure of local spectral variability "
            "or complexity across the surface. It reveals how statistically unique or common each pixel's "
            "spectrum is compared to all other spectra in the map.",
            style="normal"
        )
        
        section.add_text(
            "<b>High PRE Values</b> indicate pixels with spectra that are unique or complex "
            "relative to the majority of the map's spectra. This can result from:",
            style="normal"
        )
        section.add_text(
            "• A true, rare chemical phase or minority species<br>"
            "• High signal-to-noise variation (noise/artifacts)<br>"
            "• Pixels at phase boundaries or interfaces with mixed chemistry",
            style="normal"
        )
        
        section.add_text(
            "<b>Low PRE Values</b> indicate pixels with spectra that are highly common "
            "and statistically simple, representing the dominant or homogeneous chemical phases.",
            style="normal"
        )
        
        section.add_figure(plot_paths['pre_map'], 
                         caption="Pattern Recognition Entropy (PRE) Map - Quantitative measure of spectral complexity")
    
    # Section 3: PCA Analysis
    if pca_results:
        section = report.add_section("Principal Component Analysis")
        pca_data = create_pca_report_data(pca_results)
        
        section.add_metric("Components Extracted", pca_data['n_components'])
        section.add_metric("Total Variance Explained", f"{pca_data['total_variance_explained']*100:.1f}", "%")
        
        section.add_table(pca_data['table'], caption="Explained Variance by Component")
        
        if plot_paths and plot_paths.get('pca_scree'):
            section.add_figure(plot_paths['pca_scree'], caption="Scree Plot", width="80%")
        
        if plot_paths and plot_paths.get('pca_loadings'):
            section.add_figure(plot_paths['pca_loadings'], caption="PCA Loadings (Spectral Signatures)")
        
        for i in range(pca_data['n_components']):
            if plot_paths and plot_paths.get(f'pca_score_map_{i}'):
                section.add_figure(
                    plot_paths[f'pca_score_map_{i}'],
                    caption=f"PC{i+1} Score Map"
                )
    
    # Section 4: MCR-ALS Results
    if mcr_results:
        section = report.add_section("MCR-ALS Component Resolution")
        mcr_data = create_mcr_report_data(mcr_results)
        
        section.add_metric("Pure Components Resolved", mcr_data['n_components'])
        section.add_metric("Reconstruction Error", f"{mcr_data['reconstruction_error']:.2f}", "%")
        section.add_metric("Iterations to Convergence", mcr_data['n_iterations'])
        section.add_metric("Lack of Fit", f"{mcr_data.get('lack_of_fit', 0):.4f}")
        
        section.add_text(
            f"Convergence status: {'✓ Converged' if mcr_data['converged'] else '✗ Did not converge'}"
        )
        
        if plot_paths and plot_paths.get('mcr_spectra'):
            section.add_figure(plot_paths['mcr_spectra'], caption="MCR-ALS Resolved Spectra")
        
        if plot_paths and plot_paths.get('mcr_quality'):
            section.add_figure(plot_paths['mcr_quality'], caption="MCR-ALS Quality Metrics")
        
        for i in range(mcr_data['n_components']):
            if plot_paths and plot_paths.get(f'mcr_concentration_{i}'):
                section.add_figure(
                    plot_paths[f'mcr_concentration_{i}'],
                    caption=f"Component {i+1} Concentration Map"
                )
    
    # Section 5: Clustering Analysis
    if cluster_results:
        section = report.add_section("Cluster Analysis - Spatial Distribution of Chemical domains")
        cluster_data = create_cluster_report_data(cluster_results)
        
        # Add description of what cluster map represents
        section.add_text(
            "The cluster map below represents the spatial distribution of dominant spectral features, "
            "corresponding to different chemical domains, within the scan mapping area. Each cluster represents "
            "a distinct chemical environment identified through multivariate analysis of the XPS spectra.",
            style="normal"
        )
        
        section.add_metric("Number of Clusters", cluster_data['n_clusters'])
        section.add_metric("Silhouette Score", f"{cluster_data['silhouette_score']:.3f}")
        
        section.add_table(cluster_data['table'], caption="Cluster Size Distribution")
        
        if plot_paths and plot_paths.get('cluster_spectra'):
            section.add_figure(plot_paths['cluster_spectra'], 
                             caption="Representative Spectra for Each Chemical State (Cluster)")
        
        if plot_paths and plot_paths.get('cluster_map'):
            section.add_figure(plot_paths['cluster_map'], 
                             caption="Spatial Distribution of Chemical States (Cluster Map) - Physical coordinates shown in μm when available")
    
    # Section 6: Deconvolved Species Concentration Maps
    if mcr_results and plot_paths:
        # Add individual species concentration maps
        if plot_paths.get('individual_species_maps'):
            section = report.add_section("Individual Species Concentration Maps")
            section.add_text(
                "Quantitative concentration distribution for each deconvolved chemical species. "
                "Each species map shows the spatial distribution of that particular chemical state across the mapping area.",
                style="normal"
            )
            section.add_figure(
                plot_paths['individual_species_maps'],
                caption="Individual Species Concentration Maps - Spatial distribution of each chemical species (%)"
            )
        
        # Add combined concentration overlay map
        if plot_paths.get('combined_concentration_map'):
            section = report.add_section("Combined RGB Color Mixing Map")
            section.add_text(
                "RGB Color Mixing Concentration Map - Shows all chemical species simultaneously. "
                "Pure colors indicate single-phase regions, mixed colors reveal co-localization and interfacial regions. "
                "This visualization allows identification of phase boundaries and chemical heterogeneity.",
                style="normal"
            )
            section.add_figure(
                plot_paths['combined_concentration_map'],
                caption="Combined Concentration Map (RGB Color Mixing) - Mixed colors show co-localization of species"
            )
    
    # Section 7: NNLS Quantification
    if nnls_results:
        section = report.add_section("NNLS Quantification")
        
        if nnls_results.get('average_composition'):
            comp_data = nnls_results['average_composition']
            table = {
                "Component": list(comp_data.keys()),
                "Average (%)": [f"{v*100:.1f}" for v in comp_data.values()]
            }
            section.add_table(table, caption="Average Composition Across Map")
        
        for comp_name in nnls_results.get('components', []):
            if plot_paths and plot_paths.get(f'nnls_{comp_name}'):
                section.add_figure(
                    plot_paths[f'nnls_{comp_name}'],
                    caption=f"{comp_name} Concentration Map"
                )
    
    # Section 7: Pixel-Level Diagnostics
    if pixel_diagnostics:
        section = report.add_section("Pixel-Level Fitting Diagnostics")
        
        if pixel_diagnostics.get('mean_mse'):
            section.add_metric("Mean MSE", f"{pixel_diagnostics['mean_mse']:.4f}")
        if pixel_diagnostics.get('problematic_pixels'):
            section.add_metric("Problematic Pixels", pixel_diagnostics['problematic_pixels'])
        
        if plot_paths and plot_paths.get('mse_map'):
            section.add_figure(plot_paths['mse_map'], caption="Mean Squared Error Map")
        
        if plot_paths and plot_paths.get('shift_map'):
            section.add_figure(plot_paths['shift_map'], caption="Energy Shift Map")
    
    # Generate report in dedicated "report" subfolder with dataset-specific filename
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Use dataset filename from metadata if available
    file_basename = hmap_metadata.get('file', 'analysis_report')
    if file_basename != 'N/A' and file_basename:
        # Remove extension and sanitize filename (Path is already imported at top)
        import os
        file_basename = os.path.splitext(os.path.basename(file_basename))[0]
    else:
        file_basename = 'analysis_report'
    
    if format == "html":
        output_file = report_dir / f"{file_basename}_report.html"
        return report.to_html(output_file)
    elif format == "pdf":
        output_file = report_dir / f"{file_basename}_report.pdf"
        return report.to_pdf(output_file)
    else:
        raise ValueError(f"Unsupported format: {format}")
