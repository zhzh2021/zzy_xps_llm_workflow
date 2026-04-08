# XPS Map Processor - Architecture Flowchart

## Main Processing Pipeline

```mermaid
flowchart TD
    Start([XPS Data File]) --> Parse[Parse Input<br/>parser.py]
    Parse --> Type{Data Type?}
    
    Type -->|2D Map| Process2D[2D Processing<br/>case1_2d_processing.py]
    Type -->|Hyperspectral| ProcessHS[Hyperspectral Processing<br/>case2_hyperspectral_processing.py]
    
    Process2D --> Denoise[Denoise & Segment]
    Denoise --> ROI[ROI Analysis]
    ROI --> Plot2D[2D Plots<br/>map_plots_basic.py]
    Plot2D --> Save2D[Save Results]
    
    ProcessHS --> PreProcess[Preprocessing]
    PreProcess --> Chemo[Chemometrics Analysis]
    Chemo --> Cluster[Clustering]
    Cluster --> Quant[Quantification]
    Quant --> PlotHS[Hyperspectral Plots]
    PlotHS --> SaveHS[Save Results]
    
    Save2D --> End([Output Files])
    SaveHS --> End
    
    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style Process2D fill:#e1e5ff
    style ProcessHS fill:#ffe1f5
```

## Hyperspectral Processing Detail

```mermaid
flowchart TD
    Start([Hyperspectral Cube<br/>ny × nx × nE]) --> PRE[Compute PRE Image<br/>chemometrics_utils.py]
    
    PRE --> Quality{Check<br/>Variability}
    Quality -->|Low| Skip[Skip Chemometrics<br/>Simple Average]
    Quality -->|High| Mask[Mask Low Counts<br/>chemometrics_utils.py]
    
    Mask --> Charge{Need Charge<br/>Correction?}
    Charge -->|Yes| Align[Charge Align<br/>chemometrics_utils.py]
    Charge -->|No| MCR
    Align --> MCR
    
    MCR[MCR/NMF Decomposition<br/>chemometrics_utils.py] --> Fit[Peak Fitting<br/>mcr_fitting.py]
    Fit --> Deconv[Deconvolve Species<br/>mcr_fitting.py]
    Deconv --> Quant[Quantification<br/>mcr_fitting.py]
    
    MCR --> PCA[PCA Analysis<br/>XPS_map.py]
    PCA --> KMeans[K-Means Clustering<br/>XPS_map.py]
    KMeans --> ClusterVal[Cluster Validation<br/>cluster_validation.py]
    
    Quant --> PlotMCR[Plot MCR Results<br/>mcr_fitting.py]
    ClusterVal --> PlotCluster[Plot Clusters<br/>cluster_plots.py]
    
    PlotMCR --> Report[Generate Report<br/>report_generator.py]
    PlotCluster --> Report
    Skip --> Report
    
    Report --> End([Output:<br/>CSV, PNG, HTML])
    
    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style MCR fill:#fff5e1
    style Fit fill:#fff5e1
    style Deconv fill:#fff5e1
    style Quant fill:#fff5e1
```

## Module Dependencies

```mermaid
graph LR
    XPS_map[XPS_map.py<br/>Main Orchestrator]
    
    XPS_map --> Parser[parser.py]
    XPS_map --> Case1[case1_2d_processing.py]
    XPS_map --> Case2[case2_hyperspectral_processing.py]
    XPS_map --> ChemoUtils[chemometrics_utils.py]
    XPS_map --> MCRFit[mcr_fitting.py]
    XPS_map --> ClusterVal[cluster_validation.py]
    
    Case2 --> ChemoUtils
    MCRFit --> ChemoUtils
    MCRFit --> PeakFit[XPS_peakfitting_V2.py]
    
    XPS_map --> ChemoPlots[chemometrics_plots.py]
    XPS_map --> MapPlots[map_plots_basic.py]
    XPS_map --> ClusterPlots[cluster_plots.py]
    XPS_map --> Report[report_generator.py]
    
    ChemoPlots -.->|uses| ChemoUtils
    ClusterPlots -.->|uses| ChemoUtils
    Report -.->|uses| ChemoPlots
    Report -.->|uses| ClusterPlots
    
    style XPS_map fill:#ff9999,stroke:#333,stroke-width:4px
    style ChemoUtils fill:#99ccff
    style MCRFit fill:#ffcc99
    style ChemoPlots fill:#99ff99
    style ClusterPlots fill:#99ff99
    style MapPlots fill:#99ff99
    style Report fill:#ff99ff
```

## MCR Fitting and Quantification Pipeline

```mermaid
flowchart TD
    Start([MCR Components<br/>Component Spectra + Concentration Maps]) --> Fit[Template-Based<br/>Peak Fitting<br/>fit_mcr_components]
    
    Fit --> FitResult{Fit<br/>Success?}
    FitResult -->|No| Fallback[Fallback to<br/>Peak Matching]
    FitResult -->|Yes| Deconv[Deconvolve to<br/>Individual Species<br/>deconvolve_mcr_to_species]
    Fallback --> Save
    
    Deconv --> Merge[Merge Duplicate<br/>Species Across<br/>Components]
    Merge --> Scale[Scale Maps to<br/>Atomic Percentages]
    
    Scale --> SaveCSV[Save CSV Files<br/>save_mcr_fitting_results<br/>save_quantitative_concentration_maps]
    SaveCSV --> PlotBar[Plot Atomic %<br/>Bar & Pie Charts<br/>plot_atomic_percentages]
    PlotBar --> PlotMaps[Plot Concentration<br/>Maps<br/>plot_quantitative_concentration_maps]
    PlotMaps --> PlotCombined[Plot Combined<br/>RGB Map<br/>plot_combined_concentration_maps]
    PlotCombined --> PlotIndiv[Plot Individual<br/>Species Maps]
    
    PlotIndiv --> Save[Save All Outputs]
    
    Save --> End([Outputs:<br/>CSV + PNG Files])
    
    style Start fill:#e1f5e1
    style Deconv fill:#fff5e1
    style Merge fill:#fff5e1
    style Scale fill:#fff5e1
    style End fill:#ffe1e1
```

## Cluster Analysis Pipeline

```mermaid
flowchart TD
    Start([Hyperspectral Cube]) --> Normalize[Normalize Spectra<br/>L1 or L2]
    
    Normalize --> PCA[PCA Decomposition<br/>pca_cluster_preselect]
    PCA --> Store[Store Original<br/>Unnormalized Data]
    
    Store --> KMeans[K-Means on<br/>PCA Scores]
    KMeans --> CalcMean[Calculate Mean<br/>Spectra per Cluster<br/>from Original Data]
    
    CalcMean --> Validate[Cluster Validation<br/>cluster_validation.py]
    Validate --> Valid{Valid<br/>Clusters?}
    
    Valid -->|Yes| PlotMap[Plot Cluster Map<br/>cluster_plots.py]
    Valid -->|No| Flag[Flag Invalid<br/>Clusters]
    
    PlotMap --> PlotSpec[Plot Cluster Spectra<br/>with Proper Intensity]
    Flag --> PlotSpec
    PlotSpec --> PlotScat[Plot PCA Scatter]
    
    PlotScat --> End([Cluster Results])
    
    style Start fill:#e1f5e1
    style CalcMean fill:#fff5e1,stroke:#f66,stroke-width:2px
    style PlotSpec fill:#fff5e1,stroke:#f66,stroke-width:2px
    style End fill:#ffe1e1
```

## Report Generation Pipeline

```mermaid
flowchart TD
    Start([Analysis Complete]) --> Collect[Collect Plot Paths<br/>Scan Output Directories]
    
    Collect --> Section1[Section 1:<br/>Dataset Overview]
    Section1 --> Section2[Section 2:<br/>PRE Map & PCA]
    Section2 --> Section3[Section 3:<br/>MCR Components]
    Section3 --> Section4[Section 4:<br/>Cluster Analysis]
    Section4 --> Section5[Section 5:<br/>Individual Species Maps]
    Section5 --> Section6[Section 6:<br/>Combined RGB Map]
    
    Section6 --> HTML[Generate HTML<br/>report_generator.py]
    HTML --> PDF{Generate<br/>PDF?}
    
    PDF -->|Yes| WeasyPrint[WeasyPrint<br/>HTML to PDF]
    PDF -->|No| End
    WeasyPrint --> End([Report Files])
    
    style Start fill:#e1f5e1
    style Section5 fill:#ffcc99
    style Section6 fill:#ffcc99
    style End fill:#ffe1e1
```

## Data Flow: From Raw to Quantified

```mermaid
flowchart LR
    Raw[(Raw XPS<br/>Map File)] --> Parse[Parser]
    Parse --> Cube[Hyperspectral<br/>Cube<br/>ny × nx × nE]
    
    Cube --> MCR[MCR<br/>Decomposition]
    MCR --> MCRComp[MCR Components<br/>n_comp spectra<br/>+ conc maps]
    
    MCRComp --> Fit[Peak<br/>Fitting]
    Fit --> Peaks[Individual<br/>Chemical<br/>Peaks]
    
    Peaks --> Deconv[Deconvolution]
    Deconv --> Species[Individual<br/>Species<br/>Maps & %]
    
    Species --> RGB[RGB Color<br/>Mixing Map]
    Species --> Grid[Individual<br/>Species Grid]
    
    RGB --> Final[(Final<br/>Visualization)]
    Grid --> Final
    
    style Raw fill:#e1f5e1
    style Final fill:#ffe1e1
    style Deconv fill:#fff5e1
    style Species fill:#ffcc99
```

## Workflow Decision Tree

```mermaid
flowchart TD
    Start([Input Data]) --> Check{Data<br/>Dimension?}
    
    Check -->|2D Single Energy| Type2D{Analysis<br/>Goal?}
    Check -->|3D Hyperspectral| Type3D{Data<br/>Quality?}
    
    Type2D -->|ROI Analysis| ROI[Segmentation<br/>+ Statistics]
    Type2D -->|Spatial Dist| Spatial[Intensity Maps<br/>+ Gradients]
    
    Type3D -->|Low Variability| Avg[Average Spectrum<br/>Analysis Only]
    Type3D -->|High Variability| Full[Full Chemometrics<br/>Pipeline]
    
    Full --> Method{Primary<br/>Goal?}
    Method -->|Decomposition| MCR[MCR/NMF<br/>+ Quantification]
    Method -->|Phase ID| Cluster[PCA + Clustering<br/>+ Validation]
    Method -->|Both| Both[MCR + Clustering<br/>Complete Analysis]
    
    MCR --> Plots1[MCR Plots]
    Cluster --> Plots2[Cluster Plots]
    Both --> Plots3[All Plots + Report]
    
    ROI --> End([Outputs])
    Spatial --> End
    Avg --> End
    Plots1 --> End
    Plots2 --> End
    Plots3 --> End
    
    style Start fill:#e1f5e1
    style End fill:#ffe1e1
    style Both fill:#ffcc99
```

---

## How to View These Flowcharts

### Option 1: GitHub/GitLab (Native Support)
- Push this file to your repository
- View directly in the web interface

### Option 2: VS Code (with Mermaid Extension)
```bash
# Install extension
code --install-extension bierner.markdown-mermaid

# Open this file
code ARCHITECTURE_FLOWCHART.md
```

### Option 3: Online Viewer
- Copy the Mermaid code blocks
- Paste into [Mermaid Live Editor](https://mermaid.live)

### Option 4: Export to PNG/SVG
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate images
mmdc -i ARCHITECTURE_FLOWCHART.md -o flowchart.png
```

---

## Flowchart Legend

| Shape | Meaning |
|-------|---------|
| Rectangle | Process/Function |
| Diamond | Decision Point |
| Rounded Rectangle | Start/End |
| Cylinder | Data Storage |
| Parallelogram | Input/Output |

| Color | Module Type |
|-------|-------------|
| Blue (#e1e5ff) | Processing Module |
| Pink (#ffe1f5) | Visualization Module |
| Yellow (#fff5e1) | Core Algorithm |
| Orange (#ffcc99) | Key Feature |
| Green (#e1f5e1) | Start Point |
| Red (#ffe1e1) | End Point |

---

## Key Insights from Flowcharts

1. **Two Main Branches**: 2D vs Hyperspectral processing are completely separate paths
2. **Modular Design**: Each major step is isolated in its own module for reusability
3. **Quality Gates**: Multiple decision points check data quality before expensive computations
4. **Recent Enhancements**: Cluster validation and species deconvolution are new critical steps
5. **Visualization Separation**: Plot generation is cleanly separated from computation
6. **Report Integration**: All analysis outputs feed into unified HTML/PDF reports
