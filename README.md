# XPS Workflow Assistant

**An LLM-driven desktop tool for automated X-ray Photoelectron Spectroscopy (XPS) data analysis.**

Researchers describe their analysis in plain language. The agent interprets the request, routes it through a quality-gated pipeline, and executes the correct combination of XPS tools — no scripting required.

> Developed at Argonne National Laboratory.  
> License: see [LICENSE](LICENSE) — research / educational use.

---

## Features

| Capability | Description |
|---|---|
| **Natural language interface** | Chat-based UI powered by a local Ollama LLM |
| **Automated triage** | Detects data type (standard spectra vs 2D/hyperspectral maps) and routes accordingly |
| **Hierarchical quality gate** | Two-level SNR, resolution, energy-axis, and spatial validation before any fitting |
| **Full pipeline automation** | Reader → Fitter → Quantifier → Plotter in one command |
| **Map workflow** | Hyperspectral map processing with clustering and MCR-ALS |
| **Correlation analysis** | Cross-correlates quantified composition with experimental variables |
| **Export** | Hi-res PNG screenshots and full-conversation PDF export for publications |
| **Persistent agent state** | Remembers triage results and data context across turns |

---

## Workflow Overview

```
Raw XPS files (.spe / .vgd / .npl / .xy / .csv)
        │
        ▼
  Step 0 ── Triage + Quality Gate  ──► CRITICAL/FAILED → user notified, workflow halts
        │                                POOR/WARNING  → user alerted, proceeds with caution
        │
        ├─► Standard spectra path
        │       Step 1  XPS_Reader      raw → standardised CSV
        │       Step 2  XPS_Fitter      Shirley BG + template peak fitting
        │       Step 3  XPS_Quantifier  atomic % from peak areas
        │       Step 4  XPS_Plotter     overlay / individual fit plots
        │
        └─► Map / hyperspectral path
                Step 1  XPS_Mapper      clustering, MCR-ALS, depth profiles
                Step 2  XPS_Plotter     map visualisations
```

All outputs are organised under a numbered project folder structure:

```
project_root/
├── 00_raw_data/          ← drop your raw files here
├── 01_converted_csv/
├── 02_fitted_results/
├── 03_quantified_data/
├── 04_plots/
├── 05_map_data/
├── 06_correlator_results/
├── xps_config/           ← fit templates and element RSF table
└── _logs/                ← quality reports and workflow JSON logs
```

---

## Requirements

- Python 3.11 – 3.13
- [Ollama](https://ollama.com/download) installed and running locally
- At least one Ollama model pulled (see below)

### Python dependencies

```
pyside6               # desktop UI
langchain-ollama      # LLM backend
lmfit                 # peak fitting
scipy / numpy         # signal processing
pandas                # data handling
matplotlib / seaborn  # plotting
scikit-learn          # triage ML
pymcr                 # MCR-ALS for map data
```

Install everything with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/your-org/xps-llm-workflow.git
cd xps-llm-workflow
poetry install
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Start Ollama and pull a model

```bash
ollama pull qwen2.5        # recommended — good reasoning, fast
# or
ollama pull llama3.1
```

### 2. Launch the assistant

```bash
# from the repo root
python -m zzy_llm.ui.chat_window
```

Windows (2× DPI for crisp display):

```powershell
$env:QT_SCALE_FACTOR = "2"
python -m zzy_llm.ui.chat_window
```

### 3. Run an analysis

Drop raw data files into `project_root/00_raw_data/`, then type a request:

> *"Convert and fit the Fe 2p spectra in project_root/00_raw_data/Fe2p_batch01"*

The agent runs the full pipeline and reports back after each step.

---

## Demo Modes

Two pre-loaded demo conversations are included for testing and publication figures:

```bash
# Standard workflow demo (Si 2p etching batch)
python -m zzy_llm.ui.chat_window --demo

# Quality gate demo (Fe 2p batch with flagged spectra)
python -m zzy_llm.ui.chat_window --demo-quality
```

Use **📷 Export** for a hi-res PNG, or **📄 PDF** to export the full conversation as a paginated A4 PDF.

---

## Quality Gate

Before any fitting is attempted, every spectrum passes through the unified quality gatekeeper:

**Level 1 — Universal checks (all data types)**
- Empty / insufficient data detection
- Energy axis integrity (gaps, non-monotonic axes)
- Signal-to-noise ratio (default threshold: SNR ≥ 10)

**Level 2 — Modality-specific checks**
- *Spectra:* resolution (HR vs survey), peak detectability, fitting suitability
- *Maps:* spatial dimensions, dead/outlier pixels, spatial continuity score

Results are saved as a structured JSON report to `project_root/_logs/` and summarised in the chat.
Spectra flagged **POOR** or **CRITICAL** are highlighted by name so you can review them before accepting batch results.

---

## Project Structure

```
zzy_llm/
├── ui/
│   ├── chat_window.py        # PySide6 chat UI, demo modes, PDF/PNG export
│   └── workers.py            # QThread workers for async LLM + tool execution
├── llm_manager/
│   ├── experiment_router.py  # ChatOllama routing → run / clarify / execute / triage
│   ├── quality_gatekeeper.py # Hierarchical quality validation
│   ├── triage_router.py      # Data-type detection (spectra vs map)
│   ├── Agentstate.py         # Persistent agent memory (LangGraph state)
│   └── ollama_utils.py       # Model listing helpers
├── Tools/
│   ├── XPS_reader/           # Raw file → CSV conversion
│   ├── XPS_Fitter/           # Template-based peak fitting
│   ├── XPS_Quantifier/       # Atomic % quantification
│   ├── XPS_Plotter/          # Plot generation
│   ├── XPS_mapper/           # Hyperspectral map processing (MCR, clustering)
│   ├── XPS_Correlator/       # Cross-correlation with metadata
│   ├── XPS_utils/            # Shared utilities (Shirley background, etc.)
│   └── real_xps_workflow.py  # Pipeline orchestrator
└── experiments/              # Experiment registry for the LLM router
```

---

## Supported File Formats

| Format | Instrument / Source |
|---|---|
| `.spe` | Kratos AXIS instruments |
| `.vgd` | VG / Thermo instruments |
| `.npl` | NPL format |
| `.xy` | Generic two-column ASCII |
| `.csv` | Pre-converted CSV |
| `.vms` | VAMAS standard format |

---

## Configuration

Set `ZZY_LLM_HOME` to your project root if auto-detection fails:

```bash
export ZZY_LLM_HOME=/path/to/my/project_root   # Linux / macOS
$env:ZZY_LLM_HOME = "C:\data\my_project"        # Windows PowerShell
```

Fit templates and the element RSF table live in `project_root/xps_config/`.

---

## Citation

If you use this software in published work, please cite:

> Zhenzhen Yang, *XPS Workflow Assistant*, Argonne National Laboratory, 2025.  
> https://github.com/your-org/xps-llm-workflow

---

## Contributing

Issues and pull requests are welcome.
Please open an issue first to discuss significant changes.

---

## Acknowledgements

Developed at the **Advanced Photon Source, Argonne National Laboratory**.  
Built on [LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://ollama.com),
[lmfit](https://lmfit.github.io/lmfit-py/), and [PySide6](https://doc.qt.io/qtforpython/).