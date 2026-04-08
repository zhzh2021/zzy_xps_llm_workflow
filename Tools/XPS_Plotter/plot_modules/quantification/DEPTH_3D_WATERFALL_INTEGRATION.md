"""
3D Waterfall Plot Integration Guide
====================================

This module provides standalone 3D waterfall visualization for XPS depth profiles.

LOCATION:
  zzy_llm/Tools/XPS_Plotter/plot_modules/quantification/depth_3d_waterfall.py

USAGE - Option 1: Direct Import and Call
-------------------------------------------
from pathlib import Path
from XPS_Plotter.plot_modules.quantification.depth_3d_waterfall import (
    plot_depth_profile_3d_waterfall,
    load_depth_profile_csv
)

# Load depth profile from CSV
csv_path = Path('01_converted_csv/F1s/aggregated_F1s_allHR.csv')
layers = load_depth_profile_csv(csv_path)

# Generate 3D plot
output_dir = Path('04_plots/03_quantification')
plot_path = plot_depth_profile_3d_waterfall(
    spectra_dict=layers,
    region="F1s",
    out_dir=output_dir,
    cmap_name="viridis"  # or "plasma", "coolwarm", etc.
)


USAGE - Option 2: From Reader (After CSV Export)
-------------------------------------------
After the reader generates aggregated CSVs in 01_converted_csv/:
  
from XPS_Plotter.plot_modules.quantification.depth_3d_waterfall import (
    load_depth_profile_csv,
    plot_depth_profile_3d_waterfall
)

# For each region in depth profile
regions = ['C1s', 'O1s', 'F1s', 'Li1s', 'P2p']
for region in regions:
    csv_path = Path(f'01_converted_csv/{region}/aggregated_{region}_allHR.csv')
    layers = load_depth_profile_csv(csv_path)
    if layers and len(layers) > 1:  # Only for multi-layer depth profiles
        plot_depth_profile_3d_waterfall(
            spectra_dict=layers,
            region=region,
            out_dir=Path('04_plots/03_quantification'),
            cmap_name="viridis"
        )


USAGE - Option 3: From Quantifier (Integration Point)
-------------------------------------------
Can be called after depth profile detection:

# In XPS_Quantifier.py or similar:
if is_depth_profile and num_layers > 1:
    from XPS_Plotter.plot_modules.quantification.depth_3d_waterfall import (
        load_depth_profile_csv,
        plot_depth_profile_3d_waterfall
    )
    
    # For each region that was quantified
    for region in quantified_regions:
        csv_path = Path(f'01_converted_csv/{region}/aggregated_{region}_allHR.csv')
        layers = load_depth_profile_csv(csv_path)
        if layers:
            plot_depth_profile_3d_waterfall(layers, region, plots_dir)


FUNCTION SIGNATURES
--------------------

load_depth_profile_csv(csv_path: Path) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]
  Returns: {layer_num: (energy_array, intensity_array)} or None

plot_depth_profile_3d_waterfall(
    spectra_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    region: str,
    out_dir: Path,
    cmap_name: str = "viridis",
    config=None
) -> Optional[Path]
  Returns: Path to saved PNG figure or None if failed


VISUALIZATION FEATURES
-----------------------
• X-axis: Binding Energy (eV), reversed for XPS convention
• Y-axis: Layer number (1=surface, N=bulk)
• Z-axis: Intensity (cps)
• Color: Progressive colormap from surface to bulk
• Vertical drops: Show intensity variation at each layer
• Interactive: User can rotate, zoom with mouse

Default viewing angle: elev=20°, azim=120° (energy in front)
Users can rotate to any angle in the viewer


INTEGRATION CHECKLIST
----------------------
☐ Module file exists: depth_3d_waterfall.py
☐ Functions are callable standalone
☐ Config loading has fallback
☐ Error handling is robust
☐ Output directory is created automatically
☐ Filename is sanitized for region name
☐ Documentation is clear
☐ Test script runs successfully

STATUS: ✓ COMPLETE - Module is independent and ready to call when needed
"""

if __name__ == "__main__":
    print(__doc__)
