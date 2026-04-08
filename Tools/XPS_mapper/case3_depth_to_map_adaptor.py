"""
depth_to_map_adapter.py

Adapter module to convert depth profile data (n > 10 cycles) into hyperspectral map format
for PCA/MCR multivariate analysis.

Strategy:
- Treat each depth cycle as a "pseudo-pixel" in a 1D spatial map
- Convert depth profile structure (energy × cycles) → map structure (1 × n_cycles × energy)
- Leverage existing XPS_mapper PCA/MCR/clustering infrastructure
- Maintain depth metadata for proper interpretation of results

Benefits of treating depth as map:
1. Statistical power: n > 10 provides sufficient degrees of freedom
2. Component separation: MCR-ALS deconvolves evolving chemical states
3. Clustering: Identifies similar depth layers (interface detection)
4. Visualization: PCA score maps show chemical evolution with depth
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import sys

# Add mapper tools to path
mapper_dir = Path(__file__).parent
if str(mapper_dir) not in sys.path:
    sys.path.insert(0, str(mapper_dir))

from map_parser import HyperspectralMap, MapMetadata


class DepthProfileAdapter:
    """
    Adapter to convert depth profile CSV data into HyperspectralMap format.
    
    Depth Profile Structure (CSV):
    ```
    Energy, Cycle1, Cycle2, ..., CycleN
    284.0,  1000,   980,   ..., 850
    284.1,  1020,   990,   ..., 860
    ...
    ```
    
    Converted Map Structure:
    - Shape: (1, n_cycles, n_energy_points)
    - Interpretation: 1D "map" where X-position = depth cycle
    - Metadata: depth_cycle_number, sputter_time, etc.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def convert_depth_to_map(
        self,
        file_path: Path,
        region_name: Optional[str] = None
    ) -> Optional[HyperspectralMap]:
        """
        Convert depth profile CSV to HyperspectralMap object.
        
        Args:
            file_path: Path to depth profile CSV file
            region_name: XPS region name (e.g., "C1s", "O1s")
            
        Returns:
            HyperspectralMap object or None if conversion fails
        """
        try:
            # Parse depth profile CSV
            energy, cycles_data, n_cycles = self._parse_depth_csv(file_path)
            
            if energy is None or cycles_data is None:
                return None
            
            # Auto-detect region if not provided
            if region_name is None:
                region_name = self._detect_region(file_path)
            
            # Reshape data: (n_energy, n_cycles) → (1, n_cycles, n_energy)
            # This creates a 1D "map" where each cycle is a pseudo-pixel
            cube = cycles_data.T.reshape(1, n_cycles, len(energy))
            
            # Create metadata
            metadata = MapMetadata(
                region=region_name,
                x_start=0.0,
                x_step=1.0,  # Each step = 1 depth cycle
                nx=n_cycles,
                y_start=0.0,
                y_step=1.0,
                ny=1,  # 1D map (single row)
                energy_axis=energy,
                source_format="depth_profile_csv",
                source_file=str(file_path.name)
            )
            
            # Add depth-specific metadata as attributes
            hmap = HyperspectralMap(
                cube=cube,
                energy=energy,
                metadata=metadata
            )
            
            # Store depth profile info
            hmap._is_depth_profile = True
            hmap._n_cycles = n_cycles
            hmap._cycle_numbers = list(range(1, n_cycles + 1))
            
            if self.debug:
                print(f"OK - Converted depth profile to pseudo-map:")
                print(f"  Cycles: {n_cycles}")
                print(f"  Energy points: {len(energy)}")
                print(f"  Energy range: {energy.min():.1f} - {energy.max():.1f} eV")
                print(f"  Map shape: {hmap.shape}")
            
            return hmap
            
        except Exception as e:
            if self.debug:
                print(f"ERROR - Depth profile conversion failed: {e}")
            return None
    
    def _parse_depth_csv(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Parse depth profile CSV file.
        
        Expected format:
        - First column: Energy (eV)
        - Remaining columns: Intensity for each cycle
        - Header row optional
        
        Returns:
            (energy_array, cycles_data, n_cycles) or (None, None, 0) if parse fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Find where numeric data starts
            data_start = 0
            for i, line in enumerate(lines):
                if ',' not in line:
                    continue
                parts = [p.strip() for p in line.split(',') if p.strip()]
                try:
                    # Try to parse as floats
                    [float(x) for x in parts]
                    data_start = i
                    break
                except ValueError:
                    continue
            
            if data_start == 0 and lines:
                # Check if first line is header
                first_parts = lines[0].split(',')
                try:
                    [float(x) for x in first_parts]
                except ValueError:
                    data_start = 1  # Skip header
            
            # Parse numeric data
            data_lines = []
            for line in lines[data_start:]:
                if ',' not in line:
                    continue
                parts = [p.strip() for p in line.split(',') if p.strip()]
                try:
                    values = [float(x) for x in parts]
                    if len(values) >= 2:  # Need at least energy + 1 cycle
                        data_lines.append(values)
                except ValueError:
                    break  # Stop at first non-numeric line
            
            if len(data_lines) < 5:
                return None, None, 0
            
            # Convert to numpy array
            data_array = np.array(data_lines)
            
            # Separate energy and cycles
            energy = data_array[:, 0]
            cycles_data = data_array[:, 1:]
            n_cycles = cycles_data.shape[1]
            
            # Validate
            if n_cycles < 2:
                return None, None, 0
            
            # Check for monotonic energy axis
            energy_diffs = np.diff(energy)
            if not (np.all(energy_diffs > 0) or np.all(energy_diffs < 0)):
                if self.debug:
                    print(f"⚠️ Non-monotonic energy axis detected")
            
            return energy, cycles_data, n_cycles
            
        except Exception as e:
            if self.debug:
                print(f"ERROR - CSV parse error: {e}")
            return None, None, 0
    
    def _detect_region(self, file_path: Path) -> str:
        """
        Auto-detect XPS region from filename or file content.
        
        Args:
            file_path: Path to depth profile file
            
        Returns:
            Region name (e.g., "C1s", "O1s") or "Unknown"
        """
        # Try filename first
        filename = file_path.stem.upper()
        
        region_patterns = {
            'C1S': 'C1s',
            'O1S': 'O1s',
            'F1S': 'F1s',
            'LI1S': 'Li1s',
            'SI2P': 'Si2p',
            'P2P': 'P2p',
            'S2P': 'S2p',
            'N1S': 'N1s'
        }
        
        for pattern, region in region_patterns.items():
            if pattern in filename:
                return region
        
        # Try reading first few lines of file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header_lines = ''.join([f.readline() for _ in range(5)]).upper()
            
            for pattern, region in region_patterns.items():
                if pattern in header_lines:
                    return region
        except:
            pass
        
        return "Unknown"


# ============================================================================
# Integration Functions
# ============================================================================

def convert_depth_profile_for_mapper(
    file_path: Path,
    region_name: Optional[str] = None,
    debug: bool = False
) -> Optional[HyperspectralMap]:
    """
    Main integration function: Convert depth profile to map format.
    
    Usage in workflow:
    ```python
    from depth_to_map_adapter import convert_depth_profile_for_mapper
    
    # After triage detects depth profile with n > 10
    if triage_result['data_type'] == 'DEPTH_PROFILE' and triage_result['parameters']['use_pca_mcr']:
        hmap = convert_depth_profile_for_mapper(file_path)
        if hmap:
            # Pass to XPS_mapper for PCA/MCR analysis
            from XPS_map import process_hyperspectral_map_simple
            results = process_hyperspectral_map_simple(hmap, output_dir, config=config)
    ```
    
    Args:
        file_path: Path to depth profile CSV
        region_name: Optional region name override
        debug: Enable debug output
        
    Returns:
        HyperspectralMap ready for mapper processing
    """
    adapter = DepthProfileAdapter(debug=debug)
    return adapter.convert_depth_to_map(file_path, region_name)


def extract_depth_results(
    mapper_results: Dict[str, Any],
    n_cycles: int
) -> Dict[str, Any]:
    """
    Post-process mapper results to interpret in depth profile context.
    
    Translates spatial map terminology to depth profile terminology:
    - "pixels" → "depth cycles"
    - "spatial map" → "depth evolution"
    - "cluster map" → "layer identification"
    - PCA score maps → "chemical evolution profiles"
    
    Args:
        mapper_results: Output from process_hyperspectral_map_simple()
        n_cycles: Number of depth cycles
        
    Returns:
        Reformatted results with depth-appropriate terminology
    """
    depth_results = mapper_results.copy()
    
    # Add depth-specific interpretations
    depth_results['analysis_type'] = 'depth_profile_pca_mcr'
    depth_results['n_depth_cycles'] = n_cycles
    
    # Rename spatial terms
    if 'cluster_labels' in depth_results:
        # Cluster labels are now layer assignments
        labels = depth_results['cluster_labels'].flatten()
        depth_results['layer_assignments'] = labels
        depth_results['unique_layers'] = len(np.unique(labels))
    
    if 'pca' in depth_results:
        # PCA scores show chemical evolution
        pca = depth_results['pca']
        if 'score_maps' in pca:
            scores = pca['score_maps'].reshape(n_cycles, -1)
            depth_results['chemical_evolution_profiles'] = scores
    
    if 'mcr' in depth_results:
        # MCR concentrations show species evolution
        mcr = depth_results['mcr']
        if 'C_' in mcr:
            conc = mcr['C_']
            depth_results['species_depth_profiles'] = conc
    
    return depth_results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert depth profile to map format")
    parser.add_argument("file", type=str, help="Path to depth profile CSV")
    parser.add_argument("--region", type=str, default=None, help="Region name (e.g., C1s)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    hmap = convert_depth_profile_for_mapper(file_path, args.region, args.debug)
    
    if hmap:
        print(f"\nOK - Conversion successful!")
        print(f"  Output type: HyperspectralMap")
        print(f"  Shape: {hmap.shape}")
        print(f"  Cycles: {hmap._n_cycles}")
        print(f"  Region: {hmap.metadata.region}")
        print(f"  Ready for XPS_mapper processing")
    else:
        print(f"\nERROR - Conversion failed")
        sys.exit(1)
