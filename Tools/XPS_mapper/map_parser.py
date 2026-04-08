"""
XPS Map File Parser Module

Handles parsing of XPS map data files in various formats:
- Single-energy 2D maps
- Hyperspectral 3D data cubes
- Auto-detection of file format and dimensions

Supports robust region detection and energy axis handling.
"""
from __future__ import annotations

import re
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path

# Logging
logger = logging.getLogger("xps_map.parser")

# Default known XPS regions
DEFAULT_REGION_NAMES = {"C1s", "O1s", "F1s", "Li1s", "P2p", "S2p", "N1s", "Ag3d"}

# ============== DATA STRUCTURES ==============

@dataclass
class MapMetadata:
    """Metadata for XPS map data."""
    region: Optional[str]
    x_start: float
    x_step: float
    nx: int
    y_start: float
    y_step: float
    ny: int
    energy_axis: Optional[np.ndarray] = None
    comments: Optional[List[str]] = None
    source_format: str = "ascii_xps_map"
    source_file: Optional[str] = None


@dataclass
class Map2D:
    """2D single-energy XPS map."""
    data: np.ndarray
    metadata: MapMetadata
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    @property
    def x_axis(self) -> np.ndarray:
        return self.metadata.x_start + self.metadata.x_step * np.arange(self.metadata.nx)
    
    @property
    def y_axis(self) -> np.ndarray:
        return self.metadata.y_start + self.metadata.y_step * np.arange(self.metadata.ny)


@dataclass
class HyperspectralMap:
    """3D hyperspectral XPS map (spectrum per pixel)."""
    cube: np.ndarray   # shape (ny, nx, nE)
    energy: np.ndarray # shape (nE,)
    metadata: MapMetadata
    name: str = "map"
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.cube.shape
    
    @property
    def x_axis(self) -> np.ndarray:
        return self.metadata.x_start + self.metadata.x_step * np.arange(self.metadata.nx)
    
    @property
    def y_axis(self) -> np.ndarray:
        return self.metadata.y_start + self.metadata.y_step * np.arange(self.metadata.ny)


# ============== PARSING UTILITIES ==============

def _is_numeric_tokens(line: str) -> bool:
    """Check if line contains only numeric tokens."""
    toks = [t for t in re.split(r"[,\t\s]+", line.strip()) if t]
    if not toks:
        return False
    try:
        for t in toks:
            float(t)
        return True
    except Exception:
        return False


def _parse_numeric_tokens(line: str) -> List[float]:
    """Parse all numeric tokens from a line."""
    toks = [t for t in re.split(r"[,\t\s]+", line.strip()) if t]
    vals = []
    for t in toks:
        try:
            vals.append(float(t))
        except Exception:
            pass
    return vals


def _normalize_token(s: str) -> str:
    """Normalize token for comparison (remove non-alphanumeric, lowercase)."""
    return re.sub(r'[^A-Za-z0-9]+', '', s or '').lower()


def find_region_line(lines: List[str],
                     known_regions: Optional[set] = None,
                     search_limit: int = 20) -> Tuple[Optional[int], Optional[str]]:
    """
    Find region identifier line in file header.
    
    Searches for known region names (e.g., C1s, O1s) in the first lines.
    Falls back to first non-numeric line if no known region found.
    
    Args:
        lines: File content as list of lines
        known_regions: Set of known region names (default: DEFAULT_REGION_NAMES)
        search_limit: Number of lines to search
        
    Returns:
        (line_index, region_name) tuple, or (None, None) if not found
    """
    known = {r.lower() for r in (known_regions or DEFAULT_REGION_NAMES)}
    
    # 1) Exact known region match
    for i in range(min(search_limit, len(lines))):
        tok = _normalize_token(lines[i])
        for r in known:
            if tok == _normalize_token(r):
                # Return canonical capitalization
                canonical = next((R for R in (known_regions or DEFAULT_REGION_NAMES)
                                  if _normalize_token(R) == tok), lines[i])
                return i, canonical

    # 2) Fallback: first non-numeric line that isn't a date header
    for i in range(min(search_limit, len(lines))):
        if not _is_numeric_tokens(lines[i]):
            # Skip date-like patterns
            if re.search(r'\b\d{1,2}[-/][A-Za-z]{3}\b', lines[i]) or \
               re.search(r'\b\d{1,2}[-/]\d{1,2}\b', lines[i]):
                continue
            return i, lines[i].strip()
    
    return None, None


def extract_dims_from_header(lines: List[str]) -> Optional[Tuple[int, int]]:
    """
    Extract nx, ny dimensions from file header comments.
    
    Looks for patterns like:
    - "nx: 10", "X size: 10"
    - "ny: 10", "Y size: 10"
    
    Args:
        lines: File content as list of lines
        
    Returns:
        (nx, ny) tuple, or None if not found
    """
    nx = ny = None
    try:
        for ln in lines[:50]:  # Search header region only
            m = re.search(r'(?:^|\b)(?:nx|X\s*(?:size|points|pixels)|Width)\s*[:=]\s*(\d+)', 
                         ln, re.IGNORECASE)
            if m and nx is None:
                nx = int(m.group(1))
            
            m = re.search(r'(?:^|\b)(?:ny|Y\s*(?:size|points|pixels)|Height)\s*[:=]\s*(\d+)', 
                         ln, re.IGNORECASE)
            if m and ny is None:
                ny = int(m.group(1))
            
            if nx is not None and ny is not None:
                break
    except Exception:
        pass
    
    return (nx, ny) if (nx is not None and ny is not None) else None


def generate_energy_axis(region: str, n_points: int, 
                        region_definitions: Optional[dict] = None) -> np.ndarray:
    """
    Generate energy axis from region definitions.
    
    Args:
        region: Region name (e.g., 'C1s', 'O1s')
        n_points: Number of energy points
        region_definitions: Dict with region energy ranges (optional)
        
    Returns:
        Energy array in ascending order (binding energy in eV)
    """
    if region_definitions and region in region_definitions:
        energy_range = region_definitions[region].get('energy_range')
        if energy_range and len(energy_range) == 2:
            energy = np.linspace(energy_range[0], energy_range[1], n_points)
            logger.info(f"Generated energy axis for {region}: "
                       f"{energy_range[0]:.1f}-{energy_range[1]:.1f} eV ({n_points} pts)")
            return energy
    
    # Fallback: generic XPS range
    logger.warning(f"Region '{region}' not in definitions, using generic range")
    energy = np.linspace(50, 300, n_points)
    return energy


# ============== MAIN PARSER ==============

def detect_and_parse(file_path: str,
                    override_nx: Optional[int] = None,
                    override_ny: Optional[int] = None,
                    region_definitions: Optional[dict] = None) -> Union[Map2D, HyperspectralMap]:
    """
    Detect file format and parse XPS map data.
    
    Automatically detects:
    - 2D single-energy map vs hyperspectral 3D cube
    - Region name and spatial dimensions
    - Energy axis (if present)
    
    Args:
        file_path: Path to the data file
        override_nx: Override for number of x points (optional)
        override_ny: Override for number of y points (optional)
        region_definitions: Dict of region energy ranges for fallback generation
        
    Returns:
        Map2D or HyperspectralMap object
        
    Raises:
        ValueError: If file format is invalid or insufficient data
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        raise ValueError("Empty file")

    # Find region identifier
    region_idx, region = find_region_line(lines)
    if region_idx is None:
        raise ValueError("Could not identify region line in file header")

    # Parse spatial dimensions (next 6 numeric scalars after region line)
    scalars: List[float] = []
    j = region_idx + 1
    while j < len(lines) and len(scalars) < 6:
        vals = _parse_numeric_tokens(lines[j])
        if vals:
            scalars.extend(vals)
        j += 1

    if len(scalars) < 6:
        raise ValueError("Insufficient axis scalars after region line (expected 6)")

    x_start, x_step, nx_f, y_start, y_step, ny_f = scalars[:6]
    nx = override_nx if override_nx is not None else int(round(nx_f))
    ny = override_ny if override_ny is not None else int(round(ny_f))
    
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid dimensions nx={nx}, ny={ny}")

    # Parse comments and look for energy axis hint
    comments: List[str] = []
    energy_line_idx: Optional[int] = None
    
    for k in range(j, len(lines)):
        if lines[k].startswith("#"):
            comments.append(lines[k])
        else:
            if _is_numeric_tokens(lines[k]):
                nums = _parse_numeric_tokens(lines[k])
                # Check if this looks like energy values (not intensities)
                if len(nums) >= 5 and any(re.search(r"energy", c, re.IGNORECASE) for c in comments):
                    mean_val = np.mean(nums)
                    # XPS binding energy range validation (40-1200 eV)
                    if 40 < mean_val < 1200:
                        energy_line_idx = k
                        break

    # === HYPERSPECTRAL BRANCH ===
    if energy_line_idx is not None:
        # Energy axis found in file
        energy = np.array(_parse_numeric_tokens(lines[energy_line_idx]), dtype=float)
        nE = energy.size
        
        if nE < 5:
            raise ValueError("Energy axis too short for hyperspectral map")
        
        # Read intensity rows
        intensity_rows: List[np.ndarray] = []
        for k in range(energy_line_idx + 1, len(lines)):
            if lines[k].startswith("#"):
                continue
            vals = _parse_numeric_tokens(lines[k])
            if not vals:
                continue
            if len(vals) != nE:
                break
            intensity_rows.append(np.array(vals, dtype=float))
    
    else:
        # No energy axis in file - generate from region definitions
        logger.info(f"No energy axis in file, generating from region definition: {region}")
        
        # Find first data row to determine nE
        first_data_idx = None
        for k in range(j, len(lines)):
            if not lines[k].startswith("#") and _is_numeric_tokens(lines[k]):
                first_data_idx = k
                break
        
        if first_data_idx is None:
            raise ValueError("No intensity data found in file")
        
        first_row = _parse_numeric_tokens(lines[first_data_idx])
        nE = len(first_row)
        
        # Generate energy axis
        energy = generate_energy_axis(region, nE, region_definitions)
        
        # Read intensity rows
        intensity_rows: List[np.ndarray] = []
        for k in range(first_data_idx, len(lines)):
            if lines[k].startswith("#"):
                continue
            vals = _parse_numeric_tokens(lines[k])
            if not vals:
                continue
            if len(vals) != nE:
                break
            intensity_rows.append(np.array(vals, dtype=float))
    
    # Build hyperspectral cube
    total_pixels = nx * ny
    if len(intensity_rows) < total_pixels:
        # Pad with zeros if needed
        for _ in range(total_pixels - len(intensity_rows)):
            intensity_rows.append(np.zeros(nE, dtype=float))
    
    cube = np.zeros((ny, nx, nE), dtype=float)
    for idx, row in enumerate(intensity_rows[:total_pixels]):
        y = idx // nx
        x = idx % nx
        cube[y, x, :] = row
    
    meta = MapMetadata(
        region=region, 
        x_start=x_start, x_step=x_step, nx=nx,
        y_start=y_start, y_step=y_step, ny=ny,
        energy_axis=energy, 
        comments=comments,
        source_format="ascii_xps_hyperspectral",
        source_file=str(file_path)
    )
    
    name = Path(file_path).stem
    return HyperspectralMap(cube=cube, energy=energy, metadata=meta, name=name)


def count_hyperspec_rows_in_file(file_path: str) -> Optional[int]:
    """
    Quick scan to count spectra rows in hyperspectral file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Number of spectra rows, or None if not hyperspectral format
    """
    try:
        p = Path(file_path)
        with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        
        # Find energy line (first numeric line with >= 5 numbers)
        energy_line_idx = None
        nE = None
        for i, ln in enumerate(lines):
            nums = _parse_numeric_tokens(ln)
            if len(nums) >= 5:
                energy_line_idx = i
                nE = len(nums)
                break
        
        if energy_line_idx is None or nE is None:
            return None
        
        # Count rows with exactly nE numbers
        count = 0
        for j in range(energy_line_idx + 1, len(lines)):
            nums = _parse_numeric_tokens(lines[j])
            if len(nums) == nE:
                count += 1
        
        return count if count > 0 else None
    
    except Exception:
        return None
