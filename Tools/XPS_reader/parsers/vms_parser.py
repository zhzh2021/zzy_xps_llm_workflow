"""
Parser for VAMAS format XPS data files (.vms, .vamas).

VAMAS (Versioned ASCII format for Materials Analysis Systems) is a standard
format for surface chemical analysis data exchange.
"""
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import re
import traceback
import logging

from .base import BaseParser
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum


class VAMASParser(BaseParser):
    """Parser for VAMAS format XPS data files (.vms, .vamas)."""
    
    format_name = "vamas"
    file_extensions = ['.vms', '.vamas']
    
    def __init__(self, debug: bool = False):
        """
        Initialize VAMAS parser.
        
        Args:
            debug: Enable detailed debug output
        """
        super().__init__(debug=debug)
        self.excitation_energy = 1486.6  # Al K-alpha default
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if file is VAMAS format.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file appears to be VAMAS format
        """
        if not self._check_extension(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
            
            return self._is_vamas_format(lines)
            
        except Exception:
            return False
    
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """
        Parse VAMAS format file.
        
        Args:
            file_path: Path to VAMAS file
            
        Returns:
            List of Spectrum objects, or None if parsing failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
            
            if self.debug:
                print(f"\n   🔍 DEBUG: Reading VAMAS file: {file_path.name}")
                print(f"   🔍 DEBUG: Total lines: {len(lines)}")
                print(f"   🔍 DEBUG: First 20 lines:")
                for i, line in enumerate(lines[:20]):
                    print(f"      {i}: {repr(line.strip()[:80])}")
            
            # Verify VAMAS format
            if not self._is_vamas_format(lines):
                if self.debug:
                    print("   ⚠️ Not a VAMAS format file")
                return None
            
            # Parse all blocks
            spectra = self._parse_vamas_blocks(lines, file_path)
            
            if spectra:
                print(f"   ✅ Successfully parsed {len(spectra)} VAMAS spectrum/spectra")
                for spec in spectra:
                    print(f"      • {spec.name}: {len(spec.energy)} points, "
                          f"BE: {spec.energy.min():.2f}-{spec.energy.max():.2f} eV")
            else:
                print(f"   ⚠️ No valid spectra found in VAMAS file")
            
            return spectra
            
        except Exception as e:
            print(f"   ❌ VAMAS import failed: {e}")
            if self.debug:
                traceback.print_exc()
            return None
    
    def _is_vamas_format(self, lines: List[str]) -> bool:
        """
        Check if content is VAMAS format.
        
        Args:
            lines: List of file lines
            
        Returns:
            True if content appears to be VAMAS format
        """
        # Check first 10 lines for VAMAS signature
        first_lines = ' '.join([line.strip() for line in lines[:10]])
        
        # Look for VAMAS signature
        if 'VAMAS' in first_lines.upper() or 'Surface Chemical Analysis' in first_lines:
            return True
        
        # Also check for typical VAMAS structure (region names with /number)
        for line in lines[:100]:
            if re.search(r'[A-Z][a-z]?\s*\d[spdf]/\d+', line):
                return True
        
        return False
    
    def _parse_vamas_blocks(self, lines: List[str], file_path: Path) -> Optional[List[Spectrum]]:
        """
        Parse all VAMAS data blocks in file.
        
        Args:
            lines: List of file lines
            file_path: Path to source file
            
        Returns:
            List of parsed spectra, or None if no valid blocks found
        """
        spectra = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for block identifiers
            # Format: "region_name/block_number" (e.g., "survey/2", "C 1s/7")
            match = re.match(
                r'^([A-Z][a-z]?\s*\d[spdf]|survey|Survey|SURVEY|VB|Auger)\s*/\s*(\d+)', 
                line, 
                re.IGNORECASE
            )
            
            if match:
                region_name = match.group(1).strip()
                block_number = match.group(2)
                
                if self.debug:
                    print(f"\n   🔍 DEBUG: Found region '{region_name}' (block {block_number}) at line {i}")
                
                # Parse this block
                spectrum, next_i = self._parse_vamas_block(
                    lines, i, region_name, file_path
                )
                
                if spectrum:
                    spectra.append(spectrum)
                    if self.debug:
                        print(f"      ✅ Successfully parsed {region_name}")
                
                i = next_i
            else:
                i += 1
        
        return spectra if spectra else None
    
    def _parse_vamas_block(
        self, 
        lines: List[str], 
        start_idx: int,
        region_name: str,
        file_path: Path
    ) -> Tuple[Optional[Spectrum], int]:
        """
        Parse a single VAMAS data block.
        
        VAMAS block structure:
        1. Region identifier line (e.g., "C 1s/7")
        2. Multiple metadata lines
        3. Number of data points (single integer)
        4. Data pairs (energy, intensity)
        
        Args:
            lines: List of all file lines
            start_idx: Starting line index for this block
            region_name: Name of the region
            file_path: Path to source file
            
        Returns:
            Tuple of (Spectrum object or None, next line index)
        """
        i = start_idx + 1
        metadata = {'region_name': region_name}

        num_points = None
        data_start = None

        # Regex for labelled pass energy in block headers produced by some
        # VAMAS exporters, e.g. "analyser pass energy  20.0" or
        # "pass energy: 160".
        _pe_re = re.compile(
            r'(?:analyser\s+)?pass[_ ]?energy[\s=:]+([0-9]+(?:\.[0-9]*)?)',
            re.IGNORECASE
        )

        # Skip metadata and find number of points
        while i < len(lines) and i < start_idx + 200:  # Limit search
            line = lines[i].strip()

            if self.debug and i < start_idx + 30:
                print(f"      Line {i}: {repr(line[:60])}")

            # Empty line
            if not line:
                i += 1
                continue

            # Check if this is the next block
            if re.match(r'^[A-Z][a-z]?\s*\d[spdf]/\d+', line, re.IGNORECASE):
                if self.debug:
                    print(f"      Found next block at line {i}")
                break

            # Try to extract pass energy from a labelled metadata line
            if 'pass_energy' not in metadata:
                m = _pe_re.search(line)
                if m:
                    try:
                        metadata['pass_energy'] = float(m.group(1))
                        if self.debug:
                            print(f"      pass_energy={metadata['pass_energy']} eV from: {line!r}")
                    except ValueError:
                        pass

            # Look for number of data points
            # Usually a line with just a number (typically 100-5000)
            if line.isdigit():
                num = int(line)
                if 10 < num < 10000 and num_points is None:
                    num_points = num
                    if self.debug:
                        print(f"      Found num_points: {num_points} at line {i}")
                    i += 1
                    continue

            # Check if this looks like data (two numbers)
            if num_points is not None and self._is_data_line(line):
                data_start = i
                if self.debug:
                    print(f"      Data starts at line {i}")
                break

            i += 1
        
        if data_start is None:
            if self.debug:
                print(f"      ⚠️ No data found for {region_name}")
            return None, i
        
        # Parse data
        energy = []
        intensity = []
        i = data_start
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Stop at empty line or next region
            if not line:
                i += 1
                # Allow a few empty lines but stop if too many
                if i - data_start - len(energy) > 5:
                    break
                continue
            
            # Stop at next block
            if re.match(r'^[A-Z][a-z]?\s*\d[spdf]/\d+', line, re.IGNORECASE):
                break
            
            # Try to parse as data
            values = self._parse_data_line(line)
            if values and len(values) >= 2:
                energy.append(values[0])
                intensity.append(values[1])
                
                if self.debug and len(energy) <= 3:
                    print(f"      Data point {len(energy)}: E={values[0]:.4f}, I={values[1]:.1f}")
            else:
                # If we've collected enough data and hit non-data, stop
                if num_points and len(energy) >= num_points * 0.9:
                    break
                if len(energy) > 100:  # Have substantial data
                    break
            
            i += 1
        
        if self.debug:
            print(f"      Collected {len(energy)} data points")
        
        if len(energy) < 3:
            if self.debug:
                print(f"      ⚠️ Insufficient data points: {len(energy)}")
            return None, i
        
        # Create spectrum
        return self._create_spectrum(
            energy, intensity, region_name, metadata
        ), i
    
    def _create_spectrum(
        self,
        energy: List[float],
        intensity: List[float],
        region_name: str,
        metadata: Dict[str, Any]
    ) -> Optional[Spectrum]:
        """
        Create Spectrum object from parsed data.
        
        Args:
            energy: List of energy values
            intensity: List of intensity values
            region_name: Name of the region
            metadata: Metadata dictionary
            
        Returns:
            Spectrum object or None if creation failed
        """
        try:
            # Convert to numpy arrays
            energy_array = np.array(energy, dtype=np.float64)
            intensity_array = np.array(intensity, dtype=np.float64)
            
            if self.debug:
                print(f"      Energy range: {energy_array.min():.2f} - {energy_array.max():.2f}")
                print(f"      Energy mean: {energy_array.mean():.2f}")
            
            # VAMAS often stores kinetic energy, convert to binding energy
            # Check if values are in kinetic energy range (high values)
            if energy_array.mean() > 500:
                energy_array = self.excitation_energy - energy_array
                metadata['converted_from_ke'] = True
                metadata['excitation_energy'] = self.excitation_energy
                
                if self.debug:
                    print(f"      Converted from KE to BE using {self.excitation_energy} eV")
                    print(f"      New BE range: {energy_array.min():.2f} - {energy_array.max():.2f}")
            
            # Sort by energy (binding energy should be descending for most XPS data)
            sort_idx = np.argsort(energy_array)[::-1]
            energy_array = energy_array[sort_idx]
            intensity_array = intensity_array[sort_idx]
            
            spectrum = Spectrum(
                name=region_name,
                energy=energy_array,
                intensity=intensity_array,
                source_format=self.format_name,
                metadata=metadata
            )
            
            is_valid = spectrum.is_valid_xps()
            
            if self.debug:
                print(f"      XPS validation: {is_valid}")
            
            if is_valid:
                return spectrum
            else:
                if self.debug:
                    print(f"      ⚠️ Spectrum failed XPS validation")
                    print(f"         Energy stats: min={energy_array.min():.2f}, "
                          f"max={energy_array.max():.2f}, mean={energy_array.mean():.2f}")
                return None
                
        except Exception as e:
            if self.debug:
                print(f"      ❌ Failed to create spectrum: {e}")
                traceback.print_exc()
            return None
    
    def _is_data_line(self, line: str) -> bool:
        """
        Check if line contains data (two numeric values).
        
        Args:
            line: Line to check
            
        Returns:
            True if line contains valid data
        """
        values = self._parse_data_line(line)
        return values is not None and len(values) >= 2
    
    def _parse_data_line(self, line: str) -> Optional[List[float]]:
        """
        Parse a data line into numeric values.
        
        Args:
            line: Line to parse
            
        Returns:
            List of [energy, intensity] or None if parsing failed
        """
        try:
            # Split by whitespace
            parts = line.split()
            
            if len(parts) < 2:
                return None
            
            values = []
            for part in parts[:2]:  # Only take first two values
                try:
                    val = float(part)
                    values.append(val)
                except ValueError:
                    return None
            
            return values if len(values) == 2 else None
            
        except Exception:
            return None


# Standalone helper function for backward compatibility
def parse_vamas_format(file_path: Union[str, Path]) -> Tuple[List[np.ndarray], Dict]:
    """
    Parse VAMAS format XPS data file.
    
    This is a convenience function that wraps the VAMASParser class.
    
    Args:
        file_path: Path to the VAMAS file
        
    Returns:
        Tuple of (list of intensity arrays, metadata_dict)
        
    Raises:
        ValueError: If file format is invalid or no numeric data found
        
    Example:
        >>> intensities, metadata = parse_vamas_format("data.vms")
        >>> print(f"Found {metadata['num_regions']} regions")
    """
    file_path = Path(file_path)
    
    try:
        parser = VAMASParser(debug=True)
        spectra = parser.parse(file_path)
        
        if not spectra:
            raise ValueError("No valid XPS data found in VAMAS file")
        
        # Return all spectra data
        intensity_arrays = [spec.intensity for spec in spectra]
        
        metadata = {
            'num_regions': len(spectra),
            'region_names': [spec.name for spec in spectra],
            'source_format': 'vamas'
        }
        
        # Add info for each region
        for i, spec in enumerate(spectra):
            metadata[f'region_{i}_name'] = spec.name
            metadata[f'region_{i}_points'] = len(spec.energy)
            metadata[f'region_{i}_energy_min'] = float(spec.energy.min())
            metadata[f'region_{i}_energy_max'] = float(spec.energy.max())
        
        return intensity_arrays, metadata
        
    except Exception as e:
        raise ValueError(f"Failed to parse VAMAS format file: {str(e)}")
