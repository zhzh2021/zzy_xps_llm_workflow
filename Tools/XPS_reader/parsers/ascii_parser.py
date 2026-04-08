"""
Parser for ASCII/TXT format XPS data files.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import logging
import re
import traceback

from .base import BaseParser
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum


class ASCIIParser(BaseParser):
    """Parser for simple two-column ASCII/TXT files."""
    
    format_name = "ascii"
    file_extensions = ['.txt', '.asc', '.dat', '.xy', '.pro']
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if file is ASCII format."""
        if not self._check_extension(file_path):
            return False

        try:
            with open(file_path, 'rb') as f:
                if f.read(4) == b"SOFH":
                    return False
        except Exception:
            return False
        
        try:
            # Try to read first few lines as text
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                first_lines = f.read(1000)
            
            # Check if it looks like ASCII data
            # Should have numeric values separated by whitespace/commas
            has_numbers = bool(re.search(r'\d+\.?\d*', first_lines))
            return has_numbers
            
        except Exception:
            return False
    
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Parse ASCII file."""
        try:
            intensity, metadata = self._parse_ascii_format(file_path)
            
            if intensity is None or len(intensity) < 2:
                return None
            
            energy = metadata.get('energy_values')
            if energy is None:
                return None
            
            spectrum = Spectrum(
                name=file_path.stem,
                energy=energy,
                intensity=intensity,
                source_format=self.format_name,
                metadata=metadata
            )
            
            if spectrum.is_valid_xps():
                return [spectrum]
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ ASCII parsing failed: {e}")
                traceback.print_exc()
            return None
    
    def _parse_ascii_format(self, file_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Parse simple two-column ASCII/CSV spectra into intensity data and metadata."""
        if not file_path.exists():
            raise FileNotFoundError(f"ASCII file not found: {file_path}")

        energy_values: List[float] = []
        intensity_values: List[float] = []
        header_lines: List[str] = []
        skipped_lines: List[str] = []
        separators = [',', ';', '\t', None]
        data_started = False

        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    header_lines.append(stripped.lstrip('#').strip())
                    continue

                parsed = False
                for sep in separators:
                    if sep is None:
                        parts = stripped.split()
                    else:
                        parts = [p.strip() for p in stripped.split(sep)]
                    parts = [p for p in parts if p]
                    if len(parts) < 2:
                        continue
                    try:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                    except ValueError:
                        continue

                    energy_values.append(x_val)
                    intensity_values.append(y_val)

                    if len(parts) > 2:
                        skipped_lines.append(f"{line_no}:{stripped}")

                    data_started = True
                    parsed = True
                    break

                if not parsed:
                    target = header_lines if not data_started else skipped_lines
                    target.append(f"{line_no}:{stripped}")

        if len(energy_values) < 2:
            return None, {}

        energy_array = np.asarray(energy_values, dtype=np.float64)
        intensity_array = np.asarray(intensity_values, dtype=np.float64)

        metadata: Dict[str, Any] = {
            "source_file": file_path.name,
            "format": "ascii",
            "num_points": len(energy_array),
            "energy_min": float(np.min(energy_array)),
            "energy_max": float(np.max(energy_array)),
            "energy_values": energy_array,
        }
        if header_lines:
            metadata["header_lines"] = header_lines
        if skipped_lines:
            metadata["skipped_line_count"] = len(skipped_lines)
            metadata["skipped_line_examples"] = skipped_lines[:5]

        return intensity_array, metadata


class EnhancedASCIIParser(BaseParser):
    """Enhanced ASCII parser with multi-region support."""
    
    format_name = "enhanced_ascii"
    file_extensions = ['.txt', '.asc', '.dat', '.xy', '.csv', '.pro']
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if file is ASCII format."""
        if not self._check_extension(file_path):
            return False
        try:
            with open(file_path, 'rb') as f:
                if f.read(4) == b"SOFH":
                    return False
        except Exception:
            return False
        return True
    
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Import ASCII/CSV file with automatic format detection."""
        if self.debug:
            print(f"\n   🔍 DEBUG: Starting import of {file_path.name}")
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
            
            if self.debug:
                print(f"   🔍 DEBUG: Read {len(lines)} lines")
            
            # Try multi-region first
            if self._is_multi_region_file(file_path):
                if self.debug:
                    print("   🔍 DEBUG: Detected multi-region format")
                spectra = self._parse_multi_region(file_path, lines)
                if spectra is not None:
                    return spectra
            
            # Fall back to single-region
            if self.debug:
                print("   🔍 DEBUG: Trying single-region format")
            return self._parse_single_region(file_path)
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ DEBUG: Import failed with error: {e}")
                traceback.print_exc()
            return None
    
    def _is_multi_region_file(self, file_path: Path) -> bool:
        """Detect if file contains multiple regions."""
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                content = f.read(2000)
            
            region_pattern = r'\b[A-Z][a-z]?\d[spdf]\b'
            matches = re.findall(region_pattern, content)
            
            return len(matches) > 1
            
        except Exception:
            return False
    
    def _parse_single_region(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Parse single-region ASCII/CSV file."""
        try:
            energy = []
            intensity = []
            metadata = {}
            
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                values = self._parse_numeric_values(line)
                
                if values and len(values) >= 2:
                    energy.append(values[0])
                    intensity.append(values[1])
                elif values is None and len(energy) == 0:
                    metadata[f'header_{len(metadata)}'] = line
            
            if len(energy) >= 3:
                spectrum = Spectrum(
                    name=file_path.stem,
                    energy=np.array(energy, dtype=np.float64),
                    intensity=np.array(intensity, dtype=np.float64),
                    source_format=self.format_name,
                    metadata=metadata
                )
                
                if spectrum.is_valid_xps():
                    return [spectrum]
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ Single-region parsing failed: {e}")
            return None
    
    def _parse_numeric_values(self, line: str) -> Optional[List[float]]:
        """Parse numeric values from a line."""
        for sep in ['\t', ',', ';', None]:
            try:
                if sep:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                else:
                    parts = line.split()
                
                values = []
                for part in parts:
                    try:
                        values.append(float(part))
                    except ValueError:
                        continue
                
                if len(values) >= 2:
                    return values
                    
            except Exception:
                continue
        
        return None
    
    def _parse_multi_region(self, file_path: Path, lines: List[str]) -> Optional[List[Spectrum]]:
        """Parse multi-region file."""
        spectra = []
        return None
