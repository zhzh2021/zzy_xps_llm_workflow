"""
Parser for CSV format XPS data files.

Supports:
- Standard two-column CSV (energy, intensity)
- Multi-region CSV with paired columns (E1, I1, E2, I2, ...)
- Paired-row format (energy row followed by intensity row)
- Various separators (comma, semicolon, tab, whitespace)
- Header detection and skipping
"""
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import csv
import pandas as pd
import numpy as np
import logging
import re
import traceback

from .base import BaseParser
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum


REGION_PATTERN_STR = r'([A-Za-z][a-z]?\d[spdfSPDF])'
REGION_REGEX = re.compile(REGION_PATTERN_STR)


def extract_region_token(value: str) -> Optional[str]:
    """Extract a region token such as C1s or Li1s from a cell."""
    if value is None:
        return None
    cleaned = value.replace("\ufeff", "").strip()
    if "\t" in cleaned:
        cleaned = cleaned.split("\t")[0].strip()
    if not cleaned:
        return None
    match = REGION_REGEX.search(cleaned)
    if match:
        token = match.group(1)
        # Normalize casing (first letter uppercase, orbital lowercase)
        if len(token) >= 2:
            normalized = token[0].upper() + token[1:-2] + token[-2:].lower()
            return normalized
        return token
    return None


class CSVParserMixin:
    """Mixin providing shared CSV parsing helpers."""

    debug: bool = False

    def _parse_numeric_values(self, line: str) -> Optional[List[float]]:
        line = line.strip()
        if not line:
            return None
        for sep in ['\t', ',', ';', None]:
            try:
                parts = [p.strip() for p in line.split(sep) if p.strip()] if sep else line.split()
            except Exception:
                continue
            values = []
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    continue
            if len(values) >= 2:
                return values
        return None

    def _is_pure_metadata(self, line: str) -> bool:
        cleaned = line.replace('\t', ' ').replace(',', ' ').strip()
        if not cleaned:
            return True
        tokens = cleaned.split()
        if not tokens:
            return True
        if any(not token.replace('.', '').replace('-', '').isdigit() for token in tokens):
            return True
        if len(tokens) <= 10:
            try:
                int_values = [int(float(token)) for token in tokens]
                if len(set(int_values)) == 1 and int_values[0] < 10:
                    return True
            except ValueError:
                pass
        return False

    def _safe_float(self, value: str) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_region_data(
        self,
        lines: List[str],
        start_idx: int,
        region_name: str,
        file_path: Path
    ) -> Tuple[Optional[Spectrum], int]:
        energy = []
        intensity = []
        i = start_idx

        while i < len(lines) and self._is_pure_metadata(lines[i]):
            i += 1

        while i < len(lines):
            line = lines[i].strip()

            if not line or extract_region_token(line):
                break

            if self._is_pure_metadata(line):
                i += 1
                continue

            values = self._parse_numeric_values(line)
            if values and len(values) >= 2:
                energy.append(values[0])
                intensity.append(values[1])
                i += 1
            else:
                break

        if len(energy) >= 3:
            try:
                spectrum = Spectrum(
                    name=f"{file_path.stem}_{region_name}",
                    energy=np.array(energy, dtype=np.float64),
                    intensity=np.array(intensity, dtype=np.float64),
                    source_format=self.format_name,
                    metadata={
                        'num_points': len(energy),
                        'source_file': file_path.name,
                        'region': region_name
                    }
                )
                if spectrum.is_valid_xps():
                    if self.debug:
                        print(f"      ✅ Parsed {region_name}: {len(energy)} points")
                    return spectrum, i
            except Exception as e:
                if self.debug:
                    print(f"      ❌ Failed to create spectrum: {e}")
        return None, i


class CSVParser(CSVParserMixin, BaseParser):
    """Parser for standard CSV files (two columns: energy, intensity)."""
    
    format_name = "csv"
    file_extensions = ['.csv']
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if file is simple two-column CSV format.
        
        Returns False if file appears to be multi-region (let EnhancedCSVParser handle it).
        """
        if not self._check_extension(file_path):
            return False
        
        try:
            # Quick check: if file has multiple region names, skip this parser
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                first_lines = [f.readline() for _ in range(10)]
            
            content = ' '.join(first_lines)
            matches = re.findall(REGION_PATTERN_STR, content)
            
            # If multiple regions found, let EnhancedCSVParser handle it
            if len(matches) > 1:
                if self.debug:
                    print(f"   ⏭️ CSVParser: Skipping multi-region file (found {len(matches)} regions)")
                return False
            
            # Otherwise, this is a simple CSV
            return True
            
        except Exception:
            return True  # Default to trying if we can't determine
    
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """
        Import standard two-column CSV.
        
        Tries multiple parsing strategies:
        1. Direct pandas read with auto-detection
        2. Header detection and skiprows
        3. Multiple separator attempts
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List containing single Spectrum, or None if parsing failed
        """
        logging.debug(f"\nAttempting to import CSV file: {file_path}")
        
        # Strategy 1: Direct read with auto-detection
        spectrum = self._try_direct_read(file_path)
        if spectrum:
            return [spectrum]
        
        # Strategy 2: Header detection and manual parsing
        spectrum = self._try_header_detection(file_path)
        if spectrum:
            return [spectrum]
        
        if self.debug:
            print(f"   ❌ CSV import failed for {file_path.name}")
        
        return None
    
    def _try_direct_read(self, file_path: Path) -> Optional[Spectrum]:
        """Try direct pandas read with auto-detection."""
        try:
            logging.debug("Attempting direct CSV read...")
            df = pd.read_csv(
                file_path, 
                sep=None,  # Detect separator automatically
                engine='python',
                header=None,
                skip_blank_lines=True,
                comment='#',
                names=['energy', 'intensity']
            )
            logging.debug(f"Initial read successful. Shape: {df.shape}")
            
            # Convert to numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            logging.debug(f"After numeric conversion. Shape: {df.shape}")
            
            if len(df) >= 2 and not df.empty:
                logging.debug("Creating spectrum object...")
                spectrum = Spectrum(
                    name=file_path.stem,
                    energy=df['energy'].values,
                    intensity=df['intensity'].values,
                    source_format=self.format_name
                )
                
                if spectrum.is_valid_xps():
                    logging.debug(f"Energy range: {df['energy'].min():.2f} to {df['energy'].max():.2f}")
                    logging.debug(f"Mean energy: {df['energy'].mean():.2f}")
                    return spectrum
            
            return None
            
        except Exception as e:
            logging.debug(f"Direct read failed: {str(e)}")
            return None
    
    def _try_header_detection(self, file_path: Path) -> Optional[Spectrum]:
        """Try parsing with header detection."""
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
            
            # Find where numeric data starts
            data_start = 0
            metadata = {}
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                
                # Try to parse as numeric data
                parts = [p for p in re.split(r'[,;\t\s]+', line.strip()) if p]
                
                try:
                    floats = [float(p) for p in parts]
                    if len(floats) >= 2:
                        data_start = i
                        break
                except ValueError:
                    # Store non-numeric lines as metadata
                    if len(parts) >= 1:
                        metadata[f'header_{i}'] = ' '.join(parts)
                    continue
            
            # Try different separators
            for sep in [',', ';', '\t', None]:
                spectrum = self._try_separator(file_path, sep, data_start, metadata)
                if spectrum:
                    return spectrum
            
            return None
            
        except Exception as e:
            logging.debug(f"Header detection failed: {str(e)}")
            return None
    
    def _try_separator(
        self, 
        file_path: Path, 
        sep: Optional[str],
        data_start: int,
        metadata: Dict
    ) -> Optional[Spectrum]:
        """Try parsing with specific separator."""
        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                engine='python' if sep is None else 'c',
                header=None,
                skiprows=data_start,
                comment='#',
                skip_blank_lines=True,
                encoding='utf-8-sig'
            )
            
            # Convert to numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
            if df.empty or len(df.columns) < 2:
                return None
            
            # Take first two columns
            energy = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values
            
            if len(energy) < 5:
                return None
            
            # Build spectrum name
            name_parts = [file_path.stem]
            if 'region' in metadata:
                name_parts.append(metadata['region'])
            spectrum_name = '_'.join(name_parts)
            
            # Add separator info to metadata
            metadata['separator'] = sep or 'auto'
            metadata['source_format'] = self.format_name
            
            spectrum = Spectrum(
                name=spectrum_name,
                energy=energy,
                intensity=intensity,
                source_format=self.format_name,
                metadata=metadata
            )
            
            if spectrum.is_valid_xps():
                return spectrum
            
            return None
            
        except Exception:
            return None


class EnhancedCSVParser(CSVParserMixin, BaseParser):
    """
    Enhanced CSV parser with multi-region support.
    
    Can handle:
    - Single-region CSV files
    - Multi-region CSV files with region headers
    - Table-style multi-region files with paired columns (E1, I1, E2, I2, ...)
    - Paired-row format (energy row followed by intensity row)
    """
    
    format_name = "enhanced_csv"
    file_extensions = ['.csv']
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if file is CSV/text format AND appears to be multi-region.
        
        This parser should be tried BEFORE simple CSVParser to catch
        multi-region files.
        """
        if not self._check_extension(file_path):
            return False
        
        try:
            # Quick check: read first few lines
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = [f.readline() for _ in range(20)]
            
            # Look for multiple region names in header
            content = ' '.join(lines)
            matches = re.findall(REGION_PATTERN_STR, content)
            
            # If we find multiple regions, this parser should handle it
            if len(matches) > 1:
                if self.debug:
                    print(f"   🔍 EnhancedCSVParser: Detected {len(matches)} regions: {matches}")
                return True
            
            # Also check for paired-row format indicators
            for line in lines:
                if 'Area' in line or 'area' in line:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """
        Parse CSV file with multi-region detection.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of Spectrum objects, or None if parsing failed
        """
        if self.debug:
            print(f"\n   🔍 DEBUG: Starting enhanced CSV import of {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                lines = f.readlines()
            
            if self.debug:
                print(f"   🔍 DEBUG: Read {len(lines)} lines")
            
            # Try different parsing strategies
            
            # Strategy 1: Paired-row format (energy row, then intensity row)
            spectra = self._parse_paired_rows(file_path, lines)
            if spectra:
                if self.debug:
                    print(f"   ✅ Parsed using paired-row format: {len(spectra)} regions")
                return spectra
            
            # Strategy 2: Multi-region table with paired columns
            if self._is_multi_region_file(lines):
                if self.debug:
                    print("   🔍 DEBUG: Detected multi-region format")
                spectra = self._parse_multi_region(file_path, lines)
                if spectra:
                    return spectra
            
            # Strategy 3: Single region
            if self.debug:
                print("   🔍 DEBUG: Trying single-region format")
            return self._parse_single_region(file_path, lines)
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ DEBUG: Enhanced CSV import failed: {e}")
                traceback.print_exc()
            return None
    
    def _parse_paired_rows(
        self, 
        file_path: Path, 
        lines: List[str]
    ) -> Optional[List[Spectrum]]:
        """Parse paired-row format where energy and intensity are on consecutive rows."""
        try:
            spectra = []
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                region_name = extract_region_token(line)
                
                if region_name:
                    if self.debug:
                        print(f"   🔍 DEBUG: Found region: {region_name}")
                    
                    j = i + 1
                    while j < len(lines) and self._is_pure_metadata(lines[j]):
                        j += 1
                    
                    if j + 1 < len(lines):
                        energy_values = self._parse_numeric_values(lines[j])
                        intensity_values = self._parse_numeric_values(lines[j + 1])
                        
                        if energy_values and intensity_values:
                            if len(energy_values) == len(intensity_values) and len(energy_values) >= 3:
                                diffs = np.diff(energy_values)
                                if not (np.all(diffs >= 0) or np.all(diffs <= 0)):
                                    # Table-style data will have large jumps; let column parser handle it
                                    i += 1
                                    continue
                                try:
                                    spectrum = Spectrum(
                                        name=f"{file_path.stem}_{region_name}",
                                        energy=np.array(energy_values, dtype=np.float64),
                                        intensity=np.array(intensity_values, dtype=np.float64),
                                        source_format=self.format_name,
                                        metadata={
                                            'num_points': len(energy_values),
                                            'source_file': file_path.name,
                                            'format': 'paired_rows',
                                            'region': region_name
                                        }
                                    )
                                    
                                    if spectrum.is_valid_xps():
                                        spectra.append(spectrum)
                                        if self.debug:
                                            print(f"      ✅ Parsed {region_name}: {len(energy_values)} points")
                                        i = j + 2
                                        continue
                                        
                                except Exception as e:
                                    if self.debug:
                                        print(f"      ❌ Failed to create spectrum: {e}")
                
                i += 1
            
            return spectra if spectra else None
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ Paired-row parsing failed: {e}")
            return None
    
    def _is_multi_region_file(self, lines: List[str]) -> bool:
        """Detect if file contains multiple regions."""
        try:
            content = ' '.join(lines[:100])
            matches = re.findall(REGION_PATTERN_STR, content)
            if len(matches) > 1:
                return True
            # Also detect: >=1 XPS region token AND data rows have >=4 paired columns
            # (e.g., Survey + F1s in a two-region export where Survey is not an XPS token)
            if len(matches) >= 1:
                for line in lines:
                    values = self._parse_numeric_values(line)
                    if values and len(values) >= 4:
                        return True
            return False
        except Exception:
            return False
    
    def _parse_single_region(
        self, 
        file_path: Path, 
        lines: List[str]
    ) -> Optional[List[Spectrum]]:
        """Parse single-region CSV file."""
        try:
            energy = []
            intensity = []
            metadata = {}
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                values = self._parse_numeric_values(line)
                
                if values and len(values) >= 2:
                    energy.append(values[0])
                    intensity.append(values[1])
                elif not energy:
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
    
    def _parse_multi_region(
        self, 
        file_path: Path, 
        lines: List[str]
    ) -> Optional[List[Spectrum]]:
        """Parse multi-region CSV file."""
        spectra = self._parse_multi_region_table(file_path, lines)
        if spectra:
            return spectra
        return self._parse_multi_region_sequential(file_path, lines)
    
    def _parse_multi_region_table(
        self,
        file_path: Path,
        lines: List[str]
    ) -> Optional[List[Spectrum]]:
        """Parse table-style multi-region files with paired columns."""
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore', newline='') as handle:
                reader = csv.reader(handle)
                raw_rows = [row for row in reader]

            if not raw_rows:
                return None

            max_cols = max((len(row) for row in raw_rows), default=0)
            if max_cols < 2:
                return None

            rows: List[List[str]] = []
            for raw in raw_rows:
                cleaned = [cell.strip() for cell in raw]
                if len(cleaned) < max_cols:
                    cleaned.extend([''] * (max_cols - len(cleaned)))
                rows.append(cleaned)

            region_row_idx: Optional[int] = None
            region_columns: Dict[int, str] = {}

            for idx, row in enumerate(rows):
                names: Dict[int, str] = {}
                for col_idx, value in enumerate(row):
                    if not value:
                        continue
                    region_name = extract_region_token(value)
                    if region_name:
                        names[col_idx] = region_name
                if len(names) >= 2:
                    region_row_idx = idx
                    region_columns = dict(sorted(names.items()))
                    break
                # Single XPS region alongside non-XPS label (e.g., "Survey" + "F1s")
                # Accept if at least 1 XPS token found and there are paired columns
                if len(names) >= 1:
                    # Check if data rows actually have enough columns for pairing
                    sample_data = [r for r in rows[idx + 1:] if any(self._safe_float(c) is not None for c in r)]
                    if sample_data and len(sample_data[0]) >= 4:
                        region_row_idx = idx
                        region_columns = dict(sorted(names.items()))
                        break
                # Special case: instrument headers with alternating blank columns (e.g., Li1s,,F1s,,...)
                if not names and idx + 1 < len(rows):
                    condensed = [cell for cell in row if cell]
                    region_tokens = [extract_region_token(cell) for cell in condensed]
                    region_tokens = [token for token in region_tokens if token]
                    if len(region_tokens) >= 2:
                        region_row_idx = idx
                        region_columns = {i * 2: token for i, token in enumerate(region_tokens)}
                        break

            if region_row_idx is None or not region_columns:
                if self.debug:
                    print("   ⚠️ No header row found in table format")
                return None

            if self.debug:
                print(f"   🔍 DEBUG: Found regions: {list(region_columns.values())}")

            spectra: List[Spectrum] = []
            region_indices = list(region_columns.keys())
            data_rows = rows[region_row_idx + 1 :]

            for idx, start_col in enumerate(region_indices):
                region_name = region_columns[start_col]
                next_boundary = region_indices[idx + 1] if idx + 1 < len(region_indices) else max_cols
                end_col = next_boundary if next_boundary > start_col else start_col + 2
                intensity_cols = list(range(start_col + 1, end_col))

                energy_values: List[float] = []
                intensity_values: List[float] = []
                used_intensity_cols: set[int] = set()

                for row in data_rows:
                    if start_col >= len(row):
                        continue
                    energy_val = self._safe_float(row[start_col])
                    if energy_val is None:
                        continue

                    sample_intensities: List[float] = []
                    for col in intensity_cols:
                        if col >= len(row):
                            continue
                        intensity_val = self._safe_float(row[col])
                        if intensity_val is not None:
                            sample_intensities.append(intensity_val)
                            used_intensity_cols.add(col)

                    if not sample_intensities:
                        continue

                    if energy_val < 10 and max(sample_intensities) < 10:
                        continue

                    energy_values.append(energy_val)
                    intensity_values.append(float(np.mean(sample_intensities)))

                if len(energy_values) < 3:
                    if self.debug:
                        print(f"   ⚠️ Insufficient data for {region_name}")
                    continue

                energy_array = np.asarray(energy_values, dtype=np.float64)
                intensity_array = np.asarray(intensity_values, dtype=np.float64)

                try:
                    spectrum = Spectrum(
                        name=f"{file_path.stem}_{region_name}",
                        energy=energy_array,
                        intensity=intensity_array,
                        source_format=self.format_name,
                        metadata={
                            'num_points': len(energy_array),
                            'source_file': file_path.name,
                            'multi_region_regions': list(region_columns.values()),
                            'format': 'paired_columns',
                            'replicate_columns': len(used_intensity_cols) if used_intensity_cols else len(intensity_cols),
                            'region': region_name
                        }
                    )

                    if spectrum.is_valid_xps():
                        spectra.append(spectrum)
                        if self.debug:
                            print(
                                f"   ✅ Parsed {region_name}: {len(energy_array)} points, "
                                f"BE: {spectrum.energy.min():.2f}-{spectrum.energy.max():.2f} eV"
                            )
                    else:
                        if self.debug:
                            print(f"   ⚠️ {region_name} failed XPS validation")

                except Exception as exc:
                    if self.debug:
                        print(f"   ❌ Failed to create spectrum for {region_name}: {exc}")

            return spectra if spectra else None
        except Exception as exc:
            if self.debug:
                print(f"   ❌ Table parsing failed: {exc}")
                traceback.print_exc()
            return None
    
    def _parse_multi_region_sequential(
        self, 
        file_path: Path, 
        lines: List[str]
    ) -> Optional[List[Spectrum]]:
        """Parse sequential multi-region files."""
        spectra = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                i += 1
                continue
            
            region_name = extract_region_token(line)
            if region_name:
                if self.debug:
                    print(f"   🔍 DEBUG: Found region: {region_name}")
                
                spectrum, next_i = self._parse_region_data(
                    lines, i + 1, region_name, file_path
                )
                
                if spectrum:
                    spectra.append(spectrum)
                
                i = next_i
            else:
                i += 1
        
        return spectra if spectra else None

class DepthProfileCSVParser(CSVParserMixin, BaseParser):
    """Parser for depth-profile CSV exports with multiple layers per region."""

    format_name = "csv_depth"
    file_extensions = ['.csv']

    def can_parse(self, file_path: Path) -> bool:
        """Detect depth-profile CSVs by filename and header structure."""
        if not self._check_extension(file_path):
            return False

        try:
            rows = self._read_csv_rows(file_path, max_rows=8)
            header_idx, regions = self._find_region_header(rows)
            if header_idx is None or len(regions) < 2:
                return False
            max_cols = max((len(row) for row in rows), default=0)
            layer_counts = self._infer_layer_counts(rows, header_idx, regions, max_cols)
            return any(count > 1 for count in layer_counts.values())
        except Exception:
            return False

    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Parse depth-profile CSV where each region has multiple layer columns."""
        try:
            rows = self._read_csv_rows(file_path)
            if not rows:
                return None

            header_idx, region_positions = self._find_region_header(rows)
            if header_idx is None or not region_positions:
                return None

            max_cols = max((len(row) for row in rows), default=0)
            layer_counts = self._infer_layer_counts(rows, header_idx, region_positions, max_cols)
            data_start = header_idx + 2
            if data_start >= len(rows):
                return None

            region_layer_data: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

            for row in rows[data_start:]:
                if not any(cell.strip() for cell in row):
                    continue
                for idx, (col_idx, region_name) in enumerate(region_positions):
                    next_col = region_positions[idx + 1][0] if idx + 1 < len(region_positions) else max_cols
                    if col_idx >= len(row):
                        continue
                    energy = self._safe_float(row[col_idx])
                    if energy is None:
                        continue

                    layer_count = layer_counts.get(region_name, 1)
                    for layer_idx in range(layer_count):
                        intensity_col = col_idx + 1 + layer_idx
                        if intensity_col >= len(row):
                            continue
                        intensity = self._safe_float(row[intensity_col])
                        if intensity is None:
                            continue

                        store = region_layer_data.setdefault(region_name, {}).setdefault(
                            layer_idx + 1, {"energy": [], "intensity": []}
                        )
                        store["energy"].append(energy)
                        store["intensity"].append(intensity)

            spectra: List[Spectrum] = []
            for region_name, layers in region_layer_data.items():
                for layer_idx, data in layers.items():
                    energy = np.asarray(data["energy"], dtype=np.float64)
                    intensity = np.asarray(data["intensity"], dtype=np.float64)
                    if len(energy) < 3:
                        continue
                    try:
                        spectrum = Spectrum(
                            name=f"{file_path.stem}_{region_name}_L{layer_idx}",
                            energy=energy,
                            intensity=intensity,
                            source_format=self.format_name,
                            metadata={
                                "region": region_name,
                                "layer_index": layer_idx,
                                "source_file": file_path.name,
                                "sample_label": f"{file_path.stem}_Layer{layer_idx}",
                                "sample_base_label": file_path.stem,
                                "depth_profile": True,
                                "num_points": len(energy),
                            },
                        )
                        if spectrum.is_valid_xps():
                            spectra.append(spectrum)
                    except Exception:
                        continue

            return spectra if spectra else None
        except Exception:
            return None

    def _read_csv_rows(self, file_path: Path, max_rows: Optional[int] = None) -> List[List[str]]:
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore', newline='') as handle:
            reader = csv.reader(handle)
            rows = []
            for idx, row in enumerate(reader):
                rows.append([cell.strip() for cell in row])
                if max_rows is not None and idx + 1 >= max_rows:
                    break
        return rows

    def _find_region_header(self, rows: List[List[str]]) -> Tuple[Optional[int], List[Tuple[int, str]]]:
        for idx, row in enumerate(rows):
            region_map: Dict[int, str] = {}
            for col_idx, cell in enumerate(row):
                region = extract_region_token(cell)
                if region:
                    region_map[col_idx] = region
            if len(region_map) >= 2:
                return idx, sorted(region_map.items())
        return None, []

    def _infer_layer_counts(
        self,
        rows: List[List[str]],
        header_idx: int,
        region_positions: List[Tuple[int, str]],
        max_cols: int,
    ) -> Dict[str, int]:
        layer_row = rows[header_idx + 1] if header_idx + 1 < len(rows) else []
        counts: Dict[str, int] = {}
        for idx, (col_idx, region_name) in enumerate(region_positions):
            next_col = region_positions[idx + 1][0] if idx + 1 < len(region_positions) else max_cols
            layer_count = 0
            for layer_col in range(col_idx + 1, next_col):
                if layer_col >= len(layer_row):
                    continue
                val = layer_row[layer_col].strip()
                if not val:
                    continue
                if val.replace('.', '', 1).isdigit():
                    layer_count += 1
            counts[region_name] = layer_count if layer_count > 0 else 1
        return counts
