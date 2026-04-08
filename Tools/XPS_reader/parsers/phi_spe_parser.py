"""Parser for PHI .spe format XPS data files."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseParser

import sys

sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum
from core.metadata import XPSMetadata


class PHISPEParser(BaseParser):
    """Parser for PHI MultiPak .spe format files."""

    format_name = "phi_spe"
    file_extensions = [".spe"]

    _pynxtools_reader_cls = None
    _pynxtools_import_error: Optional[Exception] = None

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is PHI SPE format with enhanced format detection."""
        if not self._check_extension(file_path):
            return False

        try:
            file_size = file_path.stat().st_size
            if file_size < 300:  # Minimum viable .spe file size
                return False
                
            with open(file_path, "rb") as file_handle:
                # Read first 4 bytes to check format signature
                magic = file_handle.read(4)
                
                # Check for SOFH format (modern PHI format)
                try:
                    magic_str = magic.decode("ascii", errors="ignore")
                    if magic_str == "SOFH":
                        if self.debug:
                            print(f"   ✓ Detected SOFH format")
                        return True
                except UnicodeDecodeError:
                    pass
                
                # Reset and check for legacy PHI format
                file_handle.seek(0)
                
                # Try different header reading strategies
                format_detected = False
                
                # Strategy 1: Read 256-byte header and check for PHI signature
                try:
                    header_data = file_handle.read(256)
                    header_str = header_data.decode("ascii", errors="ignore").strip("\x00 ")
                    
                    if header_str.startswith("PHI") or "MultiPak" in header_str or "Physical Electronics" in header_str:
                        format_detected = True
                        if self.debug:
                            print(f"   ✓ Detected PHI signature in header")
                except (UnicodeDecodeError, Exception):
                    pass
                
                # Strategy 2: Check binary structure after header
                if not format_detected:
                    try:
                        file_handle.seek(256)  # Skip header
                        
                        # Try to read structural parameters
                        params_data = file_handle.read(28)  # n_energy, nx, ny, energy_start, energy_step
                        if len(params_data) >= 20:
                            n_energy = struct.unpack("<l", params_data[0:4])[0]  # Little endian
                            nx = struct.unpack("<l", params_data[4:8])[0]
                            ny = struct.unpack("<l", params_data[8:12])[0]
                            
                            # Validate reasonable parameters
                            if (1 <= n_energy <= 100000 and 1 <= nx <= 10000 and 1 <= ny <= 10000):
                                format_detected = True
                                if self.debug:
                                    print(f"   ✓ Valid PHI structure: n_energy={n_energy}, nx={nx}, ny={ny}")
                            
                            # Try big endian if little endian failed
                            if not format_detected:
                                n_energy = struct.unpack(">l", params_data[0:4])[0]  # Big endian
                                nx = struct.unpack(">l", params_data[4:8])[0]
                                ny = struct.unpack(">l", params_data[8:12])[0]
                                
                                if (1 <= n_energy <= 100000 and 1 <= nx <= 10000 and 1 <= ny <= 10000):
                                    format_detected = True
                                    if self.debug:
                                        print(f"   ✓ Valid PHI structure (big endian): n_energy={n_energy}, nx={nx}, ny={ny}")
                                        
                    except (struct.error, Exception):
                        pass
                
                # Strategy 3: Check for specific byte patterns that indicate PHI format
                if not format_detected:
                    file_handle.seek(0)
                    first_512_bytes = file_handle.read(512)
                    
                    # Look for PHI-specific patterns
                    phi_indicators = [
                        b"PHI", b"MultiPak", b"Physical Electronics",
                        b"ULVAC-PHI", b"Quantera", b"VersaProbe"
                    ]
                    
                    for indicator in phi_indicators:
                        if indicator in first_512_bytes:
                            format_detected = True
                            if self.debug:
                                print(f"   ✓ Found PHI indicator: {indicator.decode('ascii', errors='ignore')}")
                            break
                
                return format_detected
                
        except Exception as e:
            if self.debug:
                print(f"   ❌ Format detection error: {e}")
            logging.warning("File format recognized but rejected due to corruption/incompatibility. Check the raw SPE file in the raw_data folder.")
            return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Parse PHI SPE file with enhanced version detection and parsing."""
        if self.debug:
            print(f"   🔍 Analyzing {file_path.name} for PHI format version...")
            
        try:
            # Detect format version first
            format_info = self._detect_spe_version(file_path)
            if not format_info:
                if self.debug:
                    print(f"   ❌ Could not determine PHI format version")
                return None
                
            version = format_info['version']
            endian = format_info['endian']
            
            if self.debug:
                print(f"   📋 Detected PHI format: {version} ({endian})")
            
            # Parse based on detected version
            if version == 'SOFH':
                # Try modern pynxtools parser first for SOFH
                spectra = self._parse_with_pynxtools(file_path)
                if spectra:
                    if self.debug:
                        print(f"   ✅ SOFH format parsed successfully with pynxtools")
                    return spectra
                    
                # If pynxtools failed, try our own SOFH parser
                if self.debug:
                    print(f"   ⚠️ pynxtools failed, trying custom SOFH parser")
                spectra = self._parse_sofh_format(file_path)
                if spectra:
                    return spectra
                    
            elif version in ['legacy_v1', 'legacy_v2', 'legacy_v3']:
                # Use enhanced legacy parser with version-specific handling
                spectra = self._parse_legacy_format(file_path, version, endian)
                if spectra:
                    if self.debug:
                        print(f"   ✅ Legacy format ({version}) parsed successfully")
                    return spectra
                    
            # Final fallback - try basic single region parser
            if self.debug:
                print(f"   🔄 Trying fallback single-region parser")
            legacy_result = self._parse_single_region(file_path)
            if legacy_result:
                data, metadata = legacy_result
                spectrum = Spectrum(
                    name=file_path.stem,
                    energy=metadata.energy_values,
                    intensity=data if data.ndim == 1 else data.flatten(),
                    source_format=self.format_name,
                    metadata={
                        "region": metadata.region,
                        "scan_mode": metadata.scan_mode,
                        "pass_energy": metadata.pass_energy,
                        "nx": metadata.nx,
                        "ny": metadata.ny,
                        "comments": metadata.comments,
                        "phi_version": version,
                        "endian": endian
                    },
                )
                
                if spectrum.is_valid_xps():
                    if self.debug:
                        print(f"   ✅ Fallback parser succeeded")
                    return [spectrum]
                    
            if self.debug:
                print(f"   ❌ All parsing strategies failed")
            return None
            
        except Exception as exc:
            if self.debug:
                logging.exception("PHI SPE parsing failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # pynxtools integration
    # ------------------------------------------------------------------ #
    @classmethod
    def _get_pynxtools_reader(cls):
        """Import XPSReader so installs remain optional."""
        if cls._pynxtools_reader_cls is not None:
            return cls._pynxtools_reader_cls
        if cls._pynxtools_import_error:
            return None

        try:
            from pynxtools_xps.reader import XPSReader
            cls._pynxtools_reader_cls = XPSReader
            return XPSReader
        except ImportError as err:
            cls._pynxtools_import_error = err
            logging.warning("pynxtools-xps is unavailable. Ensure it is installed and accessible.")
            return None

    def _parse_with_pynxtools(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Use pynxtools-xps to read MultiPak files with multiple regions."""
        reader_cls = self._get_pynxtools_reader()
        if reader_cls is None:
            return None

        try:
            reader = reader_cls()
            reader.read(file_paths=(str(file_path),))
        except Exception as err:
            logging.warning("pynxtools-xps failed on %s: %s", file_path, err)
            return None

        data_dict = reader.xps_data.get("data")
        if not data_dict:
            return None

        spectra: List[Spectrum] = []
        for entry_name, xr_data in data_dict.items():
            energy_coord = xr_data.coords.get("energy")
            if energy_coord is None:
                continue

            intensity = self._select_signal_array(xr_data)
            if intensity is None:
                continue

            metadata = self._extract_entry_metadata(reader.xps_data, entry_name)
            metadata.setdefault("region", entry_name)

            spectrum = Spectrum(
                name=f"{file_path.stem}_{entry_name}",
                energy=np.asarray(getattr(energy_coord, "values", energy_coord)),
                intensity=np.asarray(intensity),
                source_format=self.format_name,
                metadata=metadata,
            )

            if spectrum.is_valid_xps():
                spectra.append(spectrum)

        return spectra or None

    @staticmethod
    def _select_signal_array(xr_data: Any) -> Optional[np.ndarray]:
        """Pick a 1D signal array from the xarray Dataset."""
        for name, data_array in xr_data.data_vars.items():
            dims = getattr(data_array, "dims", ())
            if "energy" not in dims:
                continue

            values = np.asarray(data_array.values)
            if values.ndim == 1:
                return values

            squeezed = np.squeeze(values)
            if squeezed.ndim == 1:
                logging.debug(
                    "Squeezed %s from shape %s to %s", name, values.shape, squeezed.shape
                )
                return squeezed
        return None

    def _extract_entry_metadata(self, xps_data: Dict[str, Any], entry: str) -> Dict[str, Any]:
        """Collect a small metadata subset for each entry."""
        metadata: Dict[str, Any] = {}
        base = f"/ENTRY[{entry}]"
        keys_of_interest = [
            "spectrum_type",
            "region_definition",
            "region_definition2",
            "region_background",
            "pass_energy",
            "dwell_time",
        ]

        for key in keys_of_interest:
            value_key = f"{base}/{key}"
            value = xps_data.get(value_key)
            if value is None:
                continue
            metadata[key] = self._simplify_metadata_value(value)

            units_key = f"{value_key}/@units"
            if units_key in xps_data:
                metadata[f"{key}_units"] = xps_data[units_key]

        return metadata

    @staticmethod
    def _simplify_metadata_value(value: Any) -> Any:
        """Convert numpy scalars/bytes from pynxtools to plain python types."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.item()
        return value
    
    def _detect_spe_version(self, file_path: Path) -> Optional[Dict[str, str]]:
        """Detect specific PHI .spe format version and characteristics."""
        try:
            with open(file_path, "rb") as f:
                # Check for SOFH format first
                magic = f.read(4)
                if magic == b'SOFH':
                    return {'version': 'SOFH', 'endian': 'little'}
                    
                # Reset and analyze legacy format
                f.seek(0)
                header_data = f.read(256)
                
                # Check for different legacy versions
                header_str = header_data.decode('ascii', errors='ignore')
                
                # Determine version based on header content and structure
                if 'MultiPak' in header_str and 'v9' in header_str:
                    version = 'legacy_v3'  # Newer MultiPak format
                elif 'MultiPak' in header_str:
                    version = 'legacy_v2'  # Standard MultiPak
                elif 'PHI' in header_str or 'Physical Electronics' in header_str:
                    version = 'legacy_v1'  # Older PHI format
                else:
                    # Try to infer from structure
                    f.seek(256)
                    try:
                        test_data = f.read(12)
                        if len(test_data) >= 12:
                            # Test both endianness
                            n_energy_le = struct.unpack('<l', test_data[0:4])[0]
                            n_energy_be = struct.unpack('>l', test_data[0:4])[0]
                            
                            if 1 <= n_energy_le <= 100000:
                                version = 'legacy_v1'
                                endian = 'little'
                            elif 1 <= n_energy_be <= 100000:
                                version = 'legacy_v1' 
                                endian = 'big'
                            else:
                                return None
                        else:
                            return None
                    except struct.error:
                        return None
                
                # Determine endianness for legacy formats
                if 'endian' not in locals():
                    f.seek(256)
                    try:
                        test_data = f.read(4)
                        if len(test_data) >= 4:
                            n_energy_le = struct.unpack('<l', test_data)[0]
                            n_energy_be = struct.unpack('>l', test_data)[0]
                            
                            if 1 <= n_energy_le <= 100000:
                                endian = 'little'
                            elif 1 <= n_energy_be <= 100000:
                                endian = 'big'
                            else:
                                endian = 'little'  # default
                        else:
                            endian = 'little'
                    except struct.error:
                        endian = 'little'
                        
                return {'version': version, 'endian': endian}
                
        except Exception as e:
            if self.debug:
                print(f"   ❌ Version detection error: {e}")
            return None

    def _parse_sofh_format(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Custom SOFH format parser for when pynxtools fails."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                
            # Find end of header
            eoh_idx = content.find(b"EOFH\r\n")
            if eoh_idx == -1:
                # Try just EOFH
                eoh_idx = content.find(b"EOFH")
                if eoh_idx == -1:
                    if self.debug:
                        print(f"   ❌ SOFH header end not found")
                    return None
                header_end = eoh_idx + 4
            else:
                header_end = eoh_idx + 6
                
            header_text = content[:eoh_idx].decode("utf-8", errors="ignore")
            binary_data = content[header_end:]
            
            # Parse regions from header
            regions = []
            lines = header_text.splitlines()

            # Prefer original acquisition filename when available
            source_file = file_path.name
            for line in lines:
                if line.startswith("AcqFilename:"):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        source_file = Path(value).name
                        break
            
            # Check for Full definitions first
            has_full = any("SpectralRegDefFull" in line for line in lines)
            
            if has_full:
                def_key = "SpectralRegDefFull"
            else:
                def_key = "SpectralRegDef"
                
            for line in lines:
                if line.startswith(def_key + ":"):
                    # Parse definition
                    # Format: Index ? Name ? Points Step Start End ...
                    # Example: SpectralRegDefFull: 1 1 Li1s 3 201 -0.1000 65.0000 45.0000 ...
                    parts = line.split(":")[1].strip().split()
                    if len(parts) >= 8:
                        try:
                            try:
                                region_index = int(parts[0])
                            except (ValueError, IndexError):
                                region_index = None
                            try:
                                active_flag = int(parts[1])
                            except (ValueError, IndexError):
                                active_flag = 1
                            name = parts[2]
                            n_points = int(parts[4])
                            step = float(parts[5])
                            start = float(parts[6])
                            end = float(parts[7])
                            
                            regions.append({
                                "index": region_index,
                                "active": active_flag,
                                "name": name,
                                "n_points": n_points,
                                "step": step,
                                "start": start,
                                "end": end,
                                "metadata": {
                                    "line": line,
                                    "source_file": source_file,
                                }
                            })
                        except (ValueError, IndexError):
                            continue
                            
            if not regions:
                if self.debug:
                    print(f"   ❌ No regions found in SOFH header")
                return None
                
            # Extract data from binary part
            spectra = []

            data_blocks = self._parse_sofh_data_blocks(binary_data)
            if not data_blocks:
                if self.debug:
                    print("   ? No valid SOFH data blocks found")
                return None

            # Prefer only active regions when available
            active_regions = [r for r in regions if r.get("active", 1) != 0]
            if not active_regions:
                active_regions = regions

            # Map blocks to regions in order of increasing data offset
            data_blocks.sort(key=lambda b: b["offset"])
            remaining_blocks = data_blocks.copy()

            for region in active_regions:
                expected_points = region["n_points"]

                block = None
                for candidate in remaining_blocks:
                    bytes_per_point = candidate["bytes_per_point"]
                    expected_bytes = expected_points * bytes_per_point
                    if candidate["size"] == expected_bytes:
                        block = candidate
                        break

                if block is None and remaining_blocks:
                    block = remaining_blocks[0]

                if block is None:
                    if self.debug:
                        print(f"   ?? No data block available for {region['name']}")
                    continue

                remaining_blocks.remove(block)

                data_start = block["offset"]
                data_end = data_start + block["size"]
                if data_end > len(binary_data):
                    if self.debug:
                        print(f"   ? Incomplete data for {region['name']}")
                    continue

                data_bytes = binary_data[data_start:data_end]
                intensity = np.frombuffer(data_bytes, dtype=block["dtype"])

                if len(intensity) != expected_points:
                    if len(intensity) > expected_points:
                        intensity = intensity[:expected_points]
                    else:
                        if self.debug:
                            print(
                                f"   ?? Data length mismatch for {region['name']}: "
                                f"expected {expected_points}, got {len(intensity)}"
                            )
                        continue

                # Create energy axis
                energy = region["start"] + np.arange(expected_points) * region["step"]

                spectrum = Spectrum(
                    name=region["name"],
                    energy=energy,
                    intensity=intensity.astype(np.float32, copy=False),
                    source_format=self.format_name,
                    metadata=region["metadata"]
                )

                if spectrum.is_valid_xps():
                    spectra.append(spectrum)

            return spectra if spectra else None
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ SOFH parsing error: {e}")
            return None
    

    @staticmethod
    def _parse_sofh_data_blocks(binary_data: bytes) -> List[Dict[str, Any]]:
        """Parse SOFH binary block table into data blocks with offsets."""
        type_map: Dict[bytes, Tuple[str, int]] = {
            b"f4\x00\x00": ("<f4", 4),
            b"f8\x00\x00": ("<f8", 8),
            b"i4\x00\x00": ("<i4", 4),
            b"u4\x00\x00": ("<u4", 4),
            b"i2\x00\x00": ("<i2", 2),
            b"u2\x00\x00": ("<u2", 2),
        }

        blocks: List[Dict[str, Any]] = []
        data_len = len(binary_data)

        idx = 0
        while idx + 12 <= data_len:
            marker = binary_data[idx:idx + 4]
            if marker in type_map:
                size = struct.unpack("<I", binary_data[idx + 4:idx + 8])[0]
                offset = struct.unpack("<I", binary_data[idx + 8:idx + 12])[0]
                dtype, bytes_per_point = type_map[marker]

                if size > 0 and offset >= 0 and offset + size <= data_len and offset >= idx:
                    blocks.append({
                        "dtype": dtype,
                        "bytes_per_point": bytes_per_point,
                        "size": size,
                        "offset": offset,
                        "descriptor_offset": idx,
                    })
                    idx += 12
                    continue

            idx += 1

        # Deduplicate by offset (keep first occurrence)
        seen_offsets = set()
        unique_blocks = []
        for block in blocks:
            if block["offset"] in seen_offsets:
                continue
            seen_offsets.add(block["offset"])
            unique_blocks.append(block)

        return unique_blocks

    def _parse_legacy_format(self, file_path: Path, version: str, endian: str) -> Optional[List[Spectrum]]:
        """Parse legacy PHI formats with version-specific handling."""
        try:
            endian_char = '<' if endian == 'little' else '>'
            
            with open(file_path, "rb") as f:
                # Read header
                header = f.read(256).decode("ascii", errors="ignore").strip("\x00 ")
                
                # Read structure parameters with correct endianness
                n_energy = struct.unpack(f"{endian_char}l", f.read(4))[0]
                nx = struct.unpack(f"{endian_char}l", f.read(4))[0]
                ny = struct.unpack(f"{endian_char}l", f.read(4))[0]
                
                # Validate parameters
                if not (1 <= n_energy <= 100000 and 1 <= nx <= 10000 and 1 <= ny <= 10000):
                    if self.debug:
                        print(f"   ❌ Invalid parameters: n_energy={n_energy}, nx={nx}, ny={ny}")
                    return None
                
                # Read energy calibration
                energy_start = struct.unpack(f"{endian_char}d", f.read(8))[0]
                energy_step = struct.unpack(f"{endian_char}d", f.read(8))[0]
                pass_energy = struct.unpack(f"{endian_char}d", f.read(8))[0]
                
                # Read spatial calibration (if applicable)
                x_start = struct.unpack(f"{endian_char}d", f.read(8))[0] if nx > 1 else 0.0
                x_step = struct.unpack(f"{endian_char}d", f.read(8))[0] if nx > 1 else 1.0
                y_start = struct.unpack(f"{endian_char}d", f.read(8))[0] if ny > 1 else 0.0
                y_step = struct.unpack(f"{endian_char}d", f.read(8))[0] if ny > 1 else 1.0
                
                # Read region and scan mode
                region = f.read(32).decode("ascii", errors="ignore").strip("\x00 ")
                scan_mode = f.read(32).decode("ascii", errors="ignore").strip("\x00 ")
                
                # Version-specific comment handling
                if version == 'legacy_v3':
                    # Newer format may have extended comment structure
                    comment_size = struct.unpack(f"{endian_char}l", f.read(4))[0]
                    if 0 <= comment_size <= 10000:  # Reasonable comment size
                        comments = f.read(comment_size).decode("ascii", errors="ignore").strip("\x00 ")
                    else:
                        comments = ""
                elif version in ['legacy_v1', 'legacy_v2']:
                    # Standard comment handling
                    try:
                        comment_size = struct.unpack(f"{endian_char}l", f.read(4))[0]
                        if 0 <= comment_size <= 5000:  # More conservative for older formats
                            comments = f.read(comment_size).decode("ascii", errors="ignore").strip("\x00 ")
                        else:
                            comments = ""
                    except:
                        comments = ""
                
                # Create energy axis
                energy_values = energy_start + energy_step * np.arange(n_energy)
                
                # Read spectral data
                data_size = np.int64(nx) * np.int64(ny) * np.int64(n_energy)
                
                # Safety check for file size
                max_size = 500 * 1024 * 1024 // 4  # 500MB in float32 elements
                if data_size > max_size:
                    if self.debug:
                        print(f"   ❌ Data too large: {data_size} points ({data_size*4//1024//1024} MB)")
                    return None
                
                # Read intensity data
                data_bytes = int(data_size * 4)
                intensity_data = f.read(data_bytes)
                if len(intensity_data) < data_bytes:
                    if self.debug:
                        print(f"   ❌ Incomplete data: expected {data_bytes}, got {len(intensity_data)}")
                    return None
                
                # Convert to numpy array with correct endianness
                dtype = f"{endian_char}f4"  # float32 with correct endianness
                intensities = np.frombuffer(intensity_data, dtype=dtype)
                
                # Create spectra based on dimensions
                spectra = []
                
                if nx == 1 and ny == 1:
                    # Single spectrum
                    spectrum = Spectrum(
                        name=f"{file_path.stem}",
                        energy=energy_values,
                        intensity=intensities,
                        source_format=self.format_name,
                        metadata={
                            "region": region,
                            "scan_mode": scan_mode,
                            "pass_energy": pass_energy,
                            "comments": comments,
                            "phi_version": version,
                            "endian": endian,
                            "nx": nx,
                            "ny": ny
                        }
                    )
                    
                    if spectrum.is_valid_xps():
                        spectra.append(spectrum)
                        
                else:
                    # Multi-dimensional data - create multiple spectra
                    intensities = intensities.reshape(ny, nx, n_energy)
                    
                    for y_idx in range(ny):
                        for x_idx in range(nx):
                            spectrum_name = f"{file_path.stem}_x{x_idx}_y{y_idx}"
                            if nx > 1 and ny > 1:
                                spectrum_name = f"{file_path.stem}_pos_{x_idx}_{y_idx}"
                            elif nx > 1:
                                spectrum_name = f"{file_path.stem}_x{x_idx}"
                            elif ny > 1:
                                spectrum_name = f"{file_path.stem}_y{y_idx}"
                                
                            spectrum = Spectrum(
                                name=spectrum_name,
                                energy=energy_values,
                                intensity=intensities[y_idx, x_idx, :],
                                source_format=self.format_name,
                                metadata={
                                    "region": region,
                                    "scan_mode": scan_mode,
                                    "pass_energy": pass_energy,
                                    "comments": comments,
                                    "phi_version": version,
                                    "endian": endian,
                                    "nx": nx,
                                    "ny": ny,
                                    "x_position": x_start + x_idx * x_step,
                                    "y_position": y_start + y_idx * y_step,
                                    "x_index": x_idx,
                                    "y_index": y_idx
                                }
                            )
                            
                            if spectrum.is_valid_xps():
                                spectra.append(spectrum)
                                
                return spectra if spectra else None
                
        except Exception as e:
            if self.debug:
                print(f"   ❌ Legacy parsing error: {e}")
            return None
    
    # ------------------------------------------------------------------ #
    # Legacy single-region parser (kept as fallback)
    # ------------------------------------------------------------------ #
    def _parse_single_region(
        self, file_path: Path
    ) -> Optional[Tuple[np.ndarray, XPSMetadata]]:
        """Parse PHI .spe format XPS data file (legacy single-region support)."""
        try:
            with open(file_path, "rb") as file_handle:
                header = file_handle.read(256).decode("ascii", errors="ignore").strip("\x00 ")
                if not header.startswith(("PHI", "SOFH")):
                    raise ValueError("Not a valid PHI .spe file")

                n_energy = struct.unpack("l", file_handle.read(4))[0]
                nx = struct.unpack("l", file_handle.read(4))[0]
                ny = struct.unpack("l", file_handle.read(4))[0]
                
                # Validate parameters to prevent overflow/corruption; if obviously wrong,
                # bail so other parsers (e.g., ASCII) can take over.
                if n_energy <= 0 or n_energy > 100000:
                    logging.warning(f"Skipping legacy PHI parse: invalid n_energy={n_energy} (expected 1-100000)")
                    return None
                if nx <= 0 or nx > 10000:
                    logging.warning(f"Skipping legacy PHI parse: invalid nx={nx} (expected 1-10000)")
                    return None
                if ny <= 0 or ny > 10000:
                    logging.warning(f"Skipping legacy PHI parse: invalid ny={ny} (expected 1-10000)")
                    return None

                energy_start = struct.unpack("d", file_handle.read(8))[0]
                energy_step = struct.unpack("d", file_handle.read(8))[0]
                pass_energy = struct.unpack("d", file_handle.read(8))[0]

                x_start = struct.unpack("d", file_handle.read(8))[0] if nx > 1 else 0.0
                x_step = struct.unpack("d", file_handle.read(8))[0] if nx > 1 else 1.0
                y_start = struct.unpack("d", file_handle.read(8))[0] if ny > 1 else 0.0
                y_step = struct.unpack("d", file_handle.read(8))[0] if ny > 1 else 1.0

                region = file_handle.read(32).decode("ascii", errors="ignore").strip("\x00 ")
                scan_mode = file_handle.read(32).decode("ascii", errors="ignore").strip("\x00 ")

                comment_size = struct.unpack("l", file_handle.read(4))[0]
                comments = (
                    file_handle.read(comment_size)
                    .decode("ascii", errors="ignore")
                    .strip("\x00 ")
                    if comment_size > 0
                    else None
                )

                energy_values = energy_start + energy_step * np.arange(n_energy)

                # Use int64 to prevent overflow for large files
                data_size = np.int64(nx) * np.int64(ny) * np.int64(n_energy)
                
                # Check for reasonable data size (< 500MB)
                max_size = 500 * 1024 * 1024 // 4  # 500MB in float32 elements
                if data_size > max_size:
                    raise ValueError(f"File too large: {data_size} points ({data_size*4//1024//1024} MB). "
                                   f"Maximum supported: {max_size} points ({max_size*4//1024//1024} MB)")
                
                # Convert back to int for reading (should be safe now)
                data_bytes_to_read = int(data_size * 4)
                data = np.frombuffer(file_handle.read(data_bytes_to_read), dtype=np.float32)

                if nx > 1 or ny > 1:
                    data = data.reshape(ny, nx, n_energy)
                else:
                    data = data.reshape(n_energy)

                metadata = XPSMetadata(
                    source_format=self.format_name,
                    source_file=str(file_path),
                    region=region,
                    scan_mode=scan_mode,
                    pass_energy=pass_energy,
                    x_start=x_start,
                    x_step=x_step,
                    nx=nx,
                    y_start=y_start,
                    y_step=y_step,
                    ny=ny,
                    energy_values=energy_values,
                    comments=comments,
                )

                return data, metadata

        except Exception as err:
            logging.error("Error parsing PHI file %s: %s", file_path, err)
            if self.debug:
                raise
        return None
