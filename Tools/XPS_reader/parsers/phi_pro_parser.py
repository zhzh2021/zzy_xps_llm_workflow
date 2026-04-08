"""Parser for PHI .pro (depth profile) format XPS data files."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseParser
from .phi_spe_parser import PHISPEParser

import sys

sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum


class PHIPROParser(BaseParser):
    """Parser for PHI MultiPak .pro depth profile files (SOFH)."""

    format_name = "phi_pro"
    file_extensions = [".pro"]

    def can_parse(self, file_path: Path) -> bool:
        if not self._check_extension(file_path):
            return False

        try:
            with open(file_path, "rb") as file_handle:
                magic = file_handle.read(4)
            if magic != b"SOFH":
                return False
        except Exception:
            return False

        return True

    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Parse PHI .pro file into spectra (region x depth cycle)."""
        try:
            return self._parse_sofh_depth_profile(file_path)
        except Exception as exc:
            if self.debug:
                logging.exception("PHI PRO parsing failed: %s", exc)
            return None

    def _parse_sofh_depth_profile(self, file_path: Path) -> Optional[List[Spectrum]]:
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            eoh_idx = content.find(b"EOFH\r\n")
            if eoh_idx == -1:
                eoh_idx = content.find(b"EOFH")
                if eoh_idx == -1:
                    return None
                header_end = eoh_idx + 4
            else:
                header_end = eoh_idx + 6

            header_text = content[:eoh_idx].decode("utf-8", errors="ignore")
            binary_data = content[header_end:]

            lines = header_text.splitlines()

            # Ensure this is a depth profile file when possible
            file_type_line = next((ln for ln in lines if ln.startswith("FileType:")), "")
            if file_type_line and "DEPTH" not in file_type_line.upper():
                if self.debug:
                    print(f"   [phi_pro] FileType not DEPTHPRO: {file_type_line}")

            # Prefer local filename for sample id; keep acquisition filename if present
            source_file = file_path.name
            acq_filename = None
            for line in lines:
                if line.startswith("AcqFilename:"):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        acq_filename = value
                        break
            sample_base = file_path.stem

            # Depth profile cycles
            cycles = 1
            for line in lines:
                if line.startswith("NoDPDataCyc:"):
                    try:
                        cycles = int(line.split(":", 1)[1].strip().split()[0])
                    except Exception:
                        cycles = 1
                    break

            # Parse regions from header
            regions: List[Dict[str, Any]] = []
            has_full = any("SpectralRegDefFull" in line for line in lines)
            def_key = "SpectralRegDefFull" if has_full else "SpectralRegDef"

            for line in lines:
                if line.startswith(def_key + ":"):
                    parts = line.split(":", 1)[1].strip().split()
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
                                    "acq_filename": acq_filename,
                                    "depth_profile": True,
                                    "total_cycles": cycles,
                                }
                            })
                        except (ValueError, IndexError):
                            continue

            if not regions:
                return None

            data_blocks = PHISPEParser._parse_sofh_data_blocks(binary_data)
            if not data_blocks:
                return None

            active_regions = [r for r in regions if r.get("active", 1) != 0]
            if not active_regions:
                active_regions = regions

            data_blocks.sort(key=lambda b: b["offset"])
            remaining_blocks = data_blocks.copy()

            spectra: List[Spectrum] = []

            for region in active_regions:
                expected_points = region["n_points"]
                block = None
                for candidate in remaining_blocks:
                    bytes_per_point = candidate["bytes_per_point"]
                    expected_bytes = expected_points * bytes_per_point * max(cycles, 1)
                    if candidate["size"] == expected_bytes:
                        block = candidate
                        break

                if block is None and remaining_blocks:
                    block = remaining_blocks[0]

                if block is None:
                    continue

                remaining_blocks.remove(block)

                data_start = block["offset"]
                data_end = data_start + block["size"]
                if data_end > len(binary_data):
                    continue

                data_bytes = binary_data[data_start:data_end]
                intensity = np.frombuffer(data_bytes, dtype=block["dtype"])

                # Infer cycles if header was missing or inconsistent
                total_points = len(intensity)
                inferred_cycles = cycles
                if expected_points > 0 and total_points % expected_points == 0:
                    inferred_cycles = max(1, total_points // expected_points)
                else:
                    inferred_cycles = 1

                if total_points < expected_points:
                    continue

                intensity = intensity[: expected_points * inferred_cycles]
                try:
                    intensity = intensity.reshape(inferred_cycles, expected_points)
                except Exception:
                    continue

                energy = region["start"] + np.arange(expected_points) * region["step"]

                for idx in range(inferred_cycles):
                    layer_index = idx + 1
                    name = f"{sample_base}_L{layer_index}"
                    metadata = {
                        **region["metadata"],
                        "region": region["name"],
                        "original_spectrum": sample_base,
                        "depth_profile": True,
                        "layer_index": layer_index,
                        "total_cycles": inferred_cycles,
                    }

                    spectrum = Spectrum(
                        name=name,
                        energy=energy,
                        intensity=intensity[idx].astype(np.float32, copy=False),
                        source_format=self.format_name,
                        metadata=metadata,
                    )

                    if spectrum.is_valid_xps():
                        spectra.append(spectrum)

            return spectra if spectra else None

        except Exception as exc:
            if self.debug:
                print(f"   [phi_pro] Parsing error: {exc}")
            return None


__all__ = ["PHIPROParser"]
