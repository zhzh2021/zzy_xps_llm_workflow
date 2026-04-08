"""Parser for Thermo Fisher Avantage VGD format XPS data files."""

from __future__ import annotations
from core.metadata import XPSMetadata
from core.data_structures import Spectrum

import logging
import re
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .base import BaseParser

import sys

sys.path.append(str(Path(__file__).parent.parent))

try:  # Optional dependency for structured-storage parsing
    import olefile
except ImportError:  # pragma: no cover - handled at runtime
    olefile = None


class VGDParser(BaseParser):
    """Parser for Thermo Fisher Avantage .VGD files.

    The implementation borrows the layered approach used in pynxtools-xps:
    - Use a robust container reader (`olefile`) to access vendor streams.
    - Normalize data/axes into NumPy arrays and feed them into the shared
      `Spectrum` structure with rich metadata.
    """

    format_name = "thermo_vgd"
    file_extensions = [".vgd"]
    DEFAULT_SOURCE_ENERGY_EV = 1486.6  # Al K-alpha
    SOURCE_KEYWORDS = {
        "k-alpha": 1486.6,
        "al k": 1486.6,
        "al k-alpha": 1486.6,
        "al k alpha": 1486.6,
        "al kalpha": 1486.6,
        "mg k": 1253.6,
        "mg k-alpha": 1253.6,
        "mg k alpha": 1253.6,
        "mg kalpha": 1253.6,
    }

    def can_parse(self, file_path: Path) -> bool:
        """Basic sanity check for VGD files."""
        if not self._check_extension(file_path):
            return False

        try:
            if olefile is None:
                return False

            with olefile.OleFileIO(str(file_path)) as vgd_file:
                return all(
                    vgd_file.exists(stream) for stream in ("VGData", "VGSpaceAxes")
                )
        except Exception:  # pragma: no cover - defensive
            return False

    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Parse a VGD file and return a list with one Spectrum."""
        if olefile is None:
            logging.error(
                "olefile package is required to parse %s files. "
                "Install it via `pip install olefile`.",
                self.format_name,
            )
            return None

        try:
            with olefile.OleFileIO(str(file_path)) as vgd_file:
                intensity = self._read_data_stream(vgd_file)
                axis_info = self._parse_axis_stream(vgd_file)
                source_energy, inferred = self._infer_source_energy(
                    vgd_file, file_path
                )

            kinetic_axis = axis_info["origin"] + axis_info["step"] * np.arange(
                len(intensity), dtype=np.float64
            )

            if source_energy is None:
                source_energy = self.DEFAULT_SOURCE_ENERGY_EV
                inferred = False

            binding_axis = source_energy - kinetic_axis

            valid_mask = binding_axis >= 0
            trimmed = False
            if not np.any(valid_mask):
                raise ValueError("Binding energy axis is entirely negative.")
            if not np.all(valid_mask):
                trimmed = True
                binding_axis = binding_axis[valid_mask]
                intensity = intensity[valid_mask]
                kinetic_axis = kinetic_axis[valid_mask]

            metadata = XPSMetadata(
                source_format=self.format_name,
                source_file=str(file_path),
                region=file_path.stem,
                x_start=float(binding_axis[0]),
                x_step=float(binding_axis[1] - binding_axis[0])
                if len(binding_axis) > 1
                else axis_info["step"],
                nx=len(binding_axis),
                energy_values=binding_axis,
                extra_metadata={
                    "axis_origin_ke": axis_info["origin"],
                    "axis_step_ke": axis_info["step"],
                    "point_count": len(binding_axis),
                    "axis_units": "eV",
                    "energy_scale": "binding",
                    "source_energy_ev": source_energy,
                    "source_energy_inferred": inferred,
                    "data_trimmed_negative_be": trimmed,
                },
            )

            spectrum = Spectrum(
                name=file_path.stem,
                energy=metadata.energy_values,
                intensity=intensity,
                source_format=self.format_name,
                metadata={"region": metadata.region,
                          **metadata.extra_metadata},
            )

            return [spectrum] if spectrum.is_valid_xps() else None
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logging.error("Error parsing %s: %s", file_path, exc)
            if self.debug:
                raise
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _read_data_stream(self, vgd_file: olefile.OleFileIO) -> np.ndarray:
        """Load VGData stream as float64 array."""
        try:
            raw = vgd_file.openstream("VGData").read()
        except Exception as exc:
            raise ValueError("VGData stream missing or unreadable") from exc

        if len(raw) % 8 != 0:
            raise ValueError(
                "VGData stream length is not a multiple of 8 bytes.")

        return np.frombuffer(raw, dtype="<f8").copy()

    def _parse_axis_stream(self, vgd_file: olefile.OleFileIO) -> Dict[str, float]:
        """Decode the VGSpaceAxes stream to recover axis metadata."""
        try:
            stream = vgd_file.openstream("VGSpaceAxes").read()
        except Exception as exc:
            raise ValueError(
                "VGSpaceAxes stream missing or unreadable") from exc

        if len(stream) < 46:
            raise ValueError("VGSpaceAxes stream too short.")

        # Based on reverse-engineering the binary layout:
        # bytes 26:30 -> uint32 number of points
        # bytes 30:38 -> little-endian float64 axis origin (kinetic energy)
        # bytes 38:46 -> little-endian float64 step size
        point_count = int.from_bytes(stream[26:30], "little")
        origin = struct.unpack("<d", stream[30:38])[0]
        step = struct.unpack("<d", stream[38:46])[0]

        if point_count <= 1:
            raise ValueError("Invalid point count in VGSpaceAxes.")

        return {"points": point_count, "origin": origin, "step": step}

    def _infer_source_energy(
        self, vgd_file: olefile.OleFileIO, file_path: Path
    ) -> Tuple[Optional[float], bool]:
        """Attempt to recover the excitation energy from metadata."""

        candidates: List[float] = []

        for stream in self._property_streams(vgd_file):
            try:
                props = vgd_file.getproperties(stream, convert_time=True)
            except Exception:
                continue
            hv = self._match_source_energy(props.values())
            if hv is not None:
                return hv, True
            candidates.extend(self._extract_numeric_candidates(props.values()))

        if not candidates:
            try:
                text = file_path.read_text(
                    encoding="utf-16le", errors="ignore")
            except Exception:
                text = ""
            if text:
                candidates.extend(self._extract_numeric_from_text(text))

        if candidates:
            return (
                min(
                    candidates,
                    key=lambda value: abs(
                        value - self.DEFAULT_SOURCE_ENERGY_EV),
                ),
                False,
            )

        return None, False

    def _property_streams(self, vgd_file: olefile.OleFileIO) -> Iterable[str]:
        """Yield property stream names (start with \\x05)."""
        try:
            entries = vgd_file.listdir(streams=True, storages=True)
        except Exception:
            return []

        return [
            "/".join(entry)
            for entry in entries
            if entry and entry[0].startswith("\x05")
        ]

    def _match_source_energy(self, values: Iterable) -> Optional[float]:
        """Check property values for known source descriptions."""
        for value in values:
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-16le")
                except Exception:
                    value = value.decode("utf-8", errors="ignore")

            if isinstance(value, str):
                cleaned = value.lower()
                for keyword, hv in self.SOURCE_KEYWORDS.items():
                    if keyword in cleaned:
                        return hv

                numeric = self._extract_numeric_from_text(cleaned)
                if numeric:
                    return numeric[0]

        return None

    @staticmethod
    def _extract_numeric_candidates(values: Iterable) -> List[float]:
        """Find numeric-looking substrings inside property values."""
        candidates: List[float] = []
        for value in values:
            if isinstance(value, (int, float)):
                if 200 <= value <= 3000:
                    candidates.append(float(value))
            elif isinstance(value, str):
                candidates.extend(VGDParser._extract_numeric_from_text(value))
            elif isinstance(value, bytes):
                try:
                    candidates.extend(
                        VGDParser._extract_numeric_from_text(
                            value.decode("utf-16le", errors="ignore")
                        )
                    )
                except Exception:
                    continue
        return candidates

    @staticmethod
    def _extract_numeric_from_text(text: str) -> List[float]:
        """Extract plausible excitation energies from free text."""
        matches = re.findall(r"\d+\.\d+", text)
        values = [float(match) for match in matches]
        return [val for val in values if 200 <= val <= 3000]
