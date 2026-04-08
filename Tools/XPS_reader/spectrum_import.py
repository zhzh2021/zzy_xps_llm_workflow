"""
Spectrum import module for XPS data processing.

This module handles importing and format detection of XPS data files
using the modular parser system with enhanced error handling.
"""

from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
import re
import traceback

from parsers import AVAILABLE_PARSERS
from core.data_structures import Spectrum


class FileFormat(Enum):
    PHI_MULTIPAK_SPE = "PHI MultiPak SPE"
    ASCII_TEXT = "ASCII Text"
    STANDARD_CSV = "Standard CSV"
    MULTI_REGION_CSV = "Multi-Region CSV"
    VGD = "VGD"
    KRATOS = "Kratos"
    VAMAS = "VAMAS"
    UNKNOWN = "Unknown"


class FormatDetector:
    """Detect XPS file format."""

    @staticmethod
    def detect(file_path: Path) -> FileFormat:
        """Detect file format based on extension and content."""
        ext = file_path.suffix.lower()

        # Check by extension first
        if ext in ['.vms', '.vamas']:
            return FileFormat.VAMAS

        if ext == '.spe':
            return FileFormat.PHI_MULTIPAK_SPE

        if ext == '.vgd':
            return FileFormat.VGD

        # For text-based formats, check content
        if ext in ['.txt', '.asc', '.csv', '.dat']:
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    first_lines = f.read(1000)

                # Check for VAMAS
                if 'VAMAS' in first_lines or 'Surface Chemical Analysis' in first_lines:
                    return FileFormat.VAMAS

                # Check for multi-region CSV
                if first_lines.count('\t') > 20 or first_lines.count(',') > 20:
                    region_pattern = r'\b[A-Z][a-z]?\d[spdf]\b'
                    if len(re.findall(region_pattern, first_lines)) > 1:
                        return FileFormat.MULTI_REGION_CSV

                # Check for standard CSV
                if ',' in first_lines or '\t' in first_lines:
                    return FileFormat.STANDARD_CSV

                return FileFormat.ASCII_TEXT

            except Exception:
                pass

        return FileFormat.UNKNOWN


class SpectrumImporter:
    """Unified spectrum importer using modular parsers."""

    def __init__(self, debug: bool = False):
        """Initialize with all available parsers."""
        self.debug = debug

        # Create parser instances with specific ordering
        self.parsers = []

        # Add parsers in priority order
        for parser_class in AVAILABLE_PARSERS:
            parser = parser_class(debug=debug)
            self.parsers.append(parser)

            if self.debug:
                print(
                    f"   Registered parser: {parser.format_name} ({parser.file_extensions})")

    def import_file(self, file_path: Path) -> List[Spectrum]:
        """Import spectra from file with automatic format detection."""
        print("\n" + "-" * 70)
        print(f"[FILE] Importing: {file_path.name}")
        print("-" * 70)

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Track if any parser could handle the format but returned no spectra
        format_recognized = False
        graceful_rejection = False
        
        # Try each parser in order
        for parser in self.parsers:
            parser_name = parser.format_name

            if self.debug:
                print(f"   Trying: {parser_name}...")

            try:
                # Check if parser can handle this file
                if not parser.can_parse(file_path):
                    if self.debug:
                        print(f"   ⏭️ Skipped (incompatible format)")
                    continue

                format_recognized = True
                
                # Try to parse
                spectra = parser.parse(file_path)

                if spectra:
                    print(
                        f"   [OK] Success with {parser_name}: {len(spectra)} spectrum/spectra")

                    for spec in spectra:
                        print(f"      - {spec.name}: {len(spec.energy)} points, "
                              f"BE: {spec.energy.min():.2f}-{spec.energy.max():.2f} eV, "
                              f"max intensity: {spec.intensity.max():.1f}")

                    return spectra
                else:
                    # Parser recognized format but returned None (graceful rejection)
                    graceful_rejection = True
                    print(f"   ⚠️ {parser_name}: File format recognized but file rejected (likely corrupted)")
                    if self.debug:
                        print(f"      This is normal for corrupted or incompatible file variants")

            except Exception as e:
                if self.debug:
                    print(f"   ❌ Failed: {e}")
                    traceback.print_exc()
                continue

        # Provide appropriate error message based on what happened
        if graceful_rejection:
            print(f"   ℹ️ File recognized but rejected due to corruption/incompatibility")
            # Return empty list instead of raising exception for graceful rejections
            return []
        elif format_recognized:
            raise RuntimeError(
                f"Parser recognized {file_path.name} but failed to extract spectra")
        else:
            raise RuntimeError(
                f"No parser could recognize file format: {file_path.name}")