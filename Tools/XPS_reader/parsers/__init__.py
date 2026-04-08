"""
XPS file format parsers.

This module provides parsers for various XPS data file formats.
New parsers can be added by creating a new file in this directory
and registering it in the AVAILABLE_PARSERS list.
"""
from .base import BaseParser
from .ascii_parser import ASCIIParser, EnhancedASCIIParser
from .csv_parser import CSVParser, EnhancedCSVParser, DepthProfileCSVParser
from .phi_spe_parser import PHISPEParser
from .phi_pro_parser import PHIPROParser
from .vgd_parser import VGDParser
from .kratos_parser import KratosParser
from .vms_parser import VAMASParser, parse_vamas_format

# Backwards-compatible alias
VMSParser = VAMASParser
parse_vms_format = parse_vamas_format

# Registry of all available parsers (order matters - tried in sequence)
AVAILABLE_PARSERS = [
    DepthProfileCSVParser,
    EnhancedCSVParser,   # Try enhanced CSV first
    CSVParser,           # Then simple CSV
    EnhancedASCIIParser,
    ASCIIParser,         # Fallback
    VMSParser,           # VAMAS
    PHIPROParser,        # PHI depth profile (.pro)
    PHISPEParser,        # PHI SPE
    VGDParser,
    KratosParser,
]

__all__ = [
    'BaseParser',
    'ASCIIParser',
    'EnhancedASCIIParser',
    'CSVParser',
    'EnhancedCSVParser',
    'DepthProfileCSVParser',
    'PHIPROParser',
    'PHISPEParser',
    'VGDParser',
    'KratosParser',
    'VAMASParser',
    'parse_vamas_format',
    'AVAILABLE_PARSERS',
]
