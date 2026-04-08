"""
Base parser class for all XPS file format parsers.
"""
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod

import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum


class BaseParser(ABC):
    """Base class for all XPS file format parsers."""
    
    # Class attributes to be overridden by subclasses
    format_name: str = "unknown"
    file_extensions: List[str] = []
    
    def __init__(self, debug: bool = False):
        """
        Initialize parser.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if parser can handle this file
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """
        Parse file and return list of spectra.
        
        Args:
            file_path: Path to file to parse
            
        Returns:
            List of Spectrum objects, or None if parsing failed
        """
        pass
    
    def _check_extension(self, file_path: Path) -> bool:
        """
        Check if file extension matches supported extensions.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if extension is supported
        """
        return file_path.suffix.lower() in self.file_extensions
