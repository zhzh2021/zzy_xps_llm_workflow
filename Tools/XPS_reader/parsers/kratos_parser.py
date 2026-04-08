"""
Parser for Kratos format XPS data files.
"""
from pathlib import Path
from typing import Optional, List
import numpy as np
import logging

from .base import BaseParser
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum


class KratosParser(BaseParser):
    """Parser for Kratos format files."""
    
    format_name = "kratos"
    file_extensions = ['.kal', '.dset']
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if file is Kratos format."""
        if not self._check_extension(file_path):
            return False
        
        try:
            with open(file_path, 'r') as f:
                content = f.read(500)
                return "[DATA]" in content
        except Exception:
            return False
    
    def parse(self, file_path: Path) -> Optional[List[Spectrum]]:
        """Parse Kratos file."""
        try:
            energies = []
            intensities = []
            
            with open(file_path, 'r') as f:
                # Skip to data section
                for line in f:
                    if line.startswith("[DATA]"):
                        break
                        
                # Read data
                for line in f:
                    if line.strip() and not line.startswith("["):
                        e, i = map(float, line.strip().split(','))
                        energies.append(e)
                        intensities.append(i)
                        
            spectrum = Spectrum(
                name=file_path.stem,
                energy=np.array(energies),
                intensity=np.array(intensities),
                source_format=self.format_name
            )
            
            if spectrum.is_valid_xps():
                return [spectrum]
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ Kratos parsing failed: {e}")
            return None
