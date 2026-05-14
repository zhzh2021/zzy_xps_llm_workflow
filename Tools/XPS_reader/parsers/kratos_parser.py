"""
Parser for Kratos format XPS data files.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import numpy as np
import logging

from .base import BaseParser
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Spectrum

# Compiled pattern for pass energy in Kratos key=value headers.
# Matches: PassEnergy=20, Pass Energy = 40.5, AnalyserPassEnergy=160, etc.
_KRATOS_PASS_ENERGY_RE = re.compile(
    r'(?:pass[_ ]?energy|analyser[_ ]?pass[_ ]?energy)\s*[=:]\s*([0-9]+(?:\.[0-9]*)?)'
    , re.IGNORECASE
)


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
            metadata: Dict[str, Any] = {}

            with open(file_path, 'r') as f:
                # Parse header section for pass energy before [DATA]
                for line in f:
                    if line.startswith("[DATA]"):
                        break
                    m = _KRATOS_PASS_ENERGY_RE.search(line)
                    if m:
                        try:
                            metadata['pass_energy'] = float(m.group(1))
                        except ValueError:
                            pass

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
                source_format=self.format_name,
                metadata=metadata
            )
            
            if spectrum.is_valid_xps():
                return [spectrum]
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ Kratos parsing failed: {e}")
            return None
