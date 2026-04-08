"""
Metadata structures for XPS data.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class XPSMetadata:
    """Common metadata structure for all XPS formats."""
    source_format: str = "unknown"
    source_file: str = ""
    region: Optional[str] = None
    scan_mode: Optional[str] = None
    pass_energy: Optional[float] = None
    x_start: float = 0.0
    x_step: float = 1.0
    nx: int = 1
    y_start: float = 0.0
    y_step: float = 1.0
    ny: int = 1
    energy_values: Optional[np.ndarray] = None
    comments: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

