"""XPS Utilities Package"""

from .background_correction import (
    baseline_shirley,
    shirley_background,
    linear_background,
    polynomial_background,
    apply_background_correction,
    subtract_background
)

__all__ = [
    'baseline_shirley',
    'shirley_background',
    'linear_background',
    'polynomial_background',
    'apply_background_correction',
    'subtract_background'
]
