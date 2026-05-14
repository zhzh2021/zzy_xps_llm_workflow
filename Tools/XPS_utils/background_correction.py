"""
Background Correction Utilities for XPS Analysis

Provides background subtraction algorithms including Shirley, linear, polynomial, and Tougaard methods.
Can be imported and reused across different XPS analysis modules.
"""

import numpy as np
from typing import Tuple


def cumulative_trapz_rev(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoidal integral from each x[i] to x[-1].
    
    Args:
        x: Energy/x-axis values (must be sorted ascending)
        f: Intensity/y-axis values
        
    Returns:
        Array of cumulative integrals from each point to the end
    """
    out = np.zeros_like(f, dtype=float)
    out[-1] = 0.0
    for i in range(len(f) - 2, -1, -1):
        dx = x[i + 1] - x[i]
        out[i] = out[i + 1] + 0.5 * dx * (f[i + 1] + f[i])
    return out


def baseline_shirley(x: np.ndarray, y: np.ndarray, niter: int = 100,
                      tol: float = 1e-6, n_endpoint_avg: int = 7) -> np.ndarray:
    """
    Shirley background subtraction with iterative solution.

    The Shirley background is commonly used in XPS to model the inelastic
    scattering background.  It assumes the background at each point is
    proportional to the integrated intensity of the peak above that point.

    Endpoint values are computed as the mean of the first/last
    ``n_endpoint_avg`` points so that a single noisy sample cannot push the
    background above the true baseline.  The computed background is also
    clamped to never exceed the raw data at any point, preventing negative
    background-corrected intensities.

    Args:
        x: Energy values (ascending or descending — direction is auto-handled)
        y: Intensity values
        niter: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)
        n_endpoint_avg: Number of points averaged at each end to establish
            stable baseline endpoints (default: 7)

    Returns:
        Array of background intensity values (same length and direction as input)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Work on a consistently ascending copy; restore original order at the end.
    if x[0] > x[-1]:
        x_w, y_w = x[::-1], y[::-1]
        flipped = True
    else:
        x_w, y_w = x.copy(), y.copy()
        flipped = False

    n = len(y_w)
    n_avg = max(1, min(n_endpoint_avg, n // 5))  # guard against tiny spectra

    # Stable endpoints: mean of first / last n_avg points
    y_left  = float(np.mean(y_w[:n_avg]))
    y_right = float(np.mean(y_w[-n_avg:]))

    B = np.linspace(y_left, y_right, n)
    B[-1] = y_right

    for _ in range(niter):
        prev = B.copy()
        S = cumulative_trapz_rev(x_w, y_w - B)
        k = (y_left - y_right) / (S[0] + 1e-15)
        B = y_right + k * S
        B[-1] = y_right

        if np.linalg.norm(B - prev) / (np.linalg.norm(prev) + 1e-15) < tol:
            break

    # Clamp: background must never exceed the raw data (prevents negative residuals)
    B = np.minimum(B, y_w)

    return B[::-1] if flipped else B


def shirley_background(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Simplified Shirley background for compatibility.
    
    Alias for baseline_shirley with default parameters.
    
    Args:
        x: Energy values (must be sorted ascending)
        y: Intensity values
        
    Returns:
        Array of background intensity values
    """
    return baseline_shirley(x, y)


def linear_background(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear background using first-order polynomial fit.
    
    Args:
        x: Energy values
        y: Intensity values
        
    Returns:
        Array of linear background values
    """
    return np.polyval(np.polyfit(x, y, 1), x)


def polynomial_background(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Polynomial background using nth-order polynomial fit.
    
    Args:
        x: Energy values
        y: Intensity values
        degree: Polynomial degree (default: 2 for quadratic)
        
    Returns:
        Array of polynomial background values
    """
    return np.polyval(np.polyfit(x, y, degree), x)


def tougaard_background(x: np.ndarray, y: np.ndarray, tb: float = 2866, tc: float = 1643) -> np.ndarray:
    """
    Calculates the Universal Tougaard Background for XPS data.
    
    The Tougaard background models inelastic scattering using a universal
    cross-section. It integrates intensity contributions from lower binding
    energies to higher binding energies using the Tougaard kernel.
    
    Args:
        x: Binding Energy (eV) array (handles both ascending and descending)
        y: Intensity (counts) array
        tb: Tougaard B parameter (default: 2866 eV² for Universal)
        tc: Tougaard C parameter (default: 1643 eV² for Universal)
        
    Returns:
        Array of background intensity values
        
    Example:
        >>> energy = np.linspace(280, 290, 100)
        >>> intensity = np.random.rand(100) + 5
        >>> background = tougaard_background(energy, intensity)
        >>> corrected = intensity - background
    """
    
    # Ensure working with numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Check data direction - need ascending binding energy for integration
    # Tougaard integral accumulates from low BE (high KE) to high BE (low KE)
    invert_order = False
    if x[0] > x[-1]:  # If descending (60 -> 50), flip to ascending (50 -> 60)
        x = x[::-1]
        y = y[::-1]
        invert_order = True
    
    n = len(x)
    bg = np.zeros(n, dtype=float)
    
    # Tougaard integral loop
    # B(E) = Integral [ J(E') * K(E - E') ] dE'  from E' = min to E
    # K(T) = B * T / (C + T^2)^2
    
    # We iterate through every point i (the current energy)
    for i in range(1, n):
        # The integration range is from the start (low BE) up to point i
        # T is the energy loss (current energy - source energy)
        E_current = x[i]
        E_source = x[:i]
        intensity_source = y[:i]
        
        T = E_current - E_source 
        
        # Calculate the Universal Kernel
        kernel = (tb * T) / ((tc + T**2)**2)
        
        # Integrate (Trapezoidal rule)
        # We calculate the contribution of all previous points to the background at i
        integral_slice = intensity_source * kernel
        
        # Using trapezoidal integration for the area
        bg[i] = np.trapz(integral_slice, x[:i])

    # 4. Restore original order if we flipped it
    if invert_order:
        bg = bg[::-1]
        
    return bg




def apply_background_correction(x: np.ndarray, y: np.ndarray, bg_type: str, tb: float = 2866, tc: float = 1643) -> np.ndarray:
    """
    Apply background correction based on specified method.
    
    Args:
        x: Energy values
        y: Intensity values
        bg_type: Background type ('shirley', 'linear', 'polynomial', 'tougaard')
        tb: Tougaard B parameter (default: 2866, only used for tougaard)
        tc: Tougaard C parameter (default: 1643, only used for tougaard)
        
    Returns:
        Array of background intensity values
        
    Raises:
        Warning if unknown background type is specified (defaults to Shirley)
    """
    bg_type = bg_type.lower()
    
    if bg_type == "shirley":
        return baseline_shirley(x, y)
    elif bg_type == "linear":
        return linear_background(x, y)
    elif bg_type == "polynomial":
        return polynomial_background(x, y)
    elif bg_type == "tougaard":
        return tougaard_background(x, y, tb, tc)
    else:
        print(f"[WARNING]  Unknown background type '{bg_type}', using Shirley")
        return baseline_shirley(x, y)


def subtract_background(x: np.ndarray, y: np.ndarray, bg_type: str = "shirley", tb: float = 2866, tc: float = 1643) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subtract background and return corrected data.
    
    Args:
        x: Energy values
        y: Intensity values
        bg_type: Background type ('shirley', 'linear', 'polynomial', 'tougaard')
        tb: Tougaard B parameter (default: 2866, only used for tougaard)
        tc: Tougaard C parameter (default: 1643, only used for tougaard)
        
    Returns:
        Tuple of (background, corrected_intensity)
        
    Example:
        >>> energy = np.linspace(280, 290, 100)
        >>> intensity = np.random.rand(100) + 5
        >>> background, corrected = subtract_background(energy, intensity, 'shirley')
    """
    background = apply_background_correction(x, y, bg_type, tb, tc)
    corrected = y - background
    return background, corrected
