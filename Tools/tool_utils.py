"""
utils.py
Common utility functions for XPS processing pipeline.
Author: ZY and Argo (Argonne National Laboratory)

Features:
- Time formatting and tracking
- Timing decorators and context managers
- Progress tracking
- File utilities
"""

import time
import functools
from typing import Callable, Any, Optional
from pathlib import Path
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
import os
import json
import pandas as pd

# These functions are defined more comprehensively below

def ensure_ascending_be(E: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort data by ascending binding energy."""
    idx = np.argsort(E)
    return E[idx], I[idx]

def slice_roi(E: np.ndarray, I: np.ndarray, erange: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract region of interest from spectrum."""
    e_min, e_max = float(min(erange)), float(max(erange))
    mask = (E >= e_min) & (E <= e_max)
    return E[mask], I[mask]

def load_yaml_settings(file_path: Path) -> Dict:
    """Load YAML settings from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
# ============================
# Time formatting
# ============================

def format_time(seconds: float) -> str:
    """
    Format elapsed time in human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "2m 34s", "45.2s", "1h 23m 45s")
    
    Examples:
        >>> format_time(0.5)
        '500ms'
        >>> format_time(45.7)
        '45.7s'
        >>> format_time(125)
        '2m 5s'
        >>> format_time(3725)
        '1h 2m 5s'
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


# ============================
# Timing decorators
# ============================

def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution and print result.
    
    Usage:
        @timer
        def my_function():
            # ... code ...
    
    Output:
        ⏱️  my_function completed in 2.3s
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__} completed in {format_time(elapsed)}")
        return result
    return wrapper


def timer_quiet(func: Callable) -> Callable:
    """
    Decorator to time function execution and return (result, elapsed_time).
    Does not print anything.
    
    Usage:
        @timer_quiet
        def my_function():
            # ... code ...
            return result
        
        result, elapsed = my_function()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer("Processing data"):
            # ... code ...
        
        # Output: ⏱️  Processing data: 2.3s
        
        # Or get elapsed time:
        with Timer("Processing", verbose=False) as t:
            # ... code ...
        print(f"Took {t.elapsed:.2f} seconds")
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"[TIMER] {self.name}: {format_time(self.elapsed)}")
        return False


class TimingStats:
    """
    Accumulate timing statistics for multiple operations.
    
    Usage:
        stats = TimingStats("File processing")
        
        for file in files:
            with stats.time():
                process_file(file)
        
        stats.print_summary()
    """
    
    def __init__(self, name: str = "Operations"):
        self.name = name
        self.timings = []
        self._current_start = None
    
    def time(self):
        """Return a context manager for timing one operation."""
        return self._TimingContext(self)
    
    class _TimingContext:
        def __init__(self, parent):
            self.parent = parent
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start
            self.parent.timings.append(elapsed)
            return False
    
    def add(self, elapsed: float):
        """Manually add a timing."""
        self.timings.append(elapsed)
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.timings:
            print(f"⏱️  {self.name}: No timings recorded")
            return
        
        total = sum(self.timings)
        avg = total / len(self.timings)
        min_t = min(self.timings)
        max_t = max(self.timings)
        
        print(f"[TIMER] {self.name} - {len(self.timings)} operations:")
        print(f"   Total: {format_time(total)}")
        print(f"   Average: {format_time(avg)}")
        print(f"   Fastest: {format_time(min_t)}")
        print(f"   Slowest: {format_time(max_t)}")
    
    def get_stats(self) -> dict:
        """Return statistics as a dictionary."""
        if not self.timings:
            return {}
        
        return {
            'count': len(self.timings),
            'total': sum(self.timings),
            'average': sum(self.timings) / len(self.timings),
            'min': min(self.timings),
            'max': max(self.timings),
            'timings': self.timings.copy()
        }


# ============================
# File utilities
# ============================

def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
    
    Returns:
        Path object (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str, max_length: int = 255) -> str:
    """
    Convert string to safe filename (remove/replace invalid characters).
    
    Args:
        name: Original name
        max_length: Maximum filename length
    
    Returns:
        Safe filename string
    """
    # Replace invalid characters
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    name = name.strip('. ')
    
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length]
    
    return name


# ============================
# Progress tracking
# ============================

class ProgressTracker:
    """
    Simple progress tracker for batch operations.
    
    Usage:
        tracker = ProgressTracker(total=100, name="Processing files")
        
        for i in range(100):
            # ... process ...
            tracker.update(1)
        
        tracker.finish()
    """
    
    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress."""
        if self.total == 0:
            return
        
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f", ETA: {format_time(eta)}"
        else:
            eta_str = ""
        
        print(f"\r{self.name}: {self.current}/{self.total} ({percent:.1f}%){eta_str}", end='', flush=True)
    
    def finish(self):
        """Mark as complete and print final stats."""
        self.current = self.total
        elapsed = time.time() - self.start_time
        print(f"\r{self.name}: {self.total}/{self.total} (100.0%) - Completed in {format_time(elapsed)}")


# ============================
# Demo/Test
# ============================

if __name__ == "__main__":
    # Demo/test the utilities
    print("=" * 80)
    print("Testing timing utilities")
    print("=" * 80)
    print()
    
    # Test format_time
    print("1. format_time():")
    print(f"   0.5s → {format_time(0.5)}")
    print(f"   45.7s → {format_time(45.7)}")
    print(f"   125s → {format_time(125)}")
    print(f"   3725s → {format_time(3725)}")
    print()
    
    # Test timer decorator
    print("2. @timer decorator:")
    @timer
    def slow_function():
        time.sleep(0.5)
        return "done"
    
    result = slow_function()
    print()
    
    # Test Timer context manager
    print("3. Timer context manager:")
    with Timer("Sleep test"):
        time.sleep(0.3)
    print()
    
    # Test TimingStats
    print("4. TimingStats:")
    stats = TimingStats("Batch operations")
    for i in range(5):
        with stats.time():
            time.sleep(0.1 * (i + 1))
    stats.print_summary()
    print()
    
    # Test ProgressTracker
    print("5. ProgressTracker:")
    tracker = ProgressTracker(total=50, name="Processing items")
    for i in range(50):
        time.sleep(0.02)
        tracker.update(1)
    tracker.finish()
    print()
    
    # Test file utilities
    print("6. File utilities:")
    demo_path = r"C:\test<file>.txt"
    print("   safe_filename('{}') -> '{}'".format(demo_path, safe_filename(demo_path)))
    print()
    
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)

def log(msg):
    """Simple logger."""
    print(f"[LOG] {msg}")

def read_txt_file(filepath):
    """Read a text file and return its lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def list_files(directory, file_extension):
    """List files in a directory with a given extension."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return list(directory.glob(f"*{file_extension}"))

def clean_data(df):
    """Drop NA and reset index for a DataFrame."""
    return df.dropna().reset_index(drop=True)

# Example configuration constants
DATA_PATH = "N:/zhenzhen/Python/ZZY_XPS/Data"
RESULTS_PATH = "N:/zhenzhen/Python/ZZY_XPS/Results"

