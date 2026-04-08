"""
CSV export module for XPS data processing.

This module handles exporting spectra to various CSV formats with 
configurable output options and proper data formatting for downstream analysis.
"""

import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from core.data_structures import Spectrum


class CSVExporter:
    """Export spectra to CSV format with various options."""

    def __init__(self, output_dir: Path, include_metadata: bool = True,
                 decimal_places: int = 3):
        """Initialize CSV exporter with output options."""
        self.output_dir = Path(output_dir)
        self.include_metadata = include_metadata
        self.decimal_places = decimal_places
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_spectrum(self, spectrum: Spectrum, filename_prefix: str = None,
                        filename: Optional[str] = None) -> Path:
        """Export single spectrum to CSV."""
        # Generate filename
        if filename:
            filename = filename if str(filename).lower().endswith(".csv") else f"{filename}.csv"
        elif filename_prefix:
            filename = f"{filename_prefix}_{spectrum.name}.csv"
        else:
            filename = f"{spectrum.name}.csv"
        
        output_path = self.output_dir / filename
        
        # Prepare data
        df_data = {
            'Binding_Energy_eV': np.round(spectrum.energy, self.decimal_places),
            'Intensity_cps': np.round(spectrum.intensity, self.decimal_places)
        }
        
        df = pd.DataFrame(df_data)
        
        # Write CSV with metadata header if requested
        with open(output_path, 'w', newline='') as f:
            if self.include_metadata:
                # Write metadata as comments
                f.write(f"# Spectrum: {spectrum.name}\n")
                f.write(f"# Data Points: {len(spectrum.energy)}\n")
                f.write(f"# Binding Energy Range: {spectrum.energy.min():.2f} - {spectrum.energy.max():.2f} eV\n")
                f.write(f"# Max Intensity: {spectrum.intensity.max():.1f} cps\n")
                
                if hasattr(spectrum, 'metadata') and spectrum.metadata:
                    if hasattr(spectrum.metadata, 'pass_energy') and spectrum.metadata.pass_energy:
                        f.write(f"# Pass Energy: {spectrum.metadata.pass_energy} eV\n")
                    if hasattr(spectrum.metadata, 'dwell_time') and spectrum.metadata.dwell_time:
                        f.write(f"# Dwell Time: {spectrum.metadata.dwell_time} ms\n")
                    if hasattr(spectrum.metadata, 'excitation_energy') and spectrum.metadata.excitation_energy:
                        f.write(f"# Excitation Energy: {spectrum.metadata.excitation_energy} eV\n")
                
                f.write("#\n")
            
            # Write data
            df.to_csv(f, index=False)
        
        return output_path

    def export_multiple_spectra(self, spectra: List[Spectrum], 
                              filename_prefix: str = None,
                              combine_in_single_file: bool = False) -> List[Path]:
        """Export multiple spectra to CSV."""
        output_paths = []
        
        if combine_in_single_file:
            # Combine all spectra in one file
            output_path = self._export_combined_csv(spectra, filename_prefix)
            output_paths.append(output_path)
        else:
            # Export each spectrum separately
            for spectrum in spectra:
                output_path = self.export_spectrum(spectrum, filename_prefix)
                output_paths.append(output_path)
        
        return output_paths

    def _export_combined_csv(self, spectra: List[Spectrum], 
                           filename_prefix: str = None) -> Path:
        """Export multiple spectra to a single CSV file."""
        if filename_prefix:
            filename = f"{filename_prefix}_allHR.csv"
        else:
            filename = "combined_spectra.csv"
        
        output_path = self.output_dir / filename
        
        # Normalize spectra to ascending energy for interpolation
        normalized_spectra = []
        for spec in spectra:
            energy = spec.energy
            intensity = spec.intensity
            if len(energy) == 0:
                continue
            if energy[0] > energy[-1]:
                energy = energy[::-1]
                intensity = intensity[::-1]
            metadata = getattr(spec, "metadata", {}) or {}
            # For depth profiles, spec.name already includes layer index (e.g., "Sample_F1s_L1")
            # Use spec.name directly if it's a depth profile, otherwise fall back to metadata fields
            if metadata.get('depth_profile', False):
                sample_name = spec.name
            else:
                sample_name = metadata.get('source_file') or metadata.get('original_spectrum') or spec.name
            normalized_spectra.append((energy, intensity, sample_name, metadata))

        if not normalized_spectra:
            raise ValueError("No valid spectra provided for export")

        # Find common energy range (intersection)
        min_energy = max(energy[0] for energy, _, _, _ in normalized_spectra)
        max_energy = min(energy[-1] for energy, _, _, _ in normalized_spectra)

        if max_energy <= min_energy:
            raise ValueError("Spectra do not share an overlapping energy range")

        # Determine energy step (use median absolute diff for robustness)
        first_energy = normalized_spectra[0][0]
        diffs = np.abs(np.diff(first_energy))
        energy_step = float(np.median(diffs)) if len(diffs) else 0.1
        if energy_step == 0:
            energy_step = 0.1

        common_energy = np.arange(min_energy, max_energy + energy_step/2, energy_step)
        
        # Prepare combined data
        df_data = {'Binding_Energy_eV': np.round(common_energy, self.decimal_places)}
        
        # Interpolate each spectrum to common energy axis
        column_counts: Dict[str, int] = {}
        known_suffixes = {
            ".spe", ".vgd", ".npl", ".xy", ".txt", ".asc", ".dat", ".csv", ".vms", ".vamas", ".pro"
        }
        for energy, intensity, name, metadata in normalized_spectra:
            interpolated_intensity = np.interp(common_energy, energy, intensity)
            # Clean up name: strip paths and known data file extensions
            clean_name = Path(str(name)).name
            suffix = Path(clean_name).suffix
            if suffix.lower() in known_suffixes:
                clean_name = clean_name[: -len(suffix)]
            
            # Check if this is a depth profile spectrum with layer index
            layer_idx = metadata.get('layer_index')
            is_depth = metadata.get('depth_profile', False)
            
            # Build column name: depth profiles already have layer in name from parser
            if is_depth and layer_idx is not None:
                # Depth profile: layer index already in clean_name (e.g., "15Si_G2_depth_F1s_L1")
                column_name = f"{clean_name}_cps"
            else:
                # Standard file: use collision counter for duplicates
                base_column_name = f"{clean_name}_cps"
                count = column_counts.get(base_column_name, 0)
                column_counts[base_column_name] = count + 1
                column_name = base_column_name if count == 0 else f"{base_column_name}_{count+1}"
            
            df_data[column_name] = np.round(interpolated_intensity, self.decimal_places)
        
        df = pd.DataFrame(df_data)
        
        # Write CSV with simplified metadata header
        with open(output_path, 'w', newline='') as f:
            if self.include_metadata:
                f.write(f"# Combined Spectra Export\n")
                f.write(f"# Number of Spectra: {len(spectra)}\n")
                f.write(f"# Common Energy Range: {min_energy:.2f} - {max_energy:.2f} eV\n")
                f.write("#\n")
            
            # Write data
            df.to_csv(f, index=False)
        
        return output_path

    def export_with_custom_columns(self, spectra: List[Spectrum],
                                 custom_columns: Dict[str, List[float]],
                                 filename: str) -> Path:
        """Export spectra with additional custom columns."""
        output_path = self.output_dir / filename
        
        # Start with first spectrum as base
        if not spectra:
            raise ValueError("At least one spectrum required")
        
        base_spectrum = spectra[0]
        df_data = {
            'Binding_Energy_eV': np.round(base_spectrum.energy, self.decimal_places),
            'Intensity_cps': np.round(base_spectrum.intensity, self.decimal_places)
        }
        
        # Add additional spectra
        for i, spectrum in enumerate(spectra[1:], 2):
            # Interpolate to base energy axis if needed
            if not np.array_equal(spectrum.energy, base_spectrum.energy):
                interpolated_intensity = np.interp(base_spectrum.energy, 
                                                 spectrum.energy, 
                                                 spectrum.intensity)
            else:
                interpolated_intensity = spectrum.intensity
            
            df_data[f'Intensity_{i}_cps'] = np.round(interpolated_intensity, self.decimal_places)
        
        # Add custom columns
        for col_name, col_data in custom_columns.items():
            if len(col_data) == len(base_spectrum.energy):
                df_data[col_name] = np.round(col_data, self.decimal_places)
            else:
                print(f"Warning: Custom column '{col_name}' length mismatch, skipping")
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        return output_path
