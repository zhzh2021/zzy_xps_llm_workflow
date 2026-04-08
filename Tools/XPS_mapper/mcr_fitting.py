"""
MCR Component Fitting and Quantification Module

Handles template-based peak fitting of MCR/NMF components and 
quantitative scaling of concentration maps to absolute atomic percentages.
"""
from __future__ import annotations
import logging
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger("xps_map.mcr_fitting")


def _safe_get(obj, key, default=None):
    """Safely get a key from a dict-like object.

    Returns default if obj is not a dict.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# Import fitting functions (handle gracefully if not available)
try:
    from XPS_peakfitting_V2 import fit_region_with_template, load_yaml_template
    FITTING_AVAILABLE = True
except (ImportError, UnicodeEncodeError) as e:
    logger.warning(f"Peak fitting functions not available: {e}")
    fit_region_with_template = None
    load_yaml_template = None
    FITTING_AVAILABLE = False


def fit_mcr_components(mcr_results: dict, energy: np.ndarray, region: str, 
                       template_dir: Path, output_dir: Path, 
                       base_name: str = "mcr") -> Optional[dict]:
    """
    Fit MCR/NMF component spectra using template-based peak fitting.
    
    Args:
        mcr_results: Dict from run_mcr_with_pca_init() containing 'component_spectra', 'method'
        energy: Energy axis array
        region: Region name (e.g., 'C1s', 'F1s', 'O1s')
        template_dir: Directory containing YAML templates
        output_dir: Directory to save fit results
        base_name: Base filename for outputs
    
    Returns:
        Dict with fitted results: {
            'component_fits': List of fit result dicts (one per component),
            'component_labels': List of chemical state labels,
            'component_areas': List of total peak areas per component,
            'fit_quality': List of R² values,
            'atomic_percent': List of atomic percentages
        }
        Returns None if fitting is not available or fails
    """
    if not FITTING_AVAILABLE:
        logger.warning("Peak fitting functions not available (import failed). Skipping MCR component fitting.")
        return None
    
    # Find appropriate template for this region
    region_upper = region.upper() if region else ""
    template_files = list(template_dir.glob(f"*{region_upper}*.yaml")) + \
                    list(template_dir.glob(f"*{region_upper}*.yml"))
    
    if not template_files:
        logger.warning(f"No template found for region '{region}' in {template_dir}")
        return None
    
    template_path = template_files[0]
    logger.info(f"Using template: {template_path.name}")
    
    try:
        template = load_yaml_template(template_path)
        region_config = template.get('regions', [{}])[0]  # Use first region in template
    except Exception as e:
        logger.error(f"Failed to load template {template_path}: {e}")
        return None
    
    component_spectra = mcr_results['component_spectra']  # Shape: (n_energy, n_components)
    n_components = component_spectra.shape[1]
    method = mcr_results.get('method', 'MCR')
    
    logger.info(f"Fitting {n_components} {method} component spectra with peak fitting...")
    
    fitted_results = {
        'component_fits': [],
        'component_labels': [],
        'component_areas': [],
        'fit_quality': []
    }
    
    for i in range(n_components):
        spectrum = component_spectra[:, i]
        
        logger.info(f"  Fitting Component {i+1}/{n_components}...")
        
        try:
            # Fit this component spectrum using template
            # Note: fit_region_with_template signature is (energy, intensity, region_config)
            fit_result = fit_region_with_template(
                energy,
                spectrum,
                region_config
            )
            
            # Check if fit_result is a dict (not numpy array or other type)
            if fit_result is not None and isinstance(fit_result, dict) and 'components' in fit_result:
                fitted_results['component_fits'].append(fit_result)
                
                # Extract chemical state label (dominant peak)
                components = _safe_get(fit_result, 'components', {})
                if components:
                    # components may be a dict mapping peak_name -> curve (np.ndarray)
                    # while detailed parameters live in fit_result['peaks'] (list of dicts).
                    peaks_list = _safe_get(fit_result, 'peaks', [])
                    peak_param_map = {p['name']: p for p in peaks_list} if peaks_list else {}

                    # Find peak with highest area using peak_param_map when available,
                    # otherwise fall back to computing area from the curve arrays.
                    if peak_param_map:
                        dominant_peak = max(peak_param_map.items(), key=lambda x: x[1].get('area', 0))
                        peak_name = dominant_peak[0]
                    else:
                        # components values may be curves; compute area via trapz
                        try:
                            areas = {name: float(np.trapz(curve, energy)) for name, curve in components.items()}
                            peak_name = max(areas.items(), key=lambda x: x[1])[0]
                        except Exception:
                            peak_name = list(components.keys())[0]

                    fitted_results['component_labels'].append(peak_name)

                    # Sum all peak areas for this component: prefer peak_param_map areas
                    if peak_param_map:
                        total_area = sum(p.get('area', 0.0) for p in peak_param_map.values())
                    else:
                        total_area = sum(float(np.trapz(c, energy)) for c in components.values())

                    fitted_results['component_areas'].append(total_area)

                    # Store fit quality
                    r2 = _safe_get(fit_result, 'r_squared', 0.0)
                    fitted_results['fit_quality'].append(r2)

                    logger.info(f"    ✓ {peak_name} (R²={r2:.3f}, Area={total_area:.1f})")
                else:
                    # No peaks found
                    fitted_results['component_labels'].append(f"Component_{i}")
                    fitted_results['component_areas'].append(0.0)
                    fitted_results['fit_quality'].append(0.0)
                    logger.warning(f"    ⚠ No peaks found in fit")
            else:
                # Fit failed or returned unexpected type
                if fit_result is not None and not isinstance(fit_result, dict):
                    logger.warning(f"    ⚠ Fit returned unexpected type: {type(fit_result).__name__}")
                fitted_results['component_fits'].append(None)
                fitted_results['component_labels'].append(f"Component_{i}_FitFailed")
                fitted_results['component_areas'].append(0.0)
                fitted_results['fit_quality'].append(0.0)
                logger.warning(f"    ✗ Fit failed")
        
        except Exception as e:
            logger.error(f"    ✗ Error fitting component {i}: {e}")
            fitted_results['component_fits'].append(None)
            fitted_results['component_labels'].append(f"Component_{i}_Error")
            fitted_results['component_areas'].append(0.0)
            fitted_results['fit_quality'].append(0.0)
    
    # Calculate atomic percentages from peak areas
    total_area_sum = sum(fitted_results['component_areas'])
    if total_area_sum > 0:
        atomic_percents = [(area / total_area_sum) * 100 for area in fitted_results['component_areas']]
        fitted_results['atomic_percent'] = atomic_percents
        
        logger.info(f"\n{method} Component Quantification (by MCR component):")
        for i, (label, pct) in enumerate(zip(fitted_results['component_labels'], atomic_percents)):
            logger.info(f"  Component {i}: {label} = {pct:.2f} at%")
    
    # Deconvolve MCR components into individual chemical species
    # Temporarily add fitted_results to mcr_results for deconvolution
    temp_mcr = mcr_results.copy()
    temp_mcr['fitted_components'] = fitted_results
    deconvolved = deconvolve_mcr_to_species(temp_mcr)
    if deconvolved:
        fitted_results['deconvolved_species'] = deconvolved
        logger.info(f"\n✓ Deconvolved {deconvolved['n_species']} individual chemical species")
    
    return fitted_results


def deconvolve_mcr_to_species(mcr_results: dict) -> Optional[dict]:
    """
    Deconvolve MCR components into individual chemical species based on fitted peaks.
    
    Each MCR component may contain multiple overlapping peaks (chemical species).
    This function extracts each individual peak and calculates its spatial distribution
    and quantitative atomic percentage.
    
    Args:
        mcr_results: Dict containing 'fitted_components', 'conc_maps', 'component_spectra'
    
    Returns:
        Dict with deconvolved species: {
            'species_names': List of all chemical species names,
            'species_atomic_percent': List of atomic % for each species,
            'species_concentration_maps': np.ndarray of shape (ny, nx, n_species),
            'species_to_mcr_component': List mapping species index to parent MCR component,
            'species_peak_info': List of dicts with peak parameters (BE, FWHM, etc.)
        }
    """
    fitted_components = mcr_results.get('fitted_components')
    conc_maps = mcr_results.get('conc_maps')  # Shape: (ny, nx, n_mcr_components)
    
    if fitted_components is None or conc_maps is None:
        logger.warning("Cannot deconvolve: missing fitted_components or conc_maps")
        return None
    
    component_fits = fitted_components.get('component_fits', [])
    
    species_names = []
    species_atomic_percent = []
    species_concentration_maps = []
    species_to_mcr_component = []
    species_peak_info = []
    
    total_area_all_peaks = 0.0
    all_peak_areas = []
    
    # First pass: collect all peaks from all MCR components and calculate total area
    for mcr_idx, fit_result in enumerate(component_fits):
        if fit_result is None:
            continue
        
        peaks_list = _safe_get(fit_result, 'peaks', [])
        if not peaks_list:
            continue
        
        for peak in peaks_list:
            area = peak.get('area', 0.0)
            if area > 0:
                all_peak_areas.append({
                    'mcr_idx': mcr_idx,
                    'peak': peak,
                    'area': area
                })
                total_area_all_peaks += area
    
    if total_area_all_peaks == 0:
        logger.warning("No valid peaks found for deconvolution")
        return None
    
    ny, nx, n_mcr = conc_maps.shape
    
    # Second pass: create concentration map for each peak and store by name
    logger.info(f"\nDeconvolving {len(all_peak_areas)} peaks from {n_mcr} MCR components:")
    
    # Use dict to accumulate species with same name from different MCR components
    species_dict = {}  # {peak_name: {'map': np.ndarray, 'atomic_pct': float, 'total_area': float, 'mcr_components': list, 'peak_params': list}}
    
    for peak_data in all_peak_areas:
        mcr_idx = peak_data['mcr_idx']
        peak = peak_data['peak']
        area = peak_data['area']
        
        peak_name = peak.get('name', f'Peak_{len(species_dict)}')
        
        # Calculate atomic percentage for this peak instance
        atomic_pct = (area / total_area_all_peaks) * 100
        
        # Get the MCR concentration map for the parent component
        mcr_conc_map = conc_maps[:, :, mcr_idx]  # Shape: (ny, nx)
        
        # Get all peaks in this MCR component to calculate the fraction
        fit_result = component_fits[mcr_idx]
        all_peaks_in_component = _safe_get(fit_result, 'peaks', [])
        total_area_in_component = sum(p.get('area', 0.0) for p in all_peaks_in_component)
        
        if total_area_in_component > 0:
            # Scale the MCR map by the fraction of this peak's area
            peak_fraction = area / total_area_in_component
            species_map = mcr_conc_map * peak_fraction
        else:
            species_map = np.zeros_like(mcr_conc_map)
        
        # Scale map so values represent local concentration percentage
        # The average of all non-zero pixels should equal atomic_pct
        nonzero_mask = species_map > 0
        if np.any(nonzero_mask):
            current_mean = np.mean(species_map[nonzero_mask])
            if current_mean > 0:
                species_map = species_map * (atomic_pct / current_mean)
        
        # Accumulate species with same name
        if peak_name in species_dict:
            # Add to existing species
            species_dict[peak_name]['map'] += species_map
            species_dict[peak_name]['atomic_pct'] += atomic_pct
            species_dict[peak_name]['total_area'] += area
            species_dict[peak_name]['mcr_components'].append(mcr_idx)
            species_dict[peak_name]['peak_params'].append({
                'BE': peak.get('center', 0.0),
                'FWHM': peak.get('fwhm', 0.0),
                'area': area,
                'mcr_component': mcr_idx
            })
            logger.info(f"  {peak_name}: +{atomic_pct:.2f} at% (BE={peak.get('center', 0):.2f} eV, from MCR component {mcr_idx}) [MERGED]")
        else:
            # Create new species entry
            species_dict[peak_name] = {
                'map': species_map,
                'atomic_pct': atomic_pct,
                'total_area': area,
                'mcr_components': [mcr_idx],
                'peak_params': [{
                    'BE': peak.get('center', 0.0),
                    'FWHM': peak.get('fwhm', 0.0),
                    'area': area,
                    'mcr_component': mcr_idx
                }]
            }
            logger.info(f"  {peak_name}: {atomic_pct:.2f} at% (BE={peak.get('center', 0):.2f} eV, from MCR component {mcr_idx})")
    
    # Third pass: convert dict to lists (sorted by atomic percentage)
    sorted_species = sorted(species_dict.items(), key=lambda x: x[1]['atomic_pct'], reverse=True)
    
    for peak_name, species_data in sorted_species:
        species_names.append(peak_name)
        species_atomic_percent.append(species_data['atomic_pct'])
        species_concentration_maps.append(species_data['map'])
        
        # For backward compatibility, use first MCR component if single, or -1 if merged
        if len(species_data['mcr_components']) == 1:
            species_to_mcr_component.append(species_data['mcr_components'][0])
        else:
            species_to_mcr_component.append(-1)  # Indicates merged from multiple components
        
        # Average BE and FWHM if merged from multiple components
        avg_be = np.mean([p['BE'] for p in species_data['peak_params']])
        avg_fwhm = np.mean([p['FWHM'] for p in species_data['peak_params']])
        
        species_peak_info.append({
            'name': peak_name,
            'BE': avg_be,
            'FWHM': avg_fwhm,
            'area': species_data['total_area'],
            'atomic_percent': species_data['atomic_pct'],
            'parent_mcr_component': species_data['mcr_components'] if len(species_data['mcr_components']) > 1 else species_data['mcr_components'][0],
            'merged_from_components': len(species_data['mcr_components']) > 1,
            'peak_params': species_data['peak_params']  # Keep individual peak info
        })
    
    if sorted_species:
        logger.info(f"\n✓ Final merged species: {len(sorted_species)} unique chemical species")
        for name, data in sorted_species:
            if len(data['mcr_components']) > 1:
                logger.info(f"  {name}: {data['atomic_pct']:.2f} at% (merged from MCR components {data['mcr_components']})")
    
    # Stack all species maps
    species_maps_array = np.stack(species_concentration_maps, axis=2)  # Shape: (ny, nx, n_species)
    
    return {
        'species_names': species_names,
        'species_atomic_percent': species_atomic_percent,
        'species_concentration_maps': species_maps_array,
        'species_to_mcr_component': species_to_mcr_component,
        'species_peak_info': species_peak_info,
        'n_species': len(species_names)
    }


def save_mcr_fitting_results(mcr_results: Dict, output_dir: Path, base_name: str, 
                              region: str, energy: np.ndarray):
    """
    Save detailed MCR component fitting results to CSV files.
    Exports peak parameters (BE, FWHM, area, etc.) for each component.
    Now includes deconvolved individual chemical species when available.
    
    Creates two CSV files:
    1. *_fit_parameters.csv: Detailed peak parameters for all chemical species
    2. *_peak_correlation.csv: Mapping showing all fitted species
    """
    fitted_components = mcr_results.get('fitted_components')
    if not fitted_components:
        return
    
    component_fits = fitted_components.get('component_fits', [])
    component_labels = fitted_components.get('component_labels', [])
    component_areas = fitted_components.get('component_areas', [])
    fit_quality = fitted_components.get('fit_quality', [])
    atomic_percent = fitted_components.get('atomic_percent', [])
    deconvolved = fitted_components.get('deconvolved_species')
    
    # Create comprehensive fitting parameters CSV
    fit_params_file = output_dir / f"{base_name}_{region}_fit_parameters.csv"
    
    with open(fit_params_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Peak_Name', 'BE_eV', 'FWHM_eV', 'Area', 
                        'Height', 'Atomic_Percent', 'MCR_Component', 'R_Squared'])
        
        # Use deconvolved species if available, otherwise use MCR components
        if deconvolved:
            # Write deconvolved individual chemical species
            species_info = deconvolved.get('species_peak_info', [])
            for peak_info in species_info:
                mcr_comp = peak_info.get('parent_mcr_component', 0)
                
                # Handle merged species (parent_mcr_component could be a list)
                if isinstance(mcr_comp, list):
                    mcr_comp_str = '+'.join([f"Component_{c}" for c in mcr_comp])
                    # Average R² from all components
                    r2 = np.mean([fit_quality[c] for c in mcr_comp if c < len(fit_quality)]) if mcr_comp else 0.0
                else:
                    mcr_comp_str = f"Component_{mcr_comp}"
                    r2 = fit_quality[mcr_comp] if mcr_comp < len(fit_quality) else 0.0
                
                writer.writerow([
                    peak_info.get('name', 'Unknown'),
                    f"{peak_info.get('BE', 0.0):.3f}",
                    f"{peak_info.get('FWHM', 0.0):.3f}",
                    f"{peak_info.get('area', 0.0):.2f}",
                    f"{peak_info.get('area', 0.0) / 10.0:.2f}",  # Approximate height from area
                    f"{peak_info.get('atomic_percent', 0.0):.2f}",
                    mcr_comp_str,
                    f"{r2:.4f}"
                ])
        else:
            # Fall back to MCR component-level data
            for comp_idx, fit_result in enumerate(component_fits):
                if fit_result is None:
                    writer.writerow(['FIT_FAILED', 'N/A', 'N/A', 'N/A', 'N/A', 
                                    atomic_percent[comp_idx] if comp_idx < len(atomic_percent) else 'N/A',
                                    f'Component_{comp_idx}',
                                    fit_quality[comp_idx] if comp_idx < len(fit_quality) else 'N/A'])
                    continue
                
                components = _safe_get(fit_result, 'components', {})
                r_squared = _safe_get(fit_result, 'r_squared', 0.0)
                peaks_list = _safe_get(fit_result, 'peaks', [])
                peak_param_map = {p['name']: p for p in peaks_list} if peaks_list else {}

                for peak_name, peak_data in components.items():
                    params = peak_param_map.get(peak_name, {})
                    be = params.get('center') if params else None
                    fwhm = params.get('fwhm') if params else None
                    area = params.get('area') if params else None
                    height = params.get('amplitude') if params else None

                    if be is None or fwhm is None or area is None or height is None:
                        try:
                            curve = peak_data if isinstance(peak_data, np.ndarray) else None
                            if curve is not None:
                                area = float(np.trapz(curve, energy)) if area is None else area
                                height = float(np.max(curve)) if height is None else height
                                be = be if be is not None else 0.0
                                fwhm = fwhm if fwhm is not None else 0.0
                        except Exception:
                            be = be if be is not None else 0.0
                            fwhm = fwhm if fwhm is not None else 0.0
                            area = area if area is not None else 0.0
                            height = height if height is not None else 0.0

                    writer.writerow([
                        peak_name,
                        f'{(be or 0.0):.3f}',
                        f'{(fwhm or 0.0):.3f}',
                        f'{(area or 0.0):.2f}',
                        f'{(height or 0.0):.2f}',
                        f"{atomic_percent[comp_idx]:.2f}" if comp_idx < len(atomic_percent) else 'N/A',
                        f'Component_{comp_idx}',
                        f'{r_squared:.4f}'
                    ])
    
    logger.info(f"Saved fitting parameters: {fit_params_file.name}")
    
    # Create peak correlation CSV showing all fitted species
    correlation_file = output_dir / f"{base_name}_{region}_peak_correlation.csv"
    
    with open(correlation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if deconvolved:
            # Use deconvolved species data
            writer.writerow(['Peak_Name', 'BE_eV', 'FWHM_eV', 'Area', 'Atomic_Percent_%', 'MCR_Component', 'Merged'])
            
            species_info = deconvolved.get('species_peak_info', [])
            for peak_info in species_info:
                mcr_comp = peak_info.get('parent_mcr_component', 0)
                is_merged = peak_info.get('merged_from_components', False)
                
                # Handle merged species
                if isinstance(mcr_comp, list):
                    mcr_comp_str = '+'.join([f"Component_{c}" for c in mcr_comp])
                    merged_str = 'Yes'
                else:
                    mcr_comp_str = f"Component_{mcr_comp}"
                    merged_str = 'No'
                
                writer.writerow([
                    peak_info.get('name', 'Unknown'),
                    f"{peak_info.get('BE', 0.0):.3f}",
                    f"{peak_info.get('FWHM', 0.0):.3f}",
                    f"{peak_info.get('area', 0.0):.2f}",
                    f"{peak_info.get('atomic_percent', 0.0):.2f}",
                    mcr_comp_str,
                    merged_str
                ])
        else:
            # Fall back to MCR component-level data
            writer.writerow(['MCR_Component', 'Dominant_Peak', 'BE_eV', 'Chemical_State', 
                            'Total_Area', 'Atomic_Percent_%'])
            
            for comp_idx, (fit_result, area, pct) in enumerate(zip(component_fits, component_areas, atomic_percent)):
                if fit_result is None or not _safe_get(fit_result, 'components'):
                    writer.writerow([f'Component_{comp_idx}', 'N/A', 'N/A', 'Fit Failed', '0.0', '0.0'])
                    continue
                
                components = _safe_get(fit_result, 'components', {})
                peaks_list = _safe_get(fit_result, 'peaks', [])
                peak_param_map = {p['name']: p for p in peaks_list} if peaks_list else {}
                if not components:
                    continue

                if peak_param_map:
                    dominant_peak = max(peak_param_map.items(), key=lambda x: x[1].get('area', 0))
                    peak_name = dominant_peak[0]
                    be = dominant_peak[1].get('center', 0.0)
                else:
                    try:
                        areas = {name: float(np.trapz(curve, energy)) for name, curve in components.items()}
                        peak_name = max(areas.items(), key=lambda x: x[1])[0]
                        be = 0.0
                    except Exception:
                        peak_name = list(components.keys())[0]
                        be = 0.0
                
                writer.writerow([
                    f'Component_{comp_idx}',
                    peak_name,
                    f'{be:.3f}',
                    component_labels[comp_idx] if comp_idx < len(component_labels) else 'Unknown',
                    f'{area:.2f}',
                    f'{pct:.2f}'
                ])
    
    logger.info(f"Saved peak correlation: {correlation_file.name}")

    # Save per-component fit curves (energy, raw, total fit, individual component curves)
    for comp_idx, fit_result in enumerate(component_fits):
        if fit_result is None:
            continue
        try:
            x_vals = _safe_get(fit_result, 'x', energy)
            raw = _safe_get(fit_result, 'raw', None)
            fit_total = _safe_get(fit_result, 'fit', None)
            comps = _safe_get(fit_result, 'components', {})

            cols = []
            headers = []
            cols.append(np.asarray(x_vals))
            headers.append('Energy_eV')

            if raw is not None:
                cols.append(np.asarray(raw))
                headers.append('Raw')
            if fit_total is not None:
                cols.append(np.asarray(fit_total))
                headers.append('Fit_Total')

            # Only include component curves that are arrays
            for name, curve in comps.items():
                if isinstance(curve, np.ndarray):
                    cols.append(np.asarray(curve))
                    headers.append(f'Comp_{name}')

            if len(cols) > 1:
                out_arr = np.vstack(cols).T
                curve_file = output_dir / f"{base_name}_{region}_component{comp_idx}_fit_curve.csv"
                np.savetxt(curve_file, out_arr, delimiter=',', header=','.join(headers), fmt='%.6f', comments='')
                logger.info(f"Saved component fit curves: {curve_file.name}")
        except Exception as e:
            logger.warning(f"Failed to save fit curves for component {comp_idx}: {e}")


def save_quantitative_concentration_maps(mcr_results: Dict, output_dir: Path, 
                                         base_name: str, region: str, ny: int, nx: int):
    """
    Save quantitatively scaled MCR concentration maps.
    Scales relative MCR concentrations to absolute atomic percentages based on peak fitting.
    
    Creates files:
    - *_MCR_component*_quantitative_at%.csv: Scaled concentration maps
    - *_MCR_quantification_info.txt: Scaling methodology documentation
    """
    fitted_components = mcr_results.get('fitted_components')
    if not fitted_components:
        return
    
    conc_maps = mcr_results.get('conc_maps')  # Shape: (ny, nx, n_components)
    if conc_maps is None:
        return
    
    atomic_percent = fitted_components.get('atomic_percent', [])
    component_labels = fitted_components.get('component_labels', [])
    
    n_components = conc_maps.shape[2]
    
    # Save each quantitative concentration map
    for i in range(n_components):
        relative_map = conc_maps[:, :, i]  # Relative concentration from MCR
        
        # Scale to absolute atomic percentage if available
        if i < len(atomic_percent):
            # Calculate scaling factor
            # MCR gives relative concentrations, we want them to match the average atomic %
            map_mean = np.mean(relative_map[relative_map > 0]) if np.any(relative_map > 0) else 1.0
            
            # Scale so that average matches the fitted atomic percentage
            if map_mean > 0:
                scale_factor = atomic_percent[i] / (map_mean * 100)  # Convert % to fraction
                quantitative_map = relative_map * scale_factor * 100  # Scale to percentage
            else:
                quantitative_map = relative_map  # Can't scale if no data
            
            # Save quantitative map
            quant_file = output_dir / f"{base_name}_{region}_MCR_component{i}_quantitative_at%.csv"
            np.savetxt(quant_file, quantitative_map, delimiter=',', 
                      fmt='%.4f', 
                      header=f'Quantitative atomic % map for {component_labels[i] if i < len(component_labels) else f"Component {i}"}',
                      comments='')
            
            logger.info(f"Saved quantitative concentration map {i}: {quant_file.name}")
            logger.info(f"  Scaling: MCR relative → {atomic_percent[i]:.2f} at% average")
        else:
            # No fitting data, save relative map only
            rel_file = output_dir / f"{base_name}_{region}_MCR_component{i}_relative_conc.csv"
            np.savetxt(rel_file, relative_map, delimiter=',', fmt='%.6f',
                      header=f'Relative concentration for Component {i}', comments='')
    
    # Save scaling information summary
    scaling_info_file = output_dir / f"{base_name}_{region}_quantification_info.txt"
    
    # Check if deconvolved species are available
    deconvolved = fitted_components.get('deconvolved_species')
    
    with open(scaling_info_file, 'w') as f:
        f.write("Quantitative Concentration Map Information\n")
        f.write("=" * 70 + "\n\n")
        f.write("Method: MCR relative concentrations scaled to match peak fitting atomic percentages\n")
        
        if deconvolved:
            f.write("\n*** Deconvolved Individual Chemical Species ***\n")
            f.write(f"Total species identified: {deconvolved['n_species']}\n\n")
            
            species_info = deconvolved.get('species_peak_info', [])
            f.write("Individual Species Quantification:\n")
            for peak_info in species_info:
                name = peak_info.get('name', 'Unknown')
                be = peak_info.get('BE', 0.0)
                fwhm = peak_info.get('FWHM', 0.0)
                at_pct = peak_info.get('atomic_percent', 0.0)
                mcr_comp = peak_info.get('parent_mcr_component', 0)
                is_merged = peak_info.get('merged_from_components', False)
                
                f.write(f"\n{name}:\n")
                f.write(f"  Binding Energy: {be:.2f} eV\n")
                f.write(f"  FWHM: {fwhm:.2f} eV\n")
                f.write(f"  Atomic %: {at_pct:.2f}%\n")
                
                if isinstance(mcr_comp, list):
                    f.write(f"  Parent MCR Components: {mcr_comp} (MERGED)\n")
                else:
                    f.write(f"  Parent MCR Component: {mcr_comp}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("\nOriginal MCR Component Summary:\n")
        else:
            f.write("\n")
        
        f.write("\nMCR Component Quantification:\n")
        for i in range(n_components):
            label = component_labels[i] if i < len(component_labels) else f"Component {i}"
            at_pct = atomic_percent[i] if i < len(atomic_percent) else 0.0
            
            relative_map = conc_maps[:, :, i]
            map_mean = np.mean(relative_map[relative_map > 0]) if np.any(relative_map > 0) else 0.0
            map_std = np.std(relative_map[relative_map > 0]) if np.any(relative_map > 0) else 0.0
            
            f.write(f"\n{label}:\n")
            f.write(f"  Atomic %: {at_pct:.2f}%\n")
            f.write(f"  MCR relative conc (mean ± std): {map_mean:.4f} ± {map_std:.4f}\n")
            f.write(f"  Spatial pixels with signal: {np.sum(relative_map > 0)} / {ny*nx}\n")
            
            if i < len(atomic_percent) and map_mean > 0:
                scale_factor = atomic_percent[i] / (map_mean * 100)
                f.write(f"  Scaling factor applied: {scale_factor:.4f}\n")
    
    logger.info(f"Saved quantification info: {scaling_info_file.name}")


def plot_mcr_fitting_results(mcr_results: Dict, energy: np.ndarray, output_dir: Path,
                              base_name: str, region: str, show: bool = False):
    """
    Create comprehensive visualization of MCR component fitting results.
    
    Generates plots:
    1. Fitted component spectra with peak deconvolution
    2. Atomic percentage bar chart
    3. Peak parameter comparison table
    
    Args:
        mcr_results: Dict containing 'fitted_components' and 'component_spectra'
        energy: Energy axis array
        output_dir: Directory to save plots
        base_name: Base filename for outputs
        region: Region name (e.g., 'C1s', 'F1s')
        show: Whether to display plots interactively
    """
    fitted_components = mcr_results.get('fitted_components')
    if not fitted_components:
        logger.warning("No fitted components available for plotting")
        return
    
    component_fits = fitted_components.get('component_fits', [])
    component_labels = fitted_components.get('component_labels', [])
    component_areas = fitted_components.get('component_areas', [])
    atomic_percent = fitted_components.get('atomic_percent', [])
    fit_quality = fitted_components.get('fit_quality', [])
    component_spectra = mcr_results.get('component_spectra')
    
    n_components = len(component_fits)
    method = mcr_results.get('method', 'MCR')
    
    # Create figure with fitted spectra
    fig = plt.figure(figsize=(14, 4 * n_components))
    gs = GridSpec(n_components, 1, figure=fig, hspace=0.3)
    
    for i, (fit_result, label) in enumerate(zip(component_fits, component_labels)):
        ax = fig.add_subplot(gs[i, 0])
        
        if fit_result is None or component_spectra is None:
            ax.text(0.5, 0.5, f'Component {i}: Fit Failed', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method} Component {i}: {label}')
            continue
        
        # Plot original spectrum
        spectrum = component_spectra[:, i]
        ax.plot(energy, spectrum, 'k-', linewidth=2, label='MCR Component', alpha=0.7)
        
        # Plot fitted spectrum and individual peaks
        components = _safe_get(fit_result, 'components', {})
        if components:
            # Reconstruct total fit
            fit_total = np.zeros_like(energy)

            # Try to get detailed peak params list
            peaks_list = _safe_get(fit_result, 'peaks', [])
            peak_param_map = {p['name']: p for p in peaks_list} if peaks_list else {}

            # Plot individual peaks
            colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
            for (peak_name, peak_data), color in zip(components.items(), colors):
                # If the component value is an array, it's the curve produced by the fitter
                if isinstance(peak_data, np.ndarray):
                    curve = peak_data
                    # Use parameter map if available to label center
                    params = peak_param_map.get(peak_name, {})
                    center = params.get('center', float(energy[np.argmax(curve)]))
                    ax.fill_between(energy, 0, curve, alpha=0.3, color=color,
                                   label=f'{peak_name}: {center:.2f} eV')
                    fit_total += curve
                else:
                    # peak_data may be a dict-like with parameters
                    params = peak_param_map.get(peak_name, {}) if peak_param_map else peak_data
                    center = _safe_get(params, 'center', 0)
                    fwhm = _safe_get(params, 'fwhm', 1)
                    height = _safe_get(params, 'amplitude', _safe_get(params, 'height', 0))
                    # Simple Gaussian approximation for visualization
                    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                    peak = height * np.exp(-((energy - center) ** 2) / (2 * sigma ** 2))
                    ax.fill_between(energy, 0, peak, alpha=0.3, color=color,
                                   label=f'{peak_name}: {center:.2f} eV')
                    fit_total += peak

            # Plot total fit
            try:
                ax.plot(energy, fit_total, 'r--', linewidth=1.5, label='Total Fit', alpha=0.8)
            except Exception:
                # If something goes wrong plotting fit_total, skip total fit
                pass
        
        # Formatting
        ax.set_xlabel('Binding Energy (eV)', fontsize=15)
        ax.set_ylabel('Intensity (a.u.)', fontsize=15)
        ax.set_xlim(energy.max(), energy.min())  # Reverse x-axis
        ax.legend(loc='best', fontsize=15, ncol=2)
        
        # Title with fit quality
        r2 = fit_quality[i] if i < len(fit_quality) else 0.0
        at_pct = atomic_percent[i] if i < len(atomic_percent) else 0.0
        ax.set_title(f'{method} Component {i}', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"{base_name}_{region}_MCR_fitted_components.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved fitted components plot: {output_file.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_atomic_percentages(fitted_components: Dict, output_dir: Path,
                            base_name: str, region: str, method: str = 'MCR',
                            show: bool = False):
    """
    Create bar chart of atomic percentages for each chemical species.
    Uses deconvolved individual species if available, otherwise falls back to MCR components.
    
    Args:
        fitted_components: Dict with 'component_labels', 'atomic_percent', and optionally 'deconvolved_species'
        output_dir: Directory to save plot
        base_name: Base filename
        region: Region name
        method: Analysis method (MCR or NMF)
        show: Whether to display plot
    """
    # Check if we have deconvolved species (preferred)
    deconvolved = fitted_components.get('deconvolved_species')
    
    if deconvolved:
        # Use deconvolved individual chemical species
        component_labels = deconvolved['species_names']
        atomic_percent = deconvolved['species_atomic_percent']
        logger.info(f"Plotting {len(component_labels)} deconvolved species")
    else:
        # Fall back to MCR components
        component_labels = fitted_components.get('component_labels', [])
        atomic_percent = fitted_components.get('atomic_percent', [])
        logger.info(f"Plotting {len(component_labels)} MCR components (no deconvolution)")
    
    if not atomic_percent:
        logger.warning("No atomic percentage data available for plotting")
        return
    
    # Adjust figure size based on number of species
    n_species = len(component_labels)
    fig_width = max(12, n_species * 0.8)  # Scale width with number of species
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 5))
    
    # Bar chart of atomic percentages
    x_pos = np.arange(len(component_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(component_labels)))
    
    bars = ax1.bar(x_pos, atomic_percent, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Chemical Species', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{region} Chemical Species Quantification', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(component_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(atomic_percent) * 1.15)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, atomic_percent):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Pie chart for visual proportion
    # For many species, adjust font sizes and potentially filter very small percentages
    if n_species > 10:
        # Only label species > 1% in pie chart for clarity
        labels_pie = [label if pct > 1.0 else '' for label, pct in zip(component_labels, atomic_percent)]
        fontsize_pie = 7
    else:
        labels_pie = component_labels
        fontsize_pie = 9
    
    wedges, texts, autotexts = ax2.pie(atomic_percent, labels=labels_pie, 
                                         autopct='%1.1f%%', startangle=90,
                                         colors=colors, textprops={'fontsize': fontsize_pie})
    ax2.set_title(f'{region} Composition', fontsize=14, fontweight='bold')
    
    # Bold percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    plt.tight_layout()
    output_file = output_dir / f"{base_name}_{region}_atomic_percentages.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved atomic percentage plot: {output_file.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_quantitative_concentration_maps(mcr_results: Dict, output_dir: Path,
                                         base_name: str, region: str,
                                         x_axis: Optional[np.ndarray] = None,
                                         y_axis: Optional[np.ndarray] = None,
                                         show: bool = False):
    """
    Create spatial maps of quantitative atomic percentages for each component.
    
    Args:
        mcr_results: Dict with 'conc_maps' and 'fitted_components'
        output_dir: Directory to save plots
        base_name: Base filename
        region: Region name
        x_axis: Optional x-axis values for spatial dimensions
        y_axis: Optional y-axis values for spatial dimensions
        show: Whether to display plots
    """
    logger.info(f"=== plot_quantitative_concentration_maps ENTRY ===")
    logger.info(f"  Region: {region}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Base name: {base_name}")
    
    fitted_components = mcr_results.get('fitted_components')
    conc_maps = mcr_results.get('conc_maps')
    
    logger.info(f"  fitted_components present: {fitted_components is not None}")
    logger.info(f"  conc_maps present: {conc_maps is not None}")
    if conc_maps is not None:
        logger.info(f"  conc_maps shape: {conc_maps.shape}")

    if fitted_components is None or conc_maps is None:
        logger.warning("Missing data for quantitative concentration map plots - RETURNING EARLY")
        return

    component_labels = fitted_components.get('component_labels', [])
    atomic_percent = fitted_components.get('atomic_percent', [])

    ny, nx, n_components = conc_maps.shape

    # Calculate quantitative maps
    quantitative_maps = []
    for i in range(n_components):
        relative_map = conc_maps[:, :, i]

        if i < len(atomic_percent):
            mask = relative_map > 0
            map_mean = np.mean(relative_map[mask]) if np.any(mask) else 1.0
            if map_mean > 0:
                scale_factor = atomic_percent[i] / (map_mean * 100)
                quantitative_map = relative_map * scale_factor * 100
            else:
                quantitative_map = relative_map
        else:
            quantitative_map = relative_map

        quantitative_maps.append(quantitative_map)

    # Page large numbers of components to avoid huge figures / OOM
    max_per_page = 9
    max_cols = 3
    import math, gc

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Unable to create output directory {output_dir}: {e}")

    for page_start in range(0, n_components, max_per_page):
        page_end = min(page_start + max_per_page, n_components)
        page_indices = list(range(page_start, page_end))
        page_n = len(page_indices)

        ncols = min(max_cols, page_n)
        nrows = math.ceil(page_n / ncols)

        try:
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        except Exception as e:
            logger.exception(f"Failed to create figure for quantitative maps page {page_start // max_per_page + 1}: {e}")
            return

        # Normalize axes to flat list
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = np.array([axes])

        # Extent
        if x_axis is not None and y_axis is not None:
            extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
        else:
            extent = [0, nx, ny, 0]

        for idx_in_page, comp_idx in enumerate(page_indices):
            ax = axes_flat[idx_in_page]
            quant_map = quantitative_maps[comp_idx]
            label = component_labels[comp_idx] if comp_idx < len(component_labels) else f'Component {comp_idx}'
            at_pct = atomic_percent[comp_idx] if comp_idx < len(atomic_percent) else 0.0

            try:
                # Sanitize map: replace NaN/Inf
                quant = np.nan_to_num(quant_map, nan=0.0, posinf=0.0, neginf=0.0)

                # Downsample very large maps to avoid OOM when plotting
                max_pixels = 1_000_000  # target max pixels to plot per map
                ny_map, nx_map = quant.shape
                if ny_map * nx_map > max_pixels:
                    factor = int(np.ceil(np.sqrt((ny_map * nx_map) / max_pixels)))
                    if factor > 1:
                        quant = quant[::factor, ::factor]
                        logger.debug(f"Downsampled map for component {comp_idx} by factor {factor}")

                # compute vmax robustly
                mask = quant > 0
                if np.any(mask):
                    try:
                        vmax = float(np.percentile(quant[mask], 99))
                    except Exception:
                        vmax = float(np.max(quant[mask]))
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = float(np.max(quant)) if np.isfinite(np.max(quant)) and np.max(quant) > 0 else 1.0
                else:
                    vmax = float(np.max(quant)) if np.isfinite(np.max(quant)) and np.max(quant) > 0 else 1.0

                im = ax.imshow(quant, cmap='viridis', aspect='auto', extent=extent, vmin=0, vmax=vmax, interpolation='nearest')
                ax.set_title(f'{label}\n(Avg: {at_pct:.1f} at%)', fontsize=15, fontweight='bold')
                ax.set_xlabel('X Position (µm)' if x_axis is not None else 'X Pixel', fontsize=15)
                ax.set_ylabel('Y Position (µm)' if y_axis is not None else 'Y Pixel', fontsize=15)

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Concentration %', fontsize=15)
                cbar.ax.tick_params(labelsize=12)
            except Exception as e:
                logger.exception(f"Failed plotting quantitative map for component {comp_idx}: {e}")
                try:
                    ax.text(0.5, 0.5, 'Plot failed', ha='center', va='center', transform=ax.transAxes)
                except Exception:
                    pass

        # Hide unused axes
        for j in range(len(page_indices), len(axes_flat)):
            try:
                axes_flat[j].axis('off')
            except Exception:
                pass

        plt.suptitle(f'{region} Quantitative Concentration Maps (components {page_start}-{page_end-1})', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save page
        if n_components > max_per_page:
            page_no = page_start // max_per_page + 1
            output_file = output_dir / f"{base_name}_{region}_quantitative_maps_page{page_no}.png"
        else:
            output_file = output_dir / f"{base_name}_{region}_quantitative_maps.png"

        try:
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            logger.info(f"Saved quantitative concentration maps: {output_file.name}")
        except Exception as e:
            logger.exception(f"Failed to save quantitative concentration maps PNG: {e}")

        if show:
            plt.show()
        plt.close(fig)
        gc.collect()


def plot_peak_parameter_summary(fitted_components: Dict, output_dir: Path,
                                base_name: str, region: str, show: bool = False):
    """
    Create summary table visualization of peak parameters.
    
    Args:
        fitted_components: Dict with component fitting results
        output_dir: Directory to save plot
        base_name: Base filename
        region: Region name
        show: Whether to display plot
    """
    component_fits = fitted_components.get('component_fits', [])
    component_labels = fitted_components.get('component_labels', [])
    fit_quality = fitted_components.get('fit_quality', [])
    
    # Collect data for table
    table_data = []
    for i, (fit_result, label, r2) in enumerate(zip(component_fits, component_labels, fit_quality)):
        if fit_result is None or not _safe_get(fit_result, 'components'):
            table_data.append([f'Comp {i}', label, 'N/A', 'N/A', 'N/A', f'{r2:.3f}'])
            continue
        
        components = _safe_get(fit_result, 'components', {})
        peaks_list = _safe_get(fit_result, 'peaks', [])
        peak_param_map = {p['name']: p for p in peaks_list} if peaks_list else {}

        for peak_name, peak_data in components.items():
            params = peak_param_map.get(peak_name, {})
            be = params.get('center') if params else _safe_get(peak_data, 'center', 0)
            fwhm = params.get('fwhm') if params else _safe_get(peak_data, 'fwhm', 0)
            area = params.get('area') if params else _safe_get(peak_data, 'area', 0)

            table_data.append([
                f'Comp {i}',
                label,
                peak_name,
                f'{(be or 0.0):.2f}',
                f'{(fwhm or 0.0):.2f}',
                f'{r2:.3f}'
            ])
    
    if not table_data:
        logger.warning("No peak parameter data for summary table")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(4, len(table_data) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    col_labels = ['MCR\nComp', 'Chemical\nSpecies', 'Peak\nName', 'BE\n(eV)', 'FWHM\n(eV)', 'R²']
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                    cellLoc='center', loc='center',
                    colColours=['#E0E0E0']*len(col_labels))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', fontsize=10)
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
    
    plt.title(f'{region} Peak Parameter Summary', fontsize=14, fontweight='bold', pad=20)
    
    output_file = output_dir / f"{base_name}_{region}_peak_parameters_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved peak parameter table: {output_file.name}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_concentration_maps(mcr_results: Dict, output_dir: Path,
                                     base_name: str, region: str,
                                     x_axis: Optional[np.ndarray] = None,
                                     y_axis: Optional[np.ndarray] = None,
                                     show: bool = False):
    """
    Create an overlay map showing spatial distribution of all chemical species
    with each species assigned a unique color based on its dominant presence.
    Uses deconvolved species if available, otherwise falls back to MCR components.
    
    Args:
        mcr_results: Dict with 'conc_maps' and 'fitted_components'
        output_dir: Directory to save plots
        base_name: Base filename
        region: Region name
        x_axis: Optional x-axis values for spatial dimensions
        y_axis: Optional y-axis values for spatial dimensions
        show: Whether to display plots
    """
    logger.info(f"Creating combined concentration overlay map for {region}")
    
    fitted_components = mcr_results.get('fitted_components')
    if fitted_components is None:
        logger.warning("Missing fitted_components for combined concentration map")
        return
    
    # Check if we have deconvolved species (preferred)
    deconvolved = fitted_components.get('deconvolved_species')
    
    if deconvolved:
        # Use deconvolved individual chemical species
        logger.info(f"Using deconvolved species ({deconvolved['n_species']} species)")
        component_labels = deconvolved['species_names']
        atomic_percent = deconvolved['species_atomic_percent']
        species_maps = deconvolved['species_concentration_maps']  # Shape: (ny, nx, n_species)
        ny, nx, n_components = species_maps.shape
    else:
        # Fall back to MCR components
        logger.info("Using MCR components (deconvolution not available)")
        conc_maps = mcr_results.get('conc_maps')
        if conc_maps is None:
            logger.warning("Missing conc_maps for combined concentration map")
            return
        
        component_labels = fitted_components.get('component_labels', [])
        atomic_percent = fitted_components.get('atomic_percent', [])

        ny, nx, n_components = conc_maps.shape
        
        # Calculate quantitative maps and filter valid components
        quantitative_maps = []
        valid_labels = []
        valid_at_pct = []
        
        for i in range(n_components):
            relative_map = conc_maps[:, :, i]
            
            if i < len(atomic_percent) and atomic_percent[i] > 0.01:  # Only include components with >0.01% concentration
                mask = relative_map > 0
                map_mean = np.mean(relative_map[mask]) if np.any(mask) else 1.0
                
                if map_mean > 0:
                    scale_factor = atomic_percent[i] / (map_mean * 100)
                    quantitative_map = relative_map * scale_factor * 100
                else:
                    quantitative_map = relative_map
                
                quantitative_maps.append(quantitative_map)
                valid_labels.append(component_labels[i] if i < len(component_labels) else f'Component {i}')
                valid_at_pct.append(atomic_percent[i] if i < len(atomic_percent) else 0.0)
        
        if not quantitative_maps:
            logger.warning("No valid components to plot")
            return
        
        n_valid = len(quantitative_maps)
        species_maps = np.stack(quantitative_maps, axis=2)
    
    # Filter out species with very low concentration
    valid_labels = []
    valid_at_pct = []
    valid_maps = []
    
    for i in range(n_components):
        if i < len(component_labels) and i < len(atomic_percent):
            if atomic_percent[i] > 0.01:  # Threshold: 0.01%
                valid_labels.append(component_labels[i])
                valid_at_pct.append(atomic_percent[i])
                valid_maps.append(species_maps[:, :, i])
    
    if not valid_maps:
        logger.warning("No valid species to plot")
        return
    
    n_valid = len(valid_maps)
    
    # Create RGB composite map where each pixel is colored by the dominant species
    # Stack all valid maps
    quant_stack = np.stack([np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0) for q in valid_maps], axis=2)
    
    # Normalize at each pixel to show relative contributions (makes minor species visible)
    # This way a pixel with 45% C-C, 28% CO3, 2% C-O will show the species with highest
    # *relative* contribution at that location, not just highest absolute concentration
    pixel_totals = np.sum(quant_stack, axis=2, keepdims=True)  # Shape: (ny, nx, 1)
    pixel_totals = np.where(pixel_totals > 0, pixel_totals, 1.0)  # Avoid division by zero
    normalized_stack = quant_stack / pixel_totals  # Normalize to relative fractions at each pixel
    
    # Find dominant component at each pixel based on normalized concentrations
    dominant_species = np.argmax(normalized_stack, axis=2)  # Shape: (ny, nx)
    max_normalized = np.max(normalized_stack, axis=2)    # Shape: (ny, nx)
    total_concentration = np.sum(quant_stack, axis=2)  # Total signal at each pixel
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Define distinct colors for each species using high-contrast palettes
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    import seaborn as sns
    
    # Use bright, saturated colors for better contrast in RGB mixing
    # Start with primary colors and high-contrast secondaries
    base_colors = np.array([
        [1.0, 0.0, 0.0],   # Red
        [0.0, 0.8, 0.0],   # Green
        [0.0, 0.4, 1.0],   # Blue
        [1.0, 0.8, 0.0],   # Yellow
        [1.0, 0.0, 0.8],   # Magenta
        [0.0, 0.9, 0.9],   # Cyan
        [1.0, 0.5, 0.0],   # Orange
        [0.6, 0.0, 1.0],   # Purple
        [0.0, 1.0, 0.5],   # Spring green
        [1.0, 0.3, 0.5],   # Pink
    ])
    
    # Use base colors if we have <= 10 species
    if n_valid <= len(base_colors):
        species_colors = base_colors[:n_valid]
    else:
        # For more species, use tab20 (bright) or generate from colormap
        species_colors = plt.cm.tab20(np.linspace(0, 1, n_valid))[:, :3]
    
    # Create RGB image by mixing colors based on normalized concentrations
    # Use improved normalization for better contrast
    rgb_image = np.zeros((ny, nx, 3))
    
    for i in range(n_valid):
        species_map = quant_stack[:, :, i]
        # Normalize to 0-1 range using 95th percentile for better contrast
        map_nonzero = species_map[species_map > 0]
        if len(map_nonzero) > 0:
            # Use 95th percentile instead of 99th for stronger colors
            p95 = np.percentile(map_nonzero, 95)
            # Apply gamma correction for better visual contrast (gamma = 0.8 brightens mid-tones)
            normalized_map = np.clip(species_map / p95, 0, 1) if p95 > 0 else species_map
            normalized_map = np.power(normalized_map, 0.8)  # Gamma correction
        else:
            normalized_map = species_map
        
        # Add this species' contribution to RGB channels
        for c in range(3):
            rgb_image[:, :, c] += normalized_map * species_colors[i, c]
    
    # Clip RGB values to valid range [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Apply overall contrast enhancement using histogram stretching
    for c in range(3):
        channel = rgb_image[:, :, c]
        valid_pixels = channel[channel > 0]
        if len(valid_pixels) > 100:  # Only stretch if we have enough data
            p5 = np.percentile(valid_pixels, 5)
            p95 = np.percentile(valid_pixels, 95)
            if p95 > p5:
                channel_stretched = (channel - p5) / (p95 - p5)
                rgb_image[:, :, c] = np.clip(channel_stretched, 0, 1)
    
    # Set background (pixels with no signal) to white
    signal_mask = total_concentration > 0.01
    rgb_image[~signal_mask] = [1.0, 1.0, 1.0]
    
    # Extent for imshow
    if x_axis is not None and y_axis is not None:
        extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
        xlabel = 'X Position (µm)'
        ylabel = 'Y Position (µm)'
    else:
        extent = [0, nx, ny, 0]
        xlabel = 'X Pixel'
        ylabel = 'Y Pixel'
    
    # Plot the RGB composite map
    im = ax.imshow(rgb_image, aspect='auto', extent=extent, interpolation='nearest')
    
    # Create custom legend showing species with their assigned colors and %
    legend_patches = []
    for i, (label, pct) in enumerate(zip(valid_labels, valid_at_pct)):
        # Create colored rectangle for legend
        color_rgb = species_colors[i]
        patch = mpatches.Patch(color=color_rgb, label=f'{label} ({pct:.1f}%)', 
                              edgecolor='black', linewidth=1.5)
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
             fontsize=15, frameon=True, fancybox=True, shadow=True, 
             edgecolor='black', facecolor='white')
    
    ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(f'{region} Chemical Species Distribution\n(RGB Color Mixing - Mixed Colors Show Co-localization)',
                fontsize=16, fontweight='bold', pad=13)

    # Add text annotation explaining color mixing
    ax.text(0.02, 0.98, 'RGB color mixing\nMixed colors = Co-localization', 
           transform=ax.transAxes, fontsize=15, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                    edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save combined overlay map
    output_file = output_dir / f"{base_name}_{region}_combined_concentration_map.png"
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined concentration overlay map: {output_file.name}")
    except Exception as e:
        logger.exception(f"Failed to save combined concentration map: {e}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Create second figure: individual species maps in a grid
    import math
    ncols = min(3, n_valid)
    nrows = math.ceil(n_valid / ncols)
    
    # Log map value ranges for debugging
    logger.info(f"\nIndividual species map value ranges:")
    for i in range(n_valid):
        data_min = np.min(valid_maps[i])
        data_max = np.max(valid_maps[i])
        data_mean = np.mean(valid_maps[i][valid_maps[i] > 0]) if np.any(valid_maps[i] > 0) else 0
        logger.info(f"  {valid_labels[i]}: min={data_min:.4f}, max={data_max:.4f}, mean(nonzero)={data_mean:.4f}")
    
    fig2, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i in range(n_valid):
        ax = axes_flat[i]
        species_conc = valid_maps[i]
        
        # Check actual data range
        data_max = np.max(species_conc)
        data_nonzero = species_conc[species_conc > 0]
        
        # Calculate appropriate vmax for good contrast
        if len(data_nonzero) > 0:
            p99 = np.percentile(data_nonzero, 99)
            p95 = np.percentile(data_nonzero, 95)
            # Use 99th percentile but at least 2x the mean (which should be close to valid_at_pct)
            vmax = max(p99, valid_at_pct[i] * 3, p95 * 1.5)
            logger.debug(f"  {valid_labels[i]}: data range [0, {data_max:.4f}], using vmax={vmax:.2f}")
        else:
            vmax = valid_at_pct[i] * 3
        
        im = ax.imshow(species_conc, cmap='viridis', aspect='auto', extent=extent,
                      vmin=0, vmax=vmax,
                      interpolation='nearest')
        
        ax.set_title(f'{valid_labels[i]} ({valid_at_pct[i]:.1f}%)', fontsize=15, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('%', fontsize=15)
        cbar.ax.tick_params(labelsize=15)
    
    # Hide unused subplots
    for j in range(n_valid, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.suptitle(f'{region} Individual Species Concentration Maps', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save individual species grid
    output_file2 = output_dir / f"{base_name}_{region}_individual_species_maps.png"
    try:
        plt.savefig(output_file2, dpi=250, bbox_inches='tight')
        logger.info(f"Saved individual species maps: {output_file2.name}")
    except Exception as e:
        logger.exception(f"Failed to save individual species maps: {e}")
    
    if show:
        plt.show()
    else:
        plt.close(fig2)
    
    import gc
    gc.collect()
