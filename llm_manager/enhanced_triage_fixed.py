#!/usr/bin/env python3
"""Enhanced XPS data triage system with CSV map detection and depth profile support."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
from enum import Enum


class XPSDataType(Enum):
    STANDARD_SPECTRA = "standard_spectra"
    MAP_2D = "map_2d"
    MAP_HYPERSPECTRAL = "map_hyperspectral"
    DEPTH_PROFILE = "depth_profile"
    UNKNOWN = "unknown"


class EnhancedXPSDataTriage:
    """Analyze files to classify as spectra, maps, or depth profiles."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.depth_profile_threshold = 10

    def analyze_file_structure(self, file_path: Path) -> Dict[str, Any]:
        try:
            s = str(file_path).lower()
            if s.endswith('.csv'):
                depth = self._analyze_depth_profile_pattern(file_path)
                if depth['confidence'] > 0.6:
                    return self._create_result(depth['data_type'], depth['confidence'], depth['parameters'], depth['reason'])
                cmap = self._analyze_csv_map_pattern(file_path)
                if cmap['confidence'] > 0.6:
                    return self._create_result(cmap['data_type'], cmap['confidence'], cmap['parameters'], cmap['reason'])
                standard = self._analyze_standard_csv_format(file_path)
                if standard['confidence'] > 0.4:
                    return self._create_result(standard['data_type'], standard['confidence'], standard['parameters'], standard['reason'])

            if s.endswith(('.spe', '.vgd', '.npl', '.pro')):
                return self._analyze_binary_format(file_path)

            if s.endswith(('.xy', '.txt', '.asc', '.dat')):
                return self._analyze_text_format(file_path)

            if s.endswith(('.vms', '.vamas')):
                return self._analyze_vamas_format(file_path)

            return self._create_result(XPSDataType.UNKNOWN, 0.0, {}, f"Unsupported extension {file_path.suffix}")
        except Exception as e:
            if self.debug:
                print("analyze_file_structure error:", e)
            return self._create_result(XPSDataType.UNKNOWN, 0.0, {}, f"Analysis failed: {e}")

    def _analyze_depth_profile_pattern(self, file_path: Path) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                raw = [ln.rstrip('\n') for ln in fh.readlines()[:400]]
            lines = [ln for ln in raw if ln.strip()]
            if len(lines) < 8:
                return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'File too short'}

            confidence = 0.0
            params: Dict[str, Any] = {}

            region = next((ln for ln in lines[:6] if self._is_xps_region(ln)), None)
            if region:
                confidence += 0.2
                params['region'] = region.strip()

            header_text = ' '.join(lines[:10]).lower()
            if any(k in header_text for k in ['sputter', 'cycle', 'layer', 'depth', 'etch', 'profile']):
                confidence += 0.2
                params['depth_profile_keywords'] = True

            start = self._find_data_start(lines)
            if start < 0:
                return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'No numeric CSV block found'}

            data_lines = lines[start:start + 300]
            if len(data_lines) < 5:
                return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'Insufficient numeric lines'}

            first = [p.strip() for p in data_lines[0].split(',') if p.strip()]
            try:
                _ = [float(x) for x in first]
            except Exception:
                return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'First data line not numeric'}

            ncols = len(first)
            if ncols < 2:
                return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'Too few columns'}

            ncycles = ncols - 1
            params['num_columns'] = ncols
            params['num_cycles'] = ncycles

            energy_vals = []
            for ln in data_lines[:min(60, len(data_lines))]:
                parts = [p.strip() for p in ln.split(',')]
                try:
                    energy_vals.append(float(parts[0]))
                except Exception:
                    break

            if len(energy_vals) >= 5:
                diffs = [b - a for a, b in zip(energy_vals[:-1], energy_vals[1:])]
                if diffs:
                    pos = sum(1 for d in diffs if d > 0)
                    neg = sum(1 for d in diffs if d < 0)
                    if max(pos, neg) >= 0.7 * len(diffs):
                        confidence += 0.3
                        params['energy_axis_detected'] = True
                        params['energy_points'] = len(energy_vals)
                        params['energy_range'] = (min(energy_vals), max(energy_vals))

            counts = []
            for ln in data_lines[:min(120, len(data_lines))]:
                parts = [p.strip() for p in ln.split(',') if p.strip()]
                counts.append(len(parts))
            if counts and min(counts) == max(counts) == ncols:
                confidence += 0.2
                params['consistent_structure'] = True

            spatial = self._extract_spatial_parameters(lines[:12])
            if spatial:
                confidence = max(0.0, confidence - 0.5)
                params['spatial_detected'] = True
            else:
                confidence += 0.1

            if confidence > 0.6:
                if ncycles > self.depth_profile_threshold:
                    params['recommended_workflow'] = 'pca_mcr_map'
                    reason = f"Depth profile detected ({ncycles} cycles) -> recommend PCA/MCR mapper"
                else:
                    params['recommended_workflow'] = 'standard'
                    reason = f"Depth profile detected ({ncycles} cycles) -> standard reader"

                return {'confidence': min(confidence, 0.95), 'data_type': XPSDataType.DEPTH_PROFILE, 'parameters': params, 'reason': reason}

            return {'confidence': confidence, 'data_type': XPSDataType.UNKNOWN, 'parameters': params, 'reason': 'No clear depth profile pattern'}

        except Exception as e:
            if self.debug:
                print('depth analysis error', e)
            return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': str(e)}

    def _analyze_standard_csv_format(self, file_path: Path) -> Dict[str, Any]:
        """Detect standard numeric CSV spectra (non-map, non-depth)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                raw = [ln.rstrip('\n') for ln in fh.readlines()[:200]]
            lines = [ln for ln in raw if ln.strip()]
            if len(lines) < 5:
                return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'File too short'}

            numeric_lines = 0
            numeric_cols = []
            for ln in lines[:80]:
                if ',' not in ln and '\t' not in ln:
                    continue
                parts = [p.strip() for p in ln.replace('\t', ',').split(',') if p.strip()]
                if len(parts) < 2:
                    continue
                try:
                    [float(p) for p in parts[:2]]
                except Exception:
                    continue
                numeric_lines += 1
                numeric_cols.append(len(parts))

            if numeric_lines >= 8:
                params = {
                    'data_lines': numeric_lines,
                    'num_columns': int(min(numeric_cols)) if numeric_cols else 0
                }
                return {
                    'confidence': 0.7,
                    'data_type': XPSDataType.STANDARD_SPECTRA,
                    'parameters': params,
                    'reason': 'Standard numeric CSV detected'
                }

            return {'confidence': 0.2, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'CSV format unclear'}
        except Exception as e:
            if self.debug:
                print('standard csv analysis error', e)
            return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': str(e)}

    def _find_data_start(self, lines: List[str]) -> int:
        for i, ln in enumerate(lines):
            if ',' not in ln:
                continue
            parts = [p.strip() for p in ln.split(',') if p.strip()]
            if len(parts) < 2:
                continue
            try:
                [float(p) for p in parts]
                return i
            except Exception:
                continue
        return -1

    def _analyze_csv_map_pattern(self, file_path: Path) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                lines = [ln.rstrip('\n') for ln in fh.readlines()[:200] if ln.strip()]

            if len(lines) < 8:
                return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': 'File too short'}

            confidence = 0.0
            params: Dict[str, Any] = {}

            region_line = next((ln for ln in lines[:6] if self._is_xps_region(ln)), None)
            if region_line:
                confidence += 0.25
                params['region'] = region_line.strip()

            spatial = self._extract_spatial_parameters(lines[:12])
            if spatial:
                confidence += 0.4
                params.update(spatial)
                if params.get('nx') and params.get('ny'):
                    params['expected_spectra'] = params['nx'] * params['ny']

            energy_idx = self._find_energy_axis_line(lines)
            if energy_idx >= 0:
                confidence += 0.3
                params['energy_line_idx'] = energy_idx
                params['energy_points'] = len([p for p in lines[energy_idx].split(',') if p.strip()])

            if params.get('expected_spectra') and energy_idx >= 0:
                data_lines = len(lines) - energy_idx - 1
                if abs(data_lines - params['expected_spectra']) <= max(1, 0.1 * params['expected_spectra']):
                    confidence += 0.1

            if confidence > 0.7:
                return {'confidence': min(confidence, 0.95), 'data_type': XPSDataType.MAP_HYPERSPECTRAL, 'parameters': params, 'reason': 'CSV hyperspectral map detected'}
            if confidence > 0.4:
                return {'confidence': min(confidence, 0.95), 'data_type': XPSDataType.MAP_2D, 'parameters': params, 'reason': 'CSV map pattern with spatial params'}

            return {'confidence': confidence, 'data_type': XPSDataType.UNKNOWN, 'parameters': params, 'reason': 'No clear CSV map pattern'}

        except Exception as e:
            if self.debug:
                print('map analysis error', e)
            return {'confidence': 0.0, 'data_type': XPSDataType.UNKNOWN, 'parameters': {}, 'reason': str(e)}

    def _is_xps_region(self, line: str) -> bool:
        common = ['C1s', 'C2s', 'O1s', 'N1s', 'F1s', 'Si2p', 'P2p', 'S2p', 'survey', 'wide']
        up = line.upper()
        return any(tok.upper() in up for tok in common)

    def _extract_spatial_parameters(self, lines: List[str]) -> Dict[str, Any]:
        nums = []
        for ln in lines:
            try:
                nums.append(float(ln))
            except Exception:
                continue
            if len(nums) >= 6:
                break
        if len(nums) < 6:
            return {}
        x_start, x_step, nx, y_start, y_step, ny = nums[:6]
        if not (1 <= nx <= 5000 and nx == int(nx)):
            return {}
        if not (1 <= ny <= 5000 and ny == int(ny)):
            return {}
        return {'x_start': x_start, 'x_step': x_step, 'nx': int(nx), 'y_start': y_start, 'y_step': y_step, 'ny': int(ny), 'total_pixels': int(nx * ny)}

    def _find_energy_axis_line(self, lines: List[str]) -> int:
        for i, ln in enumerate(lines):
            if ',' not in ln:
                continue
            parts = [p.strip() for p in ln.split(',') if p.strip()]
            if len(parts) < 6:
                continue
            try:
                vals = [float(p) for p in parts]
            except Exception:
                continue
            if all(-50 <= v <= 5000 for v in vals):
                diffs = [b - a for a, b in zip(vals[:-1], vals[1:])]
                if len(diffs) and sum(1 for d in diffs if d >= 0) >= 0.8 * len(diffs):
                    return i
        return -1

    def _analyze_binary_format(self, file_path: Path) -> Dict[str, Any]:
        try:
            size = file_path.stat().st_size
            if size < 100:
                return self._create_result(XPSDataType.UNKNOWN, 0.0, {}, f"File too small ({size} bytes)")
            return self._create_result(XPSDataType.STANDARD_SPECTRA, 0.9, {'file_format': file_path.suffix.lstrip('.')}, 'Binary XPS format')
        except Exception as e:
            return self._create_result(XPSDataType.UNKNOWN, 0.0, {}, str(e))

    def _analyze_text_format(self, file_path: Path) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                lines = [ln.strip() for ln in fh.readlines()[:120] if ln.strip()]
            numeric = 0
            for ln in lines[:60]:
                parts = ln.split()
                if len(parts) >= 2:
                    try:
                        float(parts[0]); float(parts[1])
                        numeric += 1
                    except Exception:
                        pass
            if numeric >= 10:
                return self._create_result(XPSDataType.STANDARD_SPECTRA, 0.85, {'data_lines': numeric}, 'Text numeric data')
            return self._create_result(XPSDataType.UNKNOWN, 0.3, {}, 'Text format unclear')
        except Exception as e:
            return self._create_result(XPSDataType.UNKNOWN, 0.0, {}, str(e))

    def _analyze_vamas_format(self, file_path: Path) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                first = fh.readline()
            if 'VAMAS' in first.upper() or first.startswith('ISO'):
                return self._create_result(XPSDataType.STANDARD_SPECTRA, 0.9, {'file_format': 'vamas'}, 'VAMAS')
            return self._create_result(XPSDataType.STANDARD_SPECTRA, 0.7, {'file_format': 'vamas'}, 'VAMAS ext')
        except Exception:
            return self._create_result(XPSDataType.UNKNOWN, 0.0, {}, 'VAMAS parse error')

    def _create_result(self, data_type: XPSDataType, confidence: float, parameters: Dict, reason: str) -> Dict[str, Any]:
        if data_type in (XPSDataType.MAP_2D, XPSDataType.MAP_HYPERSPECTRAL):
            processor = 'XPS_mapper'
        elif data_type == XPSDataType.DEPTH_PROFILE:
            num_cycles = parameters.get('num_cycles', 0)
            if num_cycles > self.depth_profile_threshold:
                processor = 'XPS_mapper'
                parameters['use_pca_mcr'] = True
            else:
                processor = 'XPS_reader'
                parameters['use_pca_mcr'] = False
        elif data_type == XPSDataType.STANDARD_SPECTRA:
            processor = 'XPS_reader'
        else:
            processor = 'manual_inspection'

        return {'data_type': data_type, 'confidence': confidence, 'parameters': parameters, 'reason': reason, 'recommended_processor': processor}


def should_route_to_mapper(file_path: Path, confidence_threshold: float = 0.7) -> Tuple[bool, Dict[str, Any]]:
    triage = EnhancedXPSDataTriage(debug=False)
    result = triage.analyze_file_structure(file_path)

    is_map = result['data_type'] in (XPSDataType.MAP_2D, XPSDataType.MAP_HYPERSPECTRAL)
    is_depth_for_pca = result['data_type'] == XPSDataType.DEPTH_PROFILE and result['parameters'].get('use_pca_mcr', False)
    has_conf = result.get('confidence', 0.0) >= confidence_threshold

    should_route = (is_map or is_depth_for_pca) and has_conf
    return should_route, result


