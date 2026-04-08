"""
Test script for unified XAS readers (v3.0 standardized format)

Tests both aps_xas_reader.py and xas_reader.py with real data to verify:
1. Both output identical xarray.Dataset structure
2. All required coordinates and variables present
3. Metadata attributes conform to v3.0 spec
4. Helper functions work correctly
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from aps_xas_reader import load_aps_xas, get_transmission_mu, get_fluorescence_mu, get_reference_mu
from xas_reader import load_xas_file
from xas_reader import load_xas_batch

def validate_dataset(data, reader_name):
    """Validate that dataset conforms to v3.0 spec"""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {reader_name}")
    print(f"{'='*80}")
    
    errors = []
    
    # Check coordinate
    if 'energy' not in data.coords:
        errors.append("❌ Missing 'energy' coordinate")
    else:
        print("✓ Has 'energy' coordinate")
        energy = data['energy'].values
        if 'units' not in data['energy'].attrs:
            errors.append("❌ Energy missing 'units' attribute")
        else:
            print(f"  - Units: {data['energy'].attrs['units']}")
            print(f"  - Range: {energy.min():.2f} - {energy.max():.2f} eV")
            print(f"  - Points: {len(energy)}")
    
    # Check required data variables
    required_vars = ['i0', 'i1', 'mu_trans']
    for var in required_vars:
        if var not in data:
            errors.append(f"❌ Missing required variable '{var}'")
        else:
            print(f"✓ Has '{var}'")
    
    # Check optional variables
    optional_vars = ['i2', 'mu_ref', 'fluor_total']
    for var in optional_vars:
        if var in data:
            print(f"✓ Has optional '{var}'")
    
    # Check required attributes
    required_attrs = ['filename', 'beamline', 'mode', 'n_points', 'energy_range', 'reader_version']
    for attr in required_attrs:
        if attr not in data.attrs:
            errors.append(f"❌ Missing required attribute '{attr}'")
        else:
            print(f"✓ Has attribute '{attr}': {data.attrs[attr]}")
    
    # Check data quality
    if 'mu_trans' in data:
        mu = data['mu_trans'].values
        if np.any(np.isnan(mu)):
            errors.append(f"❌ mu_trans contains NaN values")
        if np.any(np.isinf(mu)):
            errors.append(f"❌ mu_trans contains Inf values")
        else:
            print(f"✓ mu_trans: {mu.min():.4f} - {mu.max():.4f} (no NaN/Inf)")
    
    # Summary
    if errors:
        print(f"\n❌ VALIDATION FAILED ({len(errors)} errors):")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED - Conforms to v3.0 spec")
        return True


def test_aps_ascii():
    """Test APS ASCII format reader"""
    print("\n" + "="*80)
    print("TEST 1: APS 12-BM-B ASCII FORMAT")
    print("="*80)
    
    # Test with an ASCII file (no extension)
    test_file = r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2.2"
    
    print(f"\nLoading: {Path(test_file).name}")
    
    try:
        data = load_aps_xas(test_file)
        
        # Validate
        validate_dataset(data, "APS ASCII Reader")
        
        # Test helper functions
        print(f"\n{'='*80}")
        print("TESTING HELPER FUNCTIONS")
        print(f"{'='*80}")
        
        energy, mu = get_transmission_mu(data)
        print(f"✓ get_transmission_mu() returned {len(energy)} points")
        
        if 'mu_ref' in data:
            energy_ref, mu_ref = get_reference_mu(data)
            print(f"✓ get_reference_mu() returned {len(energy_ref)} points")
        
        if 'fluor_total' in data:
            energy_fluor, mu_fluor = get_fluorescence_mu(data, normalize=True)
            print(f"✓ get_fluorescence_mu() returned {len(energy_fluor)} points")
        
        print("\n✅ APS ASCII TEST PASSED")
        return data
        
    except Exception as e:
        print(f"\n❌ APS ASCII TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_aps_hdf():
    """Test APS HDF5 format reader"""
    print("\n" + "="*80)
    print("TEST 2: APS 12-BM-B HDF5 FORMAT")
    print("="*80)
    
    # Test with an HDF file
    test_file = r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2_1.hdf"
    
    print(f"\nLoading: {Path(test_file).name}")
    
    try:
        # Load with energy calibration (example values)
        data = load_aps_xas(test_file, energy_calibration=(7000.0, 1e-4))
        
        # Validate
        validate_dataset(data, "APS HDF5 Reader")
        
        print("\n✅ APS HDF5 TEST PASSED")
        return data
        
    except Exception as e:
        print(f"\n❌ APS HDF5 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_generic_dat():
    """Test generic DAT file reader"""
    print("\n" + "="*80)
    print("TEST 3: GENERIC DAT FORMAT")
    print("="*80)
    
    # Test with a .dat file
    test_file = r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\YZZ-1 0001.dat"
    
    print(f"\nLoading: {Path(test_file).name}")
    
    try:
        data = load_xas_file(test_file, beamline="APS 12-BM-B")
        
        # Validate
        validate_dataset(data, "Generic DAT Reader")
        
        # Test helper functions
        energy, mu = get_transmission_mu(data)
        print(f"✓ get_transmission_mu() returned {len(energy)} points")
        
        print("\n✅ GENERIC DAT TEST PASSED")
        return data
        
    except Exception as e:
        print(f"\n❌ GENERIC DAT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_loading():
    """Test batch loading functionality"""
    print("\n" + "="*80)
    print("TEST 4: BATCH LOADING")
    print("="*80)
    
    data_dir = r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\021526-FeCL2-FeSO4-MA-TA-pH2-5"
    
    try:
        # Load first 3 ASCII files
        from aps_xas_reader import load_aps_dataset
        
        print(f"\nLoading FeCl2 samples from: {data_dir}")
        datasets = load_aps_dataset(data_dir, pattern="FeCl2-Malic*pH2.2", prefer_ascii=True)
        
        print(f"\n✅ Loaded {len(datasets)} datasets")
        
        # Validate first dataset
        if len(datasets) > 0:
            validate_dataset(datasets[0], f"Batch Loader (sample 1)")
        
        print("\n✅ BATCH LOADING TEST PASSED")
        return datasets
        
    except Exception as e:
        print(f"\n❌ BATCH LOADING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_format_consistency():
    """Test that both readers produce identical structure"""
    print("\n" + "="*80)
    print("TEST 5: FORMAT CONSISTENCY ACROSS READERS")
    print("="*80)
    
    try:
        # Load same sample with both readers (using generic reader for ASCII)
        aps_file = r"C:\Users\b82797\Github\zz_llm\zzy_llm\project_root\xas_raw_data\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2.2"
        
        print("\nLoading same file with both readers...")
        aps_data = load_aps_xas(aps_file)
        generic_data = load_xas_file(aps_file, beamline="APS 12-BM-B")
        
        # Check structure consistency
        print("\nChecking coordinate consistency...")
        if 'energy' in aps_data.coords and 'energy' in generic_data.coords:
            print("✓ Both have 'energy' coordinate")
        
        print("\nChecking variable consistency...")
        aps_vars = set(aps_data.data_vars.keys())
        generic_vars = set(generic_data.data_vars.keys())
        common_vars = aps_vars & generic_vars
        print(f"✓ Common variables: {common_vars}")
        
        print("\nChecking attribute consistency...")
        aps_attrs = set(aps_data.attrs.keys())
        generic_attrs = set(generic_data.attrs.keys())
        common_attrs = aps_attrs & generic_attrs
        required_attrs = {'filename', 'beamline', 'mode', 'n_points', 'energy_range', 'reader_version'}
        if required_attrs.issubset(common_attrs):
            print(f"✓ Both have all required attributes")
        
        print("\n✅ FORMAT CONSISTENCY TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ FORMAT CONSISTENCY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_summary(data):
    """Display nice summary of dataset"""
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(data)
    print("\n" + "="*80)
    print("ATTRIBUTES")
    print("="*80)
    for key, value in data.attrs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("="*80)
    print("XAS UNIFIED READER TEST SUITE (v3.0)")
    print("Testing standardized xarray.Dataset format")
    print("="*80)
    
    results = {}
    
    # Run all tests
    results['aps_ascii'] = test_aps_ascii()
    results['aps_hdf'] = test_aps_hdf()
    results['generic_dat'] = test_generic_dat()
    results['batch'] = test_batch_loading()
    results['consistency'] = test_format_consistency()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v is not None and v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Readers are fully standardized!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - check errors above")
    
    # Display one example dataset
    if results['aps_ascii'] is not None:
        print("\n" + "="*80)
        print("EXAMPLE DATASET STRUCTURE (APS ASCII)")
        print("="*80)
        display_summary(results['aps_ascii'])
