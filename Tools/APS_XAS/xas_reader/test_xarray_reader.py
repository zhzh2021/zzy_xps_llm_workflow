"""
Test script for xarray-based APS XAS reader.

Validates that the updated reader correctly loads data from both ASCII and HDF5 
formats and returns standardized xarray.Dataset objects.
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr

# Import the updated reader
from aps_xas_reader import (
    load_aps_xas, 
    read_aps_ascii, 
    read_aps_hdf,
    get_transmission_mu,
    get_fluorescence_mu,
    get_reference_mu,
    load_aps_dataset
)


def test_ascii_reader():
    """Test reading ASCII format files."""
    print("=" * 80)
    print("TEST 1: ASCII File Reading")
    print("=" * 80)
    
    test_file = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2.2")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Load the file
        data = read_aps_ascii(test_file)
        
        # Verify it's an xarray.Dataset
        assert isinstance(data, xr.Dataset), "Reader should return xarray.Dataset"
        print("✓ Returns xarray.Dataset object")
        
        # Check required coordinates
        assert 'energy' in data.coords, "Should have energy coordinate"
        print("✓ Has energy coordinate")
        
        # Check required data variables
        required_vars = ['i0', 'i1', 'mu_trans']
        for var in required_vars:
            assert var in data, f"Should have {var} data variable"
        print(f"✓ Has required variables: {required_vars}")
        
        # Check optional data variables
        if 'i2' in data:
            print("✓ Has i2 (reference detector)")
        if 'mu_ref' in data:
            print("✓ Has mu_ref (reference absorption)")
        
        # Check fluorescence data
        fluor_vars = [v for v in data.data_vars if v.startswith('fluor_')]
        if fluor_vars:
            print(f"✓ Has fluorescence data: {len(fluor_vars)} channels")
        
        # Check attributes
        assert 'filename' in data.attrs, "Should have filename in attributes"
        assert 'beamline' in data.attrs, "Should have beamline in attributes"
        assert 'mode' in data.attrs, "Should have mode in attributes"
        assert 'n_points' in data.attrs, "Should have n_points in attributes"
        print("✓ Has required metadata attributes")
        
        # Check data shapes
        n_points = len(data['energy'])
        for var in data.data_vars:
            assert len(data[var]) == n_points, f"{var} should have same length as energy"
        print(f"✓ All variables have consistent shape: {n_points} points")
        
        # Check energy range
        energy_range = data.attrs['energy_range']
        print(f"✓ Energy range: {energy_range[0]:.2f} - {energy_range[1]:.2f} eV")
        
        # Print dataset summary
        print("\nDataset Summary:")
        print(data)
        
        print("\n✅ ASCII reader test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ASCII reader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hdf_reader():
    """Test reading HDF5 format files."""
    print("\n" + "=" * 80)
    print("TEST 2: HDF5 File Reading")
    print("=" * 80)
    
    test_file = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\JL1.1.hdf")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Load the file without calibration
        data = read_aps_hdf(test_file)
        
        # Verify it's an xarray.Dataset
        assert isinstance(data, xr.Dataset), "Reader should return xarray.Dataset"
        print("✓ Returns xarray.Dataset object")
        
        # Check required coordinates
        assert 'energy' in data.coords, "Should have energy coordinate"
        print("✓ Has energy coordinate (uncalibrated)")
        
        # Check required data variables
        required_vars = ['i0', 'i1', 'mu_trans']
        for var in required_vars:
            assert var in data, f"Should have {var} data variable"
        print(f"✓ Has required variables: {required_vars}")
        
        # Check attributes
        assert 'filename' in data.attrs, "Should have filename in attributes"
        assert 'beamline' in data.attrs, "Should have beamline in attributes"
        assert 'format' in data.attrs, "Should have format in attributes"
        assert data.attrs['format'] == 'HDF5 (raw)', "Format should indicate HDF5 raw"
        print("✓ Has required metadata attributes")
        
        # Check calibration warning
        assert 'warning' in data.attrs or 'calibrated' in data.attrs, "Should indicate calibration status"
        print(f"✓ Calibration status: {data.attrs.get('calibrated', False)}")
        
        # Print dataset summary
        print("\nDataset Summary:")
        print(data)
        
        print("\n✅ HDF5 reader test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ HDF5 reader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_helper_functions():
    """Test helper functions for extracting data."""
    print("\n" + "=" * 80)
    print("TEST 3: Helper Functions")
    print("=" * 80)
    
    test_file = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2.2")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        data = load_aps_xas(test_file)
        
        # Test get_transmission_mu
        energy, mu = get_transmission_mu(data)
        assert isinstance(energy, np.ndarray), "Energy should be numpy array"
        assert isinstance(mu, np.ndarray), "Mu should be numpy array"
        assert len(energy) == len(mu), "Energy and mu should have same length"
        print(f"✓ get_transmission_mu() works: {len(energy)} points")
        
        # Test get_reference_mu if available
        if 'mu_ref' in data:
            energy_ref, mu_ref = get_reference_mu(data)
            assert len(energy_ref) == len(mu_ref), "Energy and mu_ref should have same length"
            print(f"✓ get_reference_mu() works: {len(energy_ref)} points")
        
        # Test get_fluorescence_mu if available
        if 'fluor_total' in data:
            energy_fluor, mu_fluor = get_fluorescence_mu(data, normalize=True)
            assert len(energy_fluor) == len(mu_fluor), "Energy and fluorescence should have same length"
            print(f"✓ get_fluorescence_mu() works: {len(energy_fluor)} points")
        
        print("\n✅ Helper functions test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Helper functions test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_format_consistency():
    """Test that ASCII and HDF files produce consistent xarray structures."""
    print("\n" + "=" * 80)
    print("TEST 4: Format Consistency")
    print("=" * 80)
    
    ascii_file = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\FeCl2-Malic_acid_(0.5-0.5)-pH2.2")
    hdf_file = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5\JL1.1.hdf")
    
    if not ascii_file.exists() or not hdf_file.exists():
        print("❌ Test files not found")
        return False
    
    try:
        ascii_data = load_aps_xas(ascii_file)
        hdf_data = load_aps_xas(hdf_file)
        
        # Both should be xarray.Dataset
        assert isinstance(ascii_data, xr.Dataset), "ASCII should return Dataset"
        assert isinstance(hdf_data, xr.Dataset), "HDF should return Dataset"
        print("✓ Both return xarray.Dataset objects")
        
        # Both should have energy coordinate
        assert 'energy' in ascii_data.coords, "ASCII should have energy"
        assert 'energy' in hdf_data.coords, "HDF should have energy"
        print("✓ Both have energy coordinate")
        
        # Both should have required data variables
        required_vars = ['i0', 'i1', 'mu_trans']
        for var in required_vars:
            assert var in ascii_data, f"ASCII should have {var}"
            assert var in hdf_data, f"HDF should have {var}"
        print(f"✓ Both have required data variables: {required_vars}")
        
        # Both should have metadata attributes
        for attr in ['filename', 'beamline', 'n_points', 'energy_range']:
            assert attr in ascii_data.attrs, f"ASCII should have {attr} attribute"
            assert attr in hdf_data.attrs, f"HDF should have {attr} attribute"
        print("✓ Both have consistent metadata structure")
        
        print("\n✅ Format consistency test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Format consistency test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_loading():
    """Test loading multiple files at once."""
    print("\n" + "=" * 80)
    print("TEST 5: Batch Loading")
    print("=" * 80)
    
    data_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    try:
        # Load FeCl2 samples
        datasets = load_aps_dataset(data_dir, pattern="FeCl2*", prefer_ascii=True)
        
        assert isinstance(datasets, list), "Should return list"
        assert len(datasets) > 0, "Should load at least one file"
        print(f"✓ Loaded {len(datasets)} files")
        
        # All should be xarray.Dataset
        for i, ds in enumerate(datasets):
            assert isinstance(ds, xr.Dataset), f"Dataset {i} should be xarray.Dataset"
        print("✓ All loaded files are xarray.Dataset objects")
        
        # Show sample names
        print("\nLoaded samples:")
        for ds in datasets[:5]:  # Show first 5
            print(f"  - {ds.attrs['filename']}")
        if len(datasets) > 5:
            print(f"  ... and {len(datasets) - 5} more")
        
        print("\n✅ Batch loading test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Batch loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("APS XAS READER - XARRAY FORMAT VALIDATION")
    print("=" * 80)
    print()
    
    results = []
    
    # Run tests
    results.append(("ASCII Reader", test_ascii_reader()))
    results.append(("HDF5 Reader", test_hdf_reader()))
    results.append(("Helper Functions", test_helper_functions()))
    results.append(("Format Consistency", test_format_consistency()))
    results.append(("Batch Loading", test_batch_loading()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Reader is working correctly with xarray.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
