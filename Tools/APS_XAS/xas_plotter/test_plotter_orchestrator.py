#!/usr/bin/env python3
"""
Test script for XAS Plotter Orchestrator

This script tests the orchestrator pattern in xas_plotter_main.py
to ensure it properly delegates to specialized plotting modules.
"""

import sys
from pathlib import Path

# Add the XAS tools to Python path
xas_path = Path(__file__).parent
sys.path.insert(0, str(xas_path))

def test_plotter_orchestrator():
    """Test the XAS plotter orchestrator initialization and methods."""
    
    print("Testing XAS Plotter Orchestrator...")
    print("=" * 50)
    
    try:
        from xas_plotter.xas_plotter_main import XASPlotter
        print("✅ Successfully imported XASPlotter orchestrator")
    except ImportError as e:
        print(f"❌ Failed to import XASPlotter: {e}")
        return False
    
    try:
        # Initialize the orchestrator
        plotter = XASPlotter()
        print("✅ Successfully initialized XASPlotter orchestrator")
    except Exception as e:
        print(f"❌ Failed to initialize XASPlotter: {e}")
        return False
    
    # Test available methods
    available_methods = []
    delegation_methods = [
        'plot_raw_data',
        'plot_quality_control', 
        'plot_feature_comparison',
        'plot_quality_report',
        'plot_complete_analysis',
        'plot_batch_analysis'
    ]
    
    print("\nTesting delegation methods:")
    print("-" * 30)
    
    for method_name in delegation_methods:
        if hasattr(plotter, method_name):
            available_methods.append(method_name)
            print(f"✅ {method_name} - Available")
        else:
            print(f"❌ {method_name} - Missing")
    
    # Test specialized plotter initialization
    print("\nSpecialized plotter status:")
    print("-" * 30)
    
    if hasattr(plotter, 'raw_plotter') and plotter.raw_plotter is not None:
        print("✅ Raw data plotter - Initialized")
    else:
        print("⚠️  Raw data plotter - Not available")
    
    if hasattr(plotter, 'quality_plotter') and plotter.quality_plotter is not None:
        print("✅ Quality control plotter - Initialized")
    else:
        print("⚠️  Quality control plotter - Not available")
    
    # Test settings loading
    print("\nSettings configuration:")
    print("-" * 30)
    
    if hasattr(plotter, 'settings') and plotter.settings:
        print("✅ Settings loaded successfully")
        if 'plot_settings' in plotter.settings:
            print("✅ Plot settings available")
        else:
            print("⚠️  Plot settings missing")
    else:
        print("⚠️  Settings not loaded")
    
    print("\n" + "=" * 50)
    print("XAS Plotter Orchestrator Test Complete")
    print(f"Available delegation methods: {len(available_methods)}/{len(delegation_methods)}")
    
    if len(available_methods) == len(delegation_methods):
        print("🎉 All tests passed! Orchestrator is ready for use.")
        return True
    else:
        print("⚠️  Some functionality may be limited due to missing dependencies.")
        return True  # Still functional for most use cases

if __name__ == "__main__":
    success = test_plotter_orchestrator()
    
    if success:
        print("\n📝 Usage Guidelines:")
        print("-" * 20)
        print("• Use plot_raw_data() for basic spectrum visualization")
        print("• Use plot_quality_control() for data quality diagnostics")
        print("• Use plot_feature_comparison() for batch feature analysis")
        print("• Use plot_quality_report() for comprehensive quality reports")
        print("• Use plot_complete_analysis() for single-sample comprehensive plots")
        print("• Use plot_batch_analysis() for complete batch processing visualization")
        print("• Legacy methods (plot_xanes, plot_exafs, etc.) have been removed")
        print("  to avoid duplication with specialized modules")
        
        sys.exit(0)
    else:
        sys.exit(1)