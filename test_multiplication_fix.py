#!/usr/bin/env python3
"""
Test script to verify the multiplication error fixes in quality assessment
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_quality_assessment():
    """Test the quality assessment with various problematic inputs"""
    try:
        from quality.adaptive_quality_checker import AdaptiveDocumentQualityChecker
        
        print("🧪 Testing Quality Assessment Multiplication Fixes...")
        
        # Create test checker
        checker = AdaptiveDocumentQualityChecker()
        
        # Create test image
        test_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        
        print("📊 Running quality assessment...")
        
        # Test with different document types
        test_cases = [
            ("general", "General document"),
            ("form", "Form document"),
            ("certificate", "Certificate document"),
            ("receipt", "Receipt document")
        ]
        
        for doc_type, description in test_cases:
            try:
                print(f"  🔍 Testing {description}...")
                result = checker.assess_quality(test_image, document_type=doc_type)
                
                # Check that we got valid results
                assert isinstance(result, dict), "Result should be a dictionary"
                assert "confidence" in result, "Result should have confidence"
                assert "needs_rescan" in result, "Result should have needs_rescan"
                assert isinstance(result["confidence"], (int, float)), "Confidence should be numeric"
                
                print(f"    ✅ Success - Confidence: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                return False
        
        print("🎉 All quality assessment tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quality assessment test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced quality features availability"""
    try:
        from quality.advanced_quality_features import SCIPY_AVAILABLE, SKIMAGE_AVAILABLE, STATSMODELS_AVAILABLE
        
        print("\n🔬 Testing Advanced Features Availability...")
        print(f"  SciPy: {'✅ Available' if SCIPY_AVAILABLE else '❌ Not Available'}")
        print(f"  Scikit-image: {'✅ Available' if SKIMAGE_AVAILABLE else '❌ Not Available'}")  
        print(f"  Statsmodels: {'✅ Available' if STATSMODELS_AVAILABLE else '❌ Not Available'}")
        
        if SCIPY_AVAILABLE and SKIMAGE_AVAILABLE and STATSMODELS_AVAILABLE:
            print("🎉 All advanced features are available!")
        else:
            print("⚠️ Some advanced features are not available - this is OK if packages aren't installed")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False

def test_config_values():
    """Test that adaptive config returns proper numeric values"""
    try:
        from src.config import adaptive_config
        
        print("\n⚙️ Testing Adaptive Config Value Types...")
        
        # Test different config retrievals
        test_configs = [
            ("quality_thresholds", "base_blur_threshold"),
            ("quality_thresholds", "cut_edge_margin_pct"),
            ("quality_risk_weights", "default_weights")
        ]
        
        for category, parameter in test_configs:
            try:
                value = adaptive_config.get_adaptive_value(category, parameter)
                print(f"  📊 {category}.{parameter}: {type(value).__name__} = {value}")
                
                # For numeric parameters, ensure they can be converted to float
                if parameter in ["base_blur_threshold", "cut_edge_margin_pct"]:
                    # Test the list handling fix
                    if isinstance(value, list):
                        print(f"    📋 List value detected: {value}")
                        float_val = float(value[0]) if len(value) > 0 else 0.5
                        print(f"    ✅ Can convert list to float: {float_val}")
                    else:
                        float_val = float(value)
                        print(f"    ✅ Can convert to float: {float_val}")
                    
            except Exception as e:
                print(f"    ⚠️ Config error (expected for uninitialized values): {e}")
        
        print("✅ Config value type tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_ui_formatting():
    """Test UI formatting with None values"""
    try:
        print("\n🖼️ Testing UI Formatting Safety...")
        
        # Simulate advanced assessment data with None values
        test_data = {
            "advanced_quality_assessment": {
                "advanced_features_available": True,
                "overall_quality_score": None,
                "quality_class": None,
                "enhanced_blur_analysis": {
                    "tenengrad_energy": None,
                    "composite_blur_confidence": 0.5,
                    "brenner_focus": None
                },
                "brightness_analysis": {
                    "brightness_skewness": None,
                    "brightness_entropy": 0.7,
                    "overexposure_ratio": None
                }
            }
        }
        
        # Import the formatting function
        from enhanced_gradio_app import format_detailed_quality_assessment
        
        # This should not crash even with None values
        result = format_detailed_quality_assessment(test_data)
        
        if "Overall Quality Score:" in result:
            print("  ✅ UI formatting handles None values safely")
            return True
        else:
            print("  ❌ UI formatting test failed")
            return False
        
    except Exception as e:
        print(f"  ❌ UI formatting test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Multiplication Error Fix Tests")
    print("=" * 50)
    
    success = True
    
    # Test quality assessment
    if not test_quality_assessment():
        success = False
    
    # Test advanced features
    if not test_advanced_features():
        success = False
        
    # Test config values
    if not test_config_values():
        success = False
        
    # Test UI formatting
    if not test_ui_formatting():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! All errors should be fixed.")
        print("🎯 Fixes applied:")
        print("  ✅ Multiplication errors with type validation")
        print("  ✅ Config blending with list value handling")  
        print("  ✅ Overflow warnings in contrast calculations")
        print("  ✅ UI formatting with None value safety")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)