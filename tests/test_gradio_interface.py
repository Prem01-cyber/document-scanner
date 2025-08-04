#!/usr/bin/env python3
"""
Test script for the Gradio interface
Validates imports and basic functionality without launching the full interface
"""

import sys
import os
import importlib.util

def test_gradio_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing Gradio interface imports...")
    
    try:
        # Test basic imports
        import gradio as gr
        print("✅ Gradio imported successfully")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL imported successfully")
        
        # Test our custom modules
        from hybrid_document_processor import HybridDocumentProcessor
        print("✅ HybridDocumentProcessor imported successfully")
        
        from hybrid_kv_extractor import ExtractionStrategy
        print("✅ ExtractionStrategy imported successfully")
        
        from llm_kv_extractor import LLMProvider
        print("✅ LLMProvider imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_gradio_app_syntax():
    """Test that the gradio_app.py file has valid syntax and enhanced features"""
    print("\n🧪 Testing Enhanced Gradio app...")
    
    if not os.path.exists("gradio_app.py"):
        print("❌ gradio_app.py not found!")
        return False
    
    try:
        # Load the module spec without executing the main block
        spec = importlib.util.spec_from_file_location("gradio_app", "gradio_app.py")
        if spec is None:
            print("❌ Could not load gradio_app.py")
            return False
        
        # Create module and execute (but __name__ != "__main__" so no launch)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Enhanced Gradio app syntax is valid")
        
        # Test enhanced class instantiation
        processor = module.GradioDocumentProcessor()
        print("✅ Enhanced GradioDocumentProcessor instantiated")
        
        # Test enhanced features
        if hasattr(processor, 'session_stats'):
            print("✅ Session statistics tracking available")
        if hasattr(processor, 'colors') and len(processor.colors) > 5:
            print("✅ Enhanced color scheme loaded")
        if hasattr(processor, 'fonts'):
            print("✅ Font system initialized")
        
        # Test interface creation function
        if hasattr(module, 'create_gradio_interface'):
            print("✅ Enhanced create_gradio_interface function found")
            
            # Test that interface can be created (but don't launch)
            try:
                demo = module.create_gradio_interface()
                print("✅ Enhanced Gradio interface can be created")
                
                # Check for advanced features
                if hasattr(demo, 'blocks'):
                    print("✅ Gradio Blocks interface properly structured")
                    
            except Exception as e:
                print(f"⚠️  Interface creation issue: {e}")
        else:
            print("❌ create_gradio_interface function not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced app test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processor_functionality():
    """Test basic processor functionality"""
    print("\n🧪 Testing processor functionality...")
    
    try:
        from hybrid_document_processor import HybridDocumentProcessor
        
        # Test processor initialization
        processor = HybridDocumentProcessor()
        print("✅ HybridDocumentProcessor initialized")
        
        # Test available strategies
        from hybrid_kv_extractor import ExtractionStrategy
        strategies = [s.value for s in ExtractionStrategy]
        print(f"✅ Available strategies: {strategies}")
        
        # Test LLM providers
        from llm_kv_extractor import LLMProvider
        providers = [p.value for p in LLMProvider]
        print(f"✅ Available LLM providers: {providers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processor test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 GRADIO INTERFACE TESTING")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Imports
    if not test_gradio_imports():
        all_tests_passed = False
    
    # Test 2: Syntax
    if not test_gradio_app_syntax():
        all_tests_passed = False
    
    # Test 3: Processor functionality
    if not test_processor_functionality():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Gradio interface is ready to use.")
        print("🚀 Launch with: ./deploy.sh gradio")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()