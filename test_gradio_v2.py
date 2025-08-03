#!/usr/bin/env python3
"""
Test script for the new Gradio interface (v2)
Validates functionality and performance
"""

import sys
import os
import importlib.util
import traceback

def test_new_gradio_interface():
    """Test the modern Gradio interface"""
    print("🧪 Testing Modern Gradio Interface...")
    
    if not os.path.exists("gradio_app.py"):
        print("❌ gradio_app.py not found!")
        return False
    
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("gradio_app", "gradio_app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Modern Gradio app imported successfully")
        
        # Test DocumentProcessor class
        if hasattr(module, 'DocumentProcessor'):
            processor = module.DocumentProcessor()
            print("✅ DocumentProcessor instantiated")
            
            # Test system status
            status = processor.get_system_status()
            print(f"✅ System status: {status['status']}")
            
            # Test processor availability
            if processor.system_ready:
                print("✅ Document processor is ready")
            else:
                print("⚠️  Document processor not ready (this is expected if dependencies are missing)")
                
        else:
            print("❌ DocumentProcessor class not found")
            return False
        
        # Test interface creation
        if hasattr(module, 'create_modern_interface'):
            demo = module.create_modern_interface()
            print("✅ Modern interface created successfully")
            
            # Check if it's a proper Gradio interface
            if hasattr(demo, 'launch'):
                print("✅ Interface has launch method")
            else:
                print("❌ Interface missing launch method")
                return False
                
        else:
            print("❌ create_modern_interface function not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing new interface: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n🧪 Testing Dependencies...")
    
    required_modules = [
        "gradio",
        "cv2", 
        "numpy",
        "PIL",
        "hybrid_document_processor",
        "hybrid_kv_extractor", 
        "llm_kv_extractor"
    ]
    
    missing = []
    for module_name in required_modules:
        try:
            if module_name == "cv2":
                import cv2
            elif module_name == "PIL":
                from PIL import Image
            else:
                __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError:
            print(f"❌ {module_name}")
            missing.append(module_name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies available")
        return True

def test_performance():
    """Test basic performance metrics"""
    print("\n🧪 Testing Performance...")
    
    try:
        import time
        
        # Test import time
        start_time = time.time()
        spec = importlib.util.spec_from_file_location("gradio_app", "gradio_app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import_time = time.time() - start_time
        
        print(f"✅ Import time: {import_time:.2f} seconds")
        
        # Test interface creation time
        start_time = time.time()
        demo = module.create_modern_interface()
        creation_time = time.time() - start_time
        
        print(f"✅ Interface creation time: {creation_time:.2f} seconds")
        
        # Performance assessment
        if import_time < 5.0 and creation_time < 3.0:
            print("✅ Performance: Good")
            return True
        elif import_time < 10.0 and creation_time < 5.0:
            print("⚠️  Performance: Acceptable")
            return True
        else:
            print("❌ Performance: Slow")
            return False
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def verify_interface():
    """Verify the modern interface is available and working"""
    print("\n🧪 Verifying Interface...")
    
    interface_exists = os.path.exists("gradio_app.py")
    
    print(f"Modern interface (gradio_app.py): {'✅' if interface_exists else '❌'}")
    
    if interface_exists:
        # Get file size
        interface_size = os.path.getsize("gradio_app.py")
        print(f"Interface size: {interface_size} bytes")
        
        # Check if it's executable
        import stat
        file_stat = os.stat("gradio_app.py")
        is_executable = bool(file_stat.st_mode & stat.S_IEXEC)
        print(f"Executable: {'✅' if is_executable else '❌'}")
        
        print("✅ Modern interface is ready to use")
        return True
    else:
        print("❌ No Gradio interface available")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 MODERN GRADIO INTERFACE TESTING")
    print("=" * 60)
    
    all_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_passed = False
    
    # Test new interface
    if not test_new_gradio_interface():
        all_passed = False
    
    # Test performance  
    if not test_performance():
        all_passed = False
    
    # Verify interface
    if not verify_interface():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Modern Gradio interface is ready to use.")
        print("\n🚀 To launch the interface:")
        print("   ./deploy.sh gradio")
        print("   or")
        print("   python gradio_app.py")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        print("\n🔧 Common solutions:")
        print("   • Install missing dependencies: pip install -r requirements.txt")
        print("   • Check hybrid document processor setup")
        print("   • Verify system configuration")
    print("=" * 60)

if __name__ == "__main__":
    main()