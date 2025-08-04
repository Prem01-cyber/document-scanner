#!/usr/bin/env python3
"""
Quick Setup Script for Hybrid Quality Assessment System

This script helps set up the ML quality assessment system quickly.
"""

import os
import sys
import subprocess
import logging

# Add project root to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'sklearn', 'joblib', 'numpy', 'pandas', 'cv2', 'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            missing.append(package)
            logger.warning(f"❌ {package} is not available")
    
    return missing

def install_dependencies():
    """Install missing dependencies"""
    logger.info("Installing ML dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "scikit-learn==1.3.2", "joblib==1.3.2", 
            "matplotlib==3.8.2", "seaborn==0.13.0"
        ])
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def train_initial_model():
    """Train initial ML model with synthetic data"""
    logger.info("Training initial ML model...")
    try:
        # Change to project root directory for training
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Import and run training directly
        from quality.train_quality_classifier import main as train_main
        train_main()
        
        # Return to original directory
        os.chdir(original_cwd)
        
        logger.info("✅ Initial ML model trained successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to train initial model: {e}")
        # Try alternative approach with subprocess
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                f"import sys; sys.path.insert(0, '{project_root}'); "
                "from quality.train_quality_classifier import main; main()"
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                logger.info("✅ Initial ML model trained successfully (subprocess)")
                return True
            else:
                logger.error(f"❌ Training failed: {result.stderr}")
                return False
        except Exception as e2:
            logger.error(f"❌ Failed to train model with subprocess: {e2}")
            return False

def test_quality_assessment():
    """Test the quality assessment system"""
    logger.info("Testing quality assessment system...")
    try:
        # Change to project root for imports
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        from quality.adaptive_quality_checker import AdaptiveDocumentQualityChecker
        
        checker = AdaptiveDocumentQualityChecker()
        
        # Test with sample metrics
        test_metrics = {
            "blur_confidence": 0.75,
            "edge_cut_flags": 1,
            "text_density_violations": 0,
            "brightness_issue": False,
            "skew_angle": 3.2,
            "document_area_ratio": 0.87
        }
        
        # Test rule-based scoring
        risk_score, reasons, decision = checker.compute_quality_risk_score(test_metrics)
        logger.info(f"Rule-based: {decision} (risk: {risk_score:.3f})")
        
        # Test ML prediction (if available)
        ml_pred, ml_prob, ml_status = checker.predict_rescan_with_ml(test_metrics)
        if ml_status == "success":
            logger.info(f"ML-based: {'Rescan' if ml_pred else 'Accept'} (prob: {ml_prob:.3f})")
            logger.info("✅ Quality assessment system is working correctly")
        else:
            logger.info(f"ML status: {ml_status}")
            logger.info("✅ Rule-based quality assessment is working")
        
        # Return to original directory
        os.chdir(original_cwd)
        return True
    except Exception as e:
        logger.error(f"❌ Quality assessment test failed: {e}")
        # Return to original directory in case of error
        try:
            os.chdir(original_cwd)
        except:
            pass
        return False

def main():
    """Main setup process"""
    print("🚀 Setting up Hybrid Quality Assessment System")
    print("=" * 60)
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n📦 Missing dependencies: {', '.join(missing)}")
        install_deps = input("Install missing dependencies? (y/n): ").lower().strip()
        if install_deps == 'y':
            if not install_dependencies():
                print("❌ Setup failed due to dependency installation issues")
                return False
        else:
            print("⚠️  Setup incomplete - missing dependencies")
            return False
    
    # Train initial model
    print("\n🤖 Setting up ML model...")
    model_exists = os.path.exists("quality_rescan_model.pkl")
    
    if model_exists:
        retrain = input("ML model exists. Retrain? (y/n): ").lower().strip()
        if retrain == 'y':
            train_initial_model()
    else:
        print("No ML model found. Training initial model...")
        if not train_initial_model():
            print("⚠️  ML model training failed, but system will work with rule-based scoring only")
    
    # Test system
    print("\n🧪 Testing quality assessment system...")
    if test_quality_assessment():
        print("\n✅ Setup completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Run: python enhanced_gradio_app.py")
        print("2. Open: http://localhost:7861")
        print("3. Process documents and provide quality feedback")
        print("4. Monitor training progress in the ML Training tab")
        print("5. Retrain model when you have 100+ labeled samples")
    else:
        print("❌ Setup completed with issues. Check logs above.")
    
    return True

if __name__ == "__main__":
    main()