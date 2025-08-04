#!/usr/bin/env python3
"""
🧠 Hybrid Document Scanner - Centralized Application Runner

This is the single entry point for all operations:
- Setup and installation
- Running different modes (UI, API, demos)
- Service management (start, stop, status)
- Deployment operations
- Testing and diagnostics

Usage:
    python run.py setup           # Complete setup and installation
    python run.py ui              # Launch enhanced Gradio UI
    python run.py api             # Launch FastAPI service
    python run.py status          # Check system status
    python run.py test            # Run diagnostics
    python run.py train           # Train ML model
    python run.py deploy          # Deploy to production
    python run.py stop            # Stop all services
"""

import os
import sys
import subprocess
import logging
import time
import argparse
import json
import signal
import psutil
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

class DocumentScannerRunner:
    """Centralized runner for the Document Scanner application"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.processes = {}
        self.config = self._load_config()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """Load application configuration"""
        config_file = self.project_root / 'config' / 'scanner_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {
            "ui_port": 7861,
            "api_port": 8000,
            "hybrid_api_port": 8001
        }
    
    def print_banner(self):
        """Print application banner"""
        print(f"{Colors.CYAN}")
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║               🧠 HYBRID DOCUMENT SCANNER                     ║")
        print("║          Intelligent Quality Assessment + ML                  ║")
        print("║                      Version 5.0.0                           ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print(f"{Colors.NC}")
    
    def log_info(self, message):
        print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")
    
    def log_warn(self, message):
        print(f"{Colors.YELLOW}[WARN]{Colors.NC} {message}")
    
    def log_error(self, message):
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
    
    def log_success(self, message):
        print(f"{Colors.PURPLE}[SUCCESS]{Colors.NC} {message}")
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        self.log_info("🔍 Checking system prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log_error(f"Python 3.8+ required. Found: {sys.version_info}")
            return False
        self.log_success(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.log_warn("Not running in virtual environment. Recommended: python -m venv .venv")
        
        # Check project structure
        required_dirs = ['src', 'quality', 'config', 'docs']
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                self.log_error(f"Required directory missing: {dir_name}")
                return False
        
        self.log_success("✅ Project structure verified")
        return True
    
    def setup(self) -> bool:
        """Complete setup and installation"""
        self.print_banner()
        self.log_info("🚀 Starting complete setup process...")
        
        if not self.check_prerequisites():
            return False
        
        try:
            # Install dependencies
            self.log_info("📦 Installing dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"
            ], check=True, cwd=self.project_root)
            
            # Setup environment file if not exists
            env_file = self.project_root / '.env'
            if not env_file.exists():
                self.log_info("⚙️ Creating environment configuration...")
                env_content = """# Google Cloud Vision
GOOGLE_APPLICATION_CREDENTIALS=credentials/key.json

# LLM Providers (Optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Ollama (if using locally)
OLLAMA_BASE_URL=http://localhost:11434

# Application Settings
LOG_LEVEL=INFO
"""
                with open(env_file, 'w') as f:
                    f.write(env_content)
                self.log_info("📝 Created .env file - please update with your API keys")
            
            # Train initial ML model
            self.log_info("🤖 Training initial ML model...")
            result = subprocess.run([
                sys.executable, "-m", "quality.train_quality_classifier"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log_success("✅ ML model trained successfully")
            else:
                self.log_warn("⚠️ ML model training failed, but system will work with rule-based only")
            
            # Test system
            self.log_info("🧪 Testing system components...")
            if self.test_system():
                self.log_success("🎉 Setup completed successfully!")
                self.log_info("\n📋 Next steps:")
                self.log_info("  1. Update .env file with your API keys")
                self.log_info("  2. Run: python run.py ui")
                self.log_info("  3. Open: http://localhost:7861")
                return True
            else:
                self.log_error("❌ System test failed")
                return False
                
        except subprocess.CalledProcessError as e:
            self.log_error(f"Setup failed: {e}")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error during setup: {e}")
            return False
    
    def test_system(self) -> bool:
        """Test system components"""
        self.log_info("🧪 Testing system components...")
        
        try:
            # Test imports
            test_imports = [
                ("src.config", "Configuration"),
                ("quality.adaptive_quality_checker", "Quality Assessment"),
                ("src.hybrid_document_processor", "Document Processor"),
            ]
            
            for module, description in test_imports:
                try:
                    __import__(module)
                    self.log_success(f"✅ {description}")
                except ImportError as e:
                    self.log_error(f"❌ {description}: {e}")
                    return False
            
            # Test quality assessment with sample data
            from quality.adaptive_quality_checker import AdaptiveDocumentQualityChecker
            import numpy as np
            
            checker = AdaptiveDocumentQualityChecker()
            dummy_image = np.zeros((500, 400, 3), dtype=np.uint8)
            dummy_image.fill(128)
            
            result = checker.assess_quality(dummy_image, document_type="general")
            self.log_success("✅ Quality assessment system working")
            
            # Test ML prediction if available
            test_metrics = {
                "blur_confidence": 0.75,
                "edge_cut_flags": 1,
                "text_density_violations": 0,
                "brightness_issue": False,
                "skew_angle": 3.2,
                "document_area_ratio": 0.87
            }
            
            ml_pred, ml_prob, ml_status = checker.predict_rescan_with_ml(test_metrics)
            if ml_status == "success":
                self.log_success(f"✅ ML prediction working: {'Rescan' if ml_pred else 'Accept'} (prob: {ml_prob:.3f})")
            else:
                self.log_info(f"ℹ️ ML status: {ml_status}")
            
            return True
            
        except Exception as e:
            self.log_error(f"System test failed: {e}")
            return False
    
    def run_ui(self, port: Optional[int] = None):
        """Launch enhanced Gradio UI"""
        port = port or self.config.get("ui_port", 7861)
        
        self.log_info(f"🚀 Starting Enhanced Gradio UI on port {port}...")
        
        try:
            # Check if port is available
            if self._is_port_in_use(port):
                self.log_error(f"Port {port} is already in use")
                return False
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # Start the UI
            process = subprocess.Popen([
                sys.executable, "enhanced_gradio_app.py"
            ], cwd=self.project_root, env=env)
            
            self.processes['ui'] = process
            
            self.log_success(f"✅ Enhanced UI started successfully!")
            self.log_info(f"🌐 Open browser: http://localhost:{port}")
            self.log_info("📊 Features: Rule-based + ML quality assessment")
            self.log_info("🎓 ML training data collection enabled")
            self.log_info("\nPress Ctrl+C to stop")
            
            # Wait for process
            try:
                process.wait()
            except KeyboardInterrupt:
                self.log_info("Stopping UI...")
                process.terminate()
                process.wait()
                
        except Exception as e:
            self.log_error(f"Failed to start UI: {e}")
            return False
    
    def run_api(self, port: Optional[int] = None):
        """Launch FastAPI service"""
        port = port or self.config.get("api_port", 8000)
        
        self.log_info(f"🚀 Starting FastAPI service on port {port}...")
        
        try:
            if self._is_port_in_use(port):
                self.log_error(f"Port {port} is already in use")
                return False
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            process = subprocess.Popen([
                sys.executable, "main.py"
            ], cwd=self.project_root, env=env)
            
            self.processes['api'] = process
            
            self.log_success(f"✅ FastAPI service started!")
            self.log_info(f"🌐 API URL: http://localhost:{port}")
            self.log_info(f"📚 API Docs: http://localhost:{port}/docs")
            self.log_info("\nPress Ctrl+C to stop")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                self.log_info("Stopping API...")
                process.terminate()
                process.wait()
                
        except Exception as e:
            self.log_error(f"Failed to start API: {e}")
            return False
    
    def run_hybrid_api(self, port: Optional[int] = None):
        """Launch Hybrid API service"""
        port = port or self.config.get("hybrid_api_port", 8001)
        
        self.log_info(f"🚀 Starting Hybrid API service on port {port}...")
        
        try:
            if self._is_port_in_use(port):
                self.log_error(f"Port {port} is already in use")
                return False
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            process = subprocess.Popen([
                sys.executable, "hybrid_api_app.py"
            ], cwd=self.project_root, env=env)
            
            self.processes['hybrid_api'] = process
            
            self.log_success(f"✅ Hybrid API service started!")
            self.log_info(f"🌐 API URL: http://localhost:{port}")
            self.log_info(f"📚 API Docs: http://localhost:{port}/docs")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                self.log_info("Stopping Hybrid API...")
                process.terminate()
                process.wait()
                
        except Exception as e:
            self.log_error(f"Failed to start Hybrid API: {e}")
            return False
    
    def run_demo(self, demo_name: Optional[str] = None):
        """Run demo scripts"""
        available_demos = {
            "risk": "demos.demo_risk_scoring",
            "hybrid": "demos.demo_hybrid_quality_system", 
            "adaptive": "demos.demo_adaptive_learning"
        }
        
        if not demo_name:
            self.log_info("Available demos:")
            for name, module in available_demos.items():
                self.log_info(f"  python run.py demo {name}")
            return
        
        if demo_name not in available_demos:
            self.log_error(f"Demo '{demo_name}' not found")
            return
        
        try:
            self.log_info(f"🧪 Running {demo_name} demo...")
            subprocess.run([
                sys.executable, "-m", available_demos[demo_name]
            ], cwd=self.project_root, check=True)
            
        except subprocess.CalledProcessError as e:
            self.log_error(f"Demo failed: {e}")
    
    def train_model(self):
        """Train ML model"""
        self.log_info("🤖 Training ML model for quality assessment...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "quality.train_quality_classifier"
            ], cwd=self.project_root, check=True)
            
            self.log_success("✅ ML model training completed!")
            
        except subprocess.CalledProcessError as e:
            self.log_error(f"Training failed: {e}")
    
    def get_status(self):
        """Get system status"""
        self.log_info("📊 System Status:")
        
        # Check processes
        running_services = []
        ports_to_check = [
            (7861, "Enhanced UI"),
            (8000, "FastAPI"),
            (8001, "Hybrid API"),
            (7860, "Original UI")
        ]
        
        for port, service in ports_to_check:
            if self._is_port_in_use(port):
                running_services.append(f"{service} (port {port})")
        
        if running_services:
            self.log_success("✅ Running services:")
            for service in running_services:
                self.log_info(f"  - {service}")
        else:
            self.log_info("No services currently running")
        
        # Check files
        important_files = [
            ("quality_rescan_model.pkl", "ML Model"),
            ("quality_labeled_data.csv", "Training Data"),
            (".env", "Environment Config"),
            ("credentials/key.json", "Google Cloud Credentials")
        ]
        
        self.log_info("\n📁 Important Files:")
        for file_path, description in important_files:
            file_full_path = self.project_root / file_path
            if file_full_path.exists():
                size = file_full_path.stat().st_size
                size_str = self._format_size(size)
                self.log_success(f"  ✅ {description}: {size_str}")
            else:
                self.log_warn(f"  ❌ {description}: Missing")
        
        # Check Python environment
        self.log_info(f"\n🐍 Python Environment:")
        self.log_info(f"  Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        self.log_info(f"  Executable: {sys.executable}")
        self.log_info(f"  Working Dir: {self.project_root}")
        
        # Check key dependencies
        key_packages = ["gradio", "opencv-python", "scikit-learn", "fastapi"]
        self.log_info(f"\n📦 Key Dependencies:")
        
        for package in key_packages:
            try:
                __import__(package.replace('-', '_'))
                self.log_success(f"  ✅ {package}")
            except ImportError:
                self.log_error(f"  ❌ {package}")
    
    def stop_services(self):
        """Stop all running services"""
        self.log_info("🛑 Stopping all services...")
        
        # Stop tracked processes
        for name, process in self.processes.items():
            if process.poll() is None:  # Process is running
                self.log_info(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                    self.log_success(f"✅ {name} stopped")
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.log_warn(f"⚠️ {name} force killed")
        
        # Kill processes on known ports
        ports = [7860, 7861, 8000, 8001]
        for port in ports:
            self._kill_process_on_port(port)
        
        self.log_success("✅ All services stopped")
    
    def deploy(self):
        """Deploy to production"""
        self.log_info("🚀 Deploying to production...")
        
        # This is a simplified version - full deployment would use Docker
        self.log_warn("Production deployment requires Docker setup")
        self.log_info("For now, run: python run.py ui  # for local deployment")
        
        # Future: Docker-based deployment
        docker_compose = self.project_root / 'deploy' / 'docker-compose.yml'
        if docker_compose.exists():
            self.log_info("Docker Compose file found - you can use:")
            self.log_info("  cd deploy && docker-compose up --build")
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False
    
    def _kill_process_on_port(self, port: int):
        """Kill process running on specific port"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.pid:
                try:
                    process = psutil.Process(conn.pid)
                    process.terminate()
                    self.log_info(f"Stopped process on port {port}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    
    def _format_size(self, size: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🧠 Hybrid Document Scanner - Centralized Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py setup              # Complete setup
  python run.py ui                 # Launch enhanced UI
  python run.py api                # Launch FastAPI
  python run.py status             # Check status
  python run.py test               # Run diagnostics
  python run.py train              # Train ML model
  python run.py demo risk          # Run risk scoring demo
  python run.py stop               # Stop all services
        """
    )
    
    parser.add_argument('command', choices=[
        'setup', 'ui', 'api', 'hybrid-api', 'status', 'test', 'train', 
        'demo', 'stop', 'deploy'
    ], help='Command to execute')
    
    parser.add_argument('subcommand', nargs='?', help='Subcommand (e.g., demo name)')
    parser.add_argument('--port', type=int, help='Port number for services')
    
    args = parser.parse_args()
    
    runner = DocumentScannerRunner()
    
    try:
        if args.command == 'setup':
            success = runner.setup()
            sys.exit(0 if success else 1)
            
        elif args.command == 'ui':
            runner.run_ui(args.port)
            
        elif args.command == 'api':
            runner.run_api(args.port)
            
        elif args.command == 'hybrid-api':
            runner.run_hybrid_api(args.port)
            
        elif args.command == 'status':
            runner.get_status()
            
        elif args.command == 'test':
            success = runner.test_system()
            sys.exit(0 if success else 1)
            
        elif args.command == 'train':
            runner.train_model()
            
        elif args.command == 'demo':
            runner.run_demo(args.subcommand)
            
        elif args.command == 'stop':
            runner.stop_services()
            
        elif args.command == 'deploy':
            runner.deploy()
            
    except KeyboardInterrupt:
        runner.log_info("\n👋 Goodbye!")
        runner.stop_services()
        sys.exit(0)
    except Exception as e:
        runner.log_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()