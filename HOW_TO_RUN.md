# 🚀 How to Run the Document Scanner Application

## 🎯 Centralized Application Runner

**NEW**: We now have a centralized `run.py` script that handles all operations!

```bash
python run.py --help  # See all available commands
```

**Key Commands:**
- `python run.py setup` - Complete setup and installation
- `python run.py ui` - Launch enhanced Gradio UI  
- `python run.py api` - Launch FastAPI service
- `python run.py status` - Check system status
- `python run.py test` - Run diagnostics
- `python run.py train` - Train ML model
- `python run.py demo <name>` - Run demo scripts
- `python run.py stop` - Stop all services

**Benefits:**
- ✅ **Single entry point** for all operations
- ✅ **Automatic path management** - no more import issues
- ✅ **Service management** - start/stop/status tracking
- ✅ **Consistent interface** across all functions
- ✅ **Error handling** and user-friendly messages

---

## 📁 Project Structure

```
document-scanner/
├── 📄 Main Applications
│   ├── main.py                     # FastAPI main application 
│   ├── enhanced_gradio_app.py      # Enhanced Gradio UI with ML quality assessment
│   ├── gradio_app.py              # Original Gradio UI
│   └── hybrid_api_app.py          # Additional API endpoints
│
├── 🧠 Core Processing (src/)
│   ├── hybrid_document_processor.py    # Main document processing pipeline
│   ├── adaptive_document_processor.py  # Adaptive processing with learning
│   ├── hybrid_kv_extractor.py         # Hybrid key-value extraction
│   ├── adaptive_kv_extractor.py       # Adaptive extraction with OCR
│   ├── llm_kv_extractor.py           # LLM-based extraction
│   └── config.py                     # Configuration management
│
├── 📊 Quality Assessment (quality/)
│   ├── adaptive_quality_checker.py    # Hybrid quality assessment (Rule-based + ML)
│   ├── quality_data_collector.py     # Training data collection
│   └── train_quality_classifier.py   # ML model training
│
├── 🧪 Demos (demos/)
│   ├── demo_hybrid_quality_system.py  # Complete hybrid system demo
│   ├── demo_risk_scoring.py          # Risk scoring demonstration
│   └── demo_adaptive_learning.py     # Adaptive learning examples
│
├── 🧪 Tests (tests/)
│   ├── test_hybrid_system.py         # System integration tests
│   ├── test_client.py               # API client tests
│   ├── test_gradio_interface.py     # Gradio UI tests
│   └── test_gradio_v2.py           # Additional UI tests
│
├── 📚 Documentation (docs/)
│   ├── QUALITY_ASSESSMENT_README.md   # Quality assessment system guide
│   └── controls.md                   # System controls documentation
│
├── 🔧 Scripts (scripts/)
│   └── quick_setup.py                # Quick setup and installation
│
├── ⚙️ Configuration (config/)
│   └── scanner_config.json          # Scanner configuration
│
├── 🚀 Deployment (deploy/)
│   ├── deploy.sh                    # Deployment script
│   ├── Dockerfile                  # Docker configuration
│   ├── docker-compose.yml         # Docker Compose setup
│   └── nginx.conf                 # Nginx configuration
│
├── 🔐 Private
│   ├── credentials/               # API credentials (Google Cloud Vision)
│   ├── test_images/              # Sample test images
│   ├── uploads/                  # Temporary upload directory
│   └── logs/                     # Application logs
│
└── 📋 Project Files
    ├── requirements.txt           # Python dependencies
    ├── README.md                 # Project overview
    ├── HOW_TO_RUN.md            # This file
    └── .env                     # Environment variables
```

---

## 🔧 Prerequisites

### **System Requirements**
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM (8GB recommended for ML training)
- **Storage**: 2GB free space for dependencies and models

### **API Keys & Services**
- **Google Cloud Vision API** (for OCR)
  - Create a project at [Google Cloud Console](https://console.cloud.google.com/)
  - Enable Cloud Vision API
  - Create service account and download JSON key
  - Place key file in `credentials/key.json`

- **LLM Providers** (Optional, for advanced extraction)
  - **Ollama**: Install locally for free LLM processing
  - **OpenAI**: API key for GPT models
  - **Anthropic**: API key for Claude models

---

## 🚀 Quick Start (Centralized Runner - Recommended)

### **⚡ Super Quick Start**
```bash
# 1. Clone and setup
git clone <repository-url>
cd document-scanner
python run.py setup

# 2. Launch UI
python run.py ui

# 3. Open browser: http://localhost:7861
```

That's it! The centralized runner handles everything automatically.

### **📋 What the setup does:**
- ✅ Check system prerequisites and Python version
- ✅ Install all dependencies automatically
- ✅ Create environment configuration file
- ✅ Train initial ML model with synthetic data
- ✅ Test all system components
- ✅ Verify system readiness

### **🎯 Available Commands:**
```bash
python run.py status          # Check what's running
python run.py test            # Run diagnostics  
python run.py train           # Train ML model
python run.py demo risk       # Run demo scripts
python run.py stop            # Stop all services
```

---

## 📋 Manual Setup

### **1. Install Dependencies**
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

# For development
pip install -r requirements.txt --upgrade
```

### **2. Configure Environment**
Create `.env` file in project root:
```bash
# Google Cloud Vision
GOOGLE_APPLICATION_CREDENTIALS=credentials/key.json

# LLM Providers (Optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Ollama (if using locally)
OLLAMA_BASE_URL=http://localhost:11434
```

### **3. Train ML Model (Optional)**
```bash
# Train quality assessment ML model
python -m quality.train_quality_classifier

# This creates:
# - quality_rescan_model.pkl (trained model)
# - quality_rescan_scaler.pkl (feature scaler)
# - quality_labeled_data.csv (training data)
```

---

## 🎮 Running the Applications (Centralized Commands)

### **🌟 Enhanced Gradio UI (Recommended)**
```bash
python run.py ui
# Alternative: python enhanced_gradio_app.py
```
- **URL**: http://localhost:7861
- **Features**: 
  - Complete document processing pipeline
  - Hybrid quality assessment (Rule-based + ML)
  - User feedback collection for ML training
  - Real-time training data monitoring
  - Quality assessment visualization

### **🔗 FastAPI Web Service**
```bash
python run.py api
# Alternative: python main.py
```
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Features**: 
  - RESTful API endpoints
  - Batch processing support
  - Integration-ready JSON responses

### **🔌 Hybrid API Service**
```bash
python run.py hybrid-api
# Alternative: python hybrid_api_app.py
```
- **URL**: http://localhost:8001
- **Features**: 
  - Advanced API configurations
  - Multiple extraction strategies
  - LLM provider selection

### **📱 Original Gradio UI**
```bash
python gradio_app.py  # (No centralized command yet)
```
- **URL**: http://localhost:7860
- **Features**: 
  - Clean, simple interface
  - Core document processing
  - Basic quality assessment

---

## 🧪 Testing & Demos (Centralized Commands)

### **🎯 Quality Assessment Demos**
```bash
# Test risk scoring system
python run.py demo risk

# Test complete hybrid system
python run.py demo hybrid

# Test adaptive learning
python run.py demo adaptive

# Alternative methods:
# python -m demos.demo_risk_scoring
# python -m demos.demo_hybrid_quality_system
# python -m demos.demo_adaptive_learning
```

### **🔬 System Tests**
```bash
# Test system components (centralized)
python run.py test

# Check system status
python run.py status

# Alternative methods:
# python -m tests.test_hybrid_system
# python -m tests.test_client
```

### **🤖 ML Model Training**
```bash
# Train ML model (centralized)
python run.py train

# Alternative method:
# python -m quality.train_quality_classifier
```

---

## 📊 Quality Assessment Features

### **🧠 Hybrid Quality Analysis**
The system provides **dual quality assessment**:

**1. Rule-Based Analysis** (Computer Vision):
- Blur detection using Laplacian variance
- Edge cut detection with adaptive margins  
- Text density analysis near image borders
- Brightness and exposure validation
- Document skew and orientation checks

**2. ML Classification** (Machine Learning):
- Trained on user feedback data
- Probabilistic quality predictions
- Continuous learning from real usage
- Agreement monitoring with rule-based system

### **📈 Training Data Collection**
- **Automatic Logging**: Every processing session logged
- **User Feedback**: Quality ratings improve ML model
- **Progress Tracking**: Monitor training data collection
- **Ground Truth Labels**: User corrections enhance accuracy

---

## 🎛️ Configuration Options

### **📁 Config Files**
- `config/scanner_config.json`: Scanner-specific settings
- `src/config.py`: Adaptive configuration management
- `.env`: Environment variables and API keys

### **⚙️ Quality Thresholds**
Customize quality assessment sensitivity:
```python
# In src/config.py or via adaptive_config
quality_thresholds = {
    "blur_threshold": 100,
    "brightness_threshold": (30, 200),
    "skew_tolerance": 15,
    "cut_edge_margin_pct": 0.03
}
```

### **🤖 ML Model Settings**
```python
# Model training parameters
ml_settings = {
    "model_type": "RandomForest",  # or "LogisticRegression"
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}
```

---

## 🔍 Monitoring & Debugging

### **📊 System Status**
Check system health via:
- Enhanced Gradio UI → System Status tab
- API endpoint: `/status`
- Quick setup script diagnostics

### **📋 Logs**
Monitor application logs:
```bash
# View real-time logs
tail -f logs/dev.log
tail -f logs/gradio.log
tail -f logs/hybrid_dev.log
```

### **🔬 Debug Mode**
Enable detailed logging:
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or modify logging configuration in code
logging.basicConfig(level=logging.DEBUG)
```

---

## 🐳 Docker Deployment

### **🚀 Using Docker Compose**
```bash
# Build and start services
docker-compose up --build

# Access services:
# - Gradio UI: http://localhost:7860
# - FastAPI: http://localhost:8000
# - Nginx: http://localhost:80
```

### **📦 Manual Docker Build**
```bash
# Build image
docker build -t document-scanner .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/credentials:/app/credentials \
  -v $(pwd)/uploads:/app/uploads \
  document-scanner
```

### **☁️ Production Deployment**
```bash
# Use deployment script
./deploy/deploy.sh

# This configures:
# - Nginx reverse proxy
# - SSL certificates (Let's Encrypt)
# - Systemd services
# - Log rotation
```

---

## 🔧 Recent Fixes & Improvements

**✅ Quality Assessment System Updates (Latest)**:
- Fixed AdaptiveConfig parameter mismatch issues
- Resolved skew detection unpacking errors  
- Fixed variable scope issues in quality checker
- Eliminated sklearn feature name warnings
- Improved error handling and user messaging
- Enhanced ML model integration with proper DataFrame usage

Run `python test_quality_fixes.py` to verify all fixes are working correctly.

---

## 🚨 Troubleshooting

### **❌ Common Issues**

**1. Import Errors / "No module named 'quality'" / "No module named 'src'"**
```bash
# Solution 1: Use root-level setup script
python setup.py

# Solution 2: Set Python path explicitly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/quick_setup.py

# Solution 3: Run as module
python -m scripts.quick_setup

# Solution 4: For enhanced Gradio app
PYTHONPATH=. python enhanced_gradio_app.py
```

**2. Google Cloud Vision Setup**
```bash
# Verify credentials
export GOOGLE_APPLICATION_CREDENTIALS=credentials/key.json
python -c "from google.cloud import vision; print('✅ Google Cloud Vision OK')"
```

**3. ML Model Not Found**
```bash
# Train initial model
python -m quality.train_quality_classifier

# Or use synthetic data
python scripts/quick_setup.py
```

**4. Permission Errors**
```bash
# Fix file permissions
chmod -R 755 logs/ uploads/ credentials/
```

**5. Port Already in Use**
```bash
# Kill processes using ports
sudo lsof -ti:7861 | xargs kill -9  # Gradio
sudo lsof -ti:8000 | xargs kill -9  # FastAPI
```

### **🔍 Getting Help**

**Check System Status**:
```bash
python run.py status          # Centralized status check
python run.py test            # Run all diagnostics
# python scripts/quick_setup.py  # Alternative
```

**Stop All Services**:
```bash
python run.py stop            # Stop everything cleanly
```

**Verify Dependencies**:
```bash
pip list | grep -E "(gradio|opencv|scikit|google)"
```

**Test Core Components**:
```bash
python -c "from src.config import adaptive_config; print('✅ Config OK')"
python -c "from quality.adaptive_quality_checker import AdaptiveDocumentQualityChecker; print('✅ Quality Checker OK')"
```

---

## 🎯 Usage Recommendations

### **🏗️ For Development**
1. Use **Enhanced Gradio UI** for interactive testing
2. Enable debug logging for detailed analysis  
3. Use demo scripts to understand system components
4. Collect quality feedback to improve ML model

### **🚀 For Production**
1. Use **FastAPI main.py** for REST API services
2. Configure proper logging and monitoring
3. Set up reverse proxy (Nginx) for SSL and load balancing
4. Implement proper backup for ML models and training data

### **🧪 For Testing**
1. Use **test scripts** for automated validation
2. Run **demo scripts** to verify functionality
3. Test with various document types and qualities
4. Monitor agreement between rule-based and ML assessments

---

## 📈 Performance Optimization

### **⚡ Speed Improvements**
- Use **adaptive_only** extraction strategy for fastest processing
- Reduce image resolution for faster OCR processing
- Enable ML model caching for repeated predictions
- Use parallel processing for batch operations

### **🎯 Accuracy Improvements**
- Collect diverse training data for ML model
- Provide consistent quality feedback
- Use **parallel** extraction strategy for best accuracy
- Fine-tune quality thresholds per document type

### **💾 Memory Optimization**
- Process images in batches for large volumes
- Clear intermediate processing results
- Use streaming for large file uploads
- Monitor memory usage during ML training

---

## 🎉 Success! 

Your Enhanced Document Scanner is now ready! 

**🚀 Start Processing**: Upload documents and see the hybrid quality assessment in action.

**📊 Improve Over Time**: Provide quality feedback to enhance the ML model.

**🔧 Customize**: Adjust thresholds and configurations based on your use case.

**📈 Monitor**: Track system performance and training data collection progress.

For detailed quality assessment information, see: `docs/QUALITY_ASSESSMENT_README.md`

---

*Happy document scanning! 🎯📄*