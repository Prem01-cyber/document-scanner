#!/bin/bash

# Hybrid Document Scanner Deployment Script
# Complete deployment and management script for the hybrid extraction system
# Supports adaptive + LLM extraction with intelligent fallback

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="hybrid-document-scanner"
PYTHON_VERSION="3.9"
REQUIRED_PACKAGES=("curl" "docker" "docker-compose" "python3" "python3-pip" "python3-venv")
HYBRID_COMPONENTS=("config.py" "adaptive_quality_checker.py" "adaptive_kv_extractor.py" "llm_kv_extractor.py" "hybrid_kv_extractor.py" "hybrid_document_processor.py" "hybrid_api_app.py")

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

log_success() {
    echo -e "${PURPLE}[SUCCESS]${NC} $1"
}

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║               🧠 HYBRID DOCUMENT SCANNER                     ║"
    echo "║          Intelligent Adaptive + LLM Extraction               ║"
    echo "║                      Version 4.0.0                           ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_prerequisites() {
    log_info "🔍 Checking system prerequisites..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root. This is not recommended for production."
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        log_info "Python version: $PYTHON_VER"
        
        # Check if Python version meets minimum requirement
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "✅ Python version is compatible"
        else
            log_error "Python 3.8+ is required. Found: $PYTHON_VER"
            exit 1
        fi
    else
        log_error "Python 3 is required but not installed."
        log_info "Install with: sudo apt-get update && sudo apt-get install python3 python3-pip python3-venv"
        exit 1
    fi
    
    # Check required system packages
    log_info "Checking required system packages..."
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! command -v $pkg &> /dev/null; then
            log_error "$pkg is required but not installed."
            case $pkg in
                docker)
                    echo "Install with: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
                    ;;
                docker-compose)
                    echo "Install with: sudo apt-get install docker-compose-plugin"
                    ;;
                *)
                    echo "Install with: sudo apt-get install $pkg"
                    ;;
            esac
            exit 1
        fi
    done
    
    # Check hybrid system components
    log_info "Checking hybrid system components..."
    missing_components=()
    for component in "${HYBRID_COMPONENTS[@]}"; do
        if [ ! -f "$component" ]; then
            missing_components+=("$component")
        fi
    done
    
    if [ ${#missing_components[@]} -gt 0 ]; then
        log_error "Missing hybrid system components:"
        for component in "${missing_components[@]}"; do
            echo "  ❌ $component"
        done
        log_error "Please ensure all hybrid system files are present."
        exit 1
    else
        log_success "✅ All hybrid system components found"
    fi
    
    log_success "✅ Prerequisites check passed!"
}

setup_google_cloud() {
    log_info "🔑 Setting up Google Cloud Vision API..."
    
    # Create credentials directory
    mkdir -p credentials
    
    if [ ! -f "credentials/key.json" ]; then
        log_warn "Google Cloud credentials not found."
        echo ""
        echo "📋 To set up Google Cloud Vision API:"
        echo "   1. Go to https://console.cloud.google.com/"
        echo "   2. Create a new project or select existing"
        echo "   3. Enable Vision API"
        echo "   4. Create a service account"
        echo "   5. Download JSON key file"
        echo "   6. Place it at: credentials/key.json"
        echo ""
        read -p "Have you placed the credentials file at credentials/key.json? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Google Cloud credentials are required for OCR functionality."
            exit 1
        fi
        
        if [ ! -f "credentials/key.json" ]; then
            log_error "credentials/key.json still not found!"
            exit 1
        fi
    fi
    
    log_success "✅ Google Cloud credentials configured"
}

setup_llm_providers() {
    log_info "🤖 Setting up LLM providers..."
    
    # Check for existing .env file
    if [ ! -f ".env" ]; then
        log_info "Creating .env file for LLM providers..."
        touch .env
    fi
    
    # Source existing .env to check current values
    set -a
    [ -f .env ] && source .env
    set +a
    
    echo ""
    echo "🤖 LLM Provider Setup:"
    echo "   The hybrid system supports multiple LLM providers."
    echo "   You can configure one or more providers (at least one recommended)."
    echo ""
    
    # OpenAI Setup
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "🔹 OpenAI GPT (Recommended for best quality)"
        read -p "   Enter OpenAI API key (or press Enter to skip): " openai_key
        if [ ! -z "$openai_key" ]; then
            echo "OPENAI_API_KEY=$openai_key" >> .env
            log_success "✅ OpenAI configured"
        else
            log_info "⏭️  OpenAI skipped"
        fi
    else
        log_success "✅ OpenAI already configured"
    fi
    
    # Anthropic Setup
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "🔹 Anthropic Claude (Fast and cost-effective)"
        read -p "   Enter Anthropic API key (or press Enter to skip): " anthropic_key
        if [ ! -z "$anthropic_key" ]; then
            echo "ANTHROPIC_API_KEY=$anthropic_key" >> .env
            log_success "✅ Anthropic configured"
        else
            log_info "⏭️  Anthropic skipped"
        fi
    else
        log_success "✅ Anthropic already configured"
    fi
    
    # Azure OpenAI Setup
    if [ -z "$AZURE_OPENAI_API_KEY" ]; then
        echo "🔹 Azure OpenAI (Enterprise option)"
        read -p "   Enter Azure OpenAI API key (or press Enter to skip): " azure_key
        if [ ! -z "$azure_key" ]; then
            read -p "   Enter Azure OpenAI endpoint: " azure_endpoint
            read -p "   Enter Azure deployment name: " azure_deployment
            echo "AZURE_OPENAI_API_KEY=$azure_key" >> .env
            echo "AZURE_OPENAI_ENDPOINT=$azure_endpoint" >> .env
            echo "AZURE_OPENAI_DEPLOYMENT_NAME=$azure_deployment" >> .env
            log_success "✅ Azure OpenAI configured"
        else
            log_info "⏭️  Azure OpenAI skipped"
        fi
    else
        log_success "✅ Azure OpenAI already configured"
    fi
    
    # Ollama Setup
    echo "🔹 Ollama (Local LLM - Free but requires setup)"
    if command -v ollama &> /dev/null; then
        log_success "✅ Ollama is installed"
        
        # Check if Ollama is running
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "✅ Ollama is running"
        else
            log_warn "Ollama is installed but not running"
            read -p "   Start Ollama service? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Starting Ollama..."
                ollama serve &
                sleep 3
            fi
        fi
        
        # Check for models
        if ollama list | grep -q "llama3.1:8b"; then
            log_success "✅ Llama3.1 model available"
        else
            read -p "   Download Llama3.1 model? (~4.7GB) (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Downloading Llama3.1 model..."
                ollama pull llama3.1:8b
                log_success "✅ Llama3.1 model downloaded"
            fi
        fi
    else
        read -p "   Install Ollama for local LLM support? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installing Ollama..."
            curl -fsSL https://ollama.ai/install.sh | sh
            log_info "Starting Ollama..."
            ollama serve &
            sleep 3
            log_info "Downloading Llama3.1 model..."
            ollama pull llama3.1:8b
            log_success "✅ Ollama configured with Llama3.1"
        else
            log_info "⏭️  Ollama skipped"
        fi
    fi
    
    # Add Google Cloud credentials to .env
    if ! grep -q "GOOGLE_APPLICATION_CREDENTIALS" .env; then
        echo "GOOGLE_APPLICATION_CREDENTIALS=./credentials/key.json" >> .env
    fi
    
    log_success "✅ LLM providers setup completed"
}

create_project_structure() {
    log_info "📁 Creating project structure..."
    
    # Create directories
    mkdir -p credentials
    mkdir -p uploads
    mkdir -p logs
    mkdir -p test_images
    mkdir -p config_backups
    
    # Create hybrid requirements.txt
    if [ ! -f "requirements.txt" ]; then
        log_info "Creating requirements.txt for hybrid system..."
        cat > requirements.txt << 'EOF'
# Core FastAPI and server
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Computer Vision and OCR
opencv-python==4.8.1.78
google-cloud-vision==3.4.5
Pillow==10.0.1

# NLP and Machine Learning
spacy==3.7.2
numpy==1.24.3
scipy==1.11.4

# LLM Providers
openai>=1.0.0
anthropic>=0.8.0

# Adaptive Learning
scikit-learn==1.3.2

# Utilities
requests==2.31.0
aiohttp==3.9.0
aiofiles==23.2.0

# Development and Testing
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
EOF
    fi
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        log_info "Creating .gitignore..."
        cat > .gitignore << 'EOF'
# Environment
.env
.venv/
venv/

# Credentials
credentials/
*.json

# Logs
logs/
*.log

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Uploads and temp files
uploads/
temp/
*.tmp

# Config backups
config_backups/
scanner_config.json

# Docker
.dockerignore
EOF
    fi
    
    log_success "✅ Project structure created"
}

setup_virtual_environment() {
    log_info "🐍 Setting up Python virtual environment..."
    
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
        log_success "✅ Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    log_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    log_info "Downloading spaCy models..."
    # Try to download the large model first, fallback to smaller ones
    if python -m spacy download en_core_web_lg 2>/dev/null; then
        log_success "✅ Downloaded en_core_web_lg (large model with vectors)"
    elif python -m spacy download en_core_web_md 2>/dev/null; then
        log_success "✅ Downloaded en_core_web_md (medium model with vectors)"
    else
        python -m spacy download en_core_web_sm
        log_warn "⚠️  Downloaded en_core_web_sm (small model, no vectors)"
        log_info "For better performance, install larger models:"
        log_info "python -m spacy download en_core_web_lg"
    fi
    
    log_success "✅ Python environment setup completed"
}

setup_docker() {
    log_info "🐳 Setting up Docker configuration..."
    
    # Create Dockerfile for hybrid system
    if [ ! -f "Dockerfile" ]; then
        log_info "Creating Dockerfile for hybrid system..."
        cat > Dockerfile << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_lg || \
    python -m spacy download en_core_web_md || \
    python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads credentials config_backups

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "hybrid_api_app.py"]
EOF
    fi
    
    # Create docker-compose.yml for hybrid system
    if [ ! -f "docker-compose.yml" ]; then
        log_info "Creating docker-compose.yml for hybrid system..."
        cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  hybrid-document-scanner:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./credentials:/app/credentials:ro
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./config_backups:/app/config_backups
      - ./.env:/app/.env:ro
    environment:
      - PYTHONPATH=/app
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - hybrid-document-scanner
    restart: unless-stopped
    profiles:
      - production

  # Optional: Redis for caching (future enhancement)
  redis:
    image: redis:alpine
    restart: unless-stopped
    profiles:
      - cache
      - production

  # Optional: Monitoring with Prometheus
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    profiles:
      - monitoring
EOF
    fi
    
    # Create nginx.conf for hybrid system
    if [ ! -f "nginx.conf" ]; then
        log_info "Creating nginx configuration..."
        cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream hybrid_scanner {
        server hybrid-document-scanner:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;

    server {
        listen 80;
        server_name _;
        client_max_body_size 50M;
        client_body_timeout 60s;
        proxy_read_timeout 300s;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # API endpoints with rate limiting
        location /scan- {
            limit_req zone=upload burst=5 nodelay;
            proxy_pass http://hybrid_scanner;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Other API endpoints
        location / {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://hybrid_scanner;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check (no rate limiting)
        location /health {
            proxy_pass http://hybrid_scanner/health;
            access_log off;
        }

        # Monitoring endpoints
        location /system-status {
            proxy_pass http://hybrid_scanner/system-status;
            access_log off;
        }
    }
}
EOF
    fi
    
    log_success "✅ Docker configuration created"
}

run_comprehensive_tests() {
    log_info "🧪 Running comprehensive system tests..."
    
    source .venv/bin/activate
    
    # Test 1: Basic Python imports
    log_info "Testing basic imports..."
    python3 -c "
import sys
sys.path.append('.')

try:
    # Core dependencies
    import cv2
    import spacy
    import numpy as np
    import scipy
    from google.cloud import vision
    print('✅ Core dependencies imported successfully')
    
    # LLM providers (optional)
    try:
        import openai
        print('✅ OpenAI library available')
    except ImportError:
        print('⚠️  OpenAI library not available')
    
    try:
        import anthropic
        print('✅ Anthropic library available')
    except ImportError:
        print('⚠️  Anthropic library not available')
    
    # Hybrid system components
    import config
    from adaptive_quality_checker import AdaptiveDocumentQualityChecker
    from adaptive_kv_extractor import AdaptiveKeyValueExtractor
    from llm_kv_extractor import LLMKeyValueExtractor
    from hybrid_kv_extractor import HybridKeyValueExtractor
    from hybrid_document_processor import HybridDocumentProcessor
    print('✅ All hybrid system components imported successfully')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "✅ Import tests passed!"
    else
        log_error "❌ Import tests failed!"
        exit 1
    fi
    
    # Test 2: spaCy model
    log_info "Testing spaCy model..."
    python3 -c "
import spacy
try:
    nlp = spacy.load('en_core_web_lg')
    print('✅ en_core_web_lg loaded successfully')
except OSError:
    try:
        nlp = spacy.load('en_core_web_md')
        print('✅ en_core_web_md loaded successfully')
    except OSError:
        try:
            nlp = spacy.load('en_core_web_sm')
            print('⚠️  en_core_web_sm loaded (basic functionality)')
        except OSError:
            print('❌ No spaCy model available')
            exit(1)

# Test basic NLP functionality
doc = nlp('Test document with some text.')
print(f'✅ spaCy processing works: {len(doc)} tokens')
"
    
    # Test 3: Google Cloud credentials
    log_info "Testing Google Cloud credentials..."
    if [ -f "credentials/key.json" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="./credentials/key.json"
        python3 -c "
from google.cloud import vision
try:
    client = vision.ImageAnnotatorClient()
    print('✅ Google Cloud Vision client initialized')
except Exception as e:
    print(f'⚠️  Google Cloud setup issue: {e}')
"
    else
        log_warn "⚠️  Google Cloud credentials not found - OCR will not work"
    fi
    
    # Test 4: Hybrid system initialization
    log_info "Testing hybrid system initialization..."
    python3 -c "
import sys
sys.path.append('.')

try:
    from hybrid_document_processor import HybridDocumentProcessor
    from hybrid_kv_extractor import ExtractionStrategy
    from llm_kv_extractor import LLMProvider
    
    # Test processor initialization
    processor = HybridDocumentProcessor(
        extraction_strategy=ExtractionStrategy.ADAPTIVE_FIRST,
        enable_learning=True
    )
    print('✅ Hybrid processor initialized successfully')
    
    # Test strategy enumeration
    strategies = [s.value for s in ExtractionStrategy]
    print(f'✅ Available strategies: {strategies}')
    
    # Test LLM providers
    providers = [p.value for p in LLMProvider]
    print(f'✅ Available LLM providers: {providers}')
    
except Exception as e:
    print(f'❌ Hybrid system error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
    
    # Test 5: Configuration system
    log_info "Testing adaptive configuration..."
    python3 -c "
import sys
sys.path.append('.')

try:
    from config import adaptive_config
    
    # Test config access
    threshold = adaptive_config.get_adaptive_value('quality_thresholds', 'base_blur_threshold')
    print(f'✅ Adaptive config works: blur_threshold = {threshold}')
    
    # Test config saving
    adaptive_config.save_config()
    print('✅ Config save/load works')
    
except Exception as e:
    print(f'❌ Configuration error: {e}')
    sys.exit(1)
"
    
    # Test 6: Demo scripts
    log_info "Testing demo scripts..."
    if [ -f "test_hybrid_system.py" ]; then
        python3 -c "
import sys
sys.path.append('.')
try:
    exec(open('test_hybrid_system.py').read())
    print('✅ Test script syntax is valid')
except SystemExit:
    print('✅ Test script completed normally')
except Exception as e:
    print(f'⚠️  Test script issue: {e}')
" 2>/dev/null || log_warn "⚠️  Demo script has issues"
    fi
    
    log_success "✅ Comprehensive tests completed!"
}

deploy_hybrid_development() {
    log_info "🚀 Starting hybrid development server..."
    
    source .venv/bin/activate
    
    # Check if port is already in use
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "Port 8000 is already in use. Stopping existing processes..."
        stop_services
        sleep 2
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Start the hybrid server
    log_info "Starting hybrid API server..."
    nohup python3 hybrid_api_app.py > logs/hybrid_dev.log 2>&1 &
    DEV_PID=$!
    
    # Save deployment info
    echo "hybrid_dev" > logs/deploy_mode.txt
    echo $DEV_PID > logs/dev.pid
    
    # Wait for server to start
    log_info "Waiting for server to initialize..."
    sleep 5
    
    # Test health endpoint
    max_attempts=12
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            break
        fi
        log_info "Attempt $attempt/$max_attempts: waiting for server..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "✅ Hybrid development server started successfully!"
        echo ""
        echo "🌐 Server endpoints:"
        echo "   Main API: http://localhost:8000"
        echo "   Documentation: http://localhost:8000/docs"
        echo "   System Status: http://localhost:8000/system-status"
        echo "   Analytics: http://localhost:8000/analytics"
        echo ""
        echo "🧪 Test endpoints:"
        echo "   curl http://localhost:8000/system-status"
        echo "   curl -X POST http://localhost:8000/scan-document -F 'file=@test.jpg'"
        echo ""
        echo "📋 Logs: tail -f logs/hybrid_dev.log"
        echo "🆔 Process ID: $DEV_PID"
    else
        log_error "❌ Hybrid development server failed to start"
        if ps -p $DEV_PID > /dev/null 2>&1; then
            kill $DEV_PID 2>/dev/null || true
        fi
        log_error "Check logs: tail -50 logs/hybrid_dev.log"
        exit 1
    fi
}

deploy_hybrid_docker() {
    log_info "🐳 Building and starting hybrid Docker deployment..."
    
    # Save deployment mode
    echo "hybrid_docker" > logs/deploy_mode.txt
    
    # Build the image
    log_info "Building hybrid system Docker image..."
    docker-compose build hybrid-document-scanner
    
    # Start the service
    log_info "Starting hybrid container..."
    docker-compose up -d hybrid-document-scanner
    
    # Wait for service to be ready
    log_info "Waiting for hybrid service to be ready..."
    sleep 15
    
    # Test health endpoint
    max_attempts=12
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            break
        fi
        log_info "Attempt $attempt/$max_attempts: waiting for service..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "✅ Hybrid Docker deployment successful!"
        echo ""
        echo "🌐 Server running at: http://localhost:8000"
        echo "📚 Documentation: http://localhost:8000/docs"
        echo "🔍 System Status: http://localhost:8000/system-status"
        echo ""
        echo "📋 View logs: docker-compose logs -f hybrid-document-scanner"
        echo "🛑 Stop: docker-compose down"
    else
        log_error "❌ Hybrid Docker deployment failed"
        docker-compose logs hybrid-document-scanner
        exit 1
    fi
}

deploy_hybrid_production() {
    log_info "🏭 Starting hybrid production deployment with Nginx..."
    
    # Save deployment mode
    echo "hybrid_production" > logs/deploy_mode.txt
    
    # Build and start all services
    log_info "Building hybrid production environment..."
    docker-compose build
    
    log_info "Starting production services (Nginx + Hybrid Scanner)..."
    docker-compose --profile production up -d
    
    # Wait for services
    log_info "Waiting for production services to be ready..."
    sleep 20
    
    # Test both endpoints
    success=true
    
    # Test direct API
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "✅ Direct API endpoint is healthy"
    else
        log_error "❌ Direct API endpoint failed"
        success=false
    fi
    
    # Test Nginx proxy
    if curl -f http://localhost/health > /dev/null 2>&1; then
        log_success "✅ Nginx proxy is healthy"
    else
        log_error "❌ Nginx proxy failed"
        success=false
    fi
    
    if [ "$success" = true ]; then
        log_success "✅ Hybrid production deployment successful!"
        echo ""
        echo "🌐 Production endpoints:"
        echo "   Main API: http://localhost"
        echo "   Documentation: http://localhost/docs"
        echo "   System Status: http://localhost/system-status"
        echo "   Analytics: http://localhost/analytics"
        echo ""
        echo "🔧 Management commands:"
        echo "   docker-compose logs -f                    # View all logs"
        echo "   docker-compose logs -f hybrid-document-scanner  # App logs"
        echo "   docker-compose logs -f nginx              # Nginx logs"
        echo "   docker-compose down                       # Stop all services"
    else
        log_error "❌ Hybrid production deployment failed"
        docker-compose logs
        exit 1
    fi
}

run_hybrid_demo() {
    log_info "🎬 Running hybrid system demonstration..."
    
    source .venv/bin/activate
    
    echo ""
    echo "🎯 Available demos:"
    echo "   1. Adaptive learning demonstration"
    echo "   2. Hybrid extraction system test"
    echo "   3. Performance comparison"
    echo ""
    
    read -p "Select demo (1-3, or Enter for all): " demo_choice
    
    case $demo_choice in
        1)
            if [ -f "demo_adaptive_learning.py" ]; then
                log_info "Running adaptive learning demo..."
                python3 demo_adaptive_learning.py
            else
                log_error "demo_adaptive_learning.py not found"
            fi
            ;;
        2)
            if [ -f "test_hybrid_system.py" ]; then
                log_info "Running hybrid system test..."
                python3 test_hybrid_system.py
            else
                log_error "test_hybrid_system.py not found"
            fi
            ;;
        3)
            log_info "Running performance comparison..."
            python3 -c "
import sys
sys.path.append('.')

try:
    from test_hybrid_system import demonstrate_strategy_optimization, show_hybrid_system_benefits
    print('🚀 Running performance comparison...')
    demonstrate_strategy_optimization()
    show_hybrid_system_benefits()
    print('✅ Performance comparison completed!')
except Exception as e:
    print(f'❌ Demo error: {e}')
"
            ;;
        *)
            log_info "Running all demos..."
            for demo_file in "demo_adaptive_learning.py" "test_hybrid_system.py"; do
                if [ -f "$demo_file" ]; then
                    log_info "Running $demo_file..."
                    python3 "$demo_file" || log_warn "Demo $demo_file had issues"
                    echo ""
                fi
            done
            ;;
    esac
    
    log_success "✅ Demo completed!"
}

backup_configuration() {
    log_info "💾 Backing up system configuration..."
    
    backup_dir="config_backups/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration files
    [ -f "scanner_config.json" ] && cp scanner_config.json "$backup_dir/"
    [ -f ".env" ] && cp .env "$backup_dir/env_backup"
    [ -f "requirements.txt" ] && cp requirements.txt "$backup_dir/"
    
    # Backup deployment info
    [ -f "logs/deploy_mode.txt" ] && cp logs/deploy_mode.txt "$backup_dir/"
    
    # Create backup info
    cat > "$backup_dir/backup_info.txt" << EOF
Backup created: $(date)
System version: Hybrid Document Scanner 4.0.0
Deployment mode: $(cat logs/deploy_mode.txt 2>/dev/null || echo "unknown")
Python version: $(python3 --version)
Docker available: $(command -v docker >/dev/null && echo "yes" || echo "no")
EOF
    
    log_success "✅ Configuration backed up to: $backup_dir"
}

show_system_status() {
    log_info "📊 Hybrid System Status Report"
    echo ""
    
    # Basic system info
    echo "=== System Information ==="
    echo "Python version: $(python3 --version 2>&1)"
    echo "Docker available: $(command -v docker >/dev/null && echo "yes" || echo "no")"
    echo "Current directory: $(pwd)"
    echo "Timestamp: $(date)"
    echo ""
    
    # Deployment status
    echo "=== Deployment Status ==="
    if [ -f "logs/deploy_mode.txt" ]; then
        DEPLOY_MODE=$(cat logs/deploy_mode.txt)
        echo "Current deployment mode: $DEPLOY_MODE"
    else
        echo "No deployment mode recorded"
    fi
    echo ""
    
    # Port usage
    echo "=== Port Usage ==="
    echo "Port 8000 (API):"
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        lsof -Pi :8000 -sTCP:LISTEN
    else
        echo "  Not in use"
    fi
    
    echo ""
    echo "Port 80 (Nginx):"
    if lsof -Pi :80 -sTCP:LISTEN -t >/dev/null 2>&1; then
        lsof -Pi :80 -sTCP:LISTEN
    else
        echo "  Not in use"
    fi
    echo ""
    
    # Docker services
    echo "=== Docker Services ==="
    if command -v docker-compose &> /dev/null; then
        docker-compose ps 2>/dev/null || echo "No Docker services running"
    else
        echo "Docker Compose not available"
    fi
    echo ""
    
    # Development server
    echo "=== Development Server ==="
    if [ -f "logs/dev.pid" ]; then
        DEV_PID=$(cat logs/dev.pid)
        if ps -p $DEV_PID > /dev/null 2>&1; then
            echo "✅ Development server running (PID: $DEV_PID)"
        else
            echo "❌ Development server not running (stale PID file)"
            rm -f logs/dev.pid
        fi
    else
        echo "❌ Development server not running"
    fi
    echo ""
    
    # Health checks
    echo "=== Health Checks ==="
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API responding on port 8000"
        
        # Get detailed system status
        if curl -f http://localhost:8000/system-status > /dev/null 2>&1; then
            echo "✅ System status endpoint available"
        fi
    else
        echo "❌ API not responding on port 8000"
    fi
    
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ API responding on port 80 (Nginx)"
    else
        echo "❌ API not responding on port 80 (Nginx)"
    fi
    echo ""
    
    # Component status
    echo "=== Hybrid System Components ==="
    for component in "${HYBRID_COMPONENTS[@]}"; do
        if [ -f "$component" ]; then
            echo "✅ $component"
        else
            echo "❌ $component (missing)"
        fi
    done
    echo ""
    
    # Configuration status
    echo "=== Configuration Status ==="
    echo "Google Cloud credentials: $([ -f "credentials/key.json" ] && echo "✅ configured" || echo "❌ missing")"
    echo "Environment file: $([ -f ".env" ] && echo "✅ present" || echo "❌ missing")"
    
    if [ -f ".env" ]; then
        source .env 2>/dev/null
        echo "OpenAI API: $([ ! -z "$OPENAI_API_KEY" ] && echo "✅ configured" || echo "❌ not set")"
        echo "Anthropic API: $([ ! -z "$ANTHROPIC_API_KEY" ] && echo "✅ configured" || echo "❌ not set")"
        echo "Azure OpenAI: $([ ! -z "$AZURE_OPENAI_API_KEY" ] && echo "✅ configured" || echo "❌ not set")"
    fi
    
    echo "Ollama service: $(curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && echo "✅ running" || echo "❌ not running")"
    echo ""
    
    # Recent logs
    echo "=== Recent Activity ==="
    if [ -f "logs/hybrid_dev.log" ]; then
        echo "Last 3 log entries:"
        tail -3 logs/hybrid_dev.log 2>/dev/null || echo "No recent logs"
    else
        echo "No development logs found"
    fi
}

show_usage() {
    print_banner
    echo ""
    echo "🚀 Hybrid Document Scanner Deployment & Management"
    echo ""
    echo "SETUP COMMANDS:"
    echo "  setup              - Complete initial setup (prerequisites, environment, LLM providers)"
    echo "  setup-llm          - Setup LLM providers only (OpenAI, Anthropic, Ollama)"
    echo "  test               - Run comprehensive system tests"
    echo ""
    echo "DEPLOYMENT COMMANDS:"
    echo "  dev                - Start hybrid development server"
    echo "  docker             - Deploy with Docker (single container)"
    echo "  production         - Deploy with Docker + Nginx (production ready)"
    echo "  gradio             - Launch Gradio UI interface"
    echo ""
    echo "MANAGEMENT COMMANDS:"
    echo "  stop               - Stop all services (dev server, Docker containers)"
    echo "  restart            - Restart services using last deployment mode"
    echo "  kill               - Force kill all related processes"
    echo "  clean              - Clean up containers, images, and build artifacts"
    echo ""
    echo "MONITORING & TESTING:"
    echo "  status             - Show comprehensive system status"
    echo "  logs               - Show recent logs from all services"
    echo "  demo               - Run interactive system demonstration"
    echo "  backup             - Backup current configuration"
    echo ""
    echo "EXAMPLES:"
    echo "  ./deploy.sh setup              # First time setup"
    echo "  ./deploy.sh dev                # Start development server"
    echo "  ./deploy.sh production         # Production deployment"
    echo "  ./deploy.sh status             # Check system status"
    echo "  ./deploy.sh demo               # Run demonstrations"
    echo ""
    echo "HYBRID FEATURES:"
    echo "  ✅ Adaptive + LLM extraction strategies"
    echo "  ✅ Multi-provider LLM support (OpenAI, Anthropic, Ollama)"
    echo "  ✅ Intelligent fallback and strategy optimization"
    echo "  ✅ Continuous learning and parameter adaptation"
    echo "  ✅ Comprehensive analytics and monitoring"
    echo "  ✅ Production-ready Docker deployment"
    echo ""
    echo "📚 Documentation: http://localhost:8000/docs (when running)"
    echo "🔍 System Status: http://localhost:8000/system-status"
    echo "📊 Analytics: http://localhost:8000/analytics"
}

stop_services() {
    log_info "🛑 Stopping all hybrid system services..."
    
    # Stop Docker services
    if command -v docker-compose &> /dev/null; then
        docker-compose down 2>/dev/null || true
        log_info "✅ Docker services stopped"
    fi
    
    # Stop development server using PID file
    if [ -f "logs/dev.pid" ]; then
        DEV_PID=$(cat logs/dev.pid)
        if ps -p $DEV_PID > /dev/null 2>&1; then
            kill $DEV_PID 2>/dev/null || true
            sleep 3
            # Force kill if still running
            if ps -p $DEV_PID > /dev/null 2>&1; then
                kill -9 $DEV_PID 2>/dev/null || true
            fi
            log_info "✅ Development server stopped (PID: $DEV_PID)"
        fi
        rm -f logs/dev.pid
    fi
    
    # Stop Gradio interface using PID file
    if [ -f "logs/gradio.pid" ]; then
        GRADIO_PID=$(cat logs/gradio.pid)
        if ps -p $GRADIO_PID > /dev/null 2>&1; then
            kill $GRADIO_PID 2>/dev/null || true
            sleep 3
            # Force kill if still running
            if ps -p $GRADIO_PID > /dev/null 2>&1; then
                kill -9 $GRADIO_PID 2>/dev/null || true
            fi
            log_info "✅ Gradio interface stopped (PID: $GRADIO_PID)"
        fi
        rm -f logs/gradio.pid
    fi
    
    # Stop any remaining processes
    pkill -f "python.*hybrid_api_app.py" 2>/dev/null || true
    pkill -f "uvicorn.*hybrid_api_app" 2>/dev/null || true
    pkill -f "python.*gradio_app.py" 2>/dev/null || true
    
    # Free ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:80 | xargs kill -9 2>/dev/null || true
    lsof -ti:7860 | xargs kill -9 2>/dev/null || true
    
    log_success "✅ All services stopped"
}

restart_services() {
    log_info "🔄 Restarting hybrid system services..."
    
    # Check last deployment mode
    if [ -f "logs/deploy_mode.txt" ]; then
        DEPLOY_MODE=$(cat logs/deploy_mode.txt)
        log_info "Restarting in mode: $DEPLOY_MODE"
        
        # Stop current services
        stop_services
        sleep 3
        
        # Restart with the same mode
        case $DEPLOY_MODE in
            hybrid_dev)
                deploy_hybrid_development
                ;;
            hybrid_docker)
                deploy_hybrid_docker
                ;;
            hybrid_production)
                deploy_hybrid_production
                ;;
            gradio)
                launch_gradio_interface
                ;;
            *)
                log_warn "Unknown deployment mode: $DEPLOY_MODE"
                log_info "Starting hybrid development server as default..."
                deploy_hybrid_development
                ;;
        esac
    else
        log_warn "No previous deployment mode found. Starting hybrid development server..."
        deploy_hybrid_development
    fi
}

force_kill() {
    log_info "⚡ Force killing all hybrid system processes..."
    
    # Kill all related processes
    pkill -f "python.*hybrid" 2>/dev/null || true
    pkill -f "uvicorn.*hybrid" 2>/dev/null || true
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "uvicorn main:app" 2>/dev/null || true
    
    # Kill processes on ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:80 | xargs kill -9 2>/dev/null || true
    
    # Stop and remove all Docker containers
    docker-compose down --remove-orphans 2>/dev/null || true
    docker stop $(docker ps -q) 2>/dev/null || true
    
    # Clean up PID files
    rm -f logs/dev.pid logs/deploy_mode.txt
    
    log_success "✅ All processes forcefully terminated"
}

clean_deployment() {
    log_info "🧹 Cleaning up hybrid system deployment..."
    
    stop_services
    
    # Remove Docker containers and images
    docker-compose down --rmi all --volumes --remove-orphans 2>/dev/null || true
    
    # Clean up Docker system
    docker system prune -f 2>/dev/null || true
    
    # Clean up logs (keep recent ones)
    find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Clean up temporary files
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_success "✅ Cleanup completed"
}

show_logs() {
    echo "=== Hybrid System Logs ==="
    echo ""
    
    echo "📋 Development Logs:"
    if [ -f "logs/hybrid_dev.log" ]; then
        echo "--- Last 20 lines of hybrid_dev.log ---"
        tail -20 logs/hybrid_dev.log
    else
        echo "No development logs found"
    fi
    
    echo ""
    echo "🐳 Docker Logs:"
    if command -v docker-compose &> /dev/null; then
        echo "--- Hybrid Document Scanner Container ---"
        docker-compose logs --tail=20 hybrid-document-scanner 2>/dev/null || echo "Container not running"
        
        echo ""
        echo "--- Nginx Container ---"
        docker-compose logs --tail=20 nginx 2>/dev/null || echo "Nginx not running"
    else
        echo "Docker Compose not available"
    fi
}

launch_gradio_interface() {
    log_info "🎨 Launching Gradio UI Interface..."
    
    source .venv/bin/activate
    
    # Check if gradio is installed
    if ! python3 -c "import gradio" 2>/dev/null; then
        log_warn "Gradio not found. Installing..."
        pip install gradio==4.44.1 aiofiles==23.2.0 ffmpy==0.3.1 orjson==3.9.15
    fi
    
    # Check if gradio_app.py exists
    if [ ! -f "gradio_app.py" ]; then
        log_error "gradio_app.py not found!"
        log_info "This file contains the modern Gradio interface for the hybrid document scanner."
        exit 1
    fi
    
    # Check if port 7860 is already in use
    if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "Port 7860 is already in use. Stopping existing processes..."
        lsof -ti:7860 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Start the Gradio interface
    log_info "Starting modern Gradio interface on port 7860..."
    nohup python3 gradio_app.py > logs/gradio.log 2>&1 &
    GRADIO_PID=$!
    
    # Save deployment info
    echo "gradio" > logs/deploy_mode.txt
    echo $GRADIO_PID > logs/gradio.pid
    
    # Wait for interface to start
    log_info "Waiting for Gradio interface to initialize..."
    sleep 8
    
    # Test if interface is responding
    max_attempts=10
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:7860 > /dev/null 2>&1; then
            break
        fi
        log_info "Attempt $attempt/$max_attempts: waiting for interface..."
        sleep 3
        attempt=$((attempt + 1))
    done
    
    if curl -f http://localhost:7860 > /dev/null 2>&1; then
        log_success "✅ Gradio interface started successfully!"
        echo ""
        echo "🎨 Gradio UI running at: http://localhost:7860"
        echo "📱 Interface features:"
        echo "   • Upload documents for processing"
        echo "   • Visual bounding box overlay"
        echo "   • Real-time extraction results"
        echo "   • Strategy comparison tools"
        echo "   • Interactive confidence tuning"
        echo ""
        echo "🔧 Controls:"
        echo "   • Upload images via drag & drop"
        echo "   • Adjust extraction strategies"
        echo "   • Tune confidence thresholds"
        echo "   • Toggle OCR block visualization"
        echo ""
        echo "📋 Logs: tail -f logs/gradio.log"
        echo "🆔 Process ID: $GRADIO_PID"
        echo "🛑 Stop: ./deploy.sh stop"
    else
        log_error "❌ Gradio interface failed to start"
        if ps -p $GRADIO_PID > /dev/null 2>&1; then
            kill $GRADIO_PID 2>/dev/null || true
        fi
        log_error "Check logs: tail -50 logs/gradio.log"
        exit 1
    fi
}

# Main execution
main() {
    case "${1:-}" in
        setup)
            print_banner
            check_prerequisites
            create_project_structure
            setup_google_cloud
            setup_llm_providers
            setup_virtual_environment
            setup_docker
            run_comprehensive_tests
            backup_configuration
            log_success "🎉 Hybrid system setup completed!"
            echo ""
            echo "🚀 Next steps:"
            echo "   ./deploy.sh dev        # Start development server"
            echo "   ./deploy.sh demo       # Run system demonstration"
            echo "   ./deploy.sh status     # Check system status"
            ;;
        setup-llm)
            setup_llm_providers
            ;;
        dev)
            deploy_hybrid_development
            ;;
        docker)
            deploy_hybrid_docker
            ;;
        production)
            deploy_hybrid_production
            ;;
        test)
            run_comprehensive_tests
            ;;
        demo)
            run_hybrid_demo
            ;;
        gradio)
            launch_gradio_interface
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        kill)
            force_kill
            ;;
        clean)
            clean_deployment
            ;;
        logs)
            show_logs
            ;;
        status)
            show_system_status
            ;;
        backup)
            backup_configuration
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Make script executable and run
chmod +x "$0"

# Execute main function with all arguments
main "$@"