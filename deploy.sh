#!/bin/bash

# Document Scanner Deployment Script
# This script sets up and deploys the document scanner application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="document-scanner"
PYTHON_VERSION="3.9"
REQUIRED_PACKAGES=("curl" "docker" "docker-compose")

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running as root for system packages
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root. This is not recommended for production."
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        log_info "Python version: $PYTHON_VER"
    else
        log_error "Python 3 is required but not installed."
        exit 1
    fi
    
    # Check required packages
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! command -v $pkg &> /dev/null; then
            log_error "$pkg is required but not installed."
            echo "Please install it using: sudo apt-get install $pkg"
            exit 1
        fi
    done
    
    log_info "Prerequisites check passed!"
}

setup_google_cloud() {
    log_info "Setting up Google Cloud credentials..."
    
    if [ ! -f "credentials/key.json" ]; then
        log_warn "Google Cloud credentials not found."
        echo "Please follow these steps:"
        echo "1. Go to Google Cloud Console"
        echo "2. Create a service account with Vision API permissions"
        echo "3. Download the JSON key file"
        echo "4. Place it at credentials/key.json"
        echo ""
        read -p "Have you placed the credentials file? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Google Cloud credentials are required."
            exit 1
        fi
    fi
    
    if [ ! -f ".env" ]; then
        log_info "Creating .env file..."
        read -p "Enter your Google Cloud Project ID: " PROJECT_ID
        cat > .env << EOF
PROJECT_ID=$PROJECT_ID
GOOGLE_APPLICATION_CREDENTIALS=./credentials/key.json
EOF
        log_info ".env file created."
    fi
}

create_project_structure() {
    log_info "Creating project structure..."
    
    # Create directories
    mkdir -p credentials
    mkdir -p uploads
    mkdir -p logs
    mkdir -p test_images
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "requirements.txt" ]; then
        log_info "Creating requirements.txt..."
        cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
opencv-python==4.8.1.78
google-cloud-vision==3.4.5
spacy==3.7.2
numpy==1.24.3
scipy==1.11.4
Pillow==10.0.1
python-multipart==0.0.6
requests==2.31.0
EOF
    fi
    
    log_info "Project structure created."
}

setup_virtual_environment() {
    log_info "Setting up Python virtual environment..."
    
    if [ ! -d ".venv" ]; then
        python3 -m .venv .venv
        log_info "Virtual environment created."
    fi
    
    source .venv/bin/activate
    
    log_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    log_info "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    
    log_info "Python setup completed."
}

setup_docker() {
    log_info "Setting up Docker configuration..."
    
    # Create docker-compose.yml if it doesn't exist
    if [ ! -f "docker-compose.yml" ]; then
        log_info "Creating docker-compose.yml..."
        cat > docker-compose.yml << EOF
version: '3.8'

services:
  document-scanner:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./credentials:/app/credentials:ro
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - document-scanner
    restart: unless-stopped
    profiles:
      - production
EOF
    fi
    
    # Create nginx.conf if it doesn't exist
    if [ ! -f "nginx.conf" ]; then
        log_info "Creating nginx configuration..."
        cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream document_scanner {
        server document-scanner:8000;
    }

    server {
        listen 80;
        client_max_body_size 10M;
        client_body_timeout 60s;
        proxy_read_timeout 60s;

        location / {
            proxy_pass http://document_scanner;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://document_scanner/health;
            access_log off;
        }
    }
}
EOF
    fi
    
    log_info "Docker configuration created."
}

run_tests() {
    log_info "Running basic tests..."
    
    # Test Python imports
    source .venv/bin/activate
    python3 -c "
import cv2
import spacy
import numpy as np
from google.cloud import vision
print('✅ All imports successful')
"
    
    if [ $? -eq 0 ]; then
        log_info "Import tests passed!"
    else
        log_error "Import tests failed!"
        exit 1
    fi
}

deploy_development() {
    log_info "Starting development server..."
    
    source .venv/bin/activate
    
    # Check if port is already in use
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "Port 8000 is already in use. Stopping existing processes..."
        stop_services
        sleep 2
    fi
    
    # Start the server in background
    nohup uvicorn main:app --reload --host 0.0.0.0 --port 8000 > logs/dev.log 2>&1 &
    DEV_PID=$!
    
    # Save deployment mode
    echo "dev" > logs/deploy_mode.txt
    
    # Wait a moment for server to start
    sleep 3
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✅ Development server started successfully!"
        log_info "Server running at: http://localhost:8000"
        log_info "API docs at: http://localhost:8000/docs"
        log_info "Process ID: $DEV_PID"
        echo $DEV_PID > logs/dev.pid
    else
        log_error "❌ Development server failed to start"
        kill $DEV_PID 2>/dev/null || true
        log_error "Check logs with: tail -50 logs/dev.log"
        exit 1
    fi
}

deploy_docker() {
    log_info "Building and starting Docker containers..."
    
    # Save deployment mode
    echo "docker" > logs/deploy_mode.txt
    
    # Build the image
    docker-compose build
    
    # Start the services
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✅ Docker deployment successful!"
        log_info "Server running at: http://localhost:8000"
        log_info "View logs with: docker-compose logs -f"
    else
        log_error "❌ Docker deployment failed"
        docker-compose logs
        exit 1
    fi
}

deploy_production() {
    log_info "Starting production deployment with Nginx..."
    
    # Save deployment mode
    echo "production" > logs/deploy_mode.txt
    
    # Start all services including nginx
    docker-compose --profile production up -d
    
    # Wait for services
    sleep 15
    
    # Test both endpoints
    if curl -f http://localhost/health > /dev/null 2>&1; then
        log_info "✅ Production deployment successful!"
        log_info "Server running at: http://localhost"
        log_info "API docs at: http://localhost/docs"
    else
        log_error "❌ Production deployment failed"
        docker-compose logs
        exit 1
    fi
}

restart_services() {
    log_info "Restarting services..."
    
    # Check last deployment mode
    if [ -f "logs/deploy_mode.txt" ]; then
        DEPLOY_MODE=$(cat logs/deploy_mode.txt)
        log_info "Last deployment mode: $DEPLOY_MODE"
        
        # Stop current services
        stop_services
        sleep 2
        
        # Restart with the same mode
        case $DEPLOY_MODE in
            dev)
                deploy_development
                ;;
            docker)
                deploy_docker
                ;;
            production)
                deploy_production
                ;;
            *)
                log_warn "Unknown deployment mode: $DEPLOY_MODE"
                log_info "Starting development server as default..."
                deploy_development
                ;;
        esac
    else
        log_warn "No previous deployment mode found. Starting development server..."
        deploy_development
    fi
}

force_kill() {
    log_info "Force killing all related processes..."
    
    # Kill all uvicorn processes
    pkill -f "uvicorn main:app" 2>/dev/null || true
    
    # Kill all python processes related to our app
    pkill -f "python.*main.py" 2>/dev/null || true
    
    # Kill processes on port 8000
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    
    # Kill processes on port 80 (if nginx)
    lsof -ti:80 | xargs kill -9 2>/dev/null || true
    
    # Stop and remove all docker containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Clean up PID files
    rm -f logs/dev.pid logs/deploy_mode.txt
    
    log_info "All processes forcefully terminated."
}

show_usage() {
    echo "Document Scanner Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup        - Initial setup (create structure, install dependencies)"
    echo "  dev          - Deploy development server"
    echo "  docker       - Deploy with Docker (single container)"
    echo "  production   - Deploy with Docker + Nginx (production ready)"
    echo "  test         - Run basic tests"
    echo "  stop         - Stop all services (dev server, Docker containers)"
    echo "  restart      - Restart services (stop + start last used mode)"
    echo "  kill         - Force kill all related processes"
    echo "  clean        - Clean up containers and images"
    echo "  logs         - Show logs"
    echo "  status       - Show service status"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # First time setup"
    echo "  $0 dev        # Start development server"
    echo "  $0 stop       # Stop all running services"
    echo "  $0 restart    # Restart last used deployment mode"
    echo "  $0 production # Production deployment"
}

stop_services() {
    log_info "Stopping services..."
    
    # Stop Docker services
    if command -v docker-compose &> /dev/null; then
        docker-compose down 2>/dev/null || true
        log_info "Docker services stopped."
    fi
    
    # Stop development server using PID file
    if [ -f "logs/dev.pid" ]; then
        DEV_PID=$(cat logs/dev.pid)
        if ps -p $DEV_PID > /dev/null 2>&1; then
            kill $DEV_PID 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if ps -p $DEV_PID > /dev/null 2>&1; then
                kill -9 $DEV_PID 2>/dev/null || true
            fi
            log_info "Development server stopped (PID: $DEV_PID)."
        else
            log_info "Development server was not running (stale PID file)."
        fi
        rm logs/dev.pid
    fi
    
    # Stop any uvicorn processes on port 8000 (fallback)
    UVICORN_PIDS=$(ps aux | grep "uvicorn main:app" | grep -v grep | awk '{print $2}')
    if [ ! -z "$UVICORN_PIDS" ]; then
        log_info "Found running uvicorn processes: $UVICORN_PIDS"
        echo $UVICORN_PIDS | xargs kill 2>/dev/null || true
        sleep 2
        # Force kill any remaining
        echo $UVICORN_PIDS | xargs kill -9 2>/dev/null || true
        log_info "Stopped uvicorn processes."
    fi
    
    # Stop any processes using port 8000
    PORT_PIDS=$(lsof -ti:8000 2>/dev/null || true)
    if [ ! -z "$PORT_PIDS" ]; then
        log_info "Found processes using port 8000: $PORT_PIDS"
        echo $PORT_PIDS | xargs kill 2>/dev/null || true
        sleep 2
        # Force kill any remaining
        echo $PORT_PIDS | xargs kill -9 2>/dev/null || true
        log_info "Freed port 8000."
    fi
    
    log_info "All services stopped."
}

clean_deployment() {
    log_info "Cleaning up deployment..."
    
    stop_services
    
    # Remove Docker containers and images
    docker-compose down --rmi all --volumes --remove-orphans 2>/dev/null || true
    
    # Clean up build artifacts
    docker system prune -f 2>/dev/null || true
    
    log_info "Cleanup completed."
}

show_logs() {
    echo "=== Docker Logs ==="
    docker-compose logs --tail=50
    
    echo ""
    echo "=== Development Logs ==="
    if [ -f "logs/dev.log" ]; then
        tail -50 logs/dev.log
    else
        echo "No development logs found."
    fi
}

show_status() {
    log_info "Service Status:"
    
    echo ""
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
    echo "=== Docker Services ==="
    if command -v docker-compose &> /dev/null; then
        docker-compose ps
    else
        echo "Docker Compose not available"
    fi
    
    echo ""
    echo "=== Development Server ==="
    if [ -f "logs/dev.pid" ]; then
        DEV_PID=$(cat logs/dev.pid)
        if ps -p $DEV_PID > /dev/null 2>&1; then
            echo "✅ Development server running (PID: $DEV_PID)"
        else
            echo "❌ Development server not running (stale PID file)"
            rm logs/dev.pid
        fi
    else
        echo "❌ Development server not running"
    fi
    
    echo ""
    echo "=== Deployment Mode ==="
    if [ -f "logs/deploy_mode.txt" ]; then
        DEPLOY_MODE=$(cat logs/deploy_mode.txt)
        echo "Last deployment mode: $DEPLOY_MODE"
    else
        echo "No deployment mode recorded"
    fi
    
    echo ""
    echo "=== Health Check ==="
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is responding on port 8000"
    else
        echo "❌ API is not responding on port 8000"
    fi
    
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ API is responding on port 80 (Nginx)"
    else
        echo "❌ API is not responding on port 80 (Nginx)"
    fi
    
    echo ""
    echo "=== Process List ==="
    echo "Uvicorn processes:"
    ps aux | grep "uvicorn main:app" | grep -v grep || echo "  None found"
    
    echo ""
    echo "Python processes with main.py:"
    ps aux | grep "python.*main.py" | grep -v grep || echo "  None found"
}

# Main execution
main() {
    case "${1:-}" in
        setup)
            check_prerequisites
            create_project_structure
            setup_google_cloud
            setup_virtual_environment
            setup_docker
            run_tests
            log_info "✅ Setup completed! You can now run: $0 dev"
            ;;
        dev)
            deploy_development
            ;;
        docker)
            deploy_docker
            ;;
        production)
            deploy_production
            ;;
        test)
            run_tests
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
            show_status
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
