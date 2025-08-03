# Dockerfile for Document Scanner Application
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other requirements
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY main.py .
COPY test_client.py .

# Create directory for credentials
RUN mkdir -p /app/credentials

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

---

# docker-compose.yml for complete setup
version: '3.8'

services:
  document-scanner:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./credentials:/app/credentials:ro
      - ./uploads:/app/uploads
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json
      - PROJECT_ID=${PROJECT_ID}
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

---

# nginx.conf for load balancing and optimization
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
            
            # CORS headers
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
            add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        }

        location /health {
            proxy_pass http://document_scanner/health;
            access_log off;
        }
    }
}

---

# .dockerignore
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
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Credentials
*.json
credentials/
key.json

# Test files
test_images/
uploads/
*.jpg
*.png
*.pdf

# Logs
*.log
logs/
