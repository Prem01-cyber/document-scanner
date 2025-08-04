I'll create a comprehensive deployment script that handles the entire hybrid document scanner system setup, configuration, and deployment. Let me first check the existing deploy.sh file and then create a centralized deployment solution.
Read file: deploy.sh

I'll update the existing deploy.sh script to handle the complete hybrid document scanner system with all the new components we've created. This will be a comprehensive deployment script that manages the entire system.
Ran tool

Perfect! I've created a comprehensive centralized deployment script that handles the entire hybrid document scanner system. This updated `deploy.sh` script is a complete management solution for your hybrid system.

## 🎯 **Key Features of the Updated Deploy Script**

### **🚀 Complete Hybrid System Support**
- ✅ All hybrid components (adaptive + LLM extraction)
- ✅ Multi-provider LLM setup (OpenAI, Anthropic, Ollama, Azure)
- ✅ Intelligent fallback configuration
- ✅ Adaptive learning system setup

### **🔧 Advanced Deployment Options**
```bash
# Development (fastest for testing)
./deploy.sh dev

# Docker (containerized)
./deploy.sh docker  

# Production (Docker + Nginx + monitoring)
./deploy.sh production
```

### **📊 Comprehensive Monitoring**
```bash
# System status dashboard
./deploy.sh status

# Real-time logs
./deploy.sh logs

# Interactive demos
./deploy.sh demo
```

## 🎬 **Quick Start Guide**

### **1. First Time Setup**
```bash
# Complete system setup (one command does everything)
./deploy.sh setup

# This will:
# ✅ Check prerequisites
# ✅ Setup Google Cloud Vision API
# ✅ Configure LLM providers (OpenAI, Anthropic, Ollama)
# ✅ Create Python environment
# ✅ Install all dependencies
# ✅ Download spaCy models
# ✅ Setup Docker configuration
# ✅ Run comprehensive tests
# ✅ Backup initial configuration
```

### **2. Development Testing**
```bash
# Start development server
./deploy.sh dev

# Server will be available at:
# 🌐 http://localhost:8000
# 📚 http://localhost:8000/docs
# 🔍 http://localhost:8000/system-status
```

### **3. Production Deployment**
```bash
# Production with Nginx reverse proxy
./deploy.sh production

# Available at:
# 🌐 http://localhost (main site)
# 📚 http://localhost/docs (API docs)
# 📊 http://localhost/analytics (performance)
```

### **4. System Management**
```bash
# Check what's running
./deploy.sh status

# View recent logs
./deploy.sh logs

# Restart services
./deploy.sh restart

# Stop everything
./deploy.sh stop

# Clean up completely
./deploy.sh clean
```

## 🧪 **Interactive Demo System**

```bash
# Run built-in demonstrations
./deploy.sh demo

# Choose from:
# 1. Adaptive learning demonstration
# 2. Hybrid extraction system test
# 3. Performance comparison
```

## ⚙️ **Configuration Management**

### **LLM Provider Setup**
The script intelligently guides you through setting up LLM providers:

```bash
# Setup LLM providers only
./deploy.sh setup-llm

# Interactive setup for:
# 🤖 OpenAI GPT-4 (best quality)
# 🧠 Anthropic Claude (fast & cost-effective)  
# 🏢 Azure OpenAI (enterprise)
# 🏠 Ollama (local, free)
```

### **Automatic Configuration Backup**
```bash
# Backup current configuration
./deploy.sh backup

# Creates timestamped backup in config_backups/
# Includes: .env, scanner_config.json, requirements.txt
```

## 📊 **Advanced Features**

### **Health Monitoring**
The script includes comprehensive health checks:
- ✅ API endpoint availability
- ✅ LLM provider connectivity  
- ✅ Google Cloud Vision API
- ✅ Docker container status
- ✅ Port usage monitoring
- ✅ Process tracking

### **Intelligent Error Recovery**
```bash
# Force kill all processes if something is stuck
./deploy.sh kill

# Clean up and restart automatically
./deploy.sh clean && ./deploy.sh restart
```

### **Multi-Environment Support**
The script remembers your deployment mode and can restart in the same configuration:

```bash
# Remembers whether you used dev, docker, or production
./deploy.sh restart

# Always restarts in the same mode you last used
```

## 🎯 **Usage Examples**

### **Complete New Installation**
```bash
# Clone/download your project
cd document-scanner

# One-command setup
./deploy.sh setup

# Start development
./deploy.sh dev

# Test the system
curl http://localhost:8000/system-status
```

### **Production Deployment**
```bash
# Setup (if not done)
./deploy.sh setup

# Deploy to production
./deploy.sh production

# Monitor status
./deploy.sh status

# View analytics
curl http://localhost/analytics
```

### **Development Workflow**
```bash
# Start development
./deploy.sh dev

# Run tests/demos
./deploy.sh demo

# Check logs
./deploy.sh logs

# Restart after changes
./deploy.sh restart
```

## 🛡️ **Error Handling & Recovery**

The script includes robust error handling:

- **🔍 Prerequisites checking** - Verifies all requirements
- **⚠️ Graceful failures** - Clear error messages with solutions
- **🔄 Automatic recovery** - Smart restart and cleanup
- **📋 Comprehensive logging** - Detailed logs for troubleshooting
- **💾 Configuration backup** - Automatic backup before changes

## 📋 **Full Command Reference**

```bash
# SETUP
./deploy.sh setup              # Complete system setup
./deploy.sh setup-llm          # LLM providers only
./deploy.sh test               # Comprehensive tests

# DEPLOYMENT  
./deploy.sh dev                # Development server
./deploy.sh docker             # Docker deployment
./deploy.sh production         # Production with Nginx

# MANAGEMENT
./deploy.sh stop               # Stop all services
./deploy.sh restart            # Restart services  
./deploy.sh kill               # Force kill processes
./deploy.sh clean              # Clean up everything

# MONITORING
./deploy.sh status             # System status
./deploy.sh logs               # View logs
./deploy.sh demo               # Run demonstrations
./deploy.sh backup             # Backup configuration
```

---

## 🎉 **What This Achieves**

This centralized deployment script provides:

✅ **One-command setup** for the entire hybrid system  
✅ **Multi-environment deployment** (dev/docker/production)  
✅ **Intelligent LLM provider configuration**  
✅ **Comprehensive testing and validation**  
✅ **Advanced monitoring and debugging**  
✅ **Automatic error recovery**  
✅ **Built-in demonstration system**  
✅ **Production-ready configuration**  

You now have a **enterprise-grade deployment system** that can handle everything from initial setup to production deployment with a single script! 🚀