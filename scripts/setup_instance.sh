#!/bin/bash

# Instance Setup Script for Claude-to-CodeLlama Knowledge Distillation
# This script sets up the environment on a new compute instance

set -e  # Exit on any error

echo "🚀 Setting up Claude-to-CodeLlama Knowledge Distillation Environment"
echo "=================================================================="

# Configuration
PROJECT_DIR="/home/$(whoami)/claude_to_codellama_distillation"
DATA_DIR="/mnt/data"
LOG_FILE="/tmp/setup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if running as root
check_user() {
    if [ "$EUID" -eq 0 ]; then
        warning "Running as root. Consider using a non-root user for better security."
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            git \
            wget \
            curl \
            vim \
            htop \
            tmux \
            unzip \
            build-essential \
            python3-dev \
            python3-pip \
            python3-venv
    elif command -v yum &> /dev/null; then
        sudo yum update -y
        sudo yum install -y \
            git \
            wget \
            curl \
            vim \
            htop \
            tmux \
            unzip \
            gcc \
            python3-devel \
            python3-pip
    else
        warning "Unknown package manager. Please install dependencies manually."
    fi
    
    success "System packages updated"
}

# Setup Python environment
setup_python() {
    log "Setting up Python environment..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log "Python version: $PYTHON_VERSION"
    
    if [[ "$(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.8" ]]; then
        error "Python 3.8 or higher required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install virtual environment if not available
    python3 -m pip install virtualenv
    
    success "Python environment ready"
}

# Setup NVIDIA drivers and CUDA (if needed)
setup_nvidia() {
    log "Checking NVIDIA GPU..."
    
    if lspci | grep -i nvidia &> /dev/null; then
        log "NVIDIA GPU detected"
        
        # Check if nvidia-smi is available
        if command -v nvidia-smi &> /dev/null; then
            log "NVIDIA drivers already installed"
            nvidia-smi
        else
            warning "NVIDIA drivers not found. Installing..."
            
            # This assumes Ubuntu/Debian with apt
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y nvidia-driver-470
                sudo reboot  # Reboot required for driver installation
            else
                warning "Please install NVIDIA drivers manually"
            fi
        fi
    else
        warning "No NVIDIA GPU detected. Training will use CPU (very slow)."
    fi
}

# Clone repository
clone_repository() {
    log "Cloning repository..."
    
    if [ -d "$PROJECT_DIR" ]; then
        warning "Project directory already exists. Updating..."
        cd "$PROJECT_DIR"
        git pull
    else
        git clone https://github.com/yeditepe/claude-to-codellama-distillation.git "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi
    
    success "Repository ready"
}

# Setup virtual environment
setup_venv() {
    log "Setting up virtual environment..."
    
    cd "$PROJECT_DIR"
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Upgrade pip in virtual environment
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        log "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install project requirements
    log "Installing project requirements..."
    pip install -r requirements.txt
    
    # Install development requirements
    pip install -e ".[dev]"
    
    success "Virtual environment ready"
}

# Setup data directories
setup_directories() {
    log "Setting up data directories..."
    
    # Create data directories
    mkdir -p "$DATA_DIR"/{datasets,models,logs,checkpoints,cache}
    mkdir -p "$PROJECT_DIR"/{data,models,logs,cache}
    
    # Create symlinks if using external data directory
    if [ "$DATA_DIR" != "$PROJECT_DIR/data" ]; then
        ln -sf "$DATA_DIR/datasets" "$PROJECT_DIR/data/datasets"
        ln -sf "$DATA_DIR/models" "$PROJECT_DIR/models"
        ln -sf "$DATA_DIR/logs" "$PROJECT_DIR/logs"
        ln -sf "$DATA_DIR/cache" "$PROJECT_DIR/cache"
    fi
    
    # Set permissions
    chmod 755 "$PROJECT_DIR"/scripts/*.sh
    
    success "Directories ready"
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Create environment file
    cat > "$PROJECT_DIR/.env" << EOF
# Claude API Configuration
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}

# Weights & Biases Configuration
WANDB_API_KEY=${WANDB_API_KEY:-}
WANDB_PROJECT=claude-to-codellama-distillation

# Hugging Face Configuration
HF_HOME=$PROJECT_DIR/cache/huggingface
TRANSFORMERS_CACHE=$PROJECT_DIR/cache/transformers

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0

# Python Configuration
PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
EOF
    
    # Add to bashrc
    if ! grep -q "source $PROJECT_DIR/.env" ~/.bashrc; then
        echo "source $PROJECT_DIR/.env" >> ~/.bashrc
    fi
    
    # Source environment
    source "$PROJECT_DIR/.env"
    
    success "Environment variables configured"
}

# Run basic tests
run_tests() {
    log "Running basic tests..."
    
    cd "$PROJECT_DIR"
    source venv/bin/activate
    
    # Test Python imports
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    
    # Test CUDA availability
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    # Run unit tests
    if [ -f "requirements.txt" ] && grep -q "pytest" requirements.txt; then
        log "Running unit tests..."
        python -m pytest tests/ -v --tb=short || warning "Some tests failed"
    fi
    
    success "Basic tests completed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create monitoring script
    cat > "$PROJECT_DIR/monitor.sh" << 'EOF'
#!/bin/bash
# Simple monitoring script

echo "=== System Status ==="
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo "Disk usage:"
df -h /
echo
echo "Memory usage:"
free -h
echo
echo "GPU status:"
nvidia-smi || echo "No GPU available"
echo
echo "Running processes:"
ps aux | grep python | head -5
EOF
    
    chmod +x "$PROJECT_DIR/monitor.sh"
    
    # Setup log rotation
    sudo tee /etc/logrotate.d/claude-distillation << EOF
$PROJECT_DIR/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
    
    success "Monitoring setup completed"
}

# Create quick start script
create_quickstart() {
    log "Creating quick start script..."
    
    cat > "$PROJECT_DIR/quickstart.sh" << 'EOF'
#!/bin/bash
# Quick start script for Claude-to-CodeLlama Knowledge Distillation

echo "🚀 Claude-to-CodeLlama Knowledge Distillation Quick Start"
echo "========================================================"

# Check if in project directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run setup_instance.sh first"
    exit 1
fi

# Source environment variables
if [ -f ".env" ]; then
    source .env
    echo "✅ Environment variables loaded"
fi

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY not set. Please set it in .env file"
    exit 1
fi

echo "✅ API key configured"

# Show available commands
echo
echo "Available commands:"
echo "  1. Generate dataset:     python src/dataset_generator.py"
echo "  2. Train model:          python src/distillation_trainer.py"
echo "  3. Evaluate model:       python src/evaluation_system.py"
echo "  4. Run full pipeline:    ./scripts/run_full_pipeline.sh"
echo "  5. Monitor system:       ./monitor.sh"
echo
echo "Ready to start! 🎉"
EOF
    
    chmod +x "$PROJECT_DIR/quickstart.sh"
    
    success "Quick start script created"
}

# Main setup function
main() {
    log "Starting instance setup..."
    
    check_user
    update_system
    setup_python
    setup_nvidia
    clone_repository
    setup_venv
    setup_directories
    setup_environment
    run_tests
    setup_monitoring
    create_quickstart
    
    success "Instance setup completed!"
    
    echo
    echo "🎉 Setup Complete!"
    echo "=================="
    echo
    echo "Next steps:"
    echo "1. Set your API keys in $PROJECT_DIR/.env"
    echo "2. Run: cd $PROJECT_DIR"
    echo "3. Run: ./quickstart.sh"
    echo "4. Follow the on-screen instructions"
    echo
    echo "For help: cat README.md"
    echo "Log file: $LOG_FILE"
}

# Run main function
main "$@"