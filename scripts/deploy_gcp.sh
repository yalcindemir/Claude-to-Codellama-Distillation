#!/bin/bash

# Google Cloud Platform Deployment Script
# for Claude-to-CodeLlama Knowledge Distillation

set -e  # Exit on any error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/gcp_config.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse configuration from YAML (simplified)
get_config() {
    local key=$1
    grep "^  $key:" "$CONFIG_FILE" | sed 's/.*: "\?\([^"]*\)"\?/\1/'
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    # Check environment variables
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        warning "ANTHROPIC_API_KEY not set. Training will fail without it."
    fi
    
    success "Prerequisites check completed"
}

# Create GCP resources
create_resources() {
    log "Creating GCP resources..."
    
    local PROJECT_ID=$(get_config "project_id")
    local REGION=$(get_config "region")
    local ZONE=$(get_config "zone")
    local BUCKET_NAME=$(get_config "bucket_name")
    
    # Set project
    gcloud config set project "$PROJECT_ID"
    
    # Enable required APIs
    log "Enabling required APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable storage.googleapis.com
    gcloud services enable logging.googleapis.com
    gcloud services enable monitoring.googleapis.com
    
    # Create storage bucket
    log "Creating storage bucket: $BUCKET_NAME"
    if ! gsutil ls "gs://$BUCKET_NAME" &> /dev/null; then
        gsutil mb -l "$REGION" "gs://$BUCKET_NAME"
        success "Storage bucket created"
    else
        warning "Storage bucket already exists"
    fi
    
    # Create bucket structure
    log "Creating bucket structure..."
    gsutil -m cp /dev/null "gs://$BUCKET_NAME/datasets/.keep"
    gsutil -m cp /dev/null "gs://$BUCKET_NAME/models/.keep"
    gsutil -m cp /dev/null "gs://$BUCKET_NAME/logs/.keep"
    gsutil -m cp /dev/null "gs://$BUCKET_NAME/checkpoints/.keep"
    
    success "GCP resources created"
}

# Deploy training instance
deploy_instance() {
    log "Deploying training instance..."
    
    local PROJECT_ID=$(get_config "project_id")
    local ZONE=$(get_config "zone")
    local INSTANCE_NAME=$(get_config "instance_name")
    local MACHINE_TYPE=$(get_config "machine_type")
    local IMAGE_FAMILY="pytorch-latest-gpu"
    local IMAGE_PROJECT="deeplearning-platform-release"
    
    # Create startup script
    cat > /tmp/startup-script.sh << 'EOF'
#!/bin/bash
cd /home/jupyter
git clone https://github.com/yeditepe/claude-to-codellama-distillation.git
cd claude-to-codellama-distillation
pip install -r requirements.txt

# Set up environment
echo "export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY}'" >> ~/.bashrc
echo "export WANDB_API_KEY='${WANDB_API_KEY}'" >> ~/.bashrc

# Create directories
mkdir -p /mnt/data/{datasets,models,logs,checkpoints}

# Download any required models/data
# (This would be customized based on your needs)

echo "Setup completed at $(date)" > /tmp/setup-complete.log
EOF
    
    # Create instance
    log "Creating compute instance: $INSTANCE_NAME"
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="100GB" \
        --boot-disk-type="pd-ssd" \
        --maintenance-policy="TERMINATE" \
        --preemptible \
        --metadata-from-file startup-script=/tmp/startup-script.sh \
        --metadata="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY,WANDB_API_KEY=$WANDB_API_KEY" \
        --scopes="https://www.googleapis.com/auth/cloud-platform"
    
    # Clean up
    rm /tmp/startup-script.sh
    
    success "Training instance deployed"
    
    # Get instance IP
    local EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    
    log "Instance external IP: $EXTERNAL_IP"
    log "SSH command: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    log "Jupyter URL: http://$EXTERNAL_IP:8080"
}

# Monitor training
monitor_training() {
    log "Monitoring training progress..."
    
    local PROJECT_ID=$(get_config "project_id")
    local ZONE=$(get_config "zone")
    local INSTANCE_NAME=$(get_config "instance_name")
    
    # Check if instance exists
    if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &> /dev/null; then
        error "Instance $INSTANCE_NAME not found"
        exit 1
    fi
    
    # Show instance status
    log "Instance status:"
    gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format="table(status,machineType.basename(),scheduling.preemptible)"
    
    # Show logs
    log "Recent logs:"
    gcloud compute instances get-serial-port-output "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --start="-50"
    
    # Show GPU utilization if available
    log "GPU utilization:"
    gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --command="nvidia-smi" || warning "Could not get GPU info"
}

# Stop training instance
stop_instance() {
    log "Stopping training instance..."
    
    local PROJECT_ID=$(get_config "project_id")
    local ZONE=$(get_config "zone")
    local INSTANCE_NAME=$(get_config "instance_name")
    
    gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE"
    success "Instance stopped"
}

# Delete all resources
cleanup() {
    log "Cleaning up all resources..."
    
    local PROJECT_ID=$(get_config "project_id")
    local ZONE=$(get_config "zone")
    local INSTANCE_NAME=$(get_config "instance_name")
    local BUCKET_NAME=$(get_config "bucket_name")
    
    # Delete instance
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &> /dev/null; then
        log "Deleting instance: $INSTANCE_NAME"
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
    fi
    
    # Delete bucket (ask for confirmation)
    if gsutil ls "gs://$BUCKET_NAME" &> /dev/null; then
        read -p "Delete storage bucket gs://$BUCKET_NAME? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gsutil -m rm -r "gs://$BUCKET_NAME"
            success "Storage bucket deleted"
        fi
    fi
    
    success "Cleanup completed"
}

# Show help
show_help() {
    cat << EOF
GCP Deployment Script for Claude-to-CodeLlama Knowledge Distillation

Usage: $0 [COMMAND]

Commands:
    deploy      Deploy training instance and resources
    monitor     Monitor training progress
    stop        Stop training instance
    cleanup     Delete all resources
    help        Show this help message

Examples:
    $0 deploy       # Deploy everything
    $0 monitor      # Check training status
    $0 stop         # Stop instance
    $0 cleanup      # Delete all resources

Environment Variables:
    ANTHROPIC_API_KEY   Claude API key (required)
    WANDB_API_KEY      Weights & Biases API key (optional)

Configuration:
    Edit configs/gcp_config.yml to customize deployment settings
EOF
}

# Main script logic
main() {
    case "${1:-}" in
        "deploy")
            check_prerequisites
            create_resources
            deploy_instance
            ;;
        "monitor")
            monitor_training
            ;;
        "stop")
            stop_instance
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        "")
            error "No command specified"
            show_help
            exit 1
            ;;
        *)
            error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"