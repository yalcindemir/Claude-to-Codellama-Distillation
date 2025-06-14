# Deployment Guide

Complete guide for deploying Claude-to-CodeLlama Knowledge Distillation in various environments.

## Overview

This guide covers deployment options from development to production, including:
- Local development setup
- Google Colab for experimentation
- Google Cloud Platform for production
- Docker containerization
- API service deployment

## 🏠 Local Development

### Prerequisites

- Python 3.8+
- 16GB+ RAM (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- 50GB+ free disk space

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yalcindemir/claude-to-codellama-distillation.git
cd claude-to-codellama-distillation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run tests
python -m pytest tests/ -v

# Start training
export ANTHROPIC_API_KEY='your-api-key'
./scripts/run_full_pipeline.sh
```

### Development Dependencies

```bash
# Install development tools
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run code quality checks
black src/
flake8 src/
mypy src/
```

## 📱 Google Colab

Perfect for experimentation and learning.

### Free Tier

- **Pros**: No cost, easy setup
- **Cons**: Limited GPU hours, session timeouts
- **Best for**: Testing, small datasets (<1K examples)

### Colab Pro ($9.99/month)

- **Pros**: Better GPUs, longer sessions, priority access
- **Cons**: Monthly subscription
- **Best for**: Medium datasets (1K-10K examples)

### Setup Instructions

1. Open the Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yalcindemir/claude-to-codellama-distillation/blob/main/notebooks/Claude_Code_Model_Colab.ipynb)

2. Mount Google Drive for persistent storage:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Set API keys:
```python
import os
from getpass import getpass

os.environ['ANTHROPIC_API_KEY'] = getpass('Enter Claude API key: ')
```

4. Follow the notebook cells step by step

### Colab Optimization Tips

```python
# Colab-specific configuration
COLAB_CONFIG = {
    'target_size': 1000,      # Small dataset for demo
    'num_epochs': 1,          # Quick training
    'batch_size': 2,          # Small batch for memory
    'max_length': 1024,       # Shorter sequences
    'use_4bit': True,         # QLoRA quantization
    'lora_r': 8,              # Smaller LoRA rank
}
```

## ☁️ Google Cloud Platform

Recommended for production deployments.

### Instance Types

#### Development
```yaml
machine_type: n1-standard-4
gpu: nvidia-tesla-t4 (1x)
cost: ~$0.50/hour
memory: 15GB
use_case: Development, testing
```

#### Production
```yaml
machine_type: n1-standard-8
gpu: nvidia-tesla-v100 (1x)
cost: ~$2.50/hour
memory: 30GB
use_case: Full training runs
```

#### High Performance
```yaml
machine_type: n1-standard-16
gpu: nvidia-tesla-v100 (2x)
cost: ~$5.00/hour
memory: 60GB
use_case: Large datasets, multi-GPU training
```

### Automated Deployment

```bash
# Set up GCP credentials
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Configure environment
export ANTHROPIC_API_KEY='your-api-key'
export WANDB_API_KEY='your-wandb-key'  # optional

# Deploy training instance
./scripts/deploy_gcp.sh deploy

# Monitor training
./scripts/deploy_gcp.sh monitor

# Stop instance when done
./scripts/deploy_gcp.sh stop

# Clean up resources
./scripts/deploy_gcp.sh cleanup
```

### Manual GCP Setup

```bash
# Create instance
gcloud compute instances create claude-distillation \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --preemptible

# SSH into instance
gcloud compute ssh claude-distillation --zone=us-central1-a

# Setup environment
./scripts/setup_instance.sh
```

### Storage Configuration

```yaml
# Cloud Storage buckets
buckets:
  datasets: gs://your-bucket/datasets/
  models: gs://your-bucket/models/
  logs: gs://your-bucket/logs/
  checkpoints: gs://your-bucket/checkpoints/

# Sync commands
upload_dataset: gsutil -m cp -r ./data/generated gs://your-bucket/datasets/
download_model: gsutil -m cp -r gs://your-bucket/models/final ./models/
```

## 🐳 Docker Deployment

Containerized deployment for consistency and scalability.

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget curl vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install application
RUN pip3 install -e .

# Create directories
RUN mkdir -p /app/data /app/models /app/logs /app/cache

# Set permissions
RUN chmod +x scripts/*.sh

# Expose ports
EXPOSE 8000

# Default command
CMD ["python3", "src/distillation_trainer.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  claude-distillation:
    build: .
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### Build and Run

```bash
# Build image
docker build -t claude-distillation .

# Run training
docker run --gpus all \
    -e ANTHROPIC_API_KEY='your-key' \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    claude-distillation

# Run with Docker Compose
docker-compose up -d
```

## 🌐 API Service Deployment

Deploy trained model as a REST API service.

### FastAPI Service

```python
# api_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI(title="Claude Distilled Code Generator")

# Load model at startup
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_path = "./models/distilled_codellama"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

class CodeRequest(BaseModel):
    prompt: str
    max_length: int = 256
    temperature: float = 0.1

class CodeResponse(BaseModel):
    generated_code: str
    tokens_used: int
    generation_time: float

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    try:
        start_time = time.time()
        
        # Format prompt
        formatted_prompt = f"### Instruction:\n{request.prompt}\n\n### Response:\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_text.split("### Response:\n")[-1].strip()
        
        generation_time = time.time() - start_time
        tokens_used = outputs.shape[1] - inputs["input_ids"].shape[1]
        
        return CodeResponse(
            generated_code=generated_code,
            tokens_used=tokens_used,
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-distillation-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: claude-distillation-api
  template:
    metadata:
      labels:
        app: claude-distillation-api
    spec:
      containers:
      - name: api
        image: claude-distillation:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/distilled_codellama"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
            cpu: 2
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 4
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: claude-distillation-service
spec:
  selector:
    app: claude-distillation-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=claude-distillation-api
kubectl get services

# Scale deployment
kubectl scale deployment claude-distillation-api --replicas=3

# Update deployment
kubectl set image deployment/claude-distillation-api api=claude-distillation:v2
```

## 🚀 Production Considerations

### Security

```yaml
# Security checklist
api_authentication: JWT tokens or API keys
rate_limiting: 100 requests/minute per user
input_validation: Sanitize all inputs
model_access: Restrict direct model access
logging: Log all requests (without sensitive data)
encryption: HTTPS/TLS for all communications
```

### Monitoring

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
GENERATION_TIME = Histogram('generation_duration_seconds', 'Code generation time')
ACTIVE_REQUESTS = Gauge('active_requests', 'Currently active requests')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        REQUEST_COUNT.inc()
        return response
    finally:
        GENERATION_TIME.observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()
```

### Auto-scaling

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: claude-distillation-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: claude-distillation-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Cost Optimization

```yaml
# Cost optimization strategies
spot_instances: Use preemptible/spot instances for training
auto_shutdown: Implement idle timeout for instances
model_compression: Use quantization and pruning
caching: Cache frequent responses
batch_processing: Process multiple requests together
efficient_serving: Use TensorRT or ONNX for faster inference
```

## 📊 Performance Benchmarks

### Training Performance

```yaml
# Google Colab Pro
dataset_size: 1K examples
training_time: 2-3 hours
cost: ~$10-15
gpu: T4 (16GB)

# GCP n1-standard-8 + T4
dataset_size: 10K examples
training_time: 8-12 hours
cost: ~$40-60
gpu: T4 (16GB)

# GCP n1-standard-8 + V100
dataset_size: 25K examples
training_time: 12-16 hours
cost: ~$150-200
gpu: V100 (16GB)
```

### Inference Performance

```yaml
# CPU (Intel i7)
throughput: 0.5-1 tokens/second
latency: 2-5 seconds per request
memory: 8GB

# GPU (T4)
throughput: 10-20 tokens/second
latency: 200-500ms per request
memory: 6GB

# GPU (V100)
throughput: 20-40 tokens/second
latency: 100-300ms per request
memory: 6GB
```

## 🛠️ Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Solutions
- Reduce batch_size to 1-2
- Enable gradient_checkpointing
- Use QLoRA (4-bit quantization)
- Reduce max_length to 1024
```

#### API Rate Limits
```bash
# Solutions
- Reduce rate_limit_rpm to 30
- Increase retry delays
- Implement exponential backoff
- Use batch processing
```

#### Model Loading Errors
```bash
# Solutions
- Check disk space (need 15GB+)
- Verify model files integrity
- Update transformers library
- Check CUDA compatibility
```

#### Poor Performance
```bash
# Solutions
- Increase dataset size
- Adjust hyperparameters
- Use better teacher responses
- Implement curriculum learning
```

### Debugging Commands

```bash
# Check GPU usage
nvidia-smi

# Monitor training
tail -f logs/training.log

# Check disk space
df -h

# Monitor API costs
python scripts/cost_monitor.py

# Validate model
python scripts/validate_model.py
```

## 📞 Support

For deployment issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review [GitHub Issues](https://github.com/yalcindemir/claude-to-codellama-distillation/issues)
3. Create a new issue with deployment details
4. Join our [Discord community](https://discord.gg/claude-distillation)

## 🔄 Updates and Maintenance

### Update Process

```bash
# Backup current model
cp -r models/distilled_codellama models/backup_$(date +%Y%m%d)

# Update code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run tests
python -m pytest tests/

# Restart services
docker-compose restart
```

### Monitoring Health

```bash
# API health check
curl http://localhost:8000/health

# Model performance check
python scripts/performance_check.py

# Cost monitoring
python scripts/cost_report.py
```

This deployment guide provides comprehensive instructions for deploying the Claude-to-CodeLlama Knowledge Distillation system in various environments, from development to production.