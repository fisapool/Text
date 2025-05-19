# AI Paraphraser API

A high-quality paraphrasing API powered by fine-tuned language models. This service provides a REST API for text paraphrasing with different styles and options.

## Features

- High-quality paraphrasing using state-of-the-art language models
- Multiple paraphrasing styles (neutral, formal, creative)
- Batch processing support
- Rate limiting and API key authentication
- Prometheus metrics for monitoring
- Docker and Kubernetes support
- GPU acceleration
- Model information and statistics endpoints
- Redis-based rate limiting

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit installed
- At least 16GB RAM
- At least 20GB free disk space
- Redis (included in Docker Compose)

## How to Run

### Method 1: Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-paraphraser.git
cd ai-paraphraser
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services:
```bash
docker-compose up -d
```

4. Verify the services are running:
```bash
docker-compose ps
```

5. The API will be available at `http://localhost:8000`

### Method 2: Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-paraphraser.git
cd ai-paraphraser
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start Redis (required for rate limiting):
```bash
# Using Docker
docker run -d -p 6379:6379 redis

# Or install Redis locally:
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server
```

6. Run the development server:
```bash
uvicorn app.main:app --reload
```

7. The API will be available at `http://localhost:8000`

### Verifying the Installation

1. Test the API health endpoint:
```bash
curl http://localhost:8000/health
```

2. Test the paraphrase endpoint (replace `your_api_key` with your actual API key):
```bash
curl -X POST http://localhost:8000/api/v1/paraphrase \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test message",
    "style": "formal"
  }'
```

### Troubleshooting

1. If you encounter GPU-related issues:
   - Ensure NVIDIA drivers are installed
   - Verify CUDA installation: `nvidia-smi`
   - Check Docker GPU support: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`

2. If Redis connection fails:
   - Check Redis is running: `docker ps | grep redis`
   - Verify Redis connection: `redis-cli ping`

3. If the API is not accessible:
   - Check if the service is running: `docker-compose ps`
   - View logs: `docker-compose logs -f`
   - Ensure ports 8000 and 9090 are not in use

## Quick Start

1. Clone the repository:
```