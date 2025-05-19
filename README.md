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

## Hugging Face Authentication

This project uses models from Hugging Face Hub that require authentication. You can authenticate using any of these methods:

### Method 1: Environment Variable (Recommended)
Add your Hugging Face token to your `.env` file:
```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
```

### Method 2: Python Code
Add this to your code before loading models:
```python
from huggingface_hub import login
login(token="your_token_here")
```

### Method 3: Token File
Create a token file at `~/.huggingface/token`:
```bash
mkdir -p ~/.huggingface
echo "your_token_here" > ~/.huggingface/token
```

### Method 4: CLI Login (Alternative)
If you prefer using the CLI:
```bash
pip install huggingface_hub
huggingface-cli login
```

### Getting Your Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Copy the token and use it in any of the methods above

### Important Notes
- The token is required for accessing gated models like Mistral-7B
- Keep your token secure and never commit it to version control
- The token is used for both downloading and using the models
- You can use the same token across different authentication methods

## Model Options

### Public Models (No Authentication Required)

The project can use several public models that don't require Hugging Face authentication:

1. **BART Large CNN** (`facebook/bart-large-cnn`)
   - ✅ Pros:
     - No authentication required
     - Good for summarization and paraphrasing
     - Smaller model size (400MB)
     - Fast inference
   - ❌ Cons:
     - Less creative than larger models
     - Limited context window

2. **PEGASUS Large** (`google/pegasus-large`)
   - ✅ Pros:
     - Excellent for text generation
     - Good at maintaining context
     - No authentication needed
   - ❌ Cons:
     - Larger model size
     - Slower inference

3. **OPT-1.3B** (`facebook/opt-1.3b`)
   - ✅ Pros:
     - Efficient smaller model
     - Good balance of size and quality
     - Fast inference
   - ❌ Cons:
     - Less creative than larger models

4. **GPT-Neo 1.3B** (`EleutherAI/gpt-neo-1.3B`)
   - ✅ Pros:
     - Open source GPT model
     - Good for creative text
     - No authentication required
   - ❌ Cons:
     - Larger model size
     - Slower inference

### How to Switch Models

1. Edit your `.env` file:
```bash
# Choose one of these models
MODEL_NAME=facebook/bart-large-cnn  # Default, no auth required
# MODEL_NAME=google/pegasus-large
# MODEL_NAME=facebook/opt-1.3b
# MODEL_NAME=EleutherAI/gpt-neo-1.3B
```

2. Restart the service:
```bash
docker-compose down
docker-compose up -d
```

### Model Comparison

| Model | Size | Speed | Quality | Auth Required |
|-------|------|-------|---------|---------------|
| BART Large | 400MB | Fast | Good | No |
| PEGASUS | 568MB | Medium | Very Good | No |
| OPT-1.3B | 2.6GB | Fast | Good | No |
| GPT-Neo | 2.6GB | Medium | Very Good | No |

### Performance Tips

1. **For Low Resource Systems**:
   - Use BART Large or OPT-1.3B
   - Disable ensemble mode
   - Reduce batch size

2. **For Better Quality**:
   - Use PEGASUS or GPT-Neo
   - Enable ensemble mode
   - Increase batch size

3. **For Production**:
   - Use BART Large for reliability
   - Enable caching
   - Use GPU if available

## Choosing Your Installation Method

This project can be run in several different ways. Here's a guide to help you choose the best method for your needs:

### Method Comparison

1. **Docker (Recommended)**
   - ✅ Best for: Production environments, consistent deployments
   - ✅ Pros:
     - Isolated environment
     - Easy to set up
     - Consistent across all platforms
     - Includes all dependencies
     - GPU support out of the box
   - ❌ Cons:
     - Requires Docker knowledge
     - Slightly higher resource usage
     - Larger initial download

2. **Local Development Setup**
   - ✅ Best for: Developers, contributors
   - ✅ Pros:
     - Familiar Python virtual environment
     - Easy to modify code
     - Good for debugging
     - Smaller resource footprint
   - ❌ Cons:
     - More setup steps
     - Platform-specific configurations
     - Manual dependency management

3. **Direct Python Installation**
   - ✅ Best for: Simple testing, single-user setups
   - ✅ Pros:
     - Simplest setup
     - No container overhead
     - Direct system access
   - ❌ Cons:
     - Affects system Python
     - Potential conflicts with other packages
     - Manual dependency management
     - Platform-specific issues

4. **Conda Environment**
   - ✅ Best for: Data scientists, ML researchers
   - ✅ Pros:
     - Excellent for ML dependencies
     - Handles complex package requirements
     - Good GPU support
     - Cross-platform compatibility
   - ❌ Cons:
     - Larger installation size
     - More complex environment management
     - Steeper learning curve

5. **Poetry**
   - ✅ Best for: Modern Python development
   - ✅ Pros:
     - Modern dependency management
     - Lock file for reproducible builds
     - Clean project structure
     - Good for team development
   - ❌ Cons:
     - Newer tool, less community support
     - Additional learning curve
     - May need additional configuration

### Quick Decision Guide

Choose Docker if:
- You want the easiest setup
- You need a production environment
- You want consistent behavior across platforms
- You're not familiar with Python environment management

Choose Local Development if:
- You're a developer contributing to the project
- You need to modify the code frequently
- You prefer working with virtual environments
- You want to debug the application

Choose Direct Python if:
- You're just testing the application
- You have a clean Python environment
- You don't need isolation
- You want the simplest possible setup

Choose Conda if:
- You're working with machine learning
- You need specific package versions
- You're familiar with Conda
- You need GPU support without Docker

Choose Poetry if:
- You're building a modern Python project
- You need reproducible builds
- You're working in a team
- You want clean dependency management

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

### Method 3: Direct Python Installation

If you prefer not to use Docker or virtual environments, you can install the dependencies directly:

1. Install system dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pip python3-dev redis-server

# On macOS
brew install python3 redis

# On Windows
# Download and install Python from python.org
# Download and install Redis from https://github.com/microsoftarchive/redis/releases
```

2. Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

3. Start Redis:
```bash
# On Ubuntu/Debian
sudo service redis-server start

# On macOS
brew services start redis

# On Windows
# Start Redis server from the installed location
```

4. Run the application:
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Method 4: Using Conda Environment

For users who prefer Conda:

1. Install Miniconda or Anaconda

2. Create a new conda environment:
```bash
conda create -n ai-paraphraser python=3.8
conda activate ai-paraphraser
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Redis (using any of the methods mentioned above)

5. Run the application:
```bash
python -m uvicorn app.main:app --reload
```

### Method 5: Using Poetry (Alternative Package Manager)

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Initialize the project with Poetry:
```bash
poetry init
```

3. Install dependencies:
```bash
poetry install
```

4. Start Redis (using any of the methods mentioned above)

5. Run the application:
```bash
poetry run uvicorn app.main:app --reload
```

### Important Notes for All Methods

1. **GPU Requirements**:
   - For optimal performance, ensure you have CUDA installed
   - Verify GPU support: `nvidia-smi`
   - Install CUDA toolkit if needed

2. **Memory Requirements**:
   - Minimum 16GB RAM recommended
   - At least 20GB free disk space

3. **Environment Variables**:
   - Always set up your `.env` file
   - Required variables:
     - `API_KEY`
     - `MODEL_NAME`
     - `REDIS_HOST`
     - `REDIS_PORT`

4. **Performance Considerations**:
   - Docker method provides the most consistent environment
   - Local installation might require additional system configuration
   - GPU acceleration works best with Docker or Conda environments

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