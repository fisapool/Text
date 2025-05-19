# AI Paraphraser API

A high-quality paraphrasing API powered by fine-tuned language models. This service provides a REST API for text paraphrasing with different styles and options.

## Features

### Core Features
- High-quality paraphrasing using state-of-the-art language models
- Multiple paraphrasing styles (neutral, formal, creative)
- Batch processing support
- Rate limiting and API key authentication
- Prometheus metrics for monitoring
- Docker and Kubernetes support
- GPU acceleration
- Model information and statistics endpoints
- Redis-based rate limiting

### Premium Features
- Style-specific paraphrasing (academic, business, creative, technical, formal, casual)
- Tone adjustment and complexity control
- Grammar checking and plagiarism detection
- Readability optimization
- Multiple output variants with confidence scoring
- Advanced quality metrics (BLEU, ROUGE, semantic similarity)
- Parameter-efficient fine-tuning with LoRA

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit installed
- At least 16GB RAM
- At least 20GB free disk space
- Redis (included in Docker Compose)
- CMake (for local installation)

## Installation

### System Dependencies

#### For Ubuntu/Debian:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    redis-server \
    pkg-config \
    libprotobuf-dev \
    protobuf-compiler
```

#### For macOS:
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install cmake
brew install redis
brew install protobuf
```

### Project Setup

1. Clone the repository:
```bash
git clone https://github.com/your-repo/ai-paraphraser.git
cd ai-paraphraser
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Fine-Tuning Guide

### Data Preparation

1. Prepare your training data in CSV format with columns:
   - `original_text`: Source text
   - `paraphrased_text`: Target paraphrased text
   - `style`: (Optional) Style category

2. Run data preparation:
```bash
python app/training/prepare_data.py
```

This will:
- Calculate semantic similarities
- Filter low-quality pairs
- Split into train/validation sets
- Save processed datasets

### Training Configuration

The training configuration is in `app/training/config.py`:

```python
# Key parameters:
- num_train_epochs: 3
- learning_rate: 2e-5
- batch_size: 8
- lora_r: 16
- lora_alpha: 32
```

### Starting Fine-Tuning

1. Ensure GPU is available:
```bash
nvidia-smi
```

2. Start training:
```bash
python app/training/train.py
```

Training will:
- Use LoRA for efficient fine-tuning
- Track multiple quality metrics
- Save best model checkpoints
- Generate training statistics

### Monitoring Training

The training process tracks:
- BLEU score
- ROUGE score
- Semantic similarity
- Training loss
- Validation metrics

### Using Fine-Tuned Model

1. Update `.env`:
```
MODEL_PATH=models/premium
```

2. Restart the API service:
```bash
docker-compose up -d
```

## API Usage

### Basic Paraphrasing
```bash
curl -X POST "http://localhost:8000/api/v1/paraphrase" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text here", "style": "formal"}'
```

### Premium Features
```bash
curl -X POST "http://localhost:8000/api/v1/paraphrase/premium" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your text here",
       "style": "academic",
       "tone": "formal",
       "complexity": "high",
       "variants": 3
     }'
```

## Docker Deployment

1. Build and start services:
```bash
docker-compose up -d
```

2. Monitor logs:
```bash
docker-compose logs -f
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BART model by Facebook AI Research
- LoRA implementation by Microsoft
- Sentence Transformers by UKP Lab