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

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-paraphraser.git
cd ai-paraphraser
```

2. Copy the environment file and configure it:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services:
```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

## API Usage

### Authentication

All API requests require an API key to be included in the `X-API-Key` header.

### Rate Limiting

The API implements rate limiting using Redis:
- Default: 100 requests per hour per API key
- Configurable via environment variables
- Rate limit information included in response headers

### Endpoints

#### Paraphrase Text

```bash
curl -X POST http://localhost:8000/api/v1/paraphrase \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog",
    "style": "formal",
    "max_length": 512
  }'
```

#### Batch Paraphrase

```bash
curl -X POST http://localhost:8000/api/v1/batch-paraphrase \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "text": "First text to paraphrase",
      "style": "neutral"
    },
    {
      "text": "Second text to paraphrase",
      "style": "creative"
    }
  ]'
```

#### Model Information

```bash
curl -X GET http://localhost:8000/api/v1/model/info \
  -H "X-API-Key: your_api_key"
```

#### Model Statistics

```bash
curl -X GET http://localhost:8000/api/v1/model/stats \
  -H "X-API-Key: your_api_key"
```

### Response Headers

The API includes useful headers in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time until rate limit resets (in seconds)

## Monitoring

Prometheus metrics are available at `http://localhost:9090/metrics`

Key metrics:
- `paraphrase_requests_total`: Total number of paraphrase requests by status and style
- `paraphrase_latency_seconds`: Processing time for paraphrase requests by style
- `tokens_processed_total`: Total number of tokens processed by operation

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn app.main:app --reload
```

## Environment Variables

Key environment variables:
- `API_KEY`: Your API key for authentication
- `MODEL_NAME`: Name of the model to use
- `RATE_LIMIT_REQUESTS`: Number of requests allowed per window
- `RATE_LIMIT_WINDOW`: Time window for rate limiting in seconds
- `REDIS_HOST`: Redis server hostname
- `REDIS_PORT`: Redis server port
- `ENABLE_METRICS`: Enable Prometheus metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 