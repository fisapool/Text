from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
import time
import os
import logging
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Paraphraser API",
    description="A high-quality paraphrasing API powered by fine-tuned language models",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# Prometheus metrics
PARAPHRASE_COUNTER = Counter(
    'paraphrase_requests_total',
    'Total number of paraphrase requests',
    ['status', 'style', 'api_key']
)

PARAPHRASE_LATENCY = Histogram(
    'paraphrase_latency_seconds',
    'Time spent processing paraphrase requests',
    ['style', 'api_key']
)

TOKEN_COUNTER = Counter(
    'tokens_processed_total',
    'Total number of tokens processed',
    ['operation', 'api_key']
)

ERROR_COUNTER = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['error_type', 'endpoint']
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = request.headers.get('X-Request-ID', '')
        
        # Log request
        logger.info(f"Request started: {request.method} {request.url.path} - ID: {request_id}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Time: {process_time:.2f}s - ID: {request_id}"
            )
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.2f}s - ID: {request_id}"
            )
            raise

app.add_middleware(RequestLoggingMiddleware)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        logger.warning("API request attempted without API key")
        raise HTTPException(
            status_code=401,
            detail="API key is required"
        )
    
    if api_key != os.getenv("API_KEY"):
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = request.headers.get('X-Request-ID', '')
    logger.error(f"Unhandled exception: {str(exc)} - ID: {error_id}")
    ERROR_COUNTER.labels(error_type=type(exc).__name__, endpoint=request.url.path).inc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.get("/")
async def root():
    return {
        "message": "Welcome to the AI Paraphraser API",
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs",
        "metrics": "/metrics"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )

# Import and include routers
from app.routers import paraphrase, model_info

app.include_router(
    paraphrase.router,
    prefix="/api/v1",
    tags=["paraphrase"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    model_info.router,
    prefix="/api/v1",
    tags=["model"],
    dependencies=[Depends(verify_api_key)]
) 