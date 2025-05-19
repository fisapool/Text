from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.models.ensemble import ParaphraseEnsemble
from app.cache.redis_cache import ParaphraseCache
from app.main import PARAPHRASE_COUNTER, PARAPHRASE_LATENCY, TOKEN_COUNTER
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()
paraphraser = ParaphraseEnsemble()
cache = ParaphraseCache()
executor = ThreadPoolExecutor(max_workers=4)

class ParaphraseRequest(BaseModel):
    text: str
    style: Optional[str] = "neutral"
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    use_cache: Optional[bool] = True
    num_return_sequences: Optional[int] = 3

class ParaphraseResponse(BaseModel):
    original_text: str
    paraphrased_text: str
    style: str
    processing_time: float
    confidence: float
    ensemble_used: bool
    cache_hit: bool
    all_paraphrases: Optional[List[str]] = None
    all_scores: Optional[List[float]] = None

async def process_paraphrase(request: ParaphraseRequest) -> Dict[str, Any]:
    """Process paraphrase request in a separate thread"""
    start_time = time.time()
    
    # Check cache first
    if request.use_cache:
        cached = cache.get(request.text, request.style)
        if cached:
            return {
                **cached,
                "processing_time": time.time() - start_time,
                "cache_hit": True
            }
    
    # Generate paraphrase using ensemble
    result = await asyncio.get_event_loop().run_in_executor(
        executor,
        paraphraser.paraphrase,
        request.text,
        request.style,
        request.max_length,
        request.temperature,
        request.top_p,
        request.num_return_sequences
    )
    
    # Cache the result
    if request.use_cache:
        cache.set(
            request.text,
            request.style,
            result["paraphrased"],
            {
                "confidence": result["confidence"],
                "ensemble_used": result["ensemble_used"]
            }
        )
    
    return {
        "original_text": request.text,
        "paraphrased_text": result["paraphrased"],
        "style": request.style,
        "processing_time": time.time() - start_time,
        "confidence": result["confidence"],
        "ensemble_used": result["ensemble_used"],
        "cache_hit": False,
        "all_paraphrases": result.get("all_paraphrases"),
        "all_scores": result.get("all_scores")
    }

@router.post("/paraphrase", response_model=ParaphraseResponse)
async def paraphrase_text(request: ParaphraseRequest):
    try:
        result = await process_paraphrase(request)
        
        # Record metrics
        PARAPHRASE_COUNTER.labels(
            status="success",
            style=request.style
        ).inc()
        
        PARAPHRASE_LATENCY.labels(
            style=request.style
        ).observe(result["processing_time"])
        
        TOKEN_COUNTER.labels(
            operation="paraphrase"
        ).inc(len(request.text.split()))
        
        return ParaphraseResponse(**result)
        
    except Exception as e:
        PARAPHRASE_COUNTER.labels(
            status="error",
            style=request.style
        ).inc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing paraphrase request: {str(e)}"
        )

@router.post("/batch-paraphrase")
async def batch_paraphrase(requests: List[ParaphraseRequest]):
    """Process multiple paraphrase requests concurrently"""
    tasks = [process_paraphrase(req) for req in requests]
    results = await asyncio.gather(*tasks)
    return results

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return cache.get_stats() 