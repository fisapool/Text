from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, List
import torch
from app.models.paraphraser import Paraphraser
from app.main import verify_api_key

router = APIRouter()
paraphraser = Paraphraser()

class ModelInfo(BaseModel):
    name: str
    version: str
    device: str
    max_length: int
    supported_styles: List[str]
    gpu_memory_used: float
    total_parameters: int

class ModelStats(BaseModel):
    total_requests: int
    average_latency: float
    tokens_processed: int
    requests_by_style: Dict[str, int]

@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    return ModelInfo(
        name=paraphraser.model_name,
        version="1.0.0",
        device=str(paraphraser.device),
        max_length=paraphraser.model.config.max_position_embeddings,
        supported_styles=list(paraphraser.style_prompts.keys()),
        gpu_memory_used=torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        total_parameters=sum(p.numel() for p in paraphraser.model.parameters())
    )

@router.get("/model/stats", response_model=ModelStats)
async def get_model_stats():
    """Get current model statistics"""
    # This would typically come from a database or metrics service
    # For now, we'll return placeholder data
    return ModelStats(
        total_requests=1000,
        average_latency=0.5,
        tokens_processed=50000,
        requests_by_style={
            "neutral": 600,
            "formal": 300,
            "creative": 100
        }
    )