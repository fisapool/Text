from typing import Optional, Dict, Any
import redis
import json
import hashlib
import os
from datetime import timedelta

class ParaphraseCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=True
        )
        self.cache_ttl = int(os.getenv("CACHE_TTL", 86400))  # 24 hours default

    def _generate_key(self, text: str, style: str) -> str:
        """Generate a unique cache key for the text and style combination"""
        key_string = f"{text}:{style}".encode('utf-8')
        return f"paraphrase:{hashlib.sha256(key_string).hexdigest()}"

    def get(self, text: str, style: str) -> Optional[str]:
        """Get paraphrased text from cache if it exists"""
        key = self._generate_key(text, style)
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(self, text: str, style: str, paraphrased: str, metadata: Dict[str, Any] = None):
        """Cache paraphrased text with optional metadata"""
        key = self._generate_key(text, style)
        cache_data = {
            "original": text,
            "paraphrased": paraphrased,
            "style": style,
            "metadata": metadata or {}
        }
        self.redis_client.setex(
            key,
            timedelta(seconds=self.cache_ttl),
            json.dumps(cache_data)
        )

    def invalidate(self, text: str, style: str):
        """Remove an item from the cache"""
        key = self._generate_key(text, style)
        self.redis_client.delete(key)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        keys = self.redis_client.keys("paraphrase:*")
        return {
            "total_cached": len(keys),
            "memory_used": self.redis_client.info()["used_memory_human"]
        } 