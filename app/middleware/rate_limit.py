from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import redis
import time
import os
from typing import Optional

class RateLimitMiddleware:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=True
        )
        self.requests_limit = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
        self.window_size = int(os.getenv("RATE_LIMIT_WINDOW", 3600))  # 1 hour in seconds

    async def __call__(self, request: Request, call_next):
        # Skip rate limiting for health check and metrics endpoints
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key is required")

        # Create Redis key for this API key
        key = f"rate_limit:{api_key}"

        # Get current count and window start
        current = self.redis_client.get(key)
        if current is None:
            # First request in window
            self.redis_client.setex(key, self.window_size, 1)
        else:
            current = int(current)
            if current >= self.requests_limit:
                # Get TTL to show when limit resets
                ttl = self.redis_client.ttl(key)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "reset_in_seconds": ttl,
                        "limit": self.requests_limit,
                        "window": self.window_size
                    }
                )
            # Increment counter
            self.redis_client.incr(key)

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to the response
        remaining = self.requests_limit - int(self.redis_client.get(key))
        response.headers["X-RateLimit-Limit"] = str(self.requests_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(self.redis_client.ttl(key) if self.redis_client.ttl(key) is not None else self.window_size)

        return response 