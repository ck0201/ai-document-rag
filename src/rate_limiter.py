"""
Lightweight Sliding-Window In-Memory Rate Limiter.

Protects LLM inference and ingestion endpoints from denial-of-service
and API quota exhaustion (e.g. Gemini 15 RPM free tier).
"""

from collections import defaultdict, deque
import os
import threading
import time
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, Request, status


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    Tracks timestamps of requests within a sliding window (in seconds).
    """

    def __init__(self):
        # Maps key -> deque of float timestamps
        self._history: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, int]:
        """
        Checks whether a key has exceeded max_requests within window_seconds.
        Returns: (is_allowed, remaining_requests, retry_after_seconds)
        """
        # Bypass during automated test execution if requested
        if os.environ.get("TESTING") == "true" or os.environ.get("DISABLE_RATE_LIMIT") == "true":
            return True, max_requests, 0

        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            queue = self._history[key]

            # Prune timestamps older than window
            while queue and queue[0] < window_start:
                queue.popleft()

            if len(queue) >= max_requests:
                # Calculate time until the oldest request leaves the window
                oldest = queue[0]
                retry_after = max(1, int(oldest + window_seconds - now))
                return False, 0, retry_after

            queue.append(now)
            remaining = max_requests - len(queue)
            return True, remaining, 0

    def enforce(
        self,
        request: Request,
        action: str,
        max_requests: int,
        window_seconds: int = 60,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Enforces rate limiting on an incoming FastAPI Request.
        Raises HTTPException(429) if exceeded.
        """
        client_ip = request.client.host if request.client else "127.0.0.1"
        effective_key = f"{action}:{user_id or client_ip}"

        allowed, remaining, retry_after = self.check_rate_limit(
            key=effective_key,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {action}. Maximum {max_requests} requests per {window_seconds}s allowed on free tier. Please retry in {retry_after}s.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


limiter = SlidingWindowRateLimiter()
