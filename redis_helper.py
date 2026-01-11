"""
Simple Redis client for WhatsApp Bridge
Handles conversation storage with graceful fallback
"""
import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# In-memory fallback
_IN_MEMORY_STORE: Dict[str, Any] = {}

class SimpleRedisClient:
    """Simple Redis client with in-memory fallback"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self._client = None
        self._enabled = False
        
        if self.redis_url:
            try:
                import redis
                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                self._client.ping()
                self._enabled = True
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable: {e}. Using in-memory storage.")
        else:
            logger.warning("⚠️ REDIS_URL not set. Using in-memory storage.")
    
    def set_json(self, key: str, data: Any, ttl: int = 1800) -> bool:
        """Set JSON data with TTL (default 30 min)"""
        try:
            if self._enabled:
                self._client.setex(key, ttl, json.dumps(data, ensure_ascii=False))
            else:
                _IN_MEMORY_STORE[key] = data
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            _IN_MEMORY_STORE[key] = data  # Fallback
            return False
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON data"""
        try:
            if self._enabled:
                data = self._client.get(key)
                return json.loads(data) if data else None
            return _IN_MEMORY_STORE.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return _IN_MEMORY_STORE.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            if self._enabled:
                self._client.delete(key)
            else:
                _IN_MEMORY_STORE.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def incr(self, key: str) -> int:
        """Increment counter"""
        try:
            if self._enabled:
                count = self._client.incr(key)
                return count
            count = _IN_MEMORY_STORE.get(key, 0) + 1
            _IN_MEMORY_STORE[key] = count
            return count
        except Exception as e:
            logger.error(f"Redis incr error: {e}")
            return 0
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration"""
        try:
            if self._enabled:
                self._client.expire(key, seconds)
            return True
        except Exception as e:
            logger.error(f"Redis expire error: {e}")
            return False

# Global instance
redis_client = SimpleRedisClient()
