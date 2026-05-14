import os
import redis
import json

class RedisClient:
    def __init__(self):
        self.use_mock = False
        self.mock_store = {}
        
        try:
            self.client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                decode_responses=True,
                socket_connect_timeout=1 # fail fast if not available
            )
            self.client.ping()
            print("[Redis] Connected to Redis server successfully.")
        except (redis.ConnectionError, redis.TimeoutError, redis.exceptions.RedisError) as e:
            print(f"[Redis] WARNING: Could not connect to Redis ({e}). Falling back to in-memory store for local testing.")
            self.use_mock = True

    def set_ex(self, key: str, time: int, value: str):
        if self.use_mock:
            import time as pytime
            self.mock_store[key] = {
                "value": value,
                "expires_at": pytime.time() + time
            }
        else:
            self.client.setex(key, time, value)

    def get(self, key: str) -> str:
        if self.use_mock:
            import time as pytime
            item = self.mock_store.get(key)
            if item:
                if pytime.time() > item["expires_at"]:
                    del self.mock_store[key]
                    return None
                return item["value"]
            return None
        else:
            return self.client.get(key)
            
    def delete(self, key: str):
        if self.use_mock:
            if key in self.mock_store:
                del self.mock_store[key]
        else:
            self.client.delete(key)

# Singleton instance
redis_client = RedisClient()
