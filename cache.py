import os
import json
import redis

# Redis connection
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            # Test connection
            _redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            _redis_client = False # Use False to mark as failed
    
    return _redis_client if _redis_client else None

def cache_get(key):
    client = get_redis_client()
    if not client:
        return None
    try:
        data = client.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        print(f"Redis get error: {e}")
    return None

def cache_set(key, data, ttl):
    client = get_redis_client()
    if not client:
        return False
    try:
        client.setex(key, ttl, json.dumps(data))
        return True
    except Exception as e:
        print(f"Redis set error: {e}")
    return False

def cache_delete_pattern(pattern):
    client = get_redis_client()
    if not client:
        return False
    try:
        cursor = 0
        keys_deleted = 0
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=100)
            if keys:
                client.delete(*keys)
                keys_deleted += len(keys)
            if cursor == 0:
                break
        return keys_deleted
    except Exception as e:
        print(f"Redis delete pattern error: {e}")
    return 0
