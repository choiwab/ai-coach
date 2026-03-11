"""
Script to list all Gemini API keys currently stored in the Upstash Redis list.
Run from the repo root:
    python scripts/get_redis_keys.py
"""

import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from upstash_redis import Redis as UpstashRedis
except ImportError:
    print("Error: upstash-redis is not installed. Run `pip install upstash-redis`.")
    sys.exit(1)

def main():
    load_dotenv(override=False)

    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    list_key = os.getenv("GEMINI_KEYS_REDIS_LIST", "gemini_api_keys")

    if not redis_url or not redis_token:
        print("Error: UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set", file=sys.stderr)
        sys.exit(1)

    redis = UpstashRedis(url=redis_url, token=redis_token)

    try:
        length = int(redis.llen(list_key) or 0)
        print(f"List '{list_key}' contains {length} keys.")
        
        if length > 0:
            keys = redis.lrange(list_key, 0, -1)
            print("\nKeys currently in Redis:")
            for i, key in enumerate(keys):
                key_str = key.decode() if isinstance(key, bytes) else str(key)
                if len(key_str) > 12:
                    masked = f"{key_str[:6]}...{key_str[-4:]}"
                else:
                    masked = "***"
                print(f"  {i}: {masked}")
        else:
            print(f"The Redis list '{list_key}' is currently empty.")
            
    except Exception as exc:
        print(f"An error occurred while communicating with Redis: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
