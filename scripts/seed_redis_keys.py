"""
Script to seed Gemini API keys into Upstash Redis for circular rotation.

Run from the repo root:
    python scripts/seed_redis_keys.py KEY1 KEY2 KEY3
    
Or pass a single comma-separated string:
    python scripts/seed_redis_keys.py "KEY1,KEY2,KEY3"
"""
import argparse
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
    parser = argparse.ArgumentParser(description="Seed Gemini API keys into Upstash Redis.")
    parser.add_argument(
        "keys",
        nargs="+",
        help="One or more API keys, separated by spaces. If passing a single comma-separated string, quote it.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append keys to the existing list instead of clearing it first.",
    )

    args = parser.parse_args()

    load_dotenv(override=False)

    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    list_key = os.getenv("GEMINI_KEYS_REDIS_LIST", "gemini_api_keys")

    if not redis_url or not redis_token:
        print("Error: UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set in your environment (or .env).")
        sys.exit(1)

    redis = UpstashRedis(url=redis_url, token=redis_token)

    # Process keys (handle comma-separated string if user passed exactly one arg)
    raw_keys = args.keys
    if len(raw_keys) == 1 and "," in raw_keys[0]:
        keys = [k.strip() for k in raw_keys[0].split(",") if k.strip()]
    else:
        keys = [k.strip() for k in raw_keys if k.strip()]

    if not keys:
        print("No valid keys provided.")
        sys.exit(1)

    try:
        # Check current length before modifying
        current_len = int(redis.llen(list_key) or 0)
        print(f"Current Redis list '{list_key}' has {current_len} keys.")

        if not args.append and current_len > 0:
            print("Clearing existing list...")
            redis.delete(list_key)

        print(f"Adding {len(keys)} keys to Redis...")
        redis.rpush(list_key, *keys)

        final_len = int(redis.llen(list_key) or 0)
        print(f"Success! List '{list_key}' now contains {final_len} keys.")

    except Exception as exc:
        print(f"An error occurred while communicating with Redis: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
