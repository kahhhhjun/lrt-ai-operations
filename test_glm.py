"""Run this to test if your GLM API connection works.
Usage: python test_glm.py"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

KEY      = os.getenv("GLM_API_KEY")
ENDPOINT = os.getenv("GLM_ENDPOINT", "https://api.z.ai/api/anthropic/v1/messages")
MODEL    = os.getenv("GLM_MODEL", "glm-5.1")

print("=== GLM Connection Test ===")
print(f"API Key  : {'SET (' + KEY[:6] + '...)' if KEY else 'NOT SET ❌'}")
print(f"Endpoint : {ENDPOINT}")
print(f"Model    : {MODEL}")
print()

if not KEY:
    print("❌ GLM_API_KEY not set in .env — fix it first.")
    exit(1)

print("Sending test message to GLM...")
try:
    resp = requests.post(
        ENDPOINT,
        headers={
            "x-api-key":         KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        json={
            "model":      MODEL,
            "max_tokens": 64,
            "messages":   [{"role": "user", "content": "Say hello in one sentence."}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    print(f"HTTP status : {resp.status_code}")
    print(f"Raw response: {resp.text[:500]}")
    print()

    if resp.status_code == 200:
        content = resp.json()["content"][0]["text"]
        print(f"✅ SUCCESS! GLM replied: {content}")
    else:
        print("❌ Request failed. Check the error message above.")

except requests.exceptions.ConnectionError:
    print("❌ Cannot connect — endpoint URL might be wrong.")
except requests.exceptions.Timeout:
    print("❌ Request timed out — endpoint might be unreachable.")
except KeyError as e:
    print(f"❌ Response parsed but missing key {e} — response format might differ.")
    print(f"   Full response: {resp.json()}")
