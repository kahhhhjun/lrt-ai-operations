"""Run this to test if your GLM API connection works.
Usage: python test_glm.py"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

KEY      = os.getenv("GLM_API_KEY")
ENDPOINT = os.getenv("GLM_ENDPOINT")
MODEL    = os.getenv("GLM_MODEL")

print("=== GLM Connection Test ===")
print(f"API Key  : {'SET (' + KEY[:6] + '...)' if KEY else 'NOT SET ❌'}")
print(f"Endpoint : {ENDPOINT or 'NOT SET ❌'}")
print(f"Model    : {MODEL or 'NOT SET ❌'}")
print()

if not all([KEY, ENDPOINT, MODEL]):
    print("❌ Missing values in .env — fix them first.")
    exit(1)

print("Sending test message to GLM...")
try:
    resp = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    print(f"HTTP status : {resp.status_code}")
    print(f"Raw response: {resp.text[:500]}")
    print()

    if resp.status_code == 200:
        content = resp.json()["choices"][0]["message"]["content"]
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
