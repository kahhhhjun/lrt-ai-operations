"""Z.AI GLM wrapper.

Two public functions:
- call_glm(prompt, system, temperature) → raw text from GLM (or placeholder)
- extract_inputs_from_text(text) → dict of extracted operational factors
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

# TODO: confirm exact endpoint + model name from Z.AI docs once you have the key.
GLM_ENDPOINT = os.getenv("GLM_ENDPOINT", "https://api.z.ai/api/paas/v4/chat/completions")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4")
GLM_API_KEY = os.getenv("GLM_API_KEY")

_EXTRACT_SYSTEM = """You are a data extraction assistant for a Malaysian LRT operations system.
Read the situation description and extract operational factors.
Return ONLY a valid JSON object with exactly these fields:
{
  "day": "weekday" or "weekend",
  "weather": "clear", "cloudy", "rainy", or "stormy",
  "is_holiday": true or false,
  "emergency": a short string describing the emergency, or null,
  "line": one of ["Kelana Jaya", "Ampang", "Sri Petaling", "Kajang", "Putrajaya"] or null,
  "events": [{"name": string, "expected_attendance": integer}]
}
If a field cannot be determined from the text, use sensible defaults:
day="weekday", weather="clear", is_holiday=false, emergency=null, line=null, events=[].
Return ONLY the JSON. No explanation, no markdown, no code block."""


def _placeholder_extract(text: str) -> dict:
    """Keyword-based fallback when no API key is set."""
    t = text.lower()
    result = {
        "day": "weekday",
        "weather": "clear",
        "is_holiday": False,
        "emergency": None,
        "line": None,
        "events": [],
    }
    if any(w in t for w in ["storm", "thunderstorm"]):
        result["weather"] = "stormy"
    elif any(w in t for w in ["rain", "rainy", "raining", "wet", "drizzle"]):
        result["weather"] = "rainy"
    elif any(w in t for w in ["cloud", "cloudy", "overcast"]):
        result["weather"] = "cloudy"

    if any(w in t for w in ["weekend", "saturday", "sunday"]):
        result["day"] = "weekend"

    if any(w in t for w in ["public holiday", "holiday", "hari raya", "chinese new year", "deepavali", "merdeka"]):
        result["is_holiday"] = True

    if any(w in t for w in ["emergency", "accident", "breakdown", "failure", "incident", "evacuation"]):
        result["emergency"] = "emergency situation detected"

    for line in ["kelana jaya", "ampang", "sri petaling", "kajang", "putrajaya"]:
        if line in t:
            result["line"] = " ".join(w.capitalize() for w in line.split())
            break

    # Rough attendance extraction: look for numbers near concert/event keywords
    import re
    event_keywords = ["concert", "match", "game", "festival", "rally", "marathon", "event", "show", "gig"]
    if any(k in t for k in event_keywords):
        numbers = re.findall(r"(\d[\d,]*)\s*(?:fans?|people|attendees?|audience|spectators?|crowd|visitors?|pax)?", t)
        attendance = 0
        for n in numbers:
            val = int(n.replace(",", ""))
            if 1_000 <= val <= 200_000:
                attendance = val
                break
        event_name = next((k.capitalize() for k in event_keywords if k in t), "Event")
        result["events"] = [{"name": event_name, "expected_attendance": attendance or 30_000}]

    return result


def call_glm(prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
    if not GLM_API_KEY:
        return (
            "[PLACEHOLDER — set GLM_API_KEY in .env for real GLM responses]\n\n"
            "Based on the operational factors and the three scheduling options provided, "
            "the moderate option offers the best balance between service quality and cost. "
            "It addresses the demand surge without over-provisioning resources. "
            "Confidence: medium."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        GLM_ENDPOINT,
        headers={
            "Authorization": f"Bearer {GLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": GLM_MODEL, "messages": messages, "temperature": temperature},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_inputs_from_text(text: str) -> dict:
    """Uses GLM to parse a free-text situation description into structured inputs."""
    if not GLM_API_KEY:
        return _placeholder_extract(text)

    response = call_glm(text, system=_EXTRACT_SYSTEM, temperature=0.1)
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except Exception:
        # If GLM returns malformed JSON, fall back to keyword extraction
        return _placeholder_extract(text)
