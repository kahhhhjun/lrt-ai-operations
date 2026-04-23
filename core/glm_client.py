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
  "weather": "clear", "cloudy", "rainy", or "stormy",
  "emergency": a short string describing the emergency, or null,
  "line": one of ["Kelana Jaya", "Ampang", "Sri Petaling"] or null,
  "events": [{"name": string, "passengers_per_hr": integer}]
}
"passengers_per_hr" means extra passengers arriving at the station per hour due to the event.
Events include concerts, sports matches, public holidays, festivals, marathons — anything that draws extra people.
If a field cannot be determined from the text, use sensible defaults:
weather="clear", emergency=null, line=null, events=[].
Return ONLY the JSON. No explanation, no markdown, no code block."""


def _placeholder_extract(text: str) -> dict:
    """Keyword-based fallback when no API key is set."""
    import re
    t = text.lower()
    result = {
        "weather":   "clear",
        "emergency": None,
        "line":      None,
        "events":    [],
    }

    if any(w in t for w in ["storm", "thunderstorm"]):
        result["weather"] = "stormy"
    elif any(w in t for w in ["rain", "rainy", "raining", "wet", "drizzle"]):
        result["weather"] = "rainy"
    elif any(w in t for w in ["cloud", "cloudy", "overcast"]):
        result["weather"] = "cloudy"

    if any(w in t for w in ["emergency", "accident", "breakdown", "failure", "incident", "evacuation"]):
        result["emergency"] = "emergency situation detected"

    for line in ["kelana jaya", "ampang", "sri petaling"]:
        if line in t:
            result["line"] = " ".join(w.capitalize() for w in line.split())
            break

    event_keywords = ["concert", "match", "game", "festival", "rally", "marathon",
                      "event", "show", "gig", "holiday"]
    if any(k in t for k in event_keywords):
        # Extract passenger numbers from text (e.g. "4,000 extra passengers")
        numbers = re.findall(r"(\d[\d,]*)\s*(?:extra\s+)?(?:passengers?|pax|people|fans?|crowd)?", t)
        pax_hr = 2_000  # default if no number found
        for n in numbers:
            val = int(n.replace(",", ""))
            if 500 <= val <= 30_000:
                pax_hr = val
                break
        event_name = next((k.capitalize() for k in event_keywords if k in t), "Event")
        result["events"] = [{"name": event_name, "passengers_per_hr": pax_hr}]

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

    try:
        resp = requests.post(
            GLM_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": GLM_MODEL, "messages": messages, "temperature": temperature},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        raise TimeoutError("GLM API timed out. The server took too long to respond.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach GLM API. Check your internet connection and endpoint URL.")


def extract_inputs_from_text(text: str) -> dict:
    """Uses GLM to parse a free-text situation description into structured inputs."""
    if not GLM_API_KEY:
        return _placeholder_extract(text)

    try:
        response = call_glm(text, system=_EXTRACT_SYSTEM, temperature=0.1)
        start = response.find("{")
        end   = response.rfind("}") + 1
        return json.loads(response[start:end])
    except (TimeoutError, ConnectionError):
        raise
    except Exception:
        return _placeholder_extract(text)
