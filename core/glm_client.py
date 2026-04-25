"""Z.AI GLM wrapper.

Public functions:
- call_glm(prompt, system, temperature) → raw text from GLM (or placeholder)
- call_glm_stream(prompt, system, temperature) → generator that yields text chunks as they arrive
- extract_inputs_from_text(text) → dict of extracted operational factors
"""

import base64
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
  "emergency_type": "track_incident", "breakdown", "signal_failure", "power_failure", "evacuation", "overcrowding", or null,
  "line": one of ["Kelana Jaya", "Ampang", "Sri Petaling"] or null,
  "events": [{"name": string, "passengers_per_hr": integer, "event_type": string}]
}

Emergency type rules:
- "track_incident": person on track, suicide, suicide attempt, someone died, death, body on track, person jumped, fatality, killed on track — ALWAYS use this type for ANY situation involving a person on the tracks or death on the tracks. SERVICE MUST BE HALTED IMMEDIATELY.
- "breakdown": train malfunction, out of service, mechanical failure
- "signal_failure": signal fault, signalling problem
- "power_failure": power outage, electrical fault
- "evacuation": fire, bomb threat, security alert
- "overcrowding": crowd crush, platform full, stampede risk

Event type rules — set "event_type" for each event:
- "concert": music concert, gig, live performance, band
- "football_match": football, soccer, match, game, final, cup
- "festival": festival, fair, carnival, bazaar, pasar malam
- "marathon": marathon, run, race, cycling event
- "public_holiday": public holiday, national day, celebration, parade
- "exhibition": exhibition, convention, expo, conference, trade show
- "religious_event": prayers, church, temple, mosque, thaipusam, wesak, deepavali mass gathering
- "concert" is the default if event type is unclear

"passengers_per_hr" means extra passengers per hour due to events (not emergencies).
If a field cannot be determined, use sensible defaults: weather="clear", emergency=null, emergency_type=null, line=null, events=[].
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

    _track_kw    = ["jump", "jumps", "jumped", "suicide", "person on track", "body on track",
                    "track intrusion", "trespasser", "fell onto", "fallen onto",
                    "died", "death", "killed", "fatal", "fatality"]
    _breakdown_kw = ["breakdown", "malfunction", "out of service", "mechanical", "derail"]
    _signal_kw   = ["signal failure", "signal fault", "signalling", "signaling"]
    _power_kw    = ["power failure", "power outage", "blackout", "electrical fault"]
    _evac_kw     = ["evacuation", "fire", "bomb", "security alert", "suspicious"]
    _crowd_kw    = ["overcrowding", "crowd crush", "stampede", "platform full"]
    _generic_kw  = ["emergency", "accident", "incident", "failure", "alert"]

    if any(w in t for w in _track_kw):
        result["emergency"]      = "person on track — service suspended"
        result["emergency_type"] = "track_incident"
    elif any(w in t for w in _breakdown_kw):
        result["emergency"]      = "train breakdown"
        result["emergency_type"] = "breakdown"
    elif any(w in t for w in _signal_kw):
        result["emergency"]      = "signal failure"
        result["emergency_type"] = "signal_failure"
    elif any(w in t for w in _power_kw):
        result["emergency"]      = "power failure"
        result["emergency_type"] = "power_failure"
    elif any(w in t for w in _evac_kw):
        result["emergency"]      = "evacuation in progress"
        result["emergency_type"] = "evacuation"
    elif any(w in t for w in _crowd_kw):
        result["emergency"]      = "overcrowding emergency"
        result["emergency_type"] = "overcrowding"
    elif any(w in t for w in _generic_kw):
        result["emergency"]      = "emergency situation detected"
        result["emergency_type"] = "overcrowding"  # default: treat as needing more trains

    if result["emergency"]:
        result.setdefault("emergency_type", "overcrowding")

    for line in ["kelana jaya", "ampang", "sri petaling"]:
        if line in t:
            result["line"] = " ".join(w.capitalize() for w in line.split())
            break

    _event_type_map = [
        ("concert",         ["concert", "gig", "show", "live performance", "tour"]),
        ("football_match",  ["football", "soccer", "match", "final", "cup", "game"]),
        ("festival",        ["festival", "fair", "carnival", "bazaar", "pasar malam"]),
        ("marathon",        ["marathon", "run", "race", "cycling"]),
        ("public_holiday",  ["holiday", "merdeka", "national day", "parade", "celebration"]),
        ("exhibition",      ["exhibition", "expo", "convention", "conference", "trade show"]),
        ("religious_event", ["thaipusam", "wesak", "deepavali", "prayers", "church", "temple"]),
    ]
    event_keywords = ["concert", "match", "game", "festival", "rally", "marathon",
                      "event", "show", "gig", "holiday", "exhibition", "prayers",
                      "thaipusam", "wesak", "deepavali", "carnival", "bazaar", "run"]
    if any(k in t for k in event_keywords):
        numbers = re.findall(r"(\d[\d,]*)\s*(?:extra\s+)?(?:passengers?|pax|people|fans?|crowd)?", t)
        pax_hr = 2_000
        for n in numbers:
            val = int(n.replace(",", ""))
            if 500 <= val <= 30_000:
                pax_hr = val
                break
        detected_type = "concert"
        for ev_type, keywords in _event_type_map:
            if any(k in t for k in keywords):
                detected_type = ev_type
                break
        event_name = next((k.capitalize() for k in event_keywords if k in t), "Event")
        result["events"] = [{"name": event_name, "passengers_per_hr": pax_hr,
                              "event_type": detected_type}]

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


def call_glm_stream(prompt: str, system: str | None = None, temperature: float = 0.3):
    """Generator: yields text chunks as GLM streams them.
    Keeps the connection alive token-by-token, so no 60-second timeout on slow responses.
    Raises TimeoutError / ConnectionError if the connection itself fails."""
    if not GLM_API_KEY:
        placeholder = (
            "[PLACEHOLDER — set GLM_API_KEY in .env for real GLM responses]\n\n"
            "The moderate option offers the best balance between service quality and cost. "
            "It addresses the demand surge without over-provisioning resources.\n\n"
            "RECOMMENDATION: moderate"
        )
        for char in placeholder:
            yield char
        return

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
            json={"model": GLM_MODEL, "messages": messages, "temperature": temperature, "stream": True},
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()
        try:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                decoded = raw_line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                data = decoded[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    pass
        except requests.exceptions.ChunkedEncodingError:
            return  # stream ended prematurely — yield what we have and stop cleanly
    except requests.exceptions.Timeout:
        raise TimeoutError("GLM API timed out. The server took too long to respond.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach GLM API. Check your internet connection and endpoint URL.")


OCR_API_KEY = os.getenv("OCR_API_KEY")


def _compress_image(image_bytes: bytes, max_px: int = 1200, quality: int = 82) -> tuple[bytes, str]:
    """Resize + JPEG-compress to keep payload under OCR.space's 1 MB free-tier limit."""
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


def extract_inputs_from_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """OCR the image via OCR.space, then extract operational factors with GLM text extraction."""
    if not OCR_API_KEY:
        raise RuntimeError("OCR_API_KEY not set in .env — cannot read image.")

    # Step 1: compress before upload so we stay under OCR.space's 1 MB free-tier limit
    image_bytes, mime_type = _compress_image(image_bytes)

    b64      = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    try:
        ocr_resp = requests.post(
            "https://api.ocr.space/parse/image",
            data={
                "apikey":            OCR_API_KEY,
                "base64Image":       data_url,
                "language":          "eng",
                "isOverlayRequired": False,
                "detectOrientation": True,
                "scale":             True,
            },
            timeout=30,
        )
        ocr_resp.raise_for_status()
        ocr_result = ocr_resp.json()
    except requests.exceptions.Timeout:
        raise TimeoutError("OCR.space timed out reading the image.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach OCR.space API.")

    if ocr_result.get("IsErroredOnProcessing"):
        msgs = ocr_result.get("ErrorMessage") or ["OCR failed"]
        raise RuntimeError(f"OCR.space error: {msgs[0]}")

    parsed_text = " ".join(
        page.get("ParsedText", "")
        for page in ocr_result.get("ParsedResults", [])
    ).strip()

    if not parsed_text:
        raise RuntimeError("OCR found no readable text in the image.")

    # Step 2: feed the extracted text into GLM for structured extraction
    # Fall back to keyword extraction on the OCR text if GLM times out
    try:
        return extract_inputs_from_text(parsed_text)
    except (TimeoutError, ConnectionError):
        return _placeholder_extract(parsed_text)


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
