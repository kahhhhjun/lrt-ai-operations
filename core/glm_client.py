"""Z.AI GLM wrapper.

Public functions:
- call_glm(prompt, system, temperature) → raw text from GLM (or placeholder)
- call_glm_stream(prompt, system, temperature) → generator that yields text chunks as they arrive
- extract_inputs_from_text(text) → dict of extracted operational factors
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

GLM_ENDPOINT    = os.getenv("GLM_ENDPOINT", "https://api.z.ai/api/anthropic/v1/messages")
GLM_MODEL       = os.getenv("GLM_MODEL", "glm-5.1")        # main model: reasoning & briefing
GLM_MODEL_FAST  = os.getenv("GLM_MODEL_FAST", "glm-5-turbo")  # fast model: text extraction
GLM_API_KEY     = os.getenv("GLM_API_KEY")

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


def call_glm(prompt: str, system: str | None = None, temperature: float = 0.3, model: str | None = None) -> str:
    if not GLM_API_KEY:
        return (
            "[PLACEHOLDER — set GLM_API_KEY in .env for real GLM responses]\n\n"
            "Based on the operational factors and the three scheduling options provided, "
            "the moderate option offers the best balance between service quality and cost. "
            "It addresses the demand surge without over-provisioning resources. "
            "Confidence: medium."
        )

    body = {
        "model":       model or GLM_MODEL,
        "max_tokens":  4096,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if system:
        body["system"] = system

    try:
        resp = requests.post(
            GLM_ENDPOINT,
            headers={
                "x-api-key":         GLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type":      "application/json",
            },
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
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

    body = {
        "model":      GLM_MODEL,
        "max_tokens": 4096,
        "messages":   [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream":     True,
    }
    if system:
        body["system"] = system

    try:
        resp = requests.post(
            GLM_ENDPOINT,
            headers={
                "x-api-key":         GLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type":      "application/json",
            },
            json=body,
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
                if data.strip() in ("[DONE]", ""):
                    continue
                try:
                    chunk = json.loads(data)
                    # Anthropic SSE: event type is content_block_delta
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {}).get("text", "")
                        if delta:
                            yield delta
                except Exception:
                    pass
        except requests.exceptions.ChunkedEncodingError:
            return
    except requests.exceptions.Timeout:
        raise TimeoutError("GLM API timed out. The server took too long to respond.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach GLM API. Check your internet connection and endpoint URL.")


_PAX_FACTOR_SYSTEM = """You are a ridership analyst for Malaysian LRT operations.
Given the weather and emergency situation, predict how they affect LRT passenger numbers.
Bad weather increases LRT ridership because people switch from driving/walking to taking the train.
Return ONLY a valid JSON object with exactly these fields:
{
  "weather_pax_mult": float >= 1.0 (rain/stormy = more people take LRT instead of driving — always >= 1.0),
  "weather_event_mult": float >= 1.0 (rain pushes event-goers onto LRT instead of driving — always >= 1.0),
  "emergency_pax_mult": float between 1.0 and 2.0 (how much emergency inflates stranded/surge passengers)
}

Guidelines:
- Clear weather: weather_pax_mult=1.0, weather_event_mult=1.0
- Cloudy: weather_pax_mult=1.05, weather_event_mult=1.05
- Rainy: weather_pax_mult=1.10-1.20, weather_event_mult=1.15-1.25 (significant modal shift to LRT)
- Stormy: weather_pax_mult=1.03-1.08, weather_event_mult=1.05-1.10 (some modal shift, slightly less than rainy)
- No emergency: emergency_pax_mult=1.0
- Overcrowding: emergency_pax_mult=1.4-1.6
- Evacuation: emergency_pax_mult=1.3-1.5
- Breakdown/signal_failure/power_failure: emergency_pax_mult=1.1-1.3
- track_incident: emergency_pax_mult=1.2-1.4
Return ONLY the JSON. No explanation."""


def get_glm_pax_factors(
    weather: str,
    emergency_type: str | None,
    hour: int,
    line: str,
) -> dict:
    """Ask GLM to predict weather/emergency multipliers for passenger calculation.
    Falls back to hardcoded defaults if GLM is unavailable."""
    _DEFAULTS = {
        "weather_pax_mult":   {"clear": 1.00, "cloudy": 1.05, "rainy": 1.15, "stormy": 1.05}.get(weather, 1.0),
        "weather_event_mult": {"clear": 1.00, "cloudy": 1.05, "rainy": 1.20, "stormy": 1.08}.get(weather, 1.0),
        "emergency_pax_mult": 1.0,
    }
    if not GLM_API_KEY:
        return _DEFAULTS

    prompt = (
        f"Weather: {weather}\n"
        f"Emergency: {emergency_type or 'none'}\n"
        f"Time: {hour:02d}:00\n"
        f"Line: {line}\n"
        "Predict the ridership multipliers."
    )
    try:
        response = call_glm(prompt, system=_PAX_FACTOR_SYSTEM, temperature=0.1, model=GLM_MODEL_FAST)
        start = response.find("{")
        end   = response.rfind("}") + 1
        result = json.loads(response[start:end])
        # Clamp to safe ranges — weather multipliers always >= 1.0
        return {
            "weather_pax_mult":   max(1.0, min(1.5,  result.get("weather_pax_mult",   _DEFAULTS["weather_pax_mult"]))),
            "weather_event_mult": max(1.0, min(1.5,  result.get("weather_event_mult", _DEFAULTS["weather_event_mult"]))),
            "emergency_pax_mult": max(1.0, min(2.0,  result.get("emergency_pax_mult", _DEFAULTS["emergency_pax_mult"]))),
        }
    except Exception:
        return _DEFAULTS



_COST_JUSTIFY_SYSTEM = """You are a senior operations analyst for Malaysian LRT (Kelana Jaya, Ampang, Sri Petaling lines).
Given a schedule adjustment and its cost impact, write a concise cost justification using your knowledge of:
- How events (concerts, festivals, football matches) drive ridership surges in Malaysia
- How weather affects LRT demand (rain increases ridership as people avoid driving)
- How emergencies (signal failures, overcrowding) require costly but necessary frequency changes
- The public service obligation of LRT — safety and accessibility outweigh short-term cost savings
Reason from the situation, not just the numbers. Be practical and direct. Under 100 words. Plain text, no markdown."""

_WEEKLY_ANOMALY_SYSTEM = """You are a cost analyst for Malaysian LRT operations.
Given the weekly cost breakdown, identify any anomalies or noteworthy patterns.
Flag days where cost is significantly above or below the weekly average, explain why if obvious, and give one overall recommendation.
Use bullet points. Under 100 words."""


def get_glm_cost_justification_stream(
    line: str,
    date_str: str,
    std_cost: float,
    extra_cost: float,
    net_cost: float,
    events: list[dict],
    emergency_type: str | None,
    weather: str,
    expected_extra_pax: int = 0,
):
    """Stream GLM cost justification for today's adjusted schedule."""
    if not GLM_API_KEY:
        yield (
            f"Extra cost of RM {extra_cost:,.0f} deployed for {emergency_type or (events[0]['name'] if events else weather)} situation. "
            f"Net cost RM {net_cost:,.0f} vs standard RM {std_cost:,.0f}."
        )
        return

    event_text = ", ".join(e["name"] for e in events) if events else "none"
    situation = []
    if emergency_type:
        situation.append(f"{emergency_type.replace('_', ' ')} emergency")
    if events:
        situation.append(f"{event_text} event")
    if weather != "clear":
        situation.append(f"{weather} weather")
    situation_str = " + ".join(situation) if situation else "standard operations"

    prompt = (
        f"Line: {line} | Date: {date_str}\n"
        f"Situation: {situation_str}\n"
        f"Standard daily cost: RM {std_cost:,.0f}\n"
        f"Cost adjustment: RM {extra_cost:+,.0f}\n"
        f"Net cost: RM {net_cost:,.0f}\n\n"
        f"Based on your knowledge of this type of situation ({situation_str}), "
        f"is this cost adjustment justified? Give a brief verdict."
    )
    yield from call_glm_stream(prompt, system=_COST_JUSTIFY_SYSTEM, temperature=0.3)


def get_glm_weekly_anomalies(weekly_data: list[dict]) -> str:
    """Ask GLM to flag cost anomalies in the weekly overview. Returns plain text."""
    if not GLM_API_KEY:
        return "Connect GLM API to get weekly cost analysis."

    rows_text = "\n".join(
        f"  {r['Day']} ({r['Type']}): std=RM {r['std']:,.0f}, extra=RM {r['extra']:+,.0f}, net=RM {r['net']:,.0f}"
        for r in weekly_data
    )
    net_values = [r["net"] for r in weekly_data]
    avg = sum(net_values) / len(net_values) if net_values else 0
    prompt = (
        f"Weekly cost breakdown (average net: RM {avg:,.0f}/day):\n{rows_text}\n"
        "Identify anomalies and give one recommendation."
    )
    try:
        return call_glm(prompt, system=_WEEKLY_ANOMALY_SYSTEM, temperature=0.3, model=GLM_MODEL_FAST)
    except Exception:
        return "Could not retrieve weekly analysis."



def extract_inputs_from_text(text: str) -> dict:
    """Uses GLM (fast model) to parse a free-text situation description into structured inputs."""
    if not GLM_API_KEY:
        return _placeholder_extract(text)

    try:
        response = call_glm(text, system=_EXTRACT_SYSTEM, temperature=0.1, model=GLM_MODEL_FAST)
        start = response.find("{")
        end   = response.rfind("}") + 1
        return json.loads(response[start:end])
    except (TimeoutError, ConnectionError):
        raise
    except Exception:
        return _placeholder_extract(text)
