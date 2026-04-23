"""Orchestrator: computes three scheduling options, asks GLM to choose one and justify it,
returns a single result dict the UI can consume."""

import json
from pathlib import Path

from core.calculator import compute_options, compute_daily_schedule
from core.glm_client import GLM_API_KEY, call_glm

_HISTORY_PATH = Path(__file__).parent.parent / "data" / "history.json"


def _load_relevant_history(inputs: dict, max_records: int = 3) -> list[dict]:
    """Return the most relevant past incidents for this situation.
    Relevance: same line first, then same weather, then same event type."""
    if not _HISTORY_PATH.exists():
        return []
    history = json.loads(_HISTORY_PATH.read_text())
    line    = inputs.get("line", "").lower()
    weather = inputs.get("weather", "clear")
    has_event = bool(inputs.get("events"))

    def score(h):
        s = 0
        if line and line in h.get("line", "").lower():
            s += 3
        if h.get("weather") == weather:
            s += 2
        if has_event and h.get("event_passengers_per_hr", 0) > 0:
            s += 2
        if not has_event and h.get("event_passengers_per_hr", 0) == 0:
            s += 1
        return s

    ranked = sorted(history, key=score, reverse=True)
    return ranked[:max_records]

_OPTIONS_SYSTEM = """You are an expert operations advisor for Malaysian LRT lines (Kelana Jaya, Ampang, Sri Petaling).
You help duty managers make real-time scheduling decisions.

Key facts you must use in your reasoning:
- Normal train capacity: 600 passengers. Maximum crush load: 900.
- Load factor 75% = comfortable target. 100% = trains completely full. >100% = passengers LEFT BEHIND on platform.
- Headway = minutes between trains. Shorter headway = more trains = less waiting.
- Rainy/stormy weather: passengers crowd platforms more AND trains must slow down for safety.
- Events (concerts, holidays, matches): cause sharp passenger surges, especially at entry/exit times.
- Peak hours (7-9am, 5-7pm weekdays): highest commuter volume.
- Cost increases with every extra train deployed — minimising cost while maintaining service quality is the goal.
- The goal is NOT to maximise revenue — this is a public service. Move the most people comfortably at the lowest cost.

Your task:
1. Identify the PRIMARY factor(s) driving demand in this specific situation.
2. Evaluate the three options — quote load factor, headway, passengers served, and cost delta.
3. Choose the BEST option that maximises passenger throughput and comfort while minimising unnecessary cost.
4. Explain why the other two options are either insufficient (too few trains) or wasteful (too many trains).
5. End with one clear action sentence for the duty manager.

Return ONLY valid JSON (no markdown, no extra text):
{"choice": "conservative" | "moderate" | "aggressive", "explanation": "your detailed reasoning here"}"""


def _build_options_prompt(inputs: dict, options: list[dict]) -> str:
    from datetime import datetime as _dt
    dt        = _dt.fromisoformat(inputs["datetime"])
    day_label = "Weekend" if dt.weekday() >= 5 else "Weekday"
    weather   = inputs.get("weather", "clear")
    curr_freq = inputs["current_frequency_per_hr"]
    capacity  = inputs.get("train_capacity", 600)
    expected  = options[1]["expected_passengers_per_hr"]
    curr_cap  = curr_freq * capacity
    curr_load = round(min(expected / max(curr_cap, 1), 3.0) * 100, 1)

    if curr_load > 150:
        load_status = "SEVERELY OVERCROWDED — passengers being left behind on platform"
    elif curr_load > 100:
        load_status = "OVERCROWDED — trains at crush load, some passengers cannot board"
    elif curr_load > 80:
        load_status = "NEAR CAPACITY — limited space, risk of boarding delays"
    elif curr_load > 60:
        load_status = "MODERATE LOAD — manageable but approaching busy"
    else:
        load_status = "LOW LOAD — trains running well below capacity"

    events_text = ""
    for e in inputs.get("events", []):
        events_text += f"\n    - {e['name']}: +{e['passengers_per_hr']:,} extra passengers/hr at station"
    if not events_text:
        events_text = "\n    - None"

    emergency_text = f"\n    ⚠ EMERGENCY: {inputs['emergency']}" if inputs.get("emergency") else "\n    - None"

    options_lines = ""
    for opt in options:
        hw = round(60 / max(opt["recommended_frequency_per_hr"], 1), 1)
        options_lines += (
            f"\n  [{opt['label'].upper()}] {opt['recommended_frequency_per_hr']} trains/hr "
            f"(1 train every {hw} min)\n"
            f"    Load factor: {opt['load_factor_pct']}%  |  "
            f"Load change: {opt['congestion_change_pct']:+.1f}%  |  "
            f"Passengers served: {opt['passengers_served_per_hr']:,}/hr "
            f"(+{opt['passengers_served_delta']:,} vs now)\n"
            f"    Cost delta: RM {opt['cost_delta_rm']:+,.0f}  |  "
            f"Time saved: {opt['time_saved_min_per_passenger']} min/passenger\n"
        )

    # Load relevant historical incidents (limit to 1 to keep prompt short)
    history      = _load_relevant_history(inputs, max_records=1)
    history_text = ""
    for h in history:
        o = h.get("outcome", {})
        history_text += (
            f"\n  [{h['date']} — {h['line']}] {h['situation']}\n"
            f"    Deployed: {h['frequency_deployed']} trains/hr | "
            f"Peak load: {o.get('peak_load_factor_pct', '?')}% | "
            f"Passengers served: {o.get('passengers_served_per_hr', '?')}/hr | "
            f"Incidents: {o.get('platform_incidents', 'none')}\n"
            f"    Verdict: {o.get('verdict', '')}\n"
        )
    if not history_text:
        history_text = "\n  No historical data available."

    return f"""SITUATION SUMMARY:
  Line    : {inputs.get('line', 'Unknown')}
  Time    : {dt.strftime('%A, %d %b %Y')} at {dt.strftime('%I:%M %p')} ({day_label})
  Weather : {weather}
  Events  :{events_text}
  Emergency:{emergency_text}

CURRENT STATE:
  Running frequency : {curr_freq} trains/hr (1 train every {round(60/max(curr_freq,1),1)} min)
  Expected passengers: {expected:,}/hr
  Current capacity   : {curr_cap:,} pax/hr ({curr_freq} trains × {capacity} seats)
  Current load factor: {curr_load}% — {load_status}

THREE SCHEDULING OPTIONS:{options_lines}
HISTORICAL PRECEDENTS (past incidents on similar situations):{history_text}
Based on the situation and historical precedents above, which option should the duty manager choose?
Reference past incidents where relevant. Cite load factor, headway, cost vs revenue in your reasoning."""


def _extract_json(text: str) -> dict:
    """Try to pull a JSON object out of the GLM response, even if it has
    markdown code fences or extra text around it."""
    cleaned = text.strip()
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON object in response")
    return json.loads(cleaned[start:end])


def _get_glm_decision(inputs: dict, options: list[dict]) -> tuple[str, str]:
    if not GLM_API_KEY:
        return (
            "moderate",
            "[PLACEHOLDER — set GLM_API_KEY in .env to get real GLM reasoning]\n\n"
            "The moderate option balances service quality with cost efficiency. "
            "It meets the mathematically optimal frequency to handle the predicted demand "
            "without over-provisioning. The conservative option risks platform overcrowding "
            "during peak load, while the aggressive option increases cost beyond what the "
            "demand level justifies. Duty manager should action the moderate schedule "
            "immediately.",
        )

    prompt = _build_options_prompt(inputs, options)
    try:
        response = call_glm(prompt, system=_OPTIONS_SYSTEM, temperature=0.3)
    except (TimeoutError, ConnectionError) as exc:
        moderate = next(o for o in options if o["label"] == "moderate")
        lf = moderate["load_factor_pct"]
        hw = round(60 / max(moderate["recommended_frequency_per_hr"], 1), 1)
        return (
            "moderate",
            f"⚠ GLM unavailable ({exc}).\n\n"
            f"**Math-based recommendation: Moderate**\n"
            f"Run {moderate['recommended_frequency_per_hr']} trains/hr (every {hw} min). "
            f"Projected load factor: {lf}%. "
            f"This meets calculated demand without over- or under-provisioning. "
            f"Conservative risks overcrowding; aggressive adds unnecessary cost at current demand levels.",
        )

    # Attempt 1: parse as JSON {"choice": ..., "explanation": ...}
    try:
        parsed = _extract_json(response)
        choice = parsed.get("choice", "moderate")
        explanation = parsed.get("explanation", response)
        if choice not in ("conservative", "moderate", "aggressive"):
            choice = "moderate"
        return choice, explanation
    except Exception:
        pass

    # Attempt 2: GLM ignored JSON format — keyword-detect the choice
    choice = "moderate"
    lower = response.lower()
    if "conservative" in lower and "aggressive" not in lower:
        choice = "conservative"
    elif "aggressive" in lower and "conservative" not in lower:
        choice = "aggressive"
    return choice, response


def get_glm_recommendation(inputs: dict, options: list[dict]) -> tuple[str, str]:
    """Public: returns (choice, explanation) from GLM. Pure reasoning — no math."""
    return _get_glm_decision(inputs, options)


def recommend(inputs: dict) -> dict:
    options = compute_options(inputs)
    choice, explanation = _get_glm_decision(inputs, options)

    chosen = next((o for o in options if o["label"] == choice), options[1])

    delta = chosen["recommended_frequency_per_hr"] - chosen["current_frequency_per_hr"]
    if delta > 0:
        schedule_update = (
            f"Add {delta} train(s)/hr on {inputs.get('line', 'the line')} "
            f"around {inputs['datetime']}."
        )
    elif delta < 0:
        schedule_update = (
            f"Reduce by {abs(delta)} train(s)/hr on {inputs.get('line', 'the line')} "
            f"around {inputs['datetime']}."
        )
    else:
        schedule_update = "Maintain current schedule."

    return {
        # Chosen option metrics at top level (backward compat)
        **{k: v for k, v in chosen.items() if k != "label"},
        # All options for the UI to display
        "options": options,
        "chosen_option": choice,
        "schedule_update": schedule_update,
        "explanation": explanation,
        "confidence": "high" if abs(delta) <= 4 else "medium",
    }


# ── Daily schedule ────────────────────────────────────────────────────────────

_DAILY_SYSTEM = """You are an expert LRT operations advisor for Malaysian rail lines (Kelana Jaya, Ampang, Sri Petaling).
A duty manager is planning tomorrow's schedule and needs a clear briefing on required adjustments.

You will receive:
- The date, line, and weather forecast
- The event(s) happening and their time window
- The hour-by-hour recommended changes versus the standard schedule

Your job:
1. Explain WHY the adjustments are needed — cite the event, expected passenger surge, and any weather impact
2. Highlight the CRITICAL hours the duty manager must prepare for (typically 1 hour before and after peak)
3. Flag any risks (e.g., post-event exodus, weather compounding crowd)
4. Give a concise shift briefing the duty manager can hand over to the next shift

Keep the response practical, under 220 words, written for an operations staff member (not a technical audience)."""


def _get_glm_daily_reasoning(
    date_str: str,
    line: str,
    weather: str,
    events: list[dict],
    schedule: list[dict],
) -> str:
    from datetime import datetime as _dt
    dt        = _dt.fromisoformat(date_str + "T12:00")
    day_label = "Weekend" if dt.weekday() >= 5 else "Weekday"

    events_text = "\n".join(
        f"  - {e['name']}: {e['start_hour']:02d}:00–{e['end_hour']:02d}:00 | "
        f"{e['passengers_per_hr']:,} extra pax/hr at station"
        for e in events
    )

    event_rows = [s for s in schedule if s["has_event"]]
    affected_text = "\n".join(
        f"  {s['time_slot']}: standard {s['standard_frequency']}/hr → recommended {s['recommended_frequency']}/hr "
        f"(+{s['extra_trains']} trains | load {s['load_factor_pct']}% | extra cost RM {s['extra_cost_rm']:,.0f})"
        for s in event_rows
    )

    # Load relevant history for context
    history      = _load_relevant_history({"line": line, "weather": weather, "events": events})
    history_text = ""
    for h in history:
        o = h.get("outcome", {})
        history_text += (
            f"\n  [{h['date']} — {h['line']}] {h['situation']}\n"
            f"    Deployed: {h['frequency_deployed']}/hr | Load: {o.get('peak_load_factor_pct')}% | "
            f"Verdict: {o.get('verdict', '')}\n"
        )

    prompt = f"""DAILY SCHEDULE BRIEFING REQUEST:
Date    : {dt.strftime('%A, %d %b %Y')} ({day_label})
Line    : {line}
Weather : {weather}

EVENTS:
{events_text}

SCHEDULE ADJUSTMENTS REQUIRED:
{affected_text}

HISTORICAL PRECEDENTS:{history_text if history_text else ' None available.'}

Write a shift briefing for the duty manager covering what to action, when, and why."""

    return call_glm(prompt, system=_DAILY_SYSTEM, temperature=0.3)


def recommend_daily(
    date_str: str,
    line: str,
    weather: str,
    events: list[dict],
    cost_per_train_hr: int = 350,
) -> dict:
    schedule = compute_daily_schedule(date_str, line, weather, events, cost_per_train_hr)

    daily_std_cost   = sum(s["standard_cost_rm"] for s in schedule)
    daily_extra_cost = sum(s["extra_cost_rm"]    for s in schedule)
    daily_total_cost = sum(s["total_cost_rm"]    for s in schedule)

    event_hours = [s for s in schedule if s["has_event"]]
    if event_hours and GLM_API_KEY:
        try:
            explanation = _get_glm_daily_reasoning(date_str, line, weather, events, schedule)
        except (TimeoutError, ConnectionError) as exc:
            explanation = (
                f"⚠ GLM unavailable for shift briefing ({exc}).\n\n"
                "Schedule has been applied based on demand calculations. "
                "Highlighted rows show the adjusted time window — check load factors and deploy extra trains as indicated."
            )
    elif event_hours:
        explanation = (
            "[PLACEHOLDER — set GLM_API_KEY in .env for real briefing]\n\n"
            "Standard schedule adjusted for event window. See highlighted rows for changes."
        )
    else:
        explanation = "No events for this day. Standard schedule applies throughout."

    return {
        "schedule":              schedule,
        "explanation":           explanation,
        "daily_standard_cost_rm": daily_std_cost,
        "daily_extra_cost_rm":   daily_extra_cost,
        "daily_total_cost_rm":   daily_total_cost,
    }
