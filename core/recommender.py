"""Orchestrator: computes three scheduling options, asks GLM to choose one and justify it,
returns a single result dict the UI can consume."""

import json
from pathlib import Path

from core.calculator import compute_options, compute_daily_schedule
from core.glm_client import GLM_API_KEY, call_glm, call_glm_stream

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

CRITICAL — READ THIS FIRST:
The standard schedule (current frequency) is already calibrated for normal demand at this time of day, including peak hours.
Peak hours alone are NOT a reason to add trains — the schedule already accounts for that.
Frequency adjustments are ONLY justified by factors BEYOND the standard schedule:
  - Events (concerts, matches, festivals) creating extra passenger surge
The THREE OPTIONS are always spread around the optimal frequency:
  - CONSERVATIVE = 2 trains/hr below optimal (saves cost, may risk slight overcrowding — you must evaluate if acceptable)
  - MODERATE = optimal frequency for predicted demand
  - AGGRESSIVE = 3 trains/hr above optimal (extra buffer, higher cost — justified only if demand is uncertain or surge risk is high)
If no extra factors exist, moderate = standard schedule. Conservative = standard minus 2 (risky). Aggressive = standard plus 3 (wasteful). Pick moderate unless there is a strong reason otherwise.
  - Weather forcing a safety-based frequency reduction
  - Emergencies requiring service suspension or rapid recovery
If none of these extra factors are present, the standard schedule is correct and should be maintained.

Key facts:
- Normal train capacity: 800 passengers. Target load: 75%. >100% = passengers left behind on platform.
- Headway = minutes between trains. Shorter headway = more trains = less waiting.
- Rainy/stormy weather: trains must slow down for safety (frequency cap applies).
- Events cause sharp surges especially at arrival (1hr before) and exit (1hr after).
- Cost increases with every extra train — only add trains when the extra factors justify it.
- The goal is NOT to maximise revenue — this is a public service.
- EMERGENCY TYPES:
  * signal_failure: ALL trains must stop immediately — no exceptions. Signals protect against collisions.
  * track_incident: ABSOLUTE HIGHEST PRIORITY EMERGENCY. Someone has died, committed suicide, or is on the track. ALL trains must STOP IMMEDIATELY — zero trains, no exceptions, no compromise. This overrides EVERYTHING: events, weather, cost, passenger demand, crowd size. NOTHING justifies running even a single train while a person is on the track or has died on the track. The conservative option (0 trains, full suspension) is the ONLY acceptable choice during the active incident. The next hour AFTER clearance is the RECOVERY phase — deploy maximum frequency to clear the backlog. Two-phase response: Phase 1 = suspend completely (conservative only), Phase 2 = max frequency recovery (aggressive). Moderate and aggressive during the active incident are DANGEROUS and NEGLIGENT.
  * power_failure: reduce frequency, run on backup power only.
  * breakdown: one fewer train — slight reduction.
  * evacuation (fire/bomb): maximum frequency to clear stations rapidly.
  * overcrowding: add trains urgently.

Your task:
1. State the PRIMARY extra factor (event / weather / emergency) driving this recommendation. If none, say so.
2. Evaluate the three options — quote load factor, headway, passengers served, and cost delta.
3. Choose the BEST option. If no extra factors exist, choose the option closest to standard.
4. Explain why the other two are insufficient or wasteful.
5. End with one clear action sentence for the duty manager.

Write your reasoning as plain paragraphs. Do NOT use JSON, bullet points, or markdown headers.
On the very last line write exactly one of these (nothing else on that line):
RECOMMENDATION: conservative
RECOMMENDATION: moderate
RECOMMENDATION: aggressive"""


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

    if inputs.get("emergency"):
        etype = inputs.get("emergency_type", "")
        emergency_text = f"\n    ⚠ EMERGENCY ({etype}): {inputs['emergency']}"
    else:
        emergency_text = "\n    - None"

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


def _parse_recommendation(text: str) -> tuple[str, str]:
    """Extract choice from 'RECOMMENDATION: X' tag and return (choice, explanation_without_tag)."""
    choice = "moderate"
    explanation_lines = []
    for line in text.strip().split("\n"):
        if line.strip().upper().startswith("RECOMMENDATION:"):
            raw = line.split(":", 1)[1].strip().lower()
            for opt in ("conservative", "moderate", "aggressive"):
                if opt in raw:
                    choice = opt
                    break
        else:
            explanation_lines.append(line)
    explanation = "\n".join(explanation_lines).strip()
    # Fallback: if GLM ignored the tag entirely, keyword-detect from full text
    if not explanation:
        explanation = text.strip()
        lower = text.lower()
        if "conservative" in lower and "aggressive" not in lower:
            choice = "conservative"
        elif "aggressive" in lower and "conservative" not in lower:
            choice = "aggressive"
    return choice, explanation


def _get_glm_decision(inputs: dict, options: list[dict]) -> tuple[str, str]:
    if not GLM_API_KEY:
        return (
            "moderate",
            "[PLACEHOLDER — set GLM_API_KEY in .env to get real GLM reasoning]\n\n"
            "The moderate option balances service quality with cost efficiency. "
            "It meets the mathematically optimal frequency to handle the predicted demand "
            "without over-provisioning. The conservative option risks platform overcrowding "
            "during peak load, while the aggressive option increases cost beyond what the "
            "demand level justifies. Duty manager should action the moderate schedule immediately.",
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
            f"Math-based recommendation: Moderate — run {moderate['recommended_frequency_per_hr']} trains/hr "
            f"(every {hw} min). Projected load factor: {lf}%. Meets calculated demand without "
            f"over- or under-provisioning. Conservative risks overcrowding; aggressive adds "
            f"unnecessary cost at current demand levels.",
        )

    return _parse_recommendation(response)


def get_glm_recommendation_stream(inputs: dict, options: list[dict]):
    """Generator: yields GLM reasoning as text chunks (for st.write_stream).
    Last chunk will include 'RECOMMENDATION: <choice>' on its own line."""
    if not GLM_API_KEY:
        moderate = next(o for o in options if o["label"] == "moderate")
        lf = moderate["load_factor_pct"]
        hw = round(60 / max(moderate["recommended_frequency_per_hr"], 1), 1)
        placeholder = (
            "[PLACEHOLDER — set GLM_API_KEY in .env for real reasoning]\n\n"
            f"The moderate option offers the best balance. Running {moderate['recommended_frequency_per_hr']} "
            f"trains/hr (every {hw} min) achieves a {lf}% load factor, meeting demand without "
            f"over-provisioning. Conservative risks overcrowding; aggressive adds unnecessary cost "
            f"at current demand levels. Deploy the moderate schedule immediately.\n\n"
            f"RECOMMENDATION: moderate"
        )
        for char in placeholder:
            yield char
        return

    prompt = _build_options_prompt(inputs, options)
    yield from call_glm_stream(prompt, system=_OPTIONS_SYSTEM, temperature=0.3)


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
    emergency_type: str | None = None,
    emergency_hour: int | None = None,
    emergency_duration: int = 1,
) -> str:
    from datetime import datetime as _dt
    dt        = _dt.fromisoformat(date_str + "T12:00")
    day_label = "Weekend" if dt.weekday() >= 5 else "Weekday"

    events_text = "\n".join(
        f"  - {e['name']}: {e['start_hour']:02d}:00–{e['end_hour']:02d}:00 | "
        f"{e['passengers_per_hr']:,} extra pax/hr at station"
        for e in events
    ) or "  None"

    adjusted_rows = [s for s in schedule if s["recommended_frequency"] != s["standard_frequency"]]
    affected_text = "\n".join(
        f"  {s['time_slot']}: standard {s['standard_frequency']}/hr → recommended {s['recommended_frequency']}/hr "
        f"({s['extra_trains']:+d} trains | load {s['load_factor_pct']}% | extra cost RM {s['extra_cost_rm']:,.0f})"
        for s in adjusted_rows
    ) or "  No frequency changes from standard."

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

    if emergency_type and emergency_hour is not None:
        em_end = emergency_hour + emergency_duration
        emergency_text = (
            f"  Type    : {emergency_type}\n"
            f"  Window  : {emergency_hour:02d}:00–{em_end:02d}:00 ({emergency_duration}h)\n"
            f"  Impact  : Service suspended/reduced during window. "
            f"Recovery trains deployed at {em_end:02d}:00 to clear backlog."
        )
    else:
        emergency_text = "  None"

    prompt = f"""DAILY SCHEDULE BRIEFING REQUEST:
Date      : {dt.strftime('%A, %d %b %Y')} ({day_label})
Line      : {line}
Weather   : {weather}

EMERGENCY:
{emergency_text}

EVENTS:
{events_text}

SCHEDULE ADJUSTMENTS REQUIRED:
{affected_text}

HISTORICAL PRECEDENTS:{history_text if history_text else ' None available.'}

Write a shift briefing for the duty manager covering what to action, when, and why."""

    return call_glm(prompt, system=_DAILY_SYSTEM, temperature=0.3)


def get_glm_daily_reasoning_stream(
    date_str: str,
    line: str,
    weather: str,
    events: list[dict],
    schedule: list[dict],
    emergency_type: str | None = None,
    emergency_hour: int | None = None,
    emergency_duration: int = 1,
):
    """Generator: yields GLM shift briefing as text chunks for st.write_stream."""
    if not GLM_API_KEY:
        yield "[PLACEHOLDER — set GLM_API_KEY in .env for real briefing]\n\nSchedule adjusted. See highlighted rows for changes."
        return
    # Reuse the same prompt-building logic
    from datetime import datetime as _dt
    dt        = _dt.fromisoformat(date_str + "T12:00")
    day_label = "Weekend" if dt.weekday() >= 5 else "Weekday"
    events_text = "\n".join(
        f"  - {e['name']}: {e['start_hour']:02d}:00–{e['end_hour']:02d}:00 | "
        f"{e['passengers_per_hr']:,} extra pax/hr at station"
        for e in events
    ) or "  None"
    adjusted_rows = [s for s in schedule if s["recommended_frequency"] != s["standard_frequency"]
                     or s.get("em_status")]
    affected_text = "\n".join(
        f"  {s['time_slot']}: {s['standard_frequency']}/hr → {s['recommended_frequency']}/hr "
        f"({s['extra_trains']:+d} trains | load {s['load_factor_pct']}% | {s.get('em_status','adjusted')})"
        for s in adjusted_rows
    ) or "  No frequency changes."
    if emergency_type and emergency_hour is not None:
        em_end = emergency_hour + emergency_duration
        emergency_text = (
            f"  Type: {emergency_type} | "
            f"Window: {emergency_hour:02d}:00–{em_end:02d}:00 | "
            f"Recovery deployed at {em_end:02d}:00"
        )
    else:
        emergency_text = "  None"
    prompt = f"""DAILY SCHEDULE BRIEFING REQUEST:
Date      : {dt.strftime('%A, %d %b %Y')} ({day_label})
Line      : {line}
Weather   : {weather}
EMERGENCY : {emergency_text}
EVENTS    :
{events_text}
SCHEDULE ADJUSTMENTS:
{affected_text}
Write a shift briefing for the duty manager covering what to action, when, and why."""
    yield from call_glm_stream(prompt, system=_DAILY_SYSTEM, temperature=0.3)


def recommend_daily(
    date_str: str,
    line: str,
    weather: str,
    events: list[dict],
    cost_per_train_hr: int = 350,
    weather_window: tuple[int, int] | None = None,
    emergency_type: str | None = None,
    emergency_hour: int | None = None,
    emergency_duration: int = 1,
) -> dict:
    schedule = compute_daily_schedule(date_str, line, weather, events, cost_per_train_hr,
                                      weather_window=weather_window,
                                      emergency_type=emergency_type,
                                      emergency_hour=emergency_hour,
                                      emergency_duration=emergency_duration)

    daily_std_cost   = sum(s["standard_cost_rm"] for s in schedule)
    daily_extra_cost = sum(s["extra_cost_rm"]    for s in schedule)
    daily_total_cost = sum(s["total_cost_rm"]    for s in schedule)

    has_adjustments = bool(events or weather != "clear" or emergency_type)

    return {
        "schedule":               schedule,
        "has_adjustments":        has_adjustments,
        "daily_briefing_params":  {         # app.py streams this on demand
            "date_str":          date_str,
            "line":              line,
            "weather":           weather,
            "events":            events,
            "emergency_type":    emergency_type,
            "emergency_hour":    emergency_hour,
            "emergency_duration": emergency_duration,
        },
        "daily_standard_cost_rm": daily_std_cost,
        "daily_extra_cost_rm":    daily_extra_cost,
        "daily_total_cost_rm":    daily_total_cost,
    }
