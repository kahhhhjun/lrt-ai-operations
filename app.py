"""Streamlit UI for LRT AI Operations decision-support.
Run: streamlit run app.py"""

import json
from datetime import datetime, date as date_type, time
from pathlib import Path

import pandas as pd
import streamlit as st

from core.calculator import WEATHER_MAX_FREQ, TRAIN_CAPACITY, compute_options, default_frequency as _dfreq
from core.glm_client import extract_inputs_from_text, get_glm_pax_factors
from core.recommender import (get_glm_recommendation, get_glm_recommendation_stream,
                              get_glm_daily_reasoning_stream, recommend_daily)
from core.database import init_db, save_schedule, load_schedule, delete_schedule, list_saved

init_db()

st.set_page_config(page_title="LRT AI Operations", layout="wide")
st.title("LRT AI Operations — Decision Support")
st.caption("Powered by Z.AI GLM · Kelana Jaya · Ampang · Sri Petaling")

LINES = ["Kelana Jaya", "Ampang", "Sri Petaling"]
DAYS  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_REF_DATES = {
    "Monday": "2026-04-20", "Tuesday": "2026-04-21", "Wednesday": "2026-04-22",
    "Thursday": "2026-04-23", "Friday": "2026-04-24",
    "Saturday": "2026-04-25", "Sunday": "2026-04-26",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_cost(val: int) -> str:
    if val > 0:  return f"+RM {val:,.0f}"
    if val < 0:  return f"−RM {abs(val):,.0f}"
    return "–"


def _apply_option_to_window(schedule, chosen, tune_s, tune_e, cost_per_hr, weather):
    """Apply the chosen option's delta to all event-affected and window hours.
    Uses a consistent delta (−2 / 0 / +3) from moderate so the table always
    matches the option cards regardless of per-hour baseline differences."""
    max_freq = WEATHER_MAX_FREQ.get(weather, 20)
    delta = {"conservative": -2, "moderate": 0, "aggressive": 3}.get(chosen, 0)
    if delta == 0:
        return schedule
    for s in schedule:
        in_window      = tune_s <= s["hour"] < tune_e
        event_affected = in_window and (s.get("has_event") or s.get("is_event_tail"))
        if not (in_window or event_affected):
            continue
        new_freq = max(1, min(s["recommended_frequency"] + delta, max_freq))
        extra    = new_freq - s["standard_frequency"]
        exp_pax  = s.get("expected_passengers_per_hr", 0)
        new_cap  = new_freq * TRAIN_CAPACITY
        s["recommended_frequency"] = new_freq
        s["extra_trains"]          = extra
        s["extra_cost_rm"]         = extra * cost_per_hr
        s["total_cost_rm"]         = s["standard_cost_rm"] + extra * cost_per_hr
        s["load_factor_pct"]       = round(min(exp_pax / max(new_cap, 1), 2.0) * 100, 1)
        s["headway_rec_min"]       = int(round(60 / max(new_freq, 1)))
    return schedule


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATE & LINE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<a name="schedule-top"></a>', unsafe_allow_html=True)
st.subheader("Full Day Schedule")

_col_inputs, _col_history = st.columns(2)
with _col_inputs:
    sch_date = st.date_input("Date", value=date_type.today(), key="sch_date")
    sch_line = st.selectbox("LRT line", LINES, key="sch_line")
sch_weekday  = sch_date.weekday()
sch_day_name = DAYS[sch_weekday]

with _col_history:
    _saved_list = list_saved(sch_line)
    if _saved_list:
        _lines = []
        for r in _saved_list:
            _d = date_type.fromisoformat(r["date"]).strftime("%d %b %Y").lstrip("0")
            _parts = [r["weather"]]
            _events = json.loads(r.get("events_json") or "[]")
            if _events:
                _parts.append(", ".join(e["name"] for e in _events))
            if r.get("emergency_type"):
                _parts.append(r["emergency_type"].replace("_", " "))
            _lines.append(f"- {_d} — {' · '.join(_parts)}")
        st.info(f"📋 **Previously adjusted — {sch_line} Line**\n\n" + "\n".join(_lines))
    else:
        st.info(f"📋 **{sch_line} Line**\n\nNo previously adjusted dates.")

# Auto-load schedule when date or line changes — prefer saved DB record if exists
_key = f"{sch_date}_{sch_line}"
if st.session_state.get("_sch_key") != _key:
    try:
        _def = recommend_daily(sch_date.isoformat(), sch_line, "clear", [], 350)
    except Exception as _e:
        st.error(f"Could not load schedule: {_e}")
        st.stop()
    _saved = load_schedule(sch_date.isoformat(), sch_line)
    if _saved:
        # Reconstruct a result-like dict from the saved record
        _saved_result = {
            "schedule":               _saved["schedule"],
            "has_adjustments":        True,
            "daily_briefing_params":  {},
            "daily_standard_cost_rm": _saved["total_std_cost"],
            "daily_extra_cost_rm":    _saved["total_extra_cost"],
            "daily_total_cost_rm":    _saved["total_cost"],
        }
        st.session_state.update({
            "_sch_key": _key, "_sch_default": _def,
            "_sch_result": _saved_result, "_sch_mode": "updated",
            "_sch_events": _saved["events"],
            "_tune_start": _saved.get("tune_start", 6),
            "_tune_end":   _saved.get("tune_end", 24),
            "_upd_weather": _saved["weather"],
            "_a_em_type": _saved.get("emergency_type"),
            "_analysis": None, "_chosen_option": "moderate",
            "_db_saved_at": _saved["saved_at"],
            "_briefing_text": "",  # loaded from DB — skip briefing on reload
            "_from_db": True,
        })
    else:
        st.session_state.update({
            "_sch_key": _key, "_sch_default": _def, "_sch_result": _def,
            "_sch_mode": "default", "_sch_events": [],
            "_tune_start": 6, "_tune_end": 24,
            "_analysis": None, "_chosen_option": "moderate",
            "_db_saved_at": None,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCHEDULE TABLE (always visible, reflects current state)
# ═══════════════════════════════════════════════════════════════════════════════
result    = st.session_state.get("_sch_result")
mode      = st.session_state.get("_sch_mode", "default")
ev_active = st.session_state.get("_sch_events", [])
tune_s    = st.session_state.get("_tune_start", 6)
tune_e    = st.session_state.get("_tune_end", 23)

if result is None:
    st.stop()

schedule = result["schedule"]

title = f"### {sch_date.strftime('%d %b %Y')} ({sch_day_name}) — {sch_line} Line"
if mode == "updated" and ev_active:
    title += f"  *(with {ev_active[0]['name']})*"
st.markdown(title)

if mode == "updated":
    _banner_parts = [f"weather: {st.session_state.get('_upd_weather', '—')}"]
    _saved_em = st.session_state.get("_a_em_type")
    if _saved_em:
        _banner_parts.append(f"emergency: {_saved_em.replace('_', ' ')}")
    if ev_active:
        _banner_parts.append(f"event: {ev_active[0]['name']}")
    st.success(
        f"Situation period **{tune_s:02d}:00–{tune_e:02d}:00** | "
        + " | ".join(_banner_parts)
        + ".  Press **Reset to standard** below to revert."
    )
    st.caption("🟡 Highlighted rows = adjusted time window")
else:
    st.caption("Standard timetable. Use the panel below to adjust specific hours.")

rows = []
for s in schedule:
    is_tail   = s.get("is_event_tail", False)
    em_status = s.get("em_status")
    freq_changed = mode == "updated" and s["recommended_frequency"] != s["standard_frequency"]
    show_adj  = freq_changed or is_tail or bool(em_status) or (mode == "updated" and s.get("has_event"))
    if show_adj:
        delta = s["extra_trains"]
        if delta > 0:   tc = f"{s['recommended_frequency']} (+{delta})"
        elif delta < 0: tc = f"{s['recommended_frequency']} (−{abs(delta)})"
        else:           tc = str(s["recommended_frequency"])
        if em_status == "active":
            status = "🚨 Emergency"
        elif em_status == "recovery1":
            status = "⚠️ Recovery (backlog)"
        elif em_status == "recovery2":
            status = "↗ Tapering"
        elif s["has_event"]:
            status = ", ".join(s["event_names"])
        elif delta < 0:
            status = "Reduced"
        elif delta > 0:
            status = "Increased"
        else:
            status = "Adjusted"
        rows.append({
            "Time":             s["time_slot"],
            "Expected Pax/hr":  f"{s['expected_passengers_per_hr']:,}",
            "Frequency":        f"every {s['headway_rec_min']} min",
            "Trains/hr":        tc,
            "Cost (RM/hr)":     f"RM {s['total_cost_rm']:,.0f}",
            "Load factor":      f"{s['load_factor_pct']}%",
            "Status":           status,
        })
    else:
        rows.append({
            "Time":             s["time_slot"],
            "Expected Pax/hr":  f"{s['expected_passengers_per_hr']:,}",
            "Frequency":        f"every {s['headway_std_min']} min",
            "Trains/hr":        str(s["standard_frequency"]),
            "Cost (RM/hr)":     f"RM {s['standard_cost_rm']:,.0f}",
            "Load factor":      f"{s['standard_load_factor_pct']}%",
            "Status":           "Standard",
        })

df = pd.DataFrame(rows)

def _style_rows(row):
    s = schedule[row.name]
    em = s.get("em_status")
    if em == "active":
        return ["background-color: #7b1111; color: #ffffff; font-weight: bold"] * len(row)
    if em == "recovery1":
        return ["background-color: #7b4a00; color: #ffffff; font-weight: bold"] * len(row)
    if em == "recovery2":
        return ["background-color: #4a4a00; color: #ffffff; font-weight: bold"] * len(row)
    if mode == "updated" and s["recommended_frequency"] != s["standard_frequency"] and not s.get("em_status") and not s.get("is_event_tail") and not s.get("has_event"):
        return ["background-color: #1a3a5c; color: #ffffff; font-weight: bold"] * len(row)
    if s.get("is_event_tail") or (mode == "updated" and s.get("has_event")):
        return ["background-color: #2a4a3c; color: #ffffff; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    df.style.apply(_style_rows, axis=1),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Time":             st.column_config.TextColumn("Time",             help="Hour slot for this schedule entry."),
        "Expected Pax/hr":  st.column_config.TextColumn("Expected Pax/hr",  help="Estimated total passengers arriving at the station per hour, including event surge and weather effects."),
        "Frequency":        st.column_config.TextColumn("Frequency",        help="How often a train arrives — shorter interval means more trains."),
        "Trains/hr":        st.column_config.TextColumn("Trains/hr",        help="Number of trains per hour. Bracket shows change vs standard schedule: (+2) = 2 extra trains, (−1) = 1 fewer train."),
        "Cost (RM/hr)":     st.column_config.TextColumn("Cost (RM/hr)",     help="Total operating cost for this hour based on trains deployed × RM 350 per train-hour."),
        "Load factor":      st.column_config.TextColumn("Load factor",      help="How full the trains are. 75% = comfortable target. 100% = completely full. Above 100% = passengers left behind on platform."),
        "Status":           st.column_config.TextColumn("Status",           help="Standard = no change. Increased/Reduced = frequency adjusted. Event rows show the event name. Emergency rows show the incident type."),
    }
)

if mode == "updated":
    st.markdown("---")
    _saved_at = st.session_state.get("_db_saved_at")
    if _saved_at:
        st.caption(f"💾 Last saved: {_saved_at}")

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("💾 Save schedule", type="primary", key="save_btn"):
            save_schedule(
                date=sch_date.isoformat(),
                line=sch_line,
                schedule=schedule,
                weather=st.session_state.get("_upd_weather", "clear"),
                events=ev_active,
                emergency_type=st.session_state.get("_a_em_type"),
                tune_start=st.session_state.get("_tune_start", 6),
                tune_end=st.session_state.get("_tune_end", 24),
                total_std_cost=result["daily_standard_cost_rm"],
                total_extra_cost=result["daily_extra_cost_rm"],
                total_cost=result["daily_total_cost_rm"],
            )
            st.session_state["_db_saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.success("Schedule saved!")
            st.rerun()
    with btn2:
        if st.button("Reset to standard", key="sch_reset_table", type="primary"):
            delete_schedule(sch_date.isoformat(), sch_line)
            st.session_state.update({
                "_sch_result":   st.session_state["_sch_default"],
                "_sch_mode":     "default",
                "_sch_events":   [],
                "_tune_start":   6,
                "_tune_end":     24,
                "_analysis":     None,
                "_db_saved_at":  None,
                "_briefing_text": None,
            })
            st.rerun()

# ── Daily cost ────────────────────────────────────────────────────────────────
std_total = result["daily_standard_cost_rm"]
extra_win = sum(s["extra_cost_rm"] for s in schedule) if mode == "updated" else 0
total_day = std_total + extra_win

st.markdown("### Daily cost")
m1, m2, m3 = st.columns(3)
m1.metric(f"Standard {sch_day_name} cost", f"RM {std_total:,.0f}")
m2.metric("Extra cost (adjusted hours)", f"RM {extra_win:,.0f}",
          delta=extra_win if mode == "updated" else None, delta_color="inverse")
m3.metric("Total cost this day", f"RM {total_day:,.0f}")

# ── Weekly cost overview ──────────────────────────────────────────────────────
with st.expander("📊 Weekly cost overview (standard schedule, no events)"):
    st.caption("Standard operating cost per day for this line — same every week.")
    weekly_rows, weekly_total = [], 0
    for day_name in DAYS:
        dr = recommend_daily(date_str=_REF_DATES[day_name], line=sch_line,
                             weather="clear", events=[], cost_per_train_hr=350)
        c = dr["daily_standard_cost_rm"]
        weekly_total += c
        weekly_rows.append({
            "Day":      day_name,
            "Type":     "Weekend" if DAYS.index(day_name) >= 5 else "Weekday",
            "Cost/day": f"RM {c:,.0f}",
        })
    weekly_rows.append({"Day": "WEEKLY TOTAL", "Type": "", "Cost/day": f"RM {weekly_total:,.0f}"})
    st.dataframe(pd.DataFrame(weekly_rows), use_container_width=True, hide_index=True)

# ── GLM shift briefing (streamed once per update; skipped on page reload) ─────
if mode == "updated" and result.get("has_adjustments"):
    cached_briefing = st.session_state.get("_briefing_text")
    if cached_briefing:  # already streamed this session — show cached text
        with st.expander("Shift briefing", expanded=True):
            st.markdown(cached_briefing)
    elif cached_briefing is None:  # fresh update — stream now
        bp = result.get("daily_briefing_params", {})
        try:
            with st.status("GLM is writing shift briefing...", expanded=True) as briefing_status:
                briefing_text = st.write_stream(get_glm_daily_reasoning_stream(
                    date_str=bp.get("date_str", sch_date.isoformat()),
                    line=bp.get("line", sch_line),
                    weather=bp.get("weather", "clear"),
                    events=bp.get("events", []),
                    schedule=schedule,
                    emergency_type=bp.get("emergency_type"),
                    emergency_hour=bp.get("emergency_hour"),
                    emergency_duration=bp.get("emergency_duration", 1),
                ))
            briefing_status.update(label="Shift briefing ready", state="complete", expanded=True)
            st.session_state["_briefing_text"] = briefing_text
        except Exception:
            st.info("Schedule updated. See highlighted rows for changes.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADJUSTMENT PANEL (hidden after schedule is applied)
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "updated":
    st.stop()

st.markdown("---")
st.markdown("### Adjust schedule")

# Step 1 — time window
st.markdown("**Step 1 — Select time window to adjust**")
tw1, tw2 = st.columns(2)
with tw1:
    tune_start = st.slider("Situation starts at", 6, 23, 6,  format="%d:00", key="tune_start")
with tw2:
    tune_end   = st.slider("Situation ends at",   7, 24, 24, format="%d:00", key="tune_end")
st.caption(
    f"Weather / emergency / event occurs between **{tune_start:02d}:00 – {tune_end:02d}:00**. "
    "The rest of the day stays on the standard timetable."
)

# If sliders changed since last Analyse, invalidate the analysis so Apply can't use stale window
if st.session_state.get("_analysis") and (
    tune_start != st.session_state.get("_a_tune_s") or
    tune_end   != st.session_state.get("_a_tune_e")
):
    st.session_state["_analysis"] = None

# Step 2 — situation input
st.markdown("**Step 2 — Describe the situation**")
input_method = st.radio(
    "", ["Manual inputs", "Describe in text (GLM extracts)"],
    horizontal=True, key="input_method", label_visibility="collapsed",
)

_EMERGENCY_OPTIONS = [
    "None", "track_incident", "signal_failure", "power_failure",
    "breakdown", "evacuation", "overcrowding",
]
_EMERGENCY_LABELS = {
    "None": "None",
    "track_incident":  "Track incident (person/suicide on track)",
    "signal_failure":  "Signal failure",
    "power_failure":   "Power failure",
    "breakdown":       "Train breakdown",
    "evacuation":      "Evacuation (fire / bomb threat)",
    "overcrowding":    "Overcrowding / stampede risk",
}

sch_cost = 350  # fixed running cost — not exposed in UI

if input_method == "Manual inputs":
    ai1, ai2, ai3 = st.columns(3)
    with ai1:
        st.markdown("**Weather**")
        sch_weather = st.selectbox("Condition", ["clear", "cloudy", "rainy", "stormy"],
                                   key="sch_weather", label_visibility="collapsed")
    with ai2:
        st.markdown("**Emergency**")
        em_type_key = st.selectbox("Emergency type", _EMERGENCY_OPTIONS,
                                   format_func=lambda x: _EMERGENCY_LABELS[x],
                                   key="em_type", label_visibility="collapsed")
    with ai3:
        st.markdown("**Event**")
        ev_type = st.selectbox("Event type", [
            "none", "concert", "football_match", "festival", "marathon",
            "exhibition", "religious_event",
        ], key="ev_type", format_func=lambda x: {
            "none": "None", "concert": "Concert",
            "football_match": "Football match", "festival": "Festival",
            "marathon": "Marathon / run",
            "exhibition": "Exhibition / convention",
            "religious_event": "Religious event",
        }.get(x, x), label_visibility="collapsed")
        if ev_type != "none":
            ev_pax = st.number_input("Estimated crowd for event (pax/hr)", value=0, key="ev_pax")
        else:
            ev_pax = 0

elif input_method == "Describe in text (GLM extracts)":
    sit_text = st.text_area(
        "Describe the situation",
        height=100,
        placeholder='e.g. "Blackpink concert at Bukit Jalil, around 4,000 extra passengers per hour. Weather is rainy. Train breakdown reported."',
        key="sit_text",
    )
    sch_cost = 350

# Step 3 — Analyse
st.markdown("**Step 3 — Analyse options**")

if st.button("Analyse", type="primary", key="analyse_btn"):
    if tune_end <= tune_start:
        st.error(f"'Ends at' must be after 'Starts at' — minimum 1 hour (e.g. {tune_start:02d}:00 – {tune_start+1:02d}:00).")
        st.stop()
    if input_method == "Manual inputs":
        _pax_now  = st.session_state.get("ev_pax", 0) or 0
        _type_now = st.session_state.get("ev_type", "none")
        if _pax_now < 0 or _pax_now > 30_000:
            st.error(f"Estimated crowd must be between **0 and 30,000** pax/hr. You entered {_pax_now:,}.")
            st.stop()
        if _type_now != "none" and _pax_now == 0:
            st.error("You selected an event type but entered 0 crowd. Please enter the crowd size or set event to None.")
            st.stop()

    # Midpoint of window as the representative hour for single-hour analysis
    rep_hour = (tune_start + tune_end) // 2

    if input_method == "Describe in text (GLM extracts)":
        raw_text = st.session_state.get("sit_text", "").strip()
        if not raw_text:
            st.warning("Please describe the situation first.")
            st.stop()
        with st.spinner("GLM reading description..."):
            try:
                extracted = extract_inputs_from_text(raw_text)
            except TimeoutError:
                st.warning("GLM timed out reading the description — using keyword fallback instead.")
                from core.glm_client import _placeholder_extract
                extracted = _placeholder_extract(raw_text)
            except Exception as _ex:
                st.warning(f"GLM unavailable ({_ex}) — using keyword fallback.")
                from core.glm_client import _placeholder_extract
                extracted = _placeholder_extract(raw_text)
        a_weather = extracted.get("weather", "clear")
        a_ev_raw  = extracted.get("events", [])
        a_line    = extracted.get("line") or sch_line
        st.info(f"GLM extracted — weather: **{a_weather}** | events: {[e['name'] for e in a_ev_raw] or ['none']}")

    else:  # Manual inputs
        _ev_type_labels = {
            "concert": "Concert", "football_match": "Football match",
            "festival": "Festival", "marathon": "Marathon / run",
            "exhibition": "Exhibition / convention",
            "religious_event": "Religious event",
        }
        a_weather  = st.session_state.get("sch_weather", "clear")
        _ev_type   = st.session_state.get("ev_type", "none")
        _ev_pax    = int(st.session_state.get("ev_pax", 0) or 0)
        _ev_name   = _ev_type_labels.get(_ev_type, "Event")
        a_ev_raw   = [{"name": _ev_name, "passengers_per_hr": _ev_pax,
                       "event_type": _ev_type}] \
                     if _ev_type != "none" and _ev_pax > 0 else []
        a_line     = sch_line
        st.info(f"Manual inputs — weather: **{a_weather}** | event: {_ev_name if a_ev_raw else 'none'} | pax/hr: {_ev_pax if a_ev_raw else 0}")

    # Pull emergency fields from extraction (text/image) or manual selector
    a_emergency      = extracted.get("emergency")      if input_method != "Manual inputs" else None
    a_emergency_type = extracted.get("emergency_type") if input_method != "Manual inputs" else None
    # Manual emergency override
    if input_method == "Manual inputs":
        _em_sel = st.session_state.get("em_type", "None")
        if _em_sel != "None":
            a_emergency_type = _em_sel
            a_emergency      = _EMERGENCY_LABELS.get(_em_sel, _em_sel)
    a_emergency_dur = max(1, tune_end - tune_start)

    # Show emergency banner immediately so duty manager sees it before GLM finishes
    _EMERGENCY_LABELS = {
        "track_incident":  ("🚨 TRACK INCIDENT — Service suspended. Maximum frequency required on resumption.", "error"),
        "signal_failure":  ("⚠️ SIGNAL FAILURE — Reduce frequency for safety.", "warning"),
        "power_failure":   ("⚠️ POWER FAILURE — Service disrupted. Reduce frequency.", "warning"),
        "breakdown":       ("🔧 TRAIN BREAKDOWN — One train out of service.", "warning"),
        "evacuation":      ("🚨 EVACUATION — Maximum frequency to clear stations.", "error"),
        "overcrowding":    ("⚠️ OVERCROWDING EMERGENCY — Add trains urgently.", "warning"),
    }
    if a_emergency_type and a_emergency_type in _EMERGENCY_LABELS:
        msg, level = _EMERGENCY_LABELS[a_emergency_type]
        getattr(st, level)(f"{msg}\n\nDetected: *{a_emergency}*")
    elif a_emergency:
        st.warning(f"⚠️ Emergency detected: *{a_emergency}*")

    # GLM-predicted pax multipliers (fast model, one call per Analyse)
    _pax_factors = get_glm_pax_factors(a_weather, a_emergency_type, rep_hour, sch_line)

    inputs = {
        "datetime":                 datetime.combine(sch_date, time(hour=rep_hour)).isoformat(timespec="minutes"),
        "weather":                  a_weather,
        "events":                   a_ev_raw,
        "emergency":                a_emergency,
        "emergency_type":           a_emergency_type,
        "line":                     a_line,
        "current_frequency_per_hr": _dfreq(rep_hour, sch_date.weekday()),
        "train_capacity":           TRAIN_CAPACITY,
        "running_cost_per_train_hr": int(sch_cost),
        **_pax_factors,
    }

    # Phase 1: instant math — compute 3 options without GLM
    options = compute_options(inputs)

    st.session_state.update({
        "_analysis": {
            "options":       options,
            "chosen_option": "moderate",   # default until GLM responds
            "explanation":   None,          # None triggers Phase 2 on next render
        },
        "_glm_inputs":    inputs,           # saved for Phase 2
        "_chosen_option": "moderate",
        "_a_weather":     a_weather,
        "_a_cost":        int(sch_cost),
        "_a_ev_raw":      a_ev_raw,
        "_a_tune_s":      tune_start,
        "_a_tune_e":      tune_end,
        "_a_line":        a_line,
        "_a_em_type":     a_emergency_type,
        "_a_em_dur":      a_emergency_dur,
        "_pax_factors":   _pax_factors,
    })
    st.rerun()  # shows 3 cards immediately; Phase 2 triggers on this render


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANALYSIS RESULTS (two-phase: cards instant, GLM reasoning after)
# ═══════════════════════════════════════════════════════════════════════════════
analysis = st.session_state.get("_analysis")

if analysis and analysis.get("options"):
    options  = analysis["options"]
    glm_pick = analysis.get("chosen_option", "moderate")

    st.markdown("---")
    st.markdown("**Step 4 — Three scheduling options (calculated)**")

    # ── Phase 1 result: 3 cards appear instantly (pure math, no GLM wait) ────
    reasoning_done = analysis.get("explanation") is not None
    cols = st.columns(3)
    for col, opt in zip(cols, options):
        freq_delta = opt["recommended_frequency_per_hr"] - opt["current_frequency_per_hr"]
        with col:
            if reasoning_done and opt["label"] == glm_pick:
                st.success(f"**{opt['label'].upper()}** ✅ GLM pick")
            else:
                st.info(f"**{opt['label'].upper()}**")
            freq_val = opt["recommended_frequency_per_hr"]
            if freq_val == 0:
                freq_label = "SUSPENDED"
            elif freq_delta > 0:
                freq_label = {"conservative": "Slightly increased",
                              "moderate":     "Moderately increased",
                              "aggressive":   "Significantly increased"}.get(opt["label"], "Increased")
            elif freq_delta < 0:
                freq_label = {"conservative": "Significantly decreased",
                              "moderate":     "Moderately decreased",
                              "aggressive":   "Slightly decreased"}.get(opt["label"], "Decreased")
            else:
                freq_label = "No change"
            st.metric(
                "Frequency",
                freq_label,
                help="Whether train frequency increases, decreases, or stays the same vs current schedule.",
            )
            st.metric(
                "Load factor",
                f"{opt['load_factor_pct']}%",
                delta=opt["congestion_change_pct"],
                delta_color="inverse",
                help="How full the trains will be. 75% = comfortable target. 100% = completely full. Above 100% = passengers left behind on the platform.",
            )
            st.metric(
                "Cost delta",
                f"RM {opt['cost_delta_rm']:,.0f}",
                delta=opt["cost_delta_rm"],
                delta_color="inverse",
                help="Extra operating cost per hour vs current schedule. Positive = more spending to add trains. Negative = saving by reducing trains.",
            )
            st.metric(
                "Pax served",
                f"{opt['passengers_served_per_hr']:,}/hr",
                delta=opt["passengers_served_delta"],
                help="Passengers who can actually board trains per hour. Capped by total train capacity — when load factor exceeds 100%, not everyone can board.",
            )

    # ── Phase 2: stream GLM reasoning live (animated status + live text) ────────
    if not reasoning_done:
        glm_inputs = st.session_state.get("_glm_inputs")
        if glm_inputs:
            from core.recommender import _parse_recommendation
            try:
                with st.status("GLM is reasoning...", expanded=True) as glm_status:
                    full_text = st.write_stream(
                        get_glm_recommendation_stream(glm_inputs, options)
                    )
                    glm_status.update(label="Reasoning complete", state="complete", expanded=True)
            except Exception as _stream_err:
                # Streaming failed (connection dropped) — retry with a regular blocking call
                st.warning(f"Streaming interrupted ({type(_stream_err).__name__}) — retrying without streaming...")
                try:
                    with st.spinner("GLM reasoning..."):
                        _choice, _expl = get_glm_recommendation(glm_inputs, options)
                    full_text = _expl + f"\n\nRECOMMENDATION: {_choice}"
                except (TimeoutError, ConnectionError) as exc:
                    moderate = next(o for o in options if o["label"] == "moderate")
                    lf = moderate["load_factor_pct"]
                    hw = round(60 / max(moderate["recommended_frequency_per_hr"], 1), 1)
                    full_text = (
                        f"⚠ GLM unavailable ({exc}).\n\n"
                        f"Math-based recommendation: Moderate — {moderate['recommended_frequency_per_hr']}/hr "
                        f"(every {hw} min), load factor {lf}%. Meets demand without over-provisioning.\n\n"
                        f"RECOMMENDATION: moderate"
                    )
                    st.warning(full_text)
            choice, explanation = _parse_recommendation(full_text)
            st.session_state["_analysis"]["chosen_option"] = choice
            st.session_state["_analysis"]["explanation"]   = explanation
            st.session_state["_chosen_option"]             = choice
            st.session_state.pop("_glm_inputs", None)
            st.rerun()

    # ── Phase 2 complete: show reasoning + controls ───────────────────────────
    else:
        with st.expander("GLM reasoning", expanded=True):
            st.write(analysis["explanation"])

        cur_pick = st.session_state.get("_chosen_option", glm_pick)
        _options = ["conservative", "moderate", "aggressive"]
        _labels  = [
            f"{o}  ✦ Suggested" if o == glm_pick else o
            for o in _options
        ]
        new_pick = st.radio(
            "Apply this option to the schedule:",
            _options,
            format_func=lambda o: f"{o}  ✦ Suggested" if o == glm_pick else o,
            index=_options.index(cur_pick),
            horizontal=True, key="option_radio",
        )
        st.session_state["_chosen_option"] = new_pick

        apply_btn = st.button("Apply to schedule", type="primary", key="apply_btn")
        reset_btn = False

        if apply_btn:
            a_weather    = st.session_state.get("_a_weather", "clear")
            a_cost       = st.session_state.get("_a_cost", 350)
            a_ev_raw     = st.session_state.get("_a_ev_raw", [])
            a_ts         = st.session_state.get("_a_tune_s", 6)
            a_te         = st.session_state.get("_a_tune_e", 23)
            a_em_type    = st.session_state.get("_a_em_type")
            a_em_dur     = max(1, a_te - a_ts)
            a_pax_factors = st.session_state.get("_pax_factors", {})

            events_daily = [
                {"name": ev["name"], "start_hour": a_ts,
                 "end_hour": a_te, "passengers_per_hr": ev["passengers_per_hr"],
                 "event_type": ev.get("event_type", "concert")}
                for ev in a_ev_raw
            ]

            with st.spinner("Applying to schedule..."):
                try:
                    _upd = recommend_daily(
                        date_str=sch_date.isoformat(),
                        line=sch_line,
                        weather=a_weather,
                        events=events_daily,
                        cost_per_train_hr=a_cost,
                        weather_window=(a_ts, a_te),
                        emergency_type=a_em_type,
                        emergency_hour=a_ts,
                        emergency_duration=a_em_dur,
                        pax_factors=a_pax_factors,
                    )
                except Exception as _ex:
                    st.error(f"Error applying schedule: {_ex}")
                    st.stop()

            _upd["schedule"] = _apply_option_to_window(
                _upd["schedule"], new_pick, a_ts, a_te, a_cost, a_weather
            )

            st.session_state.update({
                "_sch_result":    _upd,
                "_sch_mode":      "updated",
                "_sch_events":    events_daily,
                "_upd_weather":   a_weather,
                "_tune_start":    a_ts,
                "_tune_end":      a_te,
                "_analysis":      None,
                "_briefing_text": None,
                "_from_db":       False,
            })
            st.rerun()

        if reset_btn:
            st.session_state.update({
                "_sch_result":  st.session_state["_sch_default"],
                "_sch_mode":    "default",
                "_sch_events":  [],
                "_tune_start":  6,
                "_tune_end":    23,
                "_analysis":    None,
            })
            st.rerun()

