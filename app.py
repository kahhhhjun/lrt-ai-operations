"""Streamlit UI for LRT AI Operations decision-support.
Run: streamlit run app.py"""

from datetime import datetime, date as date_type, time
from pathlib import Path

import pandas as pd
import streamlit as st

from core.calculator import WEATHER_MAX_FREQ, TRAIN_CAPACITY, compute_options, default_frequency as _dfreq
from core.glm_client import extract_inputs_from_image, extract_inputs_from_text
from core.recommender import (get_glm_recommendation, get_glm_recommendation_stream,
                              get_glm_daily_reasoning_stream, recommend_daily)

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
    """Shift every window-hour's recommended frequency by the option delta."""
    max_freq = WEATHER_MAX_FREQ.get(weather, 20)
    delta = {"conservative": -2, "moderate": 0, "aggressive": 3}.get(chosen, 0)
    if delta == 0:
        return schedule
    for s in schedule:
        if tune_s <= s["hour"] < tune_e:
            new_freq = max(s["standard_frequency"],          # never drop below standard
                           min(s["recommended_frequency"] + delta, max_freq))
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
st.subheader("Full Day Schedule")

sch_date = st.date_input("Date", value=date_type.today(), key="sch_date")
sch_line = st.selectbox("LRT line", LINES, key="sch_line")
sch_weekday  = sch_date.weekday()
sch_day_name = DAYS[sch_weekday]

# Auto-load default schedule when date or line changes
_key = f"{sch_date}_{sch_line}"
if st.session_state.get("_sch_key") != _key:
    try:
        _def = recommend_daily(sch_date.isoformat(), sch_line, "clear", [], 350)
    except Exception as _e:
        st.error(f"Could not load schedule: {_e}")
        st.stop()
    st.session_state.update({
        "_sch_key": _key, "_sch_default": _def, "_sch_result": _def,
        "_sch_mode": "default", "_sch_events": [],
        "_tune_start": 6, "_tune_end": 24,
        "_analysis": None, "_chosen_option": "moderate",
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
    st.success(
        f"Adjusted **{tune_s:02d}:00–{tune_e:02d}:00** | "
        f"weather: {st.session_state.get('_upd_weather', '—')}"
        + (f" | event: {ev_active[0]['name']}" if ev_active else "")
        + ".  Press **Reset to standard** below to revert."
    )
    st.caption("🟡 Highlighted rows = adjusted time window")
else:
    st.caption("Standard timetable. Use the panel below to adjust specific hours.")

rows = []
for s in schedule:
    in_win   = (mode == "updated") and (tune_s <= s["hour"] < tune_e)
    is_tail  = s.get("is_event_tail", False)
    em_status = s.get("em_status")
    show_adj = in_win or is_tail or bool(em_status)
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
    if mode == "updated" and tune_s <= s["hour"] < tune_e:
        return ["background-color: #1a3a5c; color: #ffffff; font-weight: bold"] * len(row)
    if s.get("is_event_tail"):
        return ["background-color: #2a4a3c; color: #ffffff; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(df.style.apply(_style_rows, axis=1), use_container_width=True, hide_index=True)

# ── Daily cost ────────────────────────────────────────────────────────────────
std_total = result["daily_standard_cost_rm"]
extra_win = sum(
    s["extra_cost_rm"] for s in schedule
    if mode == "updated" and tune_s <= s["hour"] < tune_e
)
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

# ── GLM shift briefing (shown after any schedule change, streamed) ────────────
if mode == "updated" and result.get("has_adjustments"):
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
    except Exception:
        st.info("Schedule updated. See highlighted rows for changes.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADJUSTMENT PANEL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Adjust schedule")

# Step 1 — time window
st.markdown("**Step 1 — Select time window to adjust**")
tw1, tw2 = st.columns(2)
with tw1:
    tune_start = st.slider("From", 6, 23, 6,  format="%d:00", key="tune_start")
with tw2:
    tune_end   = st.slider("To",   7, 24, 24, format="%d:00", key="tune_end")
st.caption(
    f"Only **{tune_start:02d}:00 – {tune_end:02d}:00** will be adjusted. "
    "The rest of the day stays on the standard timetable."
)

# Step 2 — situation input
st.markdown("**Step 2 — Describe the situation**")
input_method = st.radio(
    "", ["Manual inputs", "Describe in text (GLM extracts)", "Upload image / poster (GLM reads)"],
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

if input_method == "Manual inputs":
    ai1, ai2 = st.columns(2)
    with ai1:
        sch_weather = st.selectbox("Weather", ["clear", "cloudy", "rainy", "stormy"], key="sch_weather")
        sch_cost    = st.number_input("Running cost / train-hour (RM)", 50, 2000, 350, key="sch_cost")
        em_type_key = st.selectbox("Emergency type", _EMERGENCY_OPTIONS,
                                   format_func=lambda x: _EMERGENCY_LABELS[x], key="em_type")
        em_dur      = st.number_input("Emergency duration (hours)", 1, 4, 1, key="em_dur") if em_type_key != "None" else 1
    with ai2:
        ev_name = st.text_input("Event name (leave blank if none)", key="ev_name")
        ev_pax  = st.number_input("Extra passengers/hr at station", 0, 30_000, 0, key="ev_pax")

elif input_method == "Describe in text (GLM extracts)":
    sit_text = st.text_area(
        "Describe the situation",
        height=100,
        placeholder='e.g. "Blackpink concert tonight at 8pm, around 4,000 extra passengers per hour. Heavy rain since afternoon."',
        key="sit_text",
    )
    sch_cost = st.session_state.get("sch_cost", 350)

else:  # Upload image
    uploaded_file = st.file_uploader(
        "Upload a concert poster, event flyer, or news screenshot",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploaded_image",
    )
    if uploaded_file:
        st.image(uploaded_file, caption=uploaded_file.name, width=300)
    sch_cost = st.session_state.get("sch_cost", 350)

# Step 3 — Analyse
st.markdown("**Step 3 — Analyse options**")
if st.button("Analyse", type="primary", key="analyse_btn"):
    if tune_end <= tune_start:
        st.warning(f"'To' must be after 'From'. Minimum window is 1 hour (e.g. {tune_start:02d}:00 – {tune_start+1:02d}:00).")
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

    elif input_method == "Upload image / poster (GLM reads)":
        uploaded_file = st.session_state.get("uploaded_image")
        if not uploaded_file:
            st.warning("Please upload an image first.")
            st.stop()
        mime_type = uploaded_file.type or "image/jpeg"
        with st.spinner("Reading image with OCR, then extracting details..."):
            try:
                extracted = extract_inputs_from_image(uploaded_file.getvalue(), mime_type)
            except Exception as _ex:
                st.error(f"Could not read image: {_ex}")
                st.stop()
        a_weather = extracted.get("weather", "clear")
        a_ev_raw  = extracted.get("events", [])
        a_line    = extracted.get("line") or sch_line
        st.info(f"GLM read image — weather: **{a_weather}** | events: {[e['name'] for e in a_ev_raw] or ['none']}")

    else:  # Manual inputs
        a_weather = st.session_state.get("sch_weather", "clear")
        a_ev_raw  = [{"name": ev_name, "passengers_per_hr": int(ev_pax)}] \
                    if ev_name.strip() and ev_pax > 0 else []
        a_line    = sch_line

    # Pull emergency fields from extraction (text/image) or manual selector
    a_emergency      = extracted.get("emergency")      if input_method != "Manual inputs" else None
    a_emergency_type = extracted.get("emergency_type") if input_method != "Manual inputs" else None
    # Manual emergency override
    if input_method == "Manual inputs":
        _em_sel = st.session_state.get("em_type", "None")
        if _em_sel != "None":
            a_emergency_type = _em_sel
            a_emergency      = _EMERGENCY_LABELS.get(_em_sel, _em_sel)
    a_emergency_dur = st.session_state.get("em_dur", 1) if input_method == "Manual inputs" else 1

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
            st.metric(
                "Frequency",
                "SUSPENDED" if freq_val == 0 else f"{freq_val}/hr",
                delta=freq_delta,
                help="Trains running per hour. Arrow shows change vs current schedule. More trains = shorter wait time between arrivals.",
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

        ap1, ap2 = st.columns(2)
        with ap1:
            apply_btn = st.button("Apply to schedule", type="primary", key="apply_btn")
        with ap2:
            reset_btn = st.button("Reset to standard", key="sch_reset")

        if apply_btn:
            a_weather = st.session_state.get("_a_weather", "clear")
            a_cost    = st.session_state.get("_a_cost", 350)
            a_ev_raw  = st.session_state.get("_a_ev_raw", [])
            a_ts      = st.session_state.get("_a_tune_s", 6)
            a_te      = st.session_state.get("_a_tune_e", 23)
            a_em_type = st.session_state.get("_a_em_type")
            a_em_dur  = st.session_state.get("_a_em_dur", 1)

            events_daily = [
                {"name": ev["name"], "start_hour": a_ts,
                 "end_hour": a_te, "passengers_per_hr": ev["passengers_per_hr"]}
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
                    )
                except Exception as _ex:
                    st.error(f"Error applying schedule: {_ex}")
                    st.stop()

            _upd["schedule"] = _apply_option_to_window(
                _upd["schedule"], new_pick, a_ts, a_te, a_cost, a_weather
            )

            st.session_state.update({
                "_sch_result":  _upd,
                "_sch_mode":    "updated",
                "_sch_events":  events_daily,
                "_upd_weather": a_weather,
                "_tune_start":  a_ts,
                "_tune_end":    a_te,
                "_analysis":    None,
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

elif mode == "updated":
    if st.button("Reset to standard", key="sch_reset_plain"):
        st.session_state.update({
            "_sch_result": st.session_state["_sch_default"],
            "_sch_mode":   "default",
            "_sch_events": [],
            "_tune_start": 6,
            "_tune_end":   24,
        })
        st.rerun()
