"""Streamlit UI for LRT AI Operations decision-support.
Run: streamlit run app.py"""

from datetime import datetime, date as date_type, time
from pathlib import Path

import pandas as pd
import streamlit as st

from core.calculator import WEATHER_MAX_FREQ, TRAIN_CAPACITY, compute_options, default_frequency as _dfreq
from core.glm_client import extract_inputs_from_text
from core.recommender import get_glm_recommendation, recommend_daily

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
        if tune_s <= s["hour"] <= tune_e:
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

c1, c2 = st.columns([2, 1])
with c1:
    sch_date = st.date_input("Date", value=date_type.today(), key="sch_date")
    sch_line = st.selectbox("LRT line", LINES, key="sch_line")
with c2:
    sch_weekday  = sch_date.weekday()
    sch_day_name = DAYS[sch_weekday]
    st.info(
        f"**{sch_day_name}** — {'Weekend' if sch_weekday >= 5 else 'Weekday'}\n\n"
        f"{sch_date.strftime('%d %b %Y')} follows the **{sch_day_name}** timetable."
    )

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
        "_tune_start": 6, "_tune_end": 23,
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

title = f"### {sch_day_name} Schedule — {sch_line} ({sch_date.strftime('%d %b %Y')})"
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
    in_win = (mode == "updated") and (tune_s <= s["hour"] <= tune_e)
    if in_win:
        delta = s["extra_trains"]
        if delta > 0:   tc = f"{s['recommended_frequency']} (+{delta})"
        elif delta < 0: tc = f"{s['recommended_frequency']} (−{abs(delta)})"
        else:           tc = str(s["recommended_frequency"])
        rows.append({
            "Time":         s["time_slot"],
            "Frequency":    f"every {s['headway_rec_min']} min",
            "Trains/hr":    tc,
            "Cost (RM/hr)": f"RM {s['total_cost_rm']:,.0f}",
            "Load factor":  f"{s['load_factor_pct']}%",
            "Status":       ", ".join(s["event_names"]) if s["has_event"] else "Adjusted",
        })
    else:
        rows.append({
            "Time":         s["time_slot"],
            "Frequency":    f"every {s['headway_std_min']} min",
            "Trains/hr":    str(s["standard_frequency"]),
            "Cost (RM/hr)": f"RM {s['standard_cost_rm']:,.0f}",
            "Load factor":  f"{s['standard_load_factor_pct']}%",
            "Status":       "Standard",
        })

df = pd.DataFrame(rows)

def _style_rows(row):
    if mode == "updated" and tune_s <= schedule[row.name]["hour"] <= tune_e:
        return ["background-color: #fff3cd; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(df.style.apply(_style_rows, axis=1), use_container_width=True, hide_index=True)

# ── Daily cost ────────────────────────────────────────────────────────────────
std_total = result["daily_standard_cost_rm"]
extra_win = sum(
    s["extra_cost_rm"] for s in schedule
    if mode == "updated" and tune_s <= s["hour"] <= tune_e
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

# ── GLM shift briefing (shown after applying with an event) ───────────────────
if mode == "updated" and ev_active:
    with st.expander("GLM shift briefing", expanded=True):
        st.write(result["explanation"])


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
    tune_end   = st.slider("To",   6, 23, 23, format="%d:00", key="tune_end")
st.caption(
    f"Only **{tune_start:02d}:00 – {tune_end:02d}:00** will be adjusted. "
    "The rest of the day stays on the standard timetable."
)

# Running cost (always visible)
sch_cost = st.number_input("Running cost / train-hour (RM)", 50, 2000, 350, key="sch_cost")

# Step 2 — situation input
st.markdown("**Step 2 — Describe the situation**")
input_method = st.radio(
    "", ["Manual inputs", "Describe in text (GLM extracts)"],
    horizontal=True, key="input_method", label_visibility="collapsed",
)

if input_method == "Manual inputs":
    ai1, ai2 = st.columns(2)
    with ai1:
        sch_weather = st.selectbox("Weather", ["clear", "cloudy", "rainy", "stormy"], key="sch_weather")
    with ai2:
        ev_name = st.text_input("Event name (leave blank if none)", key="ev_name")
        ev_pax  = st.number_input("Extra passengers/hr at station", 0, 30_000, 0, key="ev_pax")
else:
    sit_text = st.text_area(
        "Describe the situation",
        height=100,
        placeholder='e.g. "Blackpink concert tonight at 8pm, around 4,000 extra passengers per hour. Heavy rain since afternoon."',
        key="sit_text",
    )

# Step 3 — Analyse
st.markdown("**Step 3 — Analyse options**")
if st.button("Analyse", type="primary", key="analyse_btn"):
    if tune_end < tune_start:
        st.warning("'To' hour must be equal to or after 'From' hour.")
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
        a_weather   = extracted.get("weather", "clear")
        a_ev_raw    = extracted.get("events", [])
        a_line      = extracted.get("line") or sch_line
        st.info(f"GLM extracted — weather: **{a_weather}** | events: {[e['name'] for e in a_ev_raw] or ['none']}")
    else:
        a_weather = st.session_state.get("sch_weather", "clear")
        a_ev_raw  = [{"name": ev_name, "passengers_per_hr": int(ev_pax)}] \
                    if ev_name.strip() and ev_pax > 0 else []
        a_line    = sch_line

    inputs = {
        "datetime":                 datetime.combine(sch_date, time(hour=rep_hour)).isoformat(timespec="minutes"),
        "weather":                  a_weather,
        "events":                   a_ev_raw,
        "emergency":                None,
        "line":                     a_line,
        "current_frequency_per_hr": _dfreq(rep_hour, sch_date.weekday(), sch_line),
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
            st.metric("Frequency",   f"{opt['recommended_frequency_per_hr']}/hr", delta=freq_delta)
            st.metric("Load factor", f"{opt['load_factor_pct']}%",
                      delta=opt["congestion_change_pct"], delta_color="inverse")
            st.metric("Cost delta",  f"RM {opt['cost_delta_rm']:,.0f}",
                      delta=opt["cost_delta_rm"], delta_color="inverse")
            st.metric("Pax served",  f"{opt['passengers_served_per_hr']:,}/hr",
                      delta=opt["passengers_served_delta"])

    # ── Phase 2: auto-trigger GLM reasoning (runs on the same render as cards) ─
    if not reasoning_done:
        glm_inputs = st.session_state.get("_glm_inputs")
        if glm_inputs:
            with st.spinner("GLM reasoning... (this may take a few seconds)"):
                try:
                    choice, explanation = get_glm_recommendation(glm_inputs, options)
                except (TimeoutError, ConnectionError) as exc:
                    moderate = next(o for o in options if o["label"] == "moderate")
                    lf = moderate["load_factor_pct"]
                    hw = round(60 / max(moderate["recommended_frequency_per_hr"], 1), 1)
                    choice = "moderate"
                    explanation = (
                        f"⚠ GLM unavailable ({exc}).\n\n"
                        f"**Math-based: Moderate** — {moderate['recommended_frequency_per_hr']}/hr "
                        f"(every {hw} min), load factor {lf}%. "
                        "Meets demand without over- or under-provisioning."
                    )
            st.session_state["_analysis"]["chosen_option"] = choice
            st.session_state["_analysis"]["explanation"]   = explanation
            st.session_state.pop("_glm_inputs", None)
            st.rerun()

    # ── Phase 2 complete: show reasoning + controls ───────────────────────────
    else:
        with st.expander("GLM reasoning", expanded=True):
            st.write(analysis["explanation"])

        cur_pick = st.session_state.get("_chosen_option", glm_pick)
        new_pick = st.radio(
            "Apply this option to the schedule:",
            ["conservative", "moderate", "aggressive"],
            index=["conservative", "moderate", "aggressive"].index(cur_pick),
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

            events_daily = [
                {"name": ev["name"], "start_hour": a_ts,
                 "end_hour": a_te + 1, "passengers_per_hr": ev["passengers_per_hr"]}
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
            "_tune_end":   23,
        })
        st.rerun()
