"""Streamlit UI for LRT AI Operations decision-support.
Run: streamlit run app.py"""

import json
from datetime import datetime, time
from pathlib import Path

import streamlit as st

from core.glm_client import extract_inputs_from_text
from core.recommender import recommend

st.set_page_config(page_title="LRT AI Operations", layout="wide")
st.title("LRT AI Operations — Decision Support")
st.caption(
    "Powered by Z.AI GLM. The GLM reads operational factors, evaluates three scheduling "
    "options, and recommends the best one with a plain-English justification."
)

LINES = ["Kelana Jaya", "Ampang", "Sri Petaling", "Kajang", "Putrajaya"]
SCENARIOS_PATH = Path("data/scenarios.json")
scenarios = json.loads(SCENARIOS_PATH.read_text()) if SCENARIOS_PATH.exists() else []

# ── Sidebar: demo presets (applies to Manual Input tab only) ──────────────────
with st.sidebar:
    st.header("Demo scenarios")
    st.caption("Presets auto-fill the Manual Input form.")
    scenario_names = ["(custom input)"] + [s["name"] for s in scenarios]
    picked = st.selectbox("Load a preset", scenario_names)
    preset = next((s["inputs"] for s in scenarios if s["name"] == picked), None)


def _get(key, default):
    return preset[key] if preset and key in preset else default


# ── Shared result display ─────────────────────────────────────────────────────
def _show_results(result: dict):
    options = result.get("options", [])
    chosen = result.get("chosen_option", "moderate")

    st.markdown("### Scheduling options — GLM comparison")
    cols = st.columns(3)
    for col, opt in zip(cols, options):
        is_chosen = opt["label"] == chosen
        freq_delta = opt["recommended_frequency_per_hr"] - opt["current_frequency_per_hr"]
        with col:
            if is_chosen:
                st.success(f"**{opt['label'].upper()}**\n\n✅ GLM recommends this")
            else:
                st.info(f"**{opt['label'].upper()}**")
            st.metric("Frequency", f"{opt['recommended_frequency_per_hr']}/hr", delta=freq_delta)
            st.metric(
                "Congestion change",
                f"{opt['congestion_change_pct']}%",
                delta=opt["congestion_change_pct"],
                delta_color="inverse",
            )
            st.metric(
                "Cost delta",
                f"RM {opt['cost_delta_rm']:,.0f}",
                delta=opt["cost_delta_rm"],
                delta_color="inverse",
            )
            st.metric("Revenue delta", f"RM {opt['revenue_delta_rm']:,.0f}", delta=opt["revenue_delta_rm"])
            st.metric("Time saved", f"{opt['time_saved_min_per_passenger']} min/pax")

    st.markdown("### GLM reasoning")
    st.write(result["explanation"])

    st.markdown("### Schedule update")
    st.info(result["schedule_update"])

    with st.expander("Raw output (for debugging)"):
        st.json(result)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Manual Input", "Describe Situation (GLM parses)"])

# ── Tab 1: Manual Input ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Operational factors")
    col1, col2, col3 = st.columns(3)

    with col1:
        default_dt = datetime.fromisoformat(_get("datetime", "2026-04-25T19:00"))
        date = st.date_input("Date", value=default_dt.date())
        hour = st.slider("Hour", 0, 23, value=default_dt.hour)
        line = st.selectbox(
            "LRT line", LINES,
            index=LINES.index(_get("line", "Kelana Jaya")),
        )

    with col2:
        day = st.radio(
            "Day type",
            ["weekday", "weekend"],
            index=["weekday", "weekend"].index(_get("day", "weekday")),
            horizontal=True,
        )
        weather = st.selectbox(
            "Weather",
            ["clear", "cloudy", "rainy", "stormy"],
            index=["clear", "cloudy", "rainy", "stormy"].index(_get("weather", "clear")),
        )
        is_holiday = st.checkbox("Public holiday", value=_get("is_holiday", False))
        emergency = st.text_input("Emergency (blank if none)", value=_get("emergency", "") or "")

    with col3:
        current_freq = st.number_input("Current frequency (trains/hr)", 1, 30, _get("current_frequency_per_hr", 6))
        capacity = st.number_input("Train capacity (passengers)", 100, 2000, _get("train_capacity", 600))
        cost = st.number_input("Running cost per train-hour (RM)", 50, 2000, _get("running_cost_per_train_hr", 350))

    st.markdown("**Nearby events**")
    preset_events = _get("events", [])
    event_name = st.text_input(
        "Event name (blank if none)",
        value=preset_events[0].get("name", "") if preset_events else "",
    )
    event_attendance = st.number_input(
        "Expected attendance", 0, 200_000,
        value=int(preset_events[0].get("expected_attendance", 0)) if preset_events else 0,
    )

    if st.button("Generate recommendation", type="primary"):
        inputs = {
            "datetime": datetime.combine(date, time(hour=hour)).isoformat(timespec="minutes"),
            "day": day,
            "weather": weather,
            "events": [{"name": event_name, "expected_attendance": int(event_attendance)}] if event_name else [],
            "is_holiday": is_holiday,
            "emergency": emergency or None,
            "line": line,
            "current_frequency_per_hr": int(current_freq),
            "train_capacity": int(capacity),
            "running_cost_per_train_hr": int(cost),
        }
        with st.spinner("GLM evaluating options..."):
            result = recommend(inputs)
        st.success("Recommendation generated.")
        _show_results(result)


# ── Tab 2: Describe Situation ─────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Type a situation description in plain language. "
        "**GLM will read it and extract the operational factors automatically** "
        "(weather, day type, events, emergency, etc.). "
        "You only need to provide the operational parameters that can't be inferred from text."
    )

    situation_text = st.text_area(
        "Situation description",
        height=130,
        placeholder=(
            'e.g. "There\'s a Coldplay concert tonight at Bukit Jalil, around 80,000 fans expected. '
            'It\'s been raining heavily since 3pm. This is a Saturday evening."'
        ),
    )

    st.markdown("**Operational parameters** (fill these — they won't be in the description)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ai_line = st.selectbox("LRT line", LINES, key="ai_line")
    with c2:
        ai_freq = st.number_input("Current frequency (trains/hr)", 1, 30, 6, key="ai_freq")
    with c3:
        ai_cap = st.number_input("Train capacity (passengers)", 100, 2000, 600, key="ai_cap")
    with c4:
        ai_cost = st.number_input("Running cost per train-hour (RM)", 50, 2000, 350, key="ai_cost")

    if st.button("Parse & Recommend", type="primary", key="ai_btn"):
        if not situation_text.strip():
            st.warning("Please describe the situation first.")
        else:
            with st.spinner("GLM reading your situation description..."):
                extracted = extract_inputs_from_text(situation_text)

            st.markdown("**What GLM extracted from your description:**")
            st.json({
                "day": extracted.get("day", "weekday"),
                "weather": extracted.get("weather", "clear"),
                "is_holiday": extracted.get("is_holiday", False),
                "emergency": extracted.get("emergency"),
                "line": extracted.get("line"),
                "events": extracted.get("events", []),
            })

            inputs = {
                "datetime": datetime.now().isoformat(timespec="minutes"),
                "day": extracted.get("day", "weekday"),
                "weather": extracted.get("weather", "clear"),
                "is_holiday": extracted.get("is_holiday", False),
                "emergency": extracted.get("emergency"),
                "events": extracted.get("events", []),
                "line": extracted.get("line") or ai_line,
                "current_frequency_per_hr": int(ai_freq),
                "train_capacity": int(ai_cap),
                "running_cost_per_train_hr": int(ai_cost),
            }

            with st.spinner("GLM evaluating scheduling options..."):
                result = recommend(inputs)
            st.success("Recommendation generated.")
            _show_results(result)
