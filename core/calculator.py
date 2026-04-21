"""Rule-based math layer. No GLM here — pure deterministic numbers."""

from datetime import datetime

WEATHER_MULTIPLIER = {
    "clear": 1.0,
    "cloudy": 1.05,
    "rainy": 1.20,
    "stormy": 1.35,
}

PEAK_HOURS = {7, 8, 9, 17, 18, 19}
FARE_PER_PASSENGER_RM = 3.50
TARGET_LOAD_FACTOR = 0.75


def _parse_hour(dt_string: str) -> int:
    return datetime.fromisoformat(dt_string).hour


def _event_demand_uplift(events: list[dict]) -> float:
    if not events:
        return 1.0
    total_attendance = sum(e.get("expected_attendance", 0) for e in events)
    return 1.0 + min(total_attendance / 100_000, 1.5)


def _baseline_passengers_per_hour(hour: int, is_holiday: bool, is_weekend: bool) -> int:
    if is_holiday:
        return 2_500
    if is_weekend:
        if 11 <= hour <= 21:
            return 3_500
        return 800
    if hour in PEAK_HOURS:
        return 8_000
    if 6 <= hour <= 22:
        return 4_000
    return 800


def compute_metrics(inputs: dict) -> dict:
    hour = _parse_hour(inputs["datetime"])
    weather_mult = WEATHER_MULTIPLIER.get(inputs.get("weather", "clear"), 1.0)
    event_mult = _event_demand_uplift(inputs.get("events", []))
    emergency = inputs.get("emergency")

    is_weekend = inputs.get("day", "weekday") == "weekend"
    baseline = _baseline_passengers_per_hour(hour, inputs.get("is_holiday", False), is_weekend)
    expected_passengers = int(baseline * weather_mult * event_mult)

    capacity = inputs["train_capacity"]
    current_freq = inputs["current_frequency_per_hr"]

    needed_freq = max(1, round(expected_passengers / (capacity * TARGET_LOAD_FACTOR)))
    if emergency:
        needed_freq = max(needed_freq, current_freq + 2)

    return {
        "expected_passengers_per_hr": expected_passengers,
        "recommended_frequency_per_hr": needed_freq,
        "current_frequency_per_hr": current_freq,
    }


def _option_metrics(inputs: dict, freq: int, expected: int) -> dict:
    capacity = inputs["train_capacity"]
    current_freq = inputs["current_frequency_per_hr"]
    cost_per_hr = inputs["running_cost_per_train_hr"]

    current_capacity = current_freq * capacity
    new_capacity = freq * capacity

    current_served = min(expected, current_capacity)
    new_served = min(expected, new_capacity)

    cost_delta = (freq - current_freq) * cost_per_hr
    revenue_delta = (new_served - current_served) * FARE_PER_PASSENGER_RM

    prev_unmet = max(0, expected - current_capacity) / max(expected, 1)
    new_unmet = max(0, expected - new_capacity) / max(expected, 1)
    congestion_change = round((new_unmet - prev_unmet) * 100, 1)

    wait_before = 60 / max(current_freq, 1) / 2
    wait_after = 60 / max(freq, 1) / 2
    time_saved = round(wait_before - wait_after, 1)

    return {
        "recommended_frequency_per_hr": freq,
        "current_frequency_per_hr": current_freq,
        "expected_passengers_per_hr": expected,
        "congestion_change_pct": congestion_change,
        "cost_delta_rm": round(cost_delta, 2),
        "revenue_delta_rm": round(revenue_delta, 2),
        "time_saved_min_per_passenger": time_saved,
    }


def compute_options(inputs: dict) -> list[dict]:
    base = compute_metrics(inputs)
    base_freq = base["recommended_frequency_per_hr"]
    expected = base["expected_passengers_per_hr"]
    current_freq = inputs["current_frequency_per_hr"]

    freq_map = {
        "conservative": max(1, base_freq - 2),
        "moderate": base_freq,
        "aggressive": base_freq + 3,
    }

    options = []
    for label, freq in freq_map.items():
        metrics = _option_metrics(inputs, freq, expected)
        options.append({"label": label, **metrics})

    return options
