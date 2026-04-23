"""Rule-based math layer grounded in LRT system assumptions:
  - Train capacity (normal): 600 pax | max: 900 pax
  - Standard headway schedule (Mon–Fri / Sat / Sun) defined in _*_SCHEDULE tables
  - Bad weather: reduces demand AND caps max frequency (trains slow down for safety)
"""

from datetime import datetime

# ── Constants from assumptions ────────────────────────────────────────────────
TRAIN_CAPACITY = 600          # normal comfort capacity per train
TRAIN_CAPACITY_MAX = 900      # absolute max (crush load)
TARGET_LOAD_FACTOR = 0.75     # aim for 75% full — leave headroom for comfort

# Peak demand hours (affects baseline passenger estimate, not just frequency)
PEAK_HOURS = {7, 8, 9, 17, 18, 19}  # 7–10am and 5–8pm

# ── Standard headway timetable per line (headway in minutes) ─────────────────
# Format: {line: {day_type: [(start_hour_incl, end_hour_excl, headway_min)]}}
_SCHEDULES = {
    "Kelana Jaya": {
        "weekday": [
            ( 6,  7,  7),   # 06–07  every 7 min
            ( 7, 10,  3),   # 07–10  every 3 min  ← morning peak
            (10, 17,  7),   # 10–17  every 7 min
            (17, 20,  3),   # 17–20  every 3 min  ← evening peak
            (20, 24, 12),   # 20–24  every 12 min
        ],
        "saturday": [
            ( 6, 22,  7),   # 06–22  every 7 min
            (22, 24, 10),   # 22–24  every 10 min
        ],
        "sunday": [
            ( 6, 10, 10),   # 06–10  every 10 min
            (10, 22,  7),   # 10–22  every 7 min
            (22, 24, 10),   # 22–24  every 10 min
        ],
    },
    "Sri Petaling": {
        "weekday": [
            ( 6,  7,  7),
            ( 7, 10,  3),
            (10, 17,  7),
            (17, 20,  3),
            (20, 24,  7),   # 20–24  every 7 min (not 12 like KJ)
        ],
        "saturday": [( 6, 24,  7)],   # all day every 7 min
        "sunday":   [( 6, 24,  7)],   # not specified — assumed same as Saturday
    },
    "Ampang": {
        "weekday": [
            ( 6,  7,  7),
            ( 7, 10,  3),
            (10, 17,  7),
            (17, 20,  3),
            (20, 24,  7),   # 20–24  every 7 min
        ],
        "saturday": [( 6, 24,  7)],   # all day every 7 min
        "sunday":   [( 6, 24,  7)],   # all day every 7 min
    },
}

# Weather — effect on passenger demand (bad weather → people stay home / drive less)
WEATHER_PAX_MULT = {
    "clear":  1.00,
    "cloudy": 0.95,
    "rainy":  0.85,
    "stormy": 0.70,
}

# Weather — max allowed frequency (trains must slow down in bad weather)
WEATHER_MAX_FREQ = {
    "clear":  20,
    "cloudy": 20,
    "rainy":  17,
    "stormy": 12,
}

# Baseline passengers per station per hour (typical busy KL LRT station)
PAX_PEAK        = 5_000   # weekday morning/evening rush
PAX_OFF_PEAK    = 2_500   # weekday daytime
PAX_NIGHT       =   800   # weekday late evening
PAX_WEEKEND_DAY = 2_000   # weekend 11am–7pm (leisure traffic)
PAX_WEEKEND_LOW =   600   # weekend morning / late night


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(dt_string: str) -> datetime:
    return datetime.fromisoformat(dt_string)


def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # Saturday=5, Sunday=6


def _baseline_pax(hour: int, is_weekend: bool) -> int:
    if is_weekend:
        return PAX_WEEKEND_DAY if 11 <= hour <= 19 else PAX_WEEKEND_LOW
    if hour in PEAK_HOURS:
        return PAX_PEAK
    if 6 <= hour <= 22:
        return PAX_OFF_PEAK
    return PAX_NIGHT


def get_standard_headway(hour: int, weekday: int, line: str = "Kelana Jaya") -> int:
    """Headway in minutes from the official per-line schedule. weekday: 0=Mon … 6=Sun."""
    line_sched = _SCHEDULES.get(line, _SCHEDULES["Kelana Jaya"])
    if weekday == 6:
        sched = line_sched["sunday"]
    elif weekday == 5:
        sched = line_sched["saturday"]
    else:
        sched = line_sched["weekday"]
    for start, end, hw in sched:
        if start <= hour < end:
            return hw
    return 7  # fallback


def default_frequency(hour: int, weekday_or_is_weekend, line: str = "Kelana Jaya") -> int:
    """Trains per hour from the official headway schedule.
    Accepts weekday int (0=Mon … 6=Sun) or legacy is_weekend bool."""
    if isinstance(weekday_or_is_weekend, bool):
        weekday = 5 if weekday_or_is_weekend else 0
    else:
        weekday = int(weekday_or_is_weekend)
    return round(60 / get_standard_headway(hour, weekday, line))


# ── Core computation ──────────────────────────────────────────────────────────

def _option_metrics(inputs: dict, freq: int, expected: int, max_freq: int) -> dict:
    capacity   = inputs.get("train_capacity", TRAIN_CAPACITY)
    curr_freq  = inputs["current_frequency_per_hr"]
    cost_per_hr = inputs.get("running_cost_per_train_hr", 350)

    curr_capacity = curr_freq * capacity
    new_capacity  = freq * capacity

    curr_served = min(expected, curr_capacity)
    new_served  = min(expected, new_capacity)

    cost_delta = (freq - curr_freq) * cost_per_hr

    # Productivity: how many passengers can actually be served per hour
    passengers_served     = new_served
    passengers_served_delta = new_served - curr_served

    # Load factor as percentage (75% = target, 100% = normal capacity full, >100% = overcrowded)
    load_before       = round(min(expected / max(curr_capacity, 1), 2.0) * 100, 1)
    load_after        = round(min(expected / max(new_capacity, 1), 2.0) * 100, 1)
    congestion_change = round(load_after - load_before, 1)

    wait_before = round(60 / max(curr_freq, 1) / 2, 1)
    wait_after  = round(60 / max(freq, 1) / 2, 1)
    time_saved  = round(wait_before - wait_after, 1)

    return {
        "recommended_frequency_per_hr":  freq,
        "current_frequency_per_hr":      curr_freq,
        "expected_passengers_per_hr":    expected,
        "passengers_served_per_hr":      passengers_served,
        "passengers_served_delta":       passengers_served_delta,
        "load_factor_pct":               load_after,
        "congestion_change_pct":         congestion_change,
        "cost_delta_rm":                 round(cost_delta, 2),
        "time_saved_min_per_passenger":  time_saved,
    }


def compute_options(inputs: dict) -> list[dict]:
    dt       = _parse_dt(inputs["datetime"])
    hour     = dt.hour
    is_wknd  = _is_weekend(dt)
    weather  = inputs.get("weather", "clear")
    events   = inputs.get("events", [])

    # Step 1: baseline passengers from time of day
    baseline = _baseline_pax(hour, is_wknd)

    # Step 2: add event passengers (directly specified as pax/hr at station)
    event_pax = sum(e.get("passengers_per_hr", 0) for e in events)

    # Step 3: weather reduces total demand (people less likely to travel)
    pax_mult = WEATHER_PAX_MULT.get(weather, 1.0)
    expected = int((baseline + event_pax) * pax_mult)

    # Step 4: compute needed frequency for target load factor
    capacity = inputs.get("train_capacity", TRAIN_CAPACITY)
    needed   = max(1, round(expected / (capacity * TARGET_LOAD_FACTOR)))

    # Step 5: cap by weather (trains must slow down in bad weather)
    max_freq = WEATHER_MAX_FREQ.get(weather, 20)
    needed   = min(needed, max_freq)

    # Step 6: emergency bumps minimum frequency
    if inputs.get("emergency"):
        curr = inputs["current_frequency_per_hr"]
        needed = min(max(needed, curr + 2), max_freq)

    # Step 7: three options
    freq_map = {
        "conservative": max(1, needed - 2),
        "moderate":     needed,
        "aggressive":   min(needed + 3, max_freq),
    }

    return [
        {"label": label, **_option_metrics(inputs, freq, expected, max_freq)}
        for label, freq in freq_map.items()
    ]


def compute_daily_schedule(
    date_str: str,
    line: str,
    weather: str,
    events: list[dict],
    cost_per_train_hr: int = 350,
    train_capacity: int = TRAIN_CAPACITY,
) -> list[dict]:
    """
    Compute full day schedule from 06:00 to 23:00 (18 slots).
    events: list of {name, start_hour, end_hour, passengers_per_hr}
    Returns one dict per hour slot.
    """
    dt_base  = _parse_dt(date_str + "T06:00")
    weekday  = dt_base.weekday()          # 0=Mon … 6=Sun
    max_freq = WEATHER_MAX_FREQ.get(weather, 20)

    schedule = []
    for hour in range(6, 24):
        # Collect event passengers active this hour
        hour_event_pax   = 0
        hour_event_names = []
        for ev in events:
            if ev.get("start_hour", 0) <= hour < ev.get("end_hour", 0):
                hour_event_pax += ev.get("passengers_per_hr", 0)
                hour_event_names.append(ev["name"])

        std_freq = default_frequency(hour, weekday, line)

        # Reuse options logic — treat standard freq as current
        inputs_hour = {
            "datetime":                  f"{date_str}T{hour:02d}:00",
            "weather":                   weather,
            "events":                    [{"name": ", ".join(hour_event_names),
                                           "passengers_per_hr": hour_event_pax}] if hour_event_pax else [],
            "emergency":                 None,
            "line":                      line,
            "current_frequency_per_hr":  std_freq,
            "train_capacity":            train_capacity,
            "running_cost_per_train_hr": cost_per_train_hr,
        }
        options  = compute_options(inputs_hour)
        moderate = next(o for o in options if o["label"] == "moderate")
        rec_freq = moderate["recommended_frequency_per_hr"]

        extra_trains   = rec_freq - std_freq
        std_cost       = std_freq * cost_per_train_hr
        extra_cost     = extra_trains * cost_per_train_hr
        total_cost     = std_cost + extra_cost
        end_hour       = hour + 1 if hour < 23 else 0
        time_slot      = f"{hour:02d}:00–{end_hour:02d}:00" if end_hour else "23:00–00:00"
        headway_std    = get_standard_headway(hour, weekday, line)
        headway_rec    = int(round(60 / max(rec_freq, 1)))

        expected_pax      = moderate["expected_passengers_per_hr"]
        std_capacity      = std_freq * train_capacity
        std_load_factor   = round(min(expected_pax / max(std_capacity, 1), 2.0) * 100, 1)

        schedule.append({
            "hour":                     hour,
            "expected_passengers_per_hr": expected_pax,
            "time_slot":                time_slot,
            "headway_std_min":          headway_std,
            "headway_rec_min":          headway_rec,
            "standard_frequency":       std_freq,
            "recommended_frequency":    rec_freq,
            "extra_trains":             extra_trains,
            "standard_cost_rm":         std_cost,
            "extra_cost_rm":            extra_cost,
            "total_cost_rm":            total_cost,
            "standard_load_factor_pct": std_load_factor,
            "load_factor_pct":          moderate["load_factor_pct"],
            "passengers_served_per_hr": moderate["passengers_served_per_hr"],
            "has_event":                bool(hour_event_pax),
            "event_names":              hour_event_names,
        })

    return schedule
