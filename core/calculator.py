"""Rule-based math layer grounded in LRT system assumptions:
  - Train capacity (normal): 800 pax | max: 900 pax
  - Weekday schedule derived from actual hourly ridership data
  - Sat/Sun schedule from official LRT timetable
  - Bad weather: reduces BASELINE demand but increases EVENT ridership (people take LRT instead of driving)
  - Bad weather also caps max frequency (trains slow down for safety)
"""

import math
from datetime import datetime

# ── Constants from assumptions ────────────────────────────────────────────────
TRAIN_CAPACITY = 800          # normal comfort capacity per train
TRAIN_CAPACITY_MAX = 900      # absolute max (crush load)
TARGET_LOAD_FACTOR = 0.75     # aim for 75% full — leave headroom for comfort
CARBON_TAX_PER_TRAIN_PER_HOUR = 4.50  # carbon tax per train per hour



# Weather — effect on BASELINE LRT demand
# Rain/cloudy → more people switch from driving/walking to LRT (modal shift)
# Stormy → severe enough that people cancel trips entirely
WEATHER_PAX_MULT = {
    "clear":  1.00,
    "cloudy": 1.05,
    "rainy":  1.15,
    "stormy": 1.05,
}

# Weather — effect on EVENT passenger load at station.
# Rain pushes event-goers onto LRT instead of driving or walking.
# Source: Malaysia Cup 2024 verdict — "rainy weather consistently adds 15-20% more passengers than forecast"
WEATHER_EVENT_MULT = {
    "clear":  1.00,
    "cloudy": 1.05,   # slight shift to LRT
    "rainy":  1.20,   # +20% — empirically validated
    "stormy": 0.85,   # fewer people attend outdoor events in a storm
}

# Weather — max allowed frequency (trains must slow down in bad weather)
WEATHER_MAX_FREQ = {
    "clear":  20,
    "cloudy": 20,
    "rainy":  17,
    "stormy": 12,
}

# Weekday baseline passengers per hour (Mon–Fri, all lines, from actual ridership data)
_WEEKDAY_BASELINE = {
    6:  2_300,
    7:  10_500,
    8:  11_000,
    9:  6_500,
    10: 5_800,
    11: 5_600,
    12: 6_800,
    13: 7_200,
    14: 7_400,
    15: 8_600,
    16: 9_800,
    17: 11_000,
    18: 12_000,
    19: 7_500,
    20: 5_300,
    21: 3_500,
    22: 3_200,
    23: 2_000,
}

_SATURDAY_BASELINE = {
    6:    800,  7:  1_500,  8:  2_200,  9:  3_500,
    10: 4_800, 11:  5_500, 12:  6_200, 13:  6_500,
    14: 6_800, 15:  6_500, 16:  6_000, 17:  6_500,
    18: 7_200, 19:  7_000, 20:  5_500, 21:  4_000,
    22: 2_500, 23:  1_200,
}

_SUNDAY_BASELINE = {
    6:    500,  7:    900,  8:  1_500,  9:  2_200,
    10: 3_500, 11:  4_500, 12:  5_500, 13:  5_800,
    14: 5_500, 15:  5_200, 16:  5_000, 17:  5_500,
    18: 6_000, 19:  5_000, 20:  3_800, 21:  2_800,
    22: 1_800, 23:    900,
}

_SATURDAY_FREQ = {
    6:  2,  7:  3,  8:  4,  9:  6,
    10: 8, 11: 10, 12: 11, 13: 11,
    14: 12, 15: 11, 16: 10, 17: 11,
    18: 12, 19: 12, 20: 10, 21:  7,
    22: 5, 23:  2,
}

_SUNDAY_FREQ = {
    6:  2,  7:  2,  8:  3,  9:  4,
    10: 6, 11:  8, 12: 10, 13: 10,
    14: 10, 15: 9, 16:  9, 17: 10,
    18: 10, 19: 9, 20:  7, 21:  5,
    22: 3, 23:  2,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

# ── Event type profiles ───────────────────────────────────────────────────────
# pre_mult  : fraction of event pax traveling TO venue in the hour BEFORE
# during_mult: fraction using LRT DURING the event (latecomers / early leavers)
# exit_mult : fraction leaving all at once in the hour AFTER
_EVENT_PROFILES = {
    "concert": {
        "pre_mult": 0.70, "during_mult": 0.00, "exit_mult": 1.00,
    },
    "football_match": {
        "pre_mult": 0.80, "during_mult": 0.30, "exit_mult": 1.00,
    },
    "festival": {
        "pre_mult": 0.70, "during_mult": 0.00, "exit_mult": 1.00,
    },
    "marathon": {
        "pre_mult": 0.90, "during_mult": 0.20, "exit_mult": 0.60,
    },
    "exhibition": {
        "pre_mult": 0.50, "during_mult": 0.40, "exit_mult": 0.60,
    },
    "religious_event": {
        "pre_mult": 0.80, "during_mult": 0.10, "exit_mult": 0.90,
    },
    "default": {
        "pre_mult": 0.70, "during_mult": 0.20, "exit_mult": 1.00,
    },
}


def _parse_dt(dt_string: str) -> datetime:
    return datetime.fromisoformat(dt_string)


def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # Saturday=5, Sunday=6


def _baseline_pax(hour: int, weekday: int) -> int:
    if weekday == 6:
        return _SUNDAY_BASELINE.get(hour, 0)
    if weekday == 5:
        return _SATURDAY_BASELINE.get(hour, 0)
    return _WEEKDAY_BASELINE.get(hour, 0)


def get_baseline_pax(hour: int, weekday: int) -> int:
    return _baseline_pax(hour, weekday)


# Weekday trains/hr derived from actual ridership data (all lines, Mon–Fri)
# Formula: ceil(expected_pax / (800 capacity × 0.75 load factor))
_WEEKDAY_FREQ = {
    6:  4,
    7:  18,
    8:  19,
    9:  11,
    10: 10,
    11: 10,
    12: 12,
    13: 12,
    14: 13,
    15: 15,
    16: 17,
    17: 19,
    18: 20,
    19: 13,
    20: 9,
    21: 6,
    22: 6,
    23: 4,
}


def get_standard_headway(hour: int, weekday: int) -> int:
    """Headway in minutes derived from per-hour frequency tables. weekday: 0=Mon … 6=Sun."""
    if weekday == 6:
        freq = _SUNDAY_FREQ.get(hour, 2)
    elif weekday == 5:
        freq = _SATURDAY_FREQ.get(hour, 2)
    else:
        freq = _WEEKDAY_FREQ.get(hour, 4)
    return round(60 / max(freq, 1))


def default_frequency(hour: int, weekday_or_is_weekend) -> int:
    """Trains per hour from the per-hour frequency tables.
    Accepts weekday int (0=Mon … 6=Sun) or legacy is_weekend bool."""
    if isinstance(weekday_or_is_weekend, bool):
        weekday = 5 if weekday_or_is_weekend else 0
    else:
        weekday = int(weekday_or_is_weekend)
    if weekday == 6:
        return _SUNDAY_FREQ.get(hour, 2)
    if weekday == 5:
        return _SATURDAY_FREQ.get(hour, 2)
    return _WEEKDAY_FREQ.get(hour, 4)


# ── Core computation ──────────────────────────────────────────────────────────

def _option_metrics(inputs: dict, freq: int, expected: int, max_freq: int) -> dict:
    capacity   = inputs.get("train_capacity", TRAIN_CAPACITY)
    curr_freq  = inputs["current_frequency_per_hr"]
    cost_per_hr = inputs.get("running_cost_per_train_hr", 350)

    curr_capacity = curr_freq * capacity
    new_capacity  = freq * capacity   # will be 0 if freq=0 (service suspended)

    curr_served = min(expected, curr_capacity)
    new_served  = min(expected, new_capacity)  # 0 if suspended

    cost_delta = (freq - curr_freq) * cost_per_hr
    carbon_tax_delta = (freq - curr_freq) * CARBON_TAX_PER_TRAIN_PER_HOUR

    passengers_served        = new_served
    passengers_served_delta  = new_served - curr_served

    load_before_raw   = expected / max(curr_capacity, 1)
    load_after_raw    = expected / max(new_capacity, 1) if new_capacity > 0 else float("inf")
    load_after        = 300.0 if load_after_raw == float("inf") else round(min(load_after_raw, 3.0) * 100, 1)
    congestion_change = round((min(load_after_raw, 9.99) - load_before_raw) * 100, 1)

    wait_before = round(60 / max(curr_freq, 1) / 2, 1)
    wait_after  = round(60 / max(freq, 1) / 2, 1) if freq > 0 else None
    time_saved  = round(wait_before - wait_after, 1) if wait_after is not None else None

    return {
        "recommended_frequency_per_hr":  freq,
        "current_frequency_per_hr":      curr_freq,
        "expected_passengers_per_hr":    expected,
        "passengers_served_per_hr":      passengers_served,
        "passengers_served_delta":       passengers_served_delta,
        "load_factor_pct":               load_after,
        "congestion_change_pct":         congestion_change,
        "cost_delta_rm":                 round(cost_delta, 2),
        "carbon_tax_delta_rm":           round(carbon_tax_delta, 2),
        "time_saved_min_per_passenger":  time_saved,
    }


def compute_options(inputs: dict) -> list[dict]:
    dt       = _parse_dt(inputs["datetime"])
    hour     = dt.hour
    weekday  = dt.weekday()
    weather  = inputs.get("weather", "clear")
    events   = inputs.get("events", [])

    # Step 1: baseline passengers — use CCTV override if available, else standard baseline
    if "cctv_pax_override" in inputs:
        baseline = inputs["cctv_pax_override"]
        raw_event_pax = 0
        expected = baseline
    else:
        baseline = _baseline_pax(hour, weekday)
        raw_event_pax = sum(e.get("passengers_per_hr", 0) for e in events)
    # Only apply GLM weather multipliers when weather is actually bad (not clear)
    if weather == "clear":
        _event_mult = 1.0
        pax_mult    = 1.0
    else:
        _event_mult = inputs.get("weather_event_mult", WEATHER_EVENT_MULT.get(weather, 1.0))
        pax_mult    = inputs.get("weather_pax_mult",   WEATHER_PAX_MULT.get(weather, 1.0))
    if "cctv_pax_override" not in inputs:
        event_pax = int(raw_event_pax * _event_mult)
        # Step 3: weather reduces baseline travel only
        expected = int(baseline * pax_mult + event_pax)

    # Step 3.5: emergency inflates expected passengers (stranded/surge)
    emergency_type = inputs.get("emergency_type") or ("overcrowding" if inputs.get("emergency") else None)
    _em_pax_mult = inputs.get("emergency_pax_mult", 1.0)
    if emergency_type == "overcrowding":
        expected = int(expected * max(1.5, _em_pax_mult))
    elif emergency_type and _em_pax_mult > 1.0:
        expected = int(expected * _em_pax_mult)

    # Step 4: how many EXTRA trains needed on top of the standard schedule.
    capacity        = inputs.get("train_capacity", TRAIN_CAPACITY)
    std_freq        = inputs["current_frequency_per_hr"]
    std_comfortable = std_freq * capacity * TARGET_LOAD_FACTOR
    extra_demand    = max(0, expected - std_comfortable)
    extra_needed    = round(extra_demand / (capacity * TARGET_LOAD_FACTOR))

    # Step 5: weather frequency cap
    max_freq = WEATHER_MAX_FREQ.get(weather, 20)

    # Step 6: build freq_map relative to the standard schedule

    if emergency_type == "signal_failure":
        # Signal failure — immediate full stop, trains cannot navigate safely
        freq_map = {
            "conservative": 0,
            "moderate":     0,
            "aggressive":   min(3, max_freq),   # only if operator manually overrides
        }
    elif emergency_type == "power_failure":
        # Power failure — reduce heavily but may keep some trains moving on backup
        reduce_to = max(1, std_freq - 2)
        max_freq  = min(max_freq, max(1, std_freq - 1))
        freq_map  = {
            "conservative": max(1, reduce_to - 1),
            "moderate":     reduce_to,
            "aggressive":   min(reduce_to + 1, max_freq),
        }
    elif emergency_type:
        # overcrowding or generic — urgently add trains above standard + event demand
        target = std_freq + extra_needed
        if target >= max_freq:
            freq_map = {
                "conservative": max(1, max_freq - 4),
                "moderate":     max(1, max_freq - 2),
                "aggressive":   max_freq,
            }
        else:
            freq_map = {
                "conservative": max(1, target - 2),
                "moderate":     target,
                "aggressive":   min(target + 3, max_freq),
            }
    else:
        if "cctv_pax_override" in inputs:
            # CCTV: compute optimal frequency purely from detected pax — can go below standard.
            # Floor at std_freq // 2 to avoid gutting service entirely.
            optimal = math.ceil(expected / (capacity * TARGET_LOAD_FACTOR))
            base    = max(optimal, max(1, std_freq // 2))
        else:
            # Normal: anchor on std_freq + extra (only ever adds trains above standard).
            base = std_freq + extra_needed

        if base > max_freq:
            freq_map = {
                "conservative": max(1, max_freq - 4),
                "moderate":     max(1, max_freq - 2),
                "aggressive":   max_freq,
            }
        else:
            freq_map = {
                "conservative": max(1, base - 2),
                "moderate":     base,
                "aggressive":   min(base + 3, max_freq),
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
    weather_window: tuple[int, int] | None = None,
    emergency_type: str | None = None,
    emergency_hour: int | None = None,
    emergency_duration: int = 1,
    pax_factors: dict | None = None,
    cctv_pax_override: int | None = None,
) -> list[dict]:
    """
    Compute full day schedule from 06:00 to 24:00 (18 slots).
    weather_window: weather only applies within (start, end) range.
    emergency_type/hour/duration: models a real-time incident with a recovery curve.
    """
    dt_base  = _parse_dt(date_str + "T06:00")
    weekday  = dt_base.weekday()          # 0=Mon … 6=Sun

    em_end      = (emergency_hour + emergency_duration) if emergency_hour is not None else None
    backlog     = 0   # unserved passengers carrying over from emergency hours
    schedule    = []
    for hour in range(6, 24):
        # Apply weather only within the selected window; clear elsewhere
        hour_weather = weather if (
            weather_window is None or
            weather_window[0] <= hour < weather_window[1]
        ) else "clear"

        # Collect event passengers active this hour
        hour_event_pax   = 0
        hour_event_names = []
        is_tail          = False
        for ev in events:
            ev_start = ev.get("start_hour", 0)
            ev_end   = ev.get("end_hour", 0)
            pax      = ev.get("passengers_per_hr", 0)
            profile  = _EVENT_PROFILES.get(ev.get("event_type", "default"),
                                           _EVENT_PROFILES["default"])
            if hour == ev_start - 1:
                hour_event_pax += int(pax * profile["pre_mult"])
                hour_event_names.append(f"{ev['name']} (arrival)")
                is_tail = True
            elif ev_start <= hour < ev_end:
                hour_event_pax += int(pax * profile["during_mult"])
                hour_event_names.append(ev["name"])
            elif hour == ev_end:
                hour_event_pax += int(pax * profile["exit_mult"])
                hour_event_names.append(f"{ev['name']} (exit rush)")
                is_tail = True

        std_freq = default_frequency(hour, weekday)

        # Reuse options logic — treat standard freq as current
        # Collect event types active this hour for profile lookup
        hour_event_types = []
        for ev in events:
            ev_start = ev.get("start_hour", 0)
            ev_end   = ev.get("end_hour", 0)
            if (hour == ev_start - 1) or (ev_start <= hour < ev_end) or (hour == ev_end):
                hour_event_types.append(ev.get("event_type", "default"))
        _ev_type = hour_event_types[0] if hour_event_types else "default"

        _is_em_active = emergency_type and emergency_hour is not None and emergency_hour <= hour < em_end
        _in_cctv_window = cctv_pax_override is not None and (
            weather_window is None or weather_window[0] <= hour < weather_window[1]
        )
        # Resolve per-hour CCTV value: override can be a {hour: pax} dict or a flat int
        if _in_cctv_window:
            if isinstance(cctv_pax_override, dict):
                _cctv_val = cctv_pax_override.get(hour)
            else:
                _cctv_val = cctv_pax_override
        else:
            _cctv_val = None
        inputs_hour = {
            "datetime":                  f"{date_str}T{hour:02d}:00",
            "weather":                   hour_weather,
            "events":                    [{"name": ", ".join(hour_event_names),
                                           "passengers_per_hr": hour_event_pax,
                                           "event_type": _ev_type}] if hour_event_pax else [],
            "emergency":                 emergency_type if _is_em_active else None,
            "emergency_type":            emergency_type if _is_em_active else None,
            "line":                      line,
            "current_frequency_per_hr":  std_freq,
            "train_capacity":            train_capacity,
            "running_cost_per_train_hr": cost_per_train_hr,
            **(pax_factors or {}),
            **( {"cctv_pax_override": _cctv_val} if _cctv_val is not None else {} ),
        }
        options      = compute_options(inputs_hour)
        moderate_opt = next(o for o in options if o["label"] == "moderate")
        rec_freq     = moderate_opt["recommended_frequency_per_hr"]
        expected_pax = moderate_opt["expected_passengers_per_hr"]

        # ── Emergency recovery curve ──────────────────────────────────────────
        em_status = None   # "active" | "recovery1" | "recovery2"
        max_freq  = WEATHER_MAX_FREQ.get(hour_weather, 20)

        if emergency_type and emergency_hour is not None:
            if emergency_hour <= hour < em_end:
                em_status = "active"
                if emergency_type == "signal_failure":
                    rec_freq = 0
                elif emergency_type == "power_failure":
                    rec_freq = max(1, std_freq - 4)
                elif emergency_type == "overcrowding":
                    # Overcrowding: compute_options already inflated expected pax by 1.5x,
                    # so rec_freq from moderate option already accounts for the surge.
                    # No override needed — keep the computed rec_freq.
                    pass
                else:
                    rec_freq = max(1, std_freq - 2)
                served  = rec_freq * train_capacity
                backlog += max(0, expected_pax - served)

            elif hour == em_end:
                em_status = "recovery1"
                # Clear backlog: boost expected by stranded passengers
                expected_pax = expected_pax + backlog
                extra_needed = max(0, round((expected_pax - std_freq * train_capacity * TARGET_LOAD_FACTOR)
                                            / (train_capacity * TARGET_LOAD_FACTOR)))
                rec_freq = min(max_freq, std_freq + extra_needed)
                backlog  = max(0, expected_pax - rec_freq * train_capacity)

            elif hour == em_end + 1 and backlog > 0:
                em_status = "recovery2"
                expected_pax = expected_pax + backlog // 2
                extra_needed = max(0, round((expected_pax - std_freq * train_capacity * TARGET_LOAD_FACTOR)
                                            / (train_capacity * TARGET_LOAD_FACTOR)))
                rec_freq = min(max_freq, std_freq + extra_needed)
                backlog  = 0
        # ─────────────────────────────────────────────────────────────────────

        new_capacity  = rec_freq * train_capacity
        load_pct      = 300.0 if new_capacity == 0 else round(min(expected_pax / max(new_capacity, 1), 3.0) * 100, 1)
        passengers_served = min(expected_pax, new_capacity)

        extra_trains   = rec_freq - std_freq
        std_cost       = std_freq * cost_per_train_hr
        extra_cost     = extra_trains * cost_per_train_hr
        total_cost     = std_cost + extra_cost

        # Carbon tax calculations
        std_carbon_tax       = std_freq * CARBON_TAX_PER_TRAIN_PER_HOUR
        extra_carbon_tax     = extra_trains * CARBON_TAX_PER_TRAIN_PER_HOUR
        total_carbon_tax     = std_carbon_tax + extra_carbon_tax

        end_hour  = hour + 1
        time_slot = f"{hour:02d}:00–{end_hour:02d}:00"
        headway_std    = get_standard_headway(hour, weekday)
        headway_rec    = int(round(60 / max(rec_freq, 1)))

        std_capacity      = std_freq * train_capacity
        std_load_factor   = round(min(expected_pax / max(std_capacity, 1), 3.0) * 100, 1)

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
            "standard_carbon_tax_rm":   std_carbon_tax,
            "extra_carbon_tax_rm":      extra_carbon_tax,
            "total_carbon_tax_rm":      total_carbon_tax,
            "standard_load_factor_pct": std_load_factor,
            "load_factor_pct":          load_pct,
            "passengers_served_per_hr": passengers_served,
            "has_event":                bool(hour_event_pax) or bool(hour_event_names),
            "is_event_tail":            is_tail,
            "event_names":              hour_event_names,
            "em_status":                em_status,
        })

    return schedule
