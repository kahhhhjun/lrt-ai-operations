"""Generates a synthetic LRT ridership CSV. Run once: python data/generator.py"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

WEATHERS  = ["clear", "clear", "clear", "cloudy", "rainy", "stormy"]
LINES     = ["Kelana Jaya", "Ampang", "Sri Petaling"]
EVENTS    = ["Coldplay Concert", "Football Match", "Marathon", "Festival"]

PAX_PEAK        = 5_000
PAX_OFF_PEAK    = 2_500
PAX_NIGHT       =   800
PAX_WEEKEND_DAY = 2_000
PAX_WEEKEND_LOW =   600

WEATHER_PAX_MULT = {"clear": 1.0, "cloudy": 0.95, "rainy": 0.85, "stormy": 0.70}
PEAK_HOURS = {7, 8, 17, 18}


def generate(rows: int = 500, output: str = "data/synthetic.csv") -> None:
    start = datetime(2026, 1, 1, 6, 0)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "datetime", "day_type", "line", "weather",
            "event_name", "event_passengers_per_hr", "expected_passengers_per_hr"
        ])

        for i in range(rows):
            ts      = start + timedelta(hours=i)
            hour    = ts.hour
            is_wknd = ts.weekday() >= 5
            day_type = "weekend" if is_wknd else "weekday"
            weather  = random.choice(WEATHERS)
            line     = random.choice(LINES)

            has_event    = random.random() < 0.08
            event_name   = random.choice(EVENTS) if has_event else ""
            event_pax_hr = random.randint(1_000, 5_000) if has_event else 0

            if is_wknd:
                base = PAX_WEEKEND_DAY if 11 <= hour <= 19 else PAX_WEEKEND_LOW
            elif hour in PEAK_HOURS:
                base = PAX_PEAK
            elif 6 <= hour <= 22:
                base = PAX_OFF_PEAK
            else:
                base = PAX_NIGHT

            mult     = WEATHER_PAX_MULT[weather]
            expected = int((base + event_pax_hr) * mult * random.uniform(0.9, 1.1))

            writer.writerow([
                ts.isoformat(timespec="minutes"), day_type, line, weather,
                event_name, event_pax_hr, expected
            ])

    print(f"Wrote {rows} rows to {out_path}")


if __name__ == "__main__":
    generate()
