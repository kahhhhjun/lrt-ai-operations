"""Generates a synthetic LRT ridership CSV. Run once: python data/generator.py"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

WEATHERS = ["clear", "clear", "clear", "cloudy", "rainy", "stormy"]
LINES = ["Kelana Jaya", "Ampang", "Sri Petaling", "Kajang", "Putrajaya"]


def generate(rows: int = 500, output: str = "data/synthetic.csv") -> None:
    start = datetime(2026, 1, 1, 6, 0)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "datetime", "day", "line", "weather", "is_holiday",
            "event_nearby", "expected_attendance", "passengers"
        ])
        for i in range(rows):
            ts = start + timedelta(hours=i)
            hour = ts.hour
            is_weekend = ts.weekday() >= 5
            day_label = "weekend" if is_weekend else "weekday"
            is_peak = (not is_weekend) and hour in {7, 8, 9, 17, 18, 19}
            is_holiday = random.random() < 0.1
            weather = random.choice(WEATHERS)
            line = random.choice(LINES)
            event_nearby = random.random() < 0.08
            attendance = random.randint(10_000, 80_000) if event_nearby else 0

            if is_holiday:
                base = 2_500
            elif is_weekend:
                base = 3_500 if 11 <= hour <= 21 else 800
            elif is_peak:
                base = 8_000
            elif 6 <= hour <= 22:
                base = 4_000
            else:
                base = 800

            mult = {"clear": 1.0, "cloudy": 1.05, "rainy": 1.20, "stormy": 1.35}[weather]
            event_mult = 1.0 + min(attendance / 100_000, 1.5)
            passengers = int(base * mult * event_mult * random.uniform(0.9, 1.1))

            writer.writerow([
                ts.isoformat(timespec="minutes"), day_label, line, weather, is_holiday,
                event_nearby, attendance, passengers
            ])

    print(f"Wrote {rows} rows to {out_path}")


if __name__ == "__main__":
    generate()
