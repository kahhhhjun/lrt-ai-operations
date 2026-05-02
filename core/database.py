"""SQLite persistence layer for LRT AI Operations.

Tables:
  saved_schedules — one row per (date, line), stores full updated schedule JSON.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "data" / "lrt_schedules.db"


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at app startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_schedules (
                date              TEXT NOT NULL,
                line              TEXT NOT NULL,
                schedule_json     TEXT NOT NULL,
                weather           TEXT,
                events_json       TEXT,
                emergency_type    TEXT,
                total_std_cost    REAL,
                total_extra_cost  REAL,
                total_cost        REAL,
                total_std_carbon_tax    REAL,
                total_extra_carbon_tax  REAL,
                total_carbon_tax        REAL,
                saved_at          TEXT,
                notes             TEXT,
                PRIMARY KEY (date, line)
            )
        """)
        # Migrate existing tables that lack newer columns
        for col, typedef in [("emergency_type", "TEXT"), ("tune_start", "INTEGER"), ("tune_end", "INTEGER"),
                            ("total_std_carbon_tax", "REAL"), ("total_extra_carbon_tax", "REAL"), ("total_carbon_tax", "REAL")]:
            try:
                conn.execute(f"ALTER TABLE saved_schedules ADD COLUMN {col} {typedef}")
            except Exception:
                pass  # column already exists


def save_schedule(
    date: str,
    line: str,
    schedule: list[dict],
    weather: str = "clear",
    events: list[dict] | None = None,
    emergency_type: str | None = None,
    tune_start: int = 6,
    tune_end: int = 24,
    total_std_cost: float = 0,
    total_extra_cost: float = 0,
    total_cost: float = 0,
    total_std_carbon_tax: float = 0,
    total_extra_carbon_tax: float = 0,
    total_carbon_tax: float = 0,
    notes: str = "",
) -> None:
    """Insert or replace a saved schedule for (date, line)."""
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO saved_schedules
            (date, line, schedule_json, weather, events_json, emergency_type,
             tune_start, tune_end,
             total_std_cost, total_extra_cost, total_cost,
             total_std_carbon_tax, total_extra_carbon_tax, total_carbon_tax,
             saved_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, line,
            json.dumps(schedule),
            weather,
            json.dumps(events or []),
            emergency_type,
            tune_start,
            tune_end,
            total_std_cost,
            total_extra_cost,
            total_cost,
            total_std_carbon_tax,
            total_extra_carbon_tax,
            total_carbon_tax,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            notes,
        ))


def load_schedule(date: str, line: str) -> dict | None:
    """Return saved record for (date, line) or None if not saved."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM saved_schedules WHERE date = ? AND line = ?",
            (date, line)
        ).fetchone()
    if row is None:
        return None
    return {
        "date":             row["date"],
        "line":             row["line"],
        "schedule":         json.loads(row["schedule_json"]),
        "weather":          row["weather"],
        "events":           json.loads(row["events_json"]),
        "emergency_type":   row["emergency_type"],
        "tune_start":       row["tune_start"] or 6,
        "tune_end":         row["tune_end"] or 24,
        "total_std_cost":   row["total_std_cost"],
        "total_extra_cost": row["total_extra_cost"],
        "total_cost":       row["total_cost"],
        "total_std_carbon_tax":   row["total_std_carbon_tax"] or 0,
        "total_extra_carbon_tax": row["total_extra_carbon_tax"] or 0,
        "total_carbon_tax":       row["total_carbon_tax"] or 0,
        "saved_at":         row["saved_at"],
        "notes":            row["notes"],
    }


def delete_schedule(date: str, line: str) -> None:
    """Delete saved record for (date, line)."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM saved_schedules WHERE date = ? AND line = ?",
            (date, line)
        )


def list_saved(line: str | None = None) -> list[dict]:
    """Return all saved records, optionally filtered by line."""
    query = "SELECT date, line, weather, events_json, emergency_type, total_cost, saved_at, notes FROM saved_schedules"
    params: tuple = ()
    if line:
        query += " WHERE line = ?"
        params = (line,)
    query += " ORDER BY date DESC"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_monthly_summary(year: int, month: int, line: str) -> list[dict]:
    """Return all saved days for a given month and line."""
    prefix = f"{year}-{month:02d}-%"
    with _connect() as conn:
        rows = conn.execute("""
            SELECT date, total_std_cost, total_extra_cost, total_cost, notes, saved_at
            FROM saved_schedules
            WHERE line = ? AND date LIKE ?
            ORDER BY date
        """, (line, prefix)).fetchall()
    return [dict(r) for r in rows]
