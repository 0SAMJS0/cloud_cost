"""
SQLite prediction log. Every endpoint call appends one row so the Phase 4 dashboard can
read the history. No ORM, no migrations (YAGNI) — a single table created on startup.

Schema: predictions(id, timestamp, endpoint, input_summary JSON, output JSON)
File:   <project_root>/predictions.db
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "predictions.db"


def _connect():
    # check_same_thread=False so the connection can be opened from request threads.
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    """Create the predictions table if it does not exist."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                endpoint      TEXT NOT NULL,
                input_summary TEXT NOT NULL,
                output        TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_prediction(endpoint: str, input_summary: dict, output: dict):
    """Append one prediction record. Logging never breaks the request."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO predictions (timestamp, endpoint, input_summary, output) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                endpoint,
                json.dumps(input_summary, default=str),
                json.dumps(output, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def recent(limit: int = 20):
    """Return the most recent rows (used by tests / quick inspection)."""
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT id, timestamp, endpoint, input_summary, output "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()
    finally:
        conn.close()
