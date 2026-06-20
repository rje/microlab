from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

READ_STATES = {"unread", "skimming", "mapped", "built", "mastered"}
DEPTHS = {"implement", "understand", "aware"}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_progress (
                paper_id   TEXT PRIMARY KEY,
                read_state TEXT NOT NULL DEFAULT 'unread',
                depth      TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_all_progress(db_path: str | Path) -> dict[str, dict[str, str | None]]:
    if not Path(db_path).exists():
        return {}
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT paper_id, read_state, depth FROM paper_progress").fetchall()
    finally:
        conn.close()
    return {
        row["paper_id"]: {"readState": row["read_state"], "depth": row["depth"]}
        for row in rows
    }


def upsert_progress(
    db_path: str | Path, paper_id: str, read_state: str, depth: str | None
) -> None:
    if read_state not in READ_STATES:
        raise ValueError(f"invalid read_state: {read_state!r}")
    if depth is not None and depth not in DEPTHS:
        raise ValueError(f"invalid depth: {depth!r}")
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO paper_progress (paper_id, read_state, depth, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                read_state = excluded.read_state,
                depth = excluded.depth,
                updated_at = excluded.updated_at
            """,
            (paper_id, read_state, depth, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
