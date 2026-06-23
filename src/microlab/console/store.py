from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

READ_STATES = {"unread", "skimming", "mapped", "built", "mastered"}
DEPTHS = {"implement", "understand", "aware"}
TASK_STATUSES = {"done", "active", "queued", "blocked"}


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_status (
                phase_id   TEXT NOT NULL,
                task_id    TEXT NOT NULL,
                status     TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (phase_id, task_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recall_state (
                card_id       TEXT PRIMARY KEY,
                paper_id      TEXT NOT NULL,
                ease          REAL NOT NULL DEFAULT 2.5,
                interval_days INTEGER NOT NULL DEFAULT 0,
                reps          INTEGER NOT NULL DEFAULT 0,
                due_at        TEXT NOT NULL,
                last_reviewed TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recall_reviews (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id     TEXT NOT NULL,
                reviewed_at TEXT NOT NULL,
                grade       INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS highlights (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id   TEXT NOT NULL,
                page       INTEGER NOT NULL,
                rects      TEXT NOT NULL,
                text       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_all_progress(db_path: str | Path) -> dict[str, dict[str, str | None]]:
    # init_db is idempotent and ensures every table exists, even on a DB file
    # created by an older schema (a plain SELECT from a missing table errors).
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT paper_id, read_state, depth FROM paper_progress").fetchall()
    finally:
        conn.close()
    return {
        row["paper_id"]: {"readState": row["read_state"], "depth": row["depth"]}
        for row in rows
    }


def get_all_task_status(db_path: str | Path) -> dict[str, dict[str, str]]:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT phase_id, task_id, status FROM task_status").fetchall()
    finally:
        conn.close()
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        result.setdefault(row["phase_id"], {})[row["task_id"]] = row["status"]
    return result


def set_task_status(db_path: str | Path, phase_id: str, task_id: str, status: str) -> None:
    if status not in TASK_STATUSES:
        raise ValueError(f"invalid task status: {status!r}")
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO task_status (phase_id, task_id, status, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(phase_id, task_id) DO UPDATE SET
                status = excluded.status, updated_at = excluded.updated_at
            """,
            (phase_id, task_id, status, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


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


def get_review_state(db_path: str | Path) -> dict[str, dict[str, object]]:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT card_id, due_at, interval_days, reps, ease FROM recall_state"
        ).fetchall()
    finally:
        conn.close()
    return {
        r["card_id"]: {
            "dueAt": r["due_at"],
            "intervalDays": r["interval_days"],
            "reps": r["reps"],
            "ease": r["ease"],
        }
        for r in rows
    }


def count_reviews(db_path: str | Path) -> int:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        return int(conn.execute("SELECT COUNT(*) AS n FROM recall_reviews").fetchone()["n"])
    finally:
        conn.close()


def record_review(
    db_path: str | Path, card_id: str, paper_id: str, grade: int, today: date
) -> dict[str, object]:
    if grade not in range(6):
        raise ValueError(f"grade must be 0..5, got {grade!r}")
    init_db(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT ease, interval_days, reps FROM recall_state WHERE card_id = ?", (card_id,)
        ).fetchone()
        ease = row["ease"] if row else 2.5
        reps = row["reps"] if row else 0
        interval = row["interval_days"] if row else 0
        if grade < 3:
            reps = 0
            interval = 1
        else:
            reps += 1
            if reps == 1:
                interval = 1
            elif reps == 2:
                interval = 6
            else:
                interval = round(interval * ease)
        ease = max(1.3, ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
        due = (today + timedelta(days=interval)).isoformat()
        conn.execute(
            """
            INSERT INTO recall_state
                (card_id, paper_id, ease, interval_days, reps, due_at, last_reviewed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(card_id) DO UPDATE SET
                ease = excluded.ease, interval_days = excluded.interval_days,
                reps = excluded.reps, due_at = excluded.due_at,
                last_reviewed = excluded.last_reviewed
            """,
            (card_id, paper_id, ease, interval, reps, due, today.isoformat()),
        )
        conn.execute(
            "INSERT INTO recall_reviews (card_id, reviewed_at, grade) VALUES (?, ?, ?)",
            (card_id, today.isoformat(), grade),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ease": ease, "intervalDays": interval, "reps": reps, "dueAt": due}


def list_highlights(db_path: str | Path, paper_id: str) -> list[dict[str, object]]:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, paper_id, page, rects, text, created_at FROM highlights "
            "WHERE paper_id = ? ORDER BY page, id",
            (paper_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "paperId": r["paper_id"],
            "page": r["page"],
            "rects": json.loads(r["rects"]),
            "text": r["text"],
            "createdAt": r["created_at"],
        }
        for r in rows
    ]


def add_highlight(
    db_path: str | Path, paper_id: str, page: int, rects: list, text: str
) -> dict[str, object]:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO highlights (paper_id, page, rects, text, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (paper_id, int(page), json.dumps(rects), str(text), datetime.now(UTC).isoformat()),
        )
        conn.commit()
        hid = cur.lastrowid
    finally:
        conn.close()
    return {"id": hid, "paperId": paper_id, "page": int(page), "rects": rects, "text": str(text)}


def delete_highlight(db_path: str | Path, paper_id: str, highlight_id: int) -> bool:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "DELETE FROM highlights WHERE paper_id = ? AND id = ?", (paper_id, int(highlight_id))
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
