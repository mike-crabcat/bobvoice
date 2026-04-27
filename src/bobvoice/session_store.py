"""SQLite-backed session message store for Bob Voice."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


_DEFAULT_DB_PATH = Path.home() / ".openclaw" / "bobvoice-sessions.db"

_LOGGER = logging.getLogger("bobvoice.session_store")


class SessionStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                language TEXT,
                created_at TEXT NOT NULL
            )
        """)
        try:
            self._conn.execute("ALTER TABLE session_messages ADD COLUMN language TEXT")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages_lookup
            ON session_messages(session_key, created_at)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS lesson_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                lesson_number INTEGER NOT NULL,
                step_index INTEGER NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                completed_at TEXT,
                UNIQUE(user_id, mode, lesson_number, step_index)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS current_lesson (
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                lesson_number INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (user_id, mode)
            )
        """)
        self._conn.commit()
        _LOGGER.info("Session store opened: %s", self._db_path)

    def add_message(self, session_key: str, role: str, text: str, language: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO session_messages (session_key, role, text, language, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_key, role, text, language, now),
        )
        self._conn.commit()

    def get_messages(self, session_key: str, limit: int = 200) -> list[dict[str, str | None]]:
        cursor = self._conn.execute(
            "SELECT role, text, language FROM session_messages WHERE session_key = ? ORDER BY created_at ASC LIMIT ?",
            (session_key, limit),
        )
        return [{"role": row[0], "text": row[1], "language": row[2]} for row in cursor]

    def delete_session(self, session_key: str) -> None:
        self._conn.execute(
            "DELETE FROM session_messages WHERE session_key = ?",
            (session_key,),
        )
        self._conn.commit()

    def delete_old_sessions(self, max_age_days: int = 30) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM session_messages WHERE created_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted:
            _LOGGER.info("Purged %d old session messages (older than %d days)", deleted, max_age_days)
        return deleted

    # ---- Lesson progress ----

    def get_current_lesson(self, user_id: str, mode: str, total_lessons: int) -> int:
        cursor = self._conn.execute(
            "SELECT lesson_number FROM current_lesson WHERE user_id = ? AND mode = ?",
            (user_id, mode),
        )
        row = cursor.fetchone()
        if row:
            return min(row[0], total_lessons)
        return 1

    def set_current_lesson(self, user_id: str, mode: str, lesson_number: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO current_lesson (user_id, mode, lesson_number) VALUES (?, ?, ?)",
            (user_id, mode, lesson_number),
        )
        self._conn.commit()

    def mark_step_complete(self, user_id: str, mode: str, lesson: int, step: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO lesson_progress (user_id, mode, lesson_number, step_index, completed, completed_at) VALUES (?, ?, ?, ?, 1, ?)",
            (user_id, mode, lesson, step, now),
        )
        self._conn.commit()

    def get_completed_steps(self, user_id: str, mode: str, lesson: int) -> list[int]:
        cursor = self._conn.execute(
            "SELECT step_index FROM lesson_progress WHERE user_id = ? AND mode = ? AND lesson_number = ? AND completed = 1 ORDER BY step_index",
            (user_id, mode, lesson),
        )
        return [row[0] for row in cursor]

    def advance_lesson(self, user_id: str, mode: str, total_lessons: int) -> int:
        current = self.get_current_lesson(user_id, mode, total_lessons)
        next_lesson = min(current + 1, total_lessons)
        self.set_current_lesson(user_id, mode, next_lesson)
        return next_lesson

    def reset_lesson(self, user_id: str, mode: str, lesson: int) -> None:
        self._conn.execute(
            "DELETE FROM lesson_progress WHERE user_id = ? AND mode = ? AND lesson_number = ?",
            (user_id, mode, lesson),
        )
        self._conn.commit()

    def reset_all_lessons(self, user_id: str, mode: str) -> None:
        self._conn.execute(
            "DELETE FROM lesson_progress WHERE user_id = ? AND mode = ?",
            (user_id, mode),
        )
        self._conn.execute(
            "DELETE FROM current_lesson WHERE user_id = ? AND mode = ?",
            (user_id, mode),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
