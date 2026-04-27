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
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages_lookup
            ON session_messages(session_key, created_at)
        """)
        self._conn.commit()
        _LOGGER.info("Session store opened: %s", self._db_path)

    def add_message(self, session_key: str, role: str, text: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO session_messages (session_key, role, text, created_at) VALUES (?, ?, ?, ?)",
            (session_key, role, text, now),
        )
        self._conn.commit()

    def get_messages(self, session_key: str, limit: int = 200) -> list[dict[str, str]]:
        cursor = self._conn.execute(
            "SELECT role, text FROM session_messages WHERE session_key = ? ORDER BY created_at ASC LIMIT ?",
            (session_key, limit),
        )
        return [{"role": row[0], "text": row[1]} for row in cursor]

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

    def close(self) -> None:
        self._conn.close()
