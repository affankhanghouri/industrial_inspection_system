"""Thread-safe SQLite persistence for the local demonstration."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from industrial_safety.events.types import EventStatus, SafetyEvent


class SQLiteEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT,
                    track_id INTEGER NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_version TEXT NOT NULL,
                    person_model_version TEXT NOT NULL,
                    ppe_model_version TEXT,
                    confidence REAL,
                    supporting_observations INTEGER NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    clip_path TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    review_action TEXT,
                    review_reason TEXT
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_detected ON events(detected_at DESC)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_rule_camera ON events(rule_id, camera_id, detected_at DESC)"
            )

    def add(self, event: SafetyEvent) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                _event_values(event),
            )

    def get(self, event_id: str) -> SafetyEvent | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return None if row is None else _event_from_row(row)

    def list_events(
        self,
        *,
        limit: int = 100,
        status: EventStatus | None = None,
    ) -> tuple[SafetyEvent, ...]:
        limit = max(1, min(limit, 1000))
        query = "SELECT * FROM events"
        parameters: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            parameters.append(status.value)
        query += " ORDER BY detected_at DESC LIMIT ?"
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def latest_for_rule(self, camera_id: str, rule_id: str) -> SafetyEvent | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM events
                WHERE camera_id = ? AND rule_id = ?
                ORDER BY detected_at DESC LIMIT 1
                """,
                (camera_id, rule_id),
            ).fetchone()
        return None if row is None else _event_from_row(row)

    def update_status(
        self,
        event_id: str,
        status: EventStatus,
        *,
        reason: str | None = None,
    ) -> SafetyEvent | None:
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE events
                SET status = ?, reviewed_at = ?, review_action = ?, review_reason = ?
                WHERE event_id = ?
                """,
                (status.value, reviewed_at, status.value, reason, event_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get(event_id)

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT event_type, status, COUNT(*) count FROM events GROUP BY event_type, status"
            ).fetchall()
        result = {"total": 0, "open": 0}
        for row in rows:
            count = int(row["count"])
            result["total"] += count
            if row["status"] == EventStatus.OPEN.value:
                result["open"] += count
            result[row["event_type"]] = result.get(row["event_type"], 0) + count
        return result

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _event_values(event: SafetyEvent) -> tuple[object, ...]:
    return (
        event.event_id,
        event.event_type,
        event.status.value,
        event.severity,
        event.detected_at.isoformat(),
        event.camera_id,
        event.zone_id,
        event.track_id,
        event.rule_id,
        event.rule_version,
        event.person_model_version,
        event.ppe_model_version,
        event.confidence,
        event.supporting_observations,
        event.snapshot_path,
        event.clip_path,
        event.created_at.isoformat(),
        None if event.reviewed_at is None else event.reviewed_at.isoformat(),
        event.review_action,
        event.review_reason,
    )


def _event_from_row(row: sqlite3.Row) -> SafetyEvent:
    return SafetyEvent(
        event_id=row["event_id"],
        event_type=row["event_type"],
        status=EventStatus(row["status"]),
        severity=row["severity"],
        detected_at=datetime.fromisoformat(row["detected_at"]),
        camera_id=row["camera_id"],
        zone_id=row["zone_id"],
        track_id=int(row["track_id"]),
        rule_id=row["rule_id"],
        rule_version=row["rule_version"],
        person_model_version=row["person_model_version"],
        ppe_model_version=row["ppe_model_version"],
        confidence=row["confidence"],
        supporting_observations=int(row["supporting_observations"]),
        snapshot_path=row["snapshot_path"],
        clip_path=row["clip_path"],
        created_at=datetime.fromisoformat(row["created_at"]),
        reviewed_at=(
            None
            if row["reviewed_at"] is None
            else datetime.fromisoformat(row["reviewed_at"])
        ),
        review_action=row["review_action"],
        review_reason=row["review_reason"],
    )
