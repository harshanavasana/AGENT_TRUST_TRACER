"""Durable, redacted run metadata for metrics links and live action timelines."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunStore:
    """Persist run metadata without raw email bodies, tool output, or audio."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    transcript TEXT NOT NULL DEFAULT '',
                    stt_latency_ms REAL,
                    task_type TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    spoken_reply TEXT NOT NULL DEFAULT '',
                    timeline_json TEXT NOT NULL DEFAULT '[]',
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    report_json TEXT NOT NULL DEFAULT '',
                    total_latency_ms REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def start(
        self,
        run_id: str,
        *,
        transcript: str = "",
        stt_latency_ms: float | None = None,
        task_type: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_runs(
                    run_id, transcript, stt_latency_ms, task_type, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?)
                """,
                (run_id, transcript[:1000], stt_latency_ms, task_type, now, now),
            )

    def event(self, run_id: str, name: str, status: str, **detail: Any) -> None:
        row = self._row(run_id)
        if row is None:
            self.start(run_id)
            row = self._row(run_id)
        timeline = json.loads(row["timeline_json"])
        timeline.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": name,
                "status": status,
                "detail": {
                    key: value
                    for key, value in detail.items()
                    if key not in {"output", "body", "raw_audio"}
                },
            }
        )
        self._update(run_id, timeline_json=json.dumps(timeline[-200:]))

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        spoken_reply: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        report: dict[str, Any] | None = None,
        total_latency_ms: float | None = None,
    ) -> None:
        self._update(
            run_id,
            status=status,
            spoken_reply=spoken_reply[:400],
            tool_calls_json=json.dumps(tool_calls or []),
            report_json=json.dumps(self._redact_report(report or {})),
            total_latency_ms=total_latency_ms,
        )

    @classmethod
    def _redact_report(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    "[redacted]"
                    if key in {"result", "output", "body", "snippet"}
                    else cls._redact_report(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._redact_report(item) for item in value]
        return value

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self._row(run_id)
        if row is None:
            return None
        result = dict(row)
        for key in ("timeline_json", "tool_calls_json", "report_json"):
            result[key[:-5] if key.endswith("_json") else key] = json.loads(
                result.pop(key) or ("[]" if key != "report_json" else "{}")
            )
        return result

    def _row(self, run_id: str) -> sqlite3.Row | None:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            return connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()

    def _update(self, run_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = datetime.now(timezone.utc).isoformat()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                f"UPDATE agent_runs SET {assignments} WHERE run_id = ?",
                (*values.values(), run_id),
            )
