"""Reversible adaptive baselines, tool-policy proposals, and feedback storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import yaml

from tracer.fusion import DEFAULT_TOOL_RISKS


def classify_task(user_request: str) -> str:
    text = user_request.casefold()
    has_gmail = any(term in text for term in ("gmail", "email", "inbox"))
    has_youtube = any(term in text for term in ("youtube", "video", "channel"))
    if has_gmail and has_youtube:
        return "cross_source_research"
    if has_youtube:
        return "youtube_research"
    if has_gmail:
        return "gmail_research"
    return "general_research"


def propose_tool_policy(
    tools: list[Any],
    path: str | Path = "policy.yaml",
) -> dict[str, Any]:
    """Write a proposed policy once; never silently promote it to active."""
    policy_path = Path(path)
    existing: dict[str, Any] = {}
    if policy_path.is_file():
        loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    entries = []
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        source = (
            "gmail"
            if name in {
                "search_gmail",
                "read_gmail_message",
                "create_gmail_draft",
                "send_gmail_message",
            }
            else "youtube"
            if name in {"search_youtube", "get_youtube_channel_details"}
            else "unknown"
        )
        expected_effect = {
            "create_gmail_draft": "creates a draft in the connected Gmail account",
            "send_gmail_message": "sends one email after explicit UI approval",
        }.get(name, "read-only; no remote state mutation")
        entries.append(
            {
                "name": name,
                "source": source,
                "risk": DEFAULT_TOOL_RISKS.get(name, "HIGH"),
                "output_trust": "untrusted",
                "expected_effect": expected_effect,
            }
        )
    policy = {
        "status": "proposed",
        "version": int(existing.get("version", 0)) + 1,
        "tools": entries,
        "confirmation_required": True,
    }
    existing_names = [
        item.get("name")
        for item in existing.get("tools", [])
        if isinstance(item, dict)
    ]
    proposed_names = [item["name"] for item in entries]
    if existing_names == proposed_names:
        return existing
    if existing.get("status") == "active":
        policy["previous_active_tools"] = existing_names
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        yaml.safe_dump(policy, sort_keys=False),
        encoding="utf-8",
    )
    return policy


def confirm_tool_policy(path: str | Path = "policy.yaml") -> dict[str, Any]:
    """Explicit human action required to promote a proposed policy."""
    policy_path = Path(path)
    if not policy_path.is_file():
        raise FileNotFoundError(f"Proposed policy not found: {policy_path}")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    if policy.get("status") != "proposed":
        raise ValueError("Only a proposed policy can be confirmed.")
    policy["status"] = "active"
    policy["confirmed_at"] = datetime.now(timezone.utc).isoformat()
    policy_path.write_text(
        yaml.safe_dump(policy, sort_keys=False),
        encoding="utf-8",
    )
    return policy


def tools_from_active_policy(policy: dict[str, Any]) -> list[str] | None:
    if policy.get("status") != "active":
        return None
    names = [
        item["name"]
        for item in policy.get("tools", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    return names


class AdaptiveStore:
    """SQLite store that excludes raw email, queries, and tool-result contents."""

    def __init__(self, path: str | Path = ".agent-trust.sqlite") -> None:
        self.path = str(path)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS baselines (
                    task_type TEXT PRIMARY KEY,
                    run_count INTEGER NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    check_id TEXT NOT NULL,
                    predicted_status TEXT NOT NULL,
                    label TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS thresholds (
                    name TEXT PRIMARY KEY,
                    value REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS threshold_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    previous_value REAL NOT NULL,
                    next_value REAL NOT NULL,
                    reverted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS baseline_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    previous_run_count INTEGER,
                    previous_data_json TEXT,
                    reverted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def current_threshold(self, name: str, default: float) -> float:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM thresholds WHERE name = ?",
                (name,),
            ).fetchone()
        return float(row["value"]) if row else default

    def anomaly_score(
        self,
        task_type: str,
        observed_tool_names: list[str],
        *,
        current_query_length: int | None = None,
        final: bool = False,
    ) -> float:
        if not observed_tool_names and not final:
            return 0.0
        with self._connection() as connection:
            row = connection.execute(
                "SELECT data_json FROM baselines WHERE task_type = ?",
                (task_type,),
            ).fetchone()
        if row is None:
            return 0.0
        baseline = json.loads(row["data_json"])
        known_tools = set(baseline["tool_counts"])
        if observed_tool_names and observed_tool_names[-1] not in known_tools:
            return 1.0
        sequences: dict[tuple[str, ...], int] = {
            tuple(item["tools"]): int(item["count"])
            for item in baseline["sequences"]
        }
        if not sequences:
            return 0.0
        prefix = tuple(observed_tool_names)
        matching = sum(
            count
            for sequence, count in sequences.items()
            if (sequence == prefix if final else sequence[: len(prefix)] == prefix)
        )
        score = 0.0 if matching else 0.9
        if final and str(len(prefix)) not in baseline.get("step_counts", {}):
            score = max(score, 0.9)
        if current_query_length is not None:
            limits = (
                baseline.get("query_length_ranges", {}).get(observed_tool_names[-1])
                if observed_tool_names
                else None
            )
            if limits and not limits["min"] <= current_query_length <= limits["max"]:
                score = max(score, 0.7)
        return score

    def learn_confirmed_good_run(
        self,
        task_type: str,
        tool_names: list[str],
        *,
        query_lengths: dict[str, list[int]] | None = None,
    ) -> None:
        """Learn only aggregate tool order/counts from explicitly confirmed runs."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT run_count, data_json FROM baselines WHERE task_type = ?",
                (task_type,),
            ).fetchone()
            previous_run_count = int(row["run_count"]) if row else None
            previous_data_json = row["data_json"] if row else None
            baseline = (
                json.loads(row["data_json"])
                if row
                else {
                    "tool_counts": {},
                    "sequences": [],
                    "step_counts": {},
                    "query_length_ranges": {},
                }
            )
            baseline.setdefault("step_counts", {})
            baseline.setdefault("query_length_ranges", {})
            sequence_key = json.dumps(tool_names, separators=(",", ":"))
            sequence_counts = {
                json.dumps(item["tools"], separators=(",", ":")): int(item["count"])
                for item in baseline["sequences"]
            }
            sequence_counts[sequence_key] = sequence_counts.get(sequence_key, 0) + 1
            baseline["sequences"] = [
                {"tools": json.loads(key), "count": count}
                for key, count in sequence_counts.items()
            ]
            for name in set(tool_names):
                counts = baseline["tool_counts"]
                counts[name] = counts.get(name, 0) + 1
            step_counts = baseline["step_counts"]
            step_counts[str(len(tool_names))] = (
                step_counts.get(str(len(tool_names)), 0) + 1
            )
            for name, lengths in (query_lengths or {}).items():
                if not lengths:
                    continue
                ranges = baseline["query_length_ranges"]
                current = ranges.get(name)
                minimum, maximum = min(lengths), max(lengths)
                ranges[name] = {
                    "min": min(current["min"], minimum) if current else minimum,
                    "max": max(current["max"], maximum) if current else maximum,
                }
            connection.execute(
                """
                INSERT INTO baselines(task_type, run_count, data_json)
                VALUES (?, 1, ?)
                ON CONFLICT(task_type) DO UPDATE SET
                    run_count = run_count + 1,
                    data_json = excluded.data_json
                """,
                (task_type, json.dumps(baseline, separators=(",", ":"))),
            )
            connection.execute(
                """
                INSERT INTO baseline_history(
                    task_type, previous_run_count, previous_data_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    task_type,
                    previous_run_count,
                    previous_data_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def record_feedback(
        self,
        *,
        run_id: str,
        task_type: str,
        check_id: str,
        predicted_status: str,
        label: str,
    ) -> int:
        """Store categorical feedback and reversibly adjust anomaly tolerance."""
        if label not in {"correct", "incorrect"}:
            raise ValueError("Feedback label must be correct or incorrect.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback(
                    run_id, task_type, check_id, predicted_status, label, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, task_type, check_id, predicted_status, label, now),
            )
            feedback_id = int(cursor.lastrowid)
            threshold_name = (
                "T_anom"
                if check_id in {"trajectory", "scope"}
                else "T_high"
                if check_id in {"semantic_judge", "grounding", "contradiction"}
                else None
            )
            if threshold_name and label == "incorrect":
                default = 0.8 if threshold_name == "T_anom" else 0.85
                row = connection.execute(
                    "SELECT value FROM thresholds WHERE name = ?",
                    (threshold_name,),
                ).fetchone()
                old = float(row["value"]) if row else default
                delta = (
                    0.05
                    if predicted_status in {"warning", "REVIEW", "FAIL"}
                    else -0.05
                )
                new = min(0.99, max(0.1, old + delta))
                connection.execute(
                    """
                    INSERT INTO thresholds(name, value) VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET value = excluded.value
                    """,
                    (threshold_name, new),
                )
                connection.execute(
                    """
                    INSERT INTO threshold_history(
                        feedback_id, name, previous_value, next_value, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (feedback_id, threshold_name, old, new, now),
                )
        return feedback_id

    def undo_last_threshold_change(self) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, name, previous_value
                FROM threshold_history
                WHERE reverted = 0
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                """
                INSERT INTO thresholds(name, value) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET value = excluded.value
                """,
                (row["name"], row["previous_value"]),
            )
            connection.execute(
                "UPDATE threshold_history SET reverted = 1 WHERE id = ?",
                (row["id"],),
            )
        return True

    def undo_last_baseline_update(self) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, task_type, previous_run_count, previous_data_json
                FROM baseline_history WHERE reverted = 0
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return False
            if row["previous_data_json"] is None:
                connection.execute(
                    "DELETE FROM baselines WHERE task_type = ?",
                    (row["task_type"],),
                )
            else:
                connection.execute(
                    """
                    UPDATE baselines
                    SET run_count = ?, data_json = ?
                    WHERE task_type = ?
                    """,
                    (
                        row["previous_run_count"],
                        row["previous_data_json"],
                        row["task_type"],
                    ),
                )
            connection.execute(
                "UPDATE baseline_history SET reverted = 1 WHERE id = ?",
                (row["id"],),
            )
        return True

    def few_shot_examples(self, limit: int = 5) -> list[dict[str, str]]:
        """Return PII-free check labels, never source text or action arguments."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT check_id, predicted_status, label
                FROM feedback ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "check_id": row["check_id"],
                "predicted_status": row["predicted_status"],
                "human_label": row["label"],
            }
            for row in rows
        ]

    def undo_last_feedback(self) -> bool:
        with self._connection() as connection:
            feedback = connection.execute(
                "SELECT id FROM feedback ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if feedback is None:
                return False
            history = connection.execute(
                """
                SELECT id, name, previous_value
                FROM threshold_history
                WHERE feedback_id = ? AND reverted = 0
                """,
                (feedback["id"],),
            ).fetchone()
            if history:
                connection.execute(
                    """
                    INSERT INTO thresholds(name, value) VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET value = excluded.value
                    """,
                    (history["name"], history["previous_value"]),
                )
                connection.execute(
                    "UPDATE threshold_history SET reverted = 1 WHERE id = ?",
                    (history["id"],),
                )
            connection.execute(
                "DELETE FROM feedback WHERE id = ?",
                (feedback["id"],),
            )
        return True

    def drift_summary(self) -> dict[str, Any]:
        with self._connection() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback"
            ).fetchone()["count"]
            incorrect = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback WHERE label = 'incorrect'"
            ).fetchone()["count"]
            baseline_runs = connection.execute(
                "SELECT COALESCE(SUM(run_count), 0) AS count FROM baselines"
            ).fetchone()["count"]
            recent_feedback = connection.execute(
                """
                SELECT label FROM feedback
                ORDER BY id DESC LIMIT 20
                """
            ).fetchall()
        recent_labels = [row["label"] for row in recent_feedback]
        recent_window = recent_labels[:10]
        prior_window = recent_labels[10:20]
        recent_rate = (
            sum(label == "incorrect" for label in recent_window) / len(recent_window)
            if recent_window
            else None
        )
        prior_rate = (
            sum(label == "incorrect" for label in prior_window) / len(prior_window)
            if prior_window
            else None
        )
        return {
            "feedback_count": int(total),
            "incorrect_count": int(incorrect),
            "incorrect_rate": round(incorrect / total, 3) if total else None,
            "recent_incorrect_rate": (
                round(recent_rate, 3) if recent_rate is not None else None
            ),
            "previous_incorrect_rate": (
                round(prior_rate, 3) if prior_rate is not None else None
            ),
            "drift_delta": (
                round(recent_rate - prior_rate, 3)
                if recent_rate is not None and prior_rate is not None
                else None
            ),
            "confirmed_good_runs": int(baseline_runs),
        }
