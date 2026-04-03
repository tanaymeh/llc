from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
import uuid
from typing import Any

from llc.logging_utils import render_kv

_DEFAULT_LOCK_LEASE_S = 90
_DEFAULT_LOCK_RENEW_S = 30
_DEFAULT_LOCK_NEAR_EXPIRY_S = 20
_DEFAULT_SHARED_NOTES_MAX = 40
_DEFAULT_INBOX_READ_MAX = 8
_DEFAULT_LOCK_REVIEW_FORCE_INTERVAL_CYCLES = 4
_CONSECUTIVE_MESSAGE_LIMIT = 5
_CONSECUTIVE_MESSAGE_WINDOW_S = 120
_CONSECUTIVE_MESSAGE_COOLDOWN_S = 120
_COORDINATION_TOOL_NAMES = {
    "SendMessage",
    "ReadInbox",
    "ReadTeamStatus",
    "ReadSharedNotes",
    "AppendSharedNote",
    "RequestLock",
    "ReleaseLock",
    "ReviewHeldLocks",
    "RespondLockReview",
}
_LOGGER = logging.getLogger("llc.subagents.coordination")


def _agent_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(record.get("id", "")).strip(),
        "name": str(record.get("name", "")).strip(),
        "status": str(record.get("status", "")).strip(),
        "spawned_by": str(record.get("spawned_by", "")).strip(),
        "task": str(record.get("task", "")).strip(),
        "last_heartbeat_at": float(record.get("last_heartbeat_at", 0.0) or 0.0),
        "updated_at": float(record.get("updated_at", 0.0) or 0.0),
        "created_at": float(record.get("created_at", 0.0) or 0.0),
        "unread_inbox_count": int(record.get("unread_inbox_count", 0) or 0),
        "runtime_notice_count": int(record.get("runtime_notice_count", 0) or 0),
        "held_lock_count": int(record.get("held_lock_count", 0) or 0),
        "lock_review_required": bool(record.get("lock_review_required", False)),
    }


class AgentCoordinationLayer:
    def __init__(
        self,
        *,
        manager: Any,
        lock: Any,
        agents: Any,
        inbox: Any,
        shared_notes: Any,
        file_locks: Any,
        config: Any,
    ) -> None:
        self._manager = manager
        self._lock = lock
        self._agents = agents
        self._inbox = inbox
        self._shared_notes = shared_notes
        self._file_locks = file_locks
        self._config = config

    @classmethod
    def create(
        cls,
        mp_context: mp.context.BaseContext,
        *,
        config: dict[str, Any] | None = None,
    ) -> "AgentCoordinationLayer":
        manager = mp_context.Manager()
        config_payload = dict(config or {})
        return cls(
            manager=manager,
            lock=manager.RLock(),
            agents=manager.dict(),
            inbox=manager.list(),
            shared_notes=manager.list(),
            file_locks=manager.dict(),
            config=manager.dict(config_payload),
        )

    @classmethod
    def from_handles(cls, handles: dict[str, Any]) -> "AgentCoordinationLayer":
        return cls(
            manager=None,
            lock=handles["lock"],
            agents=handles["agents"],
            inbox=handles["inbox"],
            shared_notes=handles["shared_notes"],
            file_locks=handles["file_locks"],
            config=handles["config"],
        )

    def export_handles(self) -> dict[str, Any]:
        return {
            "lock": self._lock,
            "agents": self._agents,
            "inbox": self._inbox,
            "shared_notes": self._shared_notes,
            "file_locks": self._file_locks,
            "config": self._config,
        }

    def _debug_enabled(self) -> bool:
        raw = self._config.get("debug_logging", False)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _log_event(self, event: str, **fields: Any) -> None:
        if not self._debug_enabled():
            return
        kv = render_kv(fields)
        if kv:
            _LOGGER.info("[subagent-debug] event=%s %s", event, kv)
            return
        _LOGGER.info("[subagent-debug] event=%s", event)

    def shutdown(self) -> None:
        manager = self._manager
        if manager is None:
            return
        try:
            manager.shutdown()
        except Exception:
            pass
        self._manager = None

    def update_config(self, values: dict[str, Any]) -> None:
        with self._lock:
            for key, value in values.items():
                self._config[str(key)] = value

    def register_agent(
        self,
        *,
        agent_id: str,
        name: str,
        task: str,
        spawned_by: str,
        status: str,
    ) -> None:
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            record = self._agent_record_locked(agent_id)
            created_at = float(record.get("created_at", 0.0) or 0.0)
            if created_at <= 0:
                created_at = now
            record.update(
                {
                    "id": agent_id,
                    "name": name,
                    "task": task,
                    "spawned_by": spawned_by,
                    "status": status,
                    "created_at": created_at,
                    "updated_at": now,
                    "last_heartbeat_at": now,
                }
            )
            record.setdefault("cycle_count", 0)
            record.setdefault("last_team_status_cycle", 0)
            record.setdefault("last_shared_notes_cycle", 0)
            record.setdefault("runtime_notice_count", 0)
            record.setdefault("pending_lock_review", False)
            record.setdefault("pending_lock_review_cycle", 0)
            record.setdefault(
                "outgoing_message_streak",
                {
                    "recipient_id": "",
                    "count": 0,
                    "window_started_at": 0.0,
                    "cooldown_until": 0.0,
                },
            )
            self._agents[agent_id] = record
            self._refresh_agent_derived_state_locked(agent_id, now)
            self._log_event(
                "register_agent",
                agent_id=agent_id,
                name=name,
                spawned_by=spawned_by,
                task_preview=task[:100],
                status=status,
            )

    def update_agent(
        self,
        *,
        agent_id: str,
        status: str | None = None,
        task: str | None = None,
        heartbeat: bool = False,
    ) -> None:
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            if agent_id not in self._agents:
                return
            record = self._agent_record_locked(agent_id)
            if status is not None and status.strip():
                record["status"] = status.strip()
            if task is not None and task.strip():
                record["task"] = task.strip()
            if heartbeat:
                record["last_heartbeat_at"] = now
            record["updated_at"] = now
            self._agents[agent_id] = record
            self._refresh_agent_derived_state_locked(agent_id, now)

    def mark_agent_stopped(self, *, agent_id: str, status: str) -> None:
        now = time.time()
        with self._lock:
            self._release_all_locks_locked(agent_id, now, reason="agent_stopped")
            if agent_id in self._agents:
                record = self._agent_record_locked(agent_id)
                record["status"] = status
                record["updated_at"] = now
                record["last_heartbeat_at"] = now
                record["pending_lock_review"] = False
                self._agents[agent_id] = record
                self._refresh_agent_derived_state_locked(agent_id, now)
            self._log_event(
                "agent_stopped",
                agent_id=agent_id,
                status=status,
            )

    def send_message(
        self,
        *,
        sender_id: str,
        recipient_id: str,
        content: str,
    ) -> dict[str, Any]:
        clean_sender = sender_id.strip()
        clean_recipient = recipient_id.strip()
        clean_content = content.strip()
        if not clean_recipient:
            return {"ok": False, "error": "recipient_subagent_id is required."}
        if not clean_content:
            return {"ok": False, "error": "Message content cannot be empty."}
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            if clean_recipient not in self._agents:
                self._log_event(
                    "send_message_failed",
                    sender_id=clean_sender,
                    recipient_id=clean_recipient,
                    reason="unknown_recipient",
                )
                return {
                    "ok": False,
                    "error": f"Unknown recipient id: {clean_recipient}",
                }
            guard = self._prepare_message_send_locked(
                sender_id=clean_sender,
                recipient_id=clean_recipient,
                now=now,
            )
            if not bool(guard.get("ok", False)):
                self._log_event(
                    "send_message_rate_limited",
                    sender_id=clean_sender or "unknown",
                    recipient_id=clean_recipient,
                    cooldown_remaining_s=guard.get("cooldown_remaining_s", 0),
                )
                return guard
            payload = {
                "id": f"msg-{uuid.uuid4().hex[:10]}",
                "kind": "agent",
                "from": clean_sender or "unknown",
                "to": clean_recipient,
                "content": clean_content,
                "created_at": now,
                "read_at": 0.0,
            }
            self._inbox.append(payload)
            self._commit_message_send_locked(
                sender_id=clean_sender,
                recipient_id=clean_recipient,
                now=now,
            )
            self._refresh_agent_derived_state_locked(clean_recipient, now)
            self._log_event(
                "send_message",
                sender_id=clean_sender or "unknown",
                recipient_id=clean_recipient,
                message_id=payload["id"],
                content_preview=clean_content[:120],
            )
            return {"ok": True, "message": payload}

    def has_unread_inbox(self, agent_id: str) -> bool:
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            return self._unread_inbox_count_locked(agent_id, kind="agent") > 0

    def read_inbox(self, agent_id: str, *, limit: int | None = None) -> dict[str, Any]:
        now = time.time()
        max_items = int(limit or self._config_value("inbox_read_max", _DEFAULT_INBOX_READ_MAX))
        if max_items <= 0:
            max_items = _DEFAULT_INBOX_READ_MAX
        messages: list[dict[str, Any]] = []
        runtime_messages: list[dict[str, Any]] = []
        with self._lock:
            self._sweep_expired_locks_locked(now)
            inbox_size = len(self._inbox)
            for idx in range(inbox_size):
                raw = self._inbox[idx]
                entry = dict(raw) if isinstance(raw, dict) else {}
                if str(entry.get("to", "")).strip() != agent_id:
                    continue
                if float(entry.get("read_at", 0.0) or 0.0) > 0:
                    continue
                entry_kind = self._inbox_entry_kind(entry)
                if entry_kind == "runtime" and len(runtime_messages) >= max_items:
                    continue
                if entry_kind != "runtime" and len(messages) >= max_items:
                    continue
                entry["read_at"] = now
                self._inbox[idx] = entry
                payload = {
                    "id": str(entry.get("id", "")).strip(),
                    "kind": entry_kind,
                    "from": str(entry.get("from", "")).strip(),
                    "content": str(entry.get("content", "")).strip(),
                    "created_at": float(entry.get("created_at", 0.0) or 0.0),
                }
                if entry_kind == "runtime":
                    runtime_messages.append(payload)
                else:
                    messages.append(payload)
            if agent_id in self._agents:
                record = self._agent_record_locked(agent_id)
                record["updated_at"] = now
                record["last_inbox_read_at"] = now
                self._agents[agent_id] = record
                self._refresh_agent_derived_state_locked(agent_id, now)
            if messages or runtime_messages:
                self._log_event(
                    "read_inbox",
                    agent_id=agent_id,
                    message_count=len(messages),
                    runtime_notice_count=len(runtime_messages),
                )
        return {
            "ok": True,
            "messages": messages,
            "runtime_messages": runtime_messages,
        }

    def append_shared_note(self, *, agent_id: str, note: str) -> dict[str, Any]:
        clean_note = note.strip()
        if not clean_note:
            return {"ok": False, "error": "Shared note cannot be empty."}
        now = time.time()
        entry = {
            "id": f"note-{uuid.uuid4().hex[:10]}",
            "author": agent_id.strip() or "unknown",
            "content": clean_note,
            "created_at": now,
        }
        with self._lock:
            self._sweep_expired_locks_locked(now)
            self._shared_notes.append(entry)
            max_notes = int(
                self._config_value("shared_notes_max_entries", _DEFAULT_SHARED_NOTES_MAX)
            )
            while len(self._shared_notes) > max_notes:
                self._shared_notes.pop(0)
            if agent_id in self._agents:
                record = self._agent_record_locked(agent_id)
                record["updated_at"] = now
                self._agents[agent_id] = record
            self._log_event(
                "append_shared_note",
                agent_id=agent_id,
                note_id=entry["id"],
                content_preview=clean_note[:120],
            )
        return {"ok": True, "note": entry}

    def read_shared_notes(self, *, agent_id: str, limit: int | None = None) -> dict[str, Any]:
        now = time.time()
        max_items = int(limit or self._config_value("shared_notes_max_entries", _DEFAULT_SHARED_NOTES_MAX))
        if max_items <= 0:
            max_items = _DEFAULT_SHARED_NOTES_MAX
        with self._lock:
            self._sweep_expired_locks_locked(now)
            notes = [
                dict(item) for item in list(self._shared_notes)[-max_items:] if isinstance(item, dict)
            ]
            if agent_id in self._agents:
                record = self._agent_record_locked(agent_id)
                record["last_shared_notes_cycle"] = int(record.get("cycle_count", 0) or 0)
                record["updated_at"] = now
                self._agents[agent_id] = record
            if notes:
                self._log_event(
                    "read_shared_notes",
                    agent_id=agent_id,
                    note_count=len(notes),
                )
        return {"ok": True, "notes": notes}

    def read_team_status(self, *, agent_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            workers: list[dict[str, Any]] = []
            for raw in self._agents.values():
                if not isinstance(raw, dict):
                    continue
                workers.append(_agent_snapshot(raw))
            workers.sort(key=lambda item: (float(item.get("created_at", 0.0) or 0.0), item.get("id", "")))
            if agent_id in self._agents:
                record = self._agent_record_locked(agent_id)
                record["last_team_status_cycle"] = int(record.get("cycle_count", 0) or 0)
                record["updated_at"] = now
                self._agents[agent_id] = record
            self._log_event(
                "read_team_status",
                agent_id=agent_id,
                worker_count=len(workers),
            )
            return {"ok": True, "workers": workers}

    def request_lock(
        self,
        *,
        agent_id: str,
        file_path: str,
        lease_seconds: int | None = None,
    ) -> dict[str, Any]:
        clean_path = file_path.strip()
        if not clean_path:
            return {"ok": False, "error": "file_path is required."}
        now = time.time()
        lease_s = int(
            lease_seconds
            if isinstance(lease_seconds, int) and lease_seconds > 0
            else self._config_value("lock_default_lease_s", _DEFAULT_LOCK_LEASE_S)
        )
        with self._lock:
            self._sweep_expired_locks_locked(now)
            raw = self._file_locks.get(clean_path)
            current = dict(raw) if isinstance(raw, dict) else {}
            owner = str(current.get("owner", "")).strip()
            if owner and owner != agent_id:
                expires_at = float(current.get("expires_at", 0.0) or 0.0)
                remaining = max(expires_at - now, 0.0)
                self._log_event(
                    "request_lock_denied",
                    agent_id=agent_id,
                    file_path=clean_path,
                    owner=owner,
                    expires_in_s=round(remaining, 2),
                )
                return {
                    "ok": False,
                    "error": f"Lock denied. {clean_path} is currently held by {owner}.",
                    "owner": owner,
                    "expires_in_s": round(remaining, 2),
                }
            acquired_at = float(current.get("acquired_at", now) or now)
            lock_entry = {
                "path": clean_path,
                "owner": agent_id,
                "acquired_at": acquired_at,
                "updated_at": now,
                "lease_seconds": lease_s,
                "expires_at": now + lease_s,
            }
            self._file_locks[clean_path] = lock_entry
            self._refresh_agent_derived_state_locked(agent_id, now)
            self._log_event(
                "request_lock_granted",
                agent_id=agent_id,
                file_path=clean_path,
                lease_seconds=lease_s,
            )
            return {"ok": True, "lock": lock_entry}

    def release_lock(
        self,
        *,
        agent_id: str,
        file_path: str,
        reason: str = "",
    ) -> dict[str, Any]:
        clean_path = file_path.strip()
        if not clean_path:
            return {"ok": False, "error": "file_path is required."}
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            raw = self._file_locks.get(clean_path)
            lock_entry = dict(raw) if isinstance(raw, dict) else {}
            owner = str(lock_entry.get("owner", "")).strip()
            if not owner:
                self._log_event(
                    "release_lock_failed",
                    agent_id=agent_id,
                    file_path=clean_path,
                    reason="missing_lock",
                )
                return {"ok": False, "error": "No lock exists for this file."}
            if owner != agent_id:
                self._log_event(
                    "release_lock_failed",
                    agent_id=agent_id,
                    file_path=clean_path,
                    reason=f"owned_by_{owner}",
                )
                return {
                    "ok": False,
                    "error": f"Cannot release lock owned by {owner}.",
                }
            self._file_locks.pop(clean_path, None)
            self._refresh_agent_derived_state_locked(agent_id, now)
            self._log_event(
                "release_lock",
                agent_id=agent_id,
                file_path=clean_path,
                reason=reason.strip(),
            )
            return {
                "ok": True,
                "released": clean_path,
                "reason": reason.strip(),
            }

    def release_all_locks(self, agent_id: str, *, reason: str = "") -> dict[str, Any]:
        now = time.time()
        with self._lock:
            released = self._release_all_locks_locked(agent_id, now, reason=reason.strip())
            self._refresh_agent_derived_state_locked(agent_id, now)
            if released:
                self._log_event(
                    "release_all_locks",
                    agent_id=agent_id,
                    count=len(released),
                    reason=reason.strip(),
                )
            return {"ok": True, "released": released}

    def review_held_locks(self, agent_id: str) -> dict[str, Any]:
        now = time.time()
        near_expiry_s = int(self._config_value("lock_near_expiry_s", _DEFAULT_LOCK_NEAR_EXPIRY_S))
        with self._lock:
            self._sweep_expired_locks_locked(now)
            held = self._held_locks_locked(agent_id, now)
            if agent_id in self._agents:
                record = self._agent_record_locked(agent_id)
                record["pending_lock_review"] = bool(held)
                record["pending_lock_review_cycle"] = int(record.get("cycle_count", 0) or 0)
                record["updated_at"] = now
                self._agents[agent_id] = record
            review = [
                {
                    "file_path": item["path"],
                    "duration_held_s": round(item["duration_held_s"], 2),
                    "expires_in_s": round(item["expires_in_s"], 2),
                    "near_expiry": item["expires_in_s"] <= near_expiry_s,
                }
                for item in held
            ]
            self._log_event(
                "review_held_locks",
                agent_id=agent_id,
                count=len(review),
            )
            return {"ok": True, "locks": review}

    def respond_lock_review(
        self,
        *,
        agent_id: str,
        decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = time.time()
        renew_s = int(self._config_value("lock_renew_s", _DEFAULT_LOCK_RENEW_S))
        normalized: dict[str, str] = {}
        for item in decisions:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file_path", "")).strip()
            if not file_path:
                continue
            action = str(item.get("action", "")).strip().upper()
            if action not in {"KEEP", "RELEASE"}:
                continue
            normalized[file_path] = action
        with self._lock:
            self._sweep_expired_locks_locked(now)
            held = self._held_locks_locked(agent_id, now)
            held_paths = {item["path"] for item in held}
            missing = [path for path in sorted(held_paths) if path not in normalized]
            if missing:
                self._log_event(
                    "respond_lock_review_failed",
                    agent_id=agent_id,
                    missing_count=len(missing),
                )
                return {
                    "ok": False,
                    "error": "Missing lock decisions.",
                    "missing_paths": missing,
                }
            kept: list[str] = []
            released: list[str] = []
            for path, action in normalized.items():
                if path not in held_paths:
                    continue
                if action == "RELEASE":
                    self._file_locks.pop(path, None)
                    released.append(path)
                    continue
                raw = self._file_locks.get(path)
                lock_entry = dict(raw) if isinstance(raw, dict) else {}
                if str(lock_entry.get("owner", "")).strip() != agent_id:
                    continue
                lock_entry["updated_at"] = now
                lock_entry["expires_at"] = now + renew_s
                lock_entry["lease_seconds"] = renew_s
                self._file_locks[path] = lock_entry
                kept.append(path)
            if agent_id in self._agents:
                record = self._agent_record_locked(agent_id)
                record["pending_lock_review"] = False
                record["pending_lock_review_cycle"] = 0
                record["updated_at"] = now
                self._agents[agent_id] = record
                self._refresh_agent_derived_state_locked(agent_id, now)
            self._log_event(
                "respond_lock_review",
                agent_id=agent_id,
                kept_count=len(kept),
                released_count=len(released),
            )
            return {
                "ok": True,
                "kept": kept,
                "released": released,
            }

    def ensure_mutation_allowed(self, *, agent_id: str, file_path: str) -> dict[str, Any]:
        clean_path = file_path.strip()
        if not clean_path:
            return {"ok": False, "error": "file_path is required."}
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            raw = self._file_locks.get(clean_path)
            lock_entry = dict(raw) if isinstance(raw, dict) else {}
            owner = str(lock_entry.get("owner", "")).strip()
            if owner == agent_id:
                return {"ok": True}
            if not owner:
                self._log_event(
                    "mutation_blocked",
                    agent_id=agent_id,
                    file_path=clean_path,
                    reason="no_lock",
                )
                return {
                    "ok": False,
                    "error": f"Lock required before editing {clean_path}.",
                }
            self._log_event(
                "mutation_blocked",
                agent_id=agent_id,
                file_path=clean_path,
                reason=f"owned_by_{owner}",
            )
            return {
                "ok": False,
                "error": f"Lock required before editing {clean_path}. Current owner: {owner}.",
            }

    def next_forced_tool_call(self, agent_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            if agent_id not in self._agents:
                return None
            record = self._agent_record_locked(agent_id)
            cycle_count = int(record.get("cycle_count", 0) or 0) + 1
            record["cycle_count"] = cycle_count
            record["updated_at"] = now
            self._agents[agent_id] = record
            if self._unread_inbox_count_locked(agent_id, kind="agent") > 0:
                forced = {"tool_name": "ReadInbox", "args": {}}
                self._log_event(
                    "forced_tool_call",
                    agent_id=agent_id,
                    cycle_count=cycle_count,
                    tool_name=forced["tool_name"],
                )
                return forced
            if self._should_inject_lock_review_locked(record, now):
                forced = {"tool_name": "ReviewHeldLocks", "args": {}}
                self._log_event(
                    "forced_tool_call",
                    agent_id=agent_id,
                    cycle_count=cycle_count,
                    tool_name=forced["tool_name"],
                )
                return forced
            return None

    def record_tool_invocation(self, agent_id: str, tool_name: str) -> None:
        now = time.time()
        clean_tool = tool_name.strip()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            if agent_id not in self._agents:
                return
            record = self._agent_record_locked(agent_id)
            if clean_tool == "ReadTeamStatus":
                record["last_team_status_cycle"] = int(record.get("cycle_count", 0) or 0)
            elif clean_tool == "ReadSharedNotes":
                record["last_shared_notes_cycle"] = int(record.get("cycle_count", 0) or 0)
            elif clean_tool == "ReadInbox":
                record["last_inbox_read_at"] = now
            elif clean_tool == "ReviewHeldLocks":
                record["pending_lock_review"] = False
                record["pending_lock_review_cycle"] = 0
            elif clean_tool == "RespondLockReview":
                record["pending_lock_review"] = False
                record["pending_lock_review_cycle"] = 0
            elif clean_tool not in _COORDINATION_TOOL_NAMES:
                held_locks = self._held_locks_locked(agent_id, now)
                if held_locks:
                    if not bool(record.get("pending_lock_review", False)):
                        current_cycle = int(record.get("cycle_count", 0) or 0)
                        record["pending_lock_review"] = True
                        record["pending_lock_review_cycle"] = (
                            current_cycle + _DEFAULT_LOCK_REVIEW_FORCE_INTERVAL_CYCLES
                        )
                else:
                    record["pending_lock_review"] = False
                    record["pending_lock_review_cycle"] = 0
            record["updated_at"] = now
            self._agents[agent_id] = record
            self._refresh_agent_derived_state_locked(agent_id, now)
            if clean_tool in _COORDINATION_TOOL_NAMES:
                self._log_event(
                    "tool_invocation",
                    agent_id=agent_id,
                    tool_name=clean_tool,
                    cycle_count=int(record.get("cycle_count", 0) or 0),
                )

    def sweep_expired_locks(self) -> None:
        now = time.time()
        with self._lock:
            expired = self._sweep_expired_locks_locked(now)
            if expired:
                self._log_event(
                    "sweep_expired_locks",
                    expired_count=len(expired),
                )

    def coordination_snapshot_for_agent(self, agent_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._sweep_expired_locks_locked(now)
            if agent_id not in self._agents:
                return {
                    "unread_inbox_count": self._unread_inbox_count_locked(
                        agent_id,
                        kind="agent",
                    ),
                    "runtime_notice_count": self._unread_inbox_count_locked(
                        agent_id,
                        kind="runtime",
                    ),
                    "held_lock_count": len(self._held_locks_locked(agent_id, now)),
                    "lock_review_required": False,
                }
            record = self._agent_record_locked(agent_id)
            return {
                "unread_inbox_count": int(record.get("unread_inbox_count", 0) or 0),
                "runtime_notice_count": int(record.get("runtime_notice_count", 0) or 0),
                "held_lock_count": int(record.get("held_lock_count", 0) or 0),
                "lock_review_required": bool(record.get("lock_review_required", False)),
            }

    def _config_value(self, key: str, default: int) -> int:
        raw = self._config.get(key, default)
        try:
            value = int(raw)
        except Exception:
            value = default
        return value if value > 0 else default

    def _agent_record_locked(self, agent_id: str) -> dict[str, Any]:
        raw = self._agents.get(agent_id)
        return dict(raw) if isinstance(raw, dict) else {"id": agent_id}

    def _message_streak_state(self, record: dict[str, Any]) -> dict[str, Any]:
        raw = record.get("outgoing_message_streak")
        state = dict(raw) if isinstance(raw, dict) else {}
        state.setdefault("recipient_id", "")
        state.setdefault("count", 0)
        state.setdefault("window_started_at", 0.0)
        state.setdefault("cooldown_until", 0.0)
        return state

    def _persist_message_streak_state_locked(
        self,
        *,
        sender_id: str,
        record: dict[str, Any],
        state: dict[str, Any],
        now: float,
    ) -> None:
        record["outgoing_message_streak"] = {
            "recipient_id": str(state.get("recipient_id", "")).strip(),
            "count": max(int(state.get("count", 0) or 0), 0),
            "window_started_at": float(state.get("window_started_at", 0.0) or 0.0),
            "cooldown_until": float(state.get("cooldown_until", 0.0) or 0.0),
        }
        record["updated_at"] = now
        self._agents[sender_id] = record
        self._refresh_agent_derived_state_locked(sender_id, now)

    def _message_cooldown_payload(self, *, recipient_id: str, cooldown_remaining_s: int) -> dict[str, Any]:
        return {
            "ok": False,
            "error": (
                f"Message cooldown active for recipient {recipient_id}. "
                f"Try again in {cooldown_remaining_s}s."
            ),
            "message": (
                "[System Ping]: You have sent 5 consecutive messages to this teammate. "
                f"Cooldown active for {cooldown_remaining_s}s. "
                "Work on another task or wait for the cooldown to expire."
            ),
            "system_ping": True,
            "recipient_id": recipient_id,
            "cooldown_remaining_s": cooldown_remaining_s,
            "cooldown_total_s": _CONSECUTIVE_MESSAGE_COOLDOWN_S,
            "consecutive_limit": _CONSECUTIVE_MESSAGE_LIMIT,
            "window_s": _CONSECUTIVE_MESSAGE_WINDOW_S,
        }

    def _prepare_message_send_locked(
        self,
        *,
        sender_id: str,
        recipient_id: str,
        now: float,
    ) -> dict[str, Any]:
        if not sender_id or sender_id not in self._agents:
            return {"ok": True}
        record = self._agent_record_locked(sender_id)
        state = self._message_streak_state(record)
        current_recipient = str(state.get("recipient_id", "")).strip()
        current_count = max(int(state.get("count", 0) or 0), 0)
        window_started_at = float(state.get("window_started_at", 0.0) or 0.0)
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)

        if current_recipient != recipient_id:
            state["recipient_id"] = recipient_id
            state["count"] = 0
            state["window_started_at"] = now
            state["cooldown_until"] = 0.0
            self._persist_message_streak_state_locked(
                sender_id=sender_id,
                record=record,
                state=state,
                now=now,
            )
            return {"ok": True}

        if window_started_at <= 0.0 or (now - window_started_at) > _CONSECUTIVE_MESSAGE_WINDOW_S:
            state["count"] = 0
            state["window_started_at"] = now
            state["cooldown_until"] = 0.0
            self._persist_message_streak_state_locked(
                sender_id=sender_id,
                record=record,
                state=state,
                now=now,
            )
            return {"ok": True}

        if cooldown_until > now:
            remaining = max(int(math.ceil(cooldown_until - now)), 0)
            self._persist_message_streak_state_locked(
                sender_id=sender_id,
                record=record,
                state=state,
                now=now,
            )
            return self._message_cooldown_payload(
                recipient_id=recipient_id,
                cooldown_remaining_s=remaining,
            )

        if current_count >= _CONSECUTIVE_MESSAGE_LIMIT:
            state["cooldown_until"] = now + _CONSECUTIVE_MESSAGE_COOLDOWN_S
            remaining = _CONSECUTIVE_MESSAGE_COOLDOWN_S
            self._persist_message_streak_state_locked(
                sender_id=sender_id,
                record=record,
                state=state,
                now=now,
            )
            return self._message_cooldown_payload(
                recipient_id=recipient_id,
                cooldown_remaining_s=remaining,
            )

        return {"ok": True}

    def _commit_message_send_locked(
        self,
        *,
        sender_id: str,
        recipient_id: str,
        now: float,
    ) -> None:
        if not sender_id or sender_id not in self._agents:
            return
        record = self._agent_record_locked(sender_id)
        state = self._message_streak_state(record)
        current_recipient = str(state.get("recipient_id", "")).strip()
        window_started_at = float(state.get("window_started_at", 0.0) or 0.0)
        current_count = max(int(state.get("count", 0) or 0), 0)

        if current_recipient != recipient_id or window_started_at <= 0.0:
            state["recipient_id"] = recipient_id
            state["count"] = 1
            state["window_started_at"] = now
            state["cooldown_until"] = 0.0
        elif (now - window_started_at) > _CONSECUTIVE_MESSAGE_WINDOW_S:
            state["count"] = 1
            state["window_started_at"] = now
            state["cooldown_until"] = 0.0
        else:
            state["count"] = current_count + 1
            state["recipient_id"] = recipient_id
        self._persist_message_streak_state_locked(
            sender_id=sender_id,
            record=record,
            state=state,
            now=now,
        )

    def _inbox_entry_kind(self, entry: dict[str, Any]) -> str:
        raw_kind = str(entry.get("kind", "")).strip().lower()
        if raw_kind in {"agent", "runtime"}:
            return raw_kind
        sender = str(entry.get("from", "")).strip()
        if sender == "runtime":
            return "runtime"
        return "agent"

    def _unread_inbox_count_locked(self, agent_id: str, *, kind: str | None = None) -> int:
        count = 0
        for raw in self._inbox:
            entry = raw if isinstance(raw, dict) else {}
            if str(entry.get("to", "")).strip() != agent_id:
                continue
            if float(entry.get("read_at", 0.0) or 0.0) > 0:
                continue
            if kind is not None and self._inbox_entry_kind(entry) != kind:
                continue
            count += 1
        return count

    def _held_locks_locked(self, agent_id: str, now: float) -> list[dict[str, Any]]:
        held: list[dict[str, Any]] = []
        for raw in self._file_locks.values():
            lock_entry = raw if isinstance(raw, dict) else {}
            owner = str(lock_entry.get("owner", "")).strip()
            if owner != agent_id:
                continue
            path = str(lock_entry.get("path", "")).strip()
            acquired_at = float(lock_entry.get("acquired_at", now) or now)
            expires_at = float(lock_entry.get("expires_at", now) or now)
            held.append(
                {
                    "path": path,
                    "acquired_at": acquired_at,
                    "expires_at": expires_at,
                    "duration_held_s": max(now - acquired_at, 0.0),
                    "expires_in_s": max(expires_at - now, 0.0),
                }
            )
        held.sort(key=lambda item: item["path"])
        return held

    def _release_all_locks_locked(
        self,
        agent_id: str,
        now: float,
        *,
        reason: str,
    ) -> list[str]:
        released: list[str] = []
        for path in list(self._file_locks.keys()):
            raw = self._file_locks.get(path)
            lock_entry = raw if isinstance(raw, dict) else {}
            owner = str(lock_entry.get("owner", "")).strip()
            if owner != agent_id:
                continue
            self._file_locks.pop(path, None)
            released.append(path)
        if reason:
            for path in released:
                self._append_runtime_message_locked(
                    recipient_id=agent_id,
                    content=f"Lock released for {path} ({reason}).",
                    now=now,
                )
        return released

    def _refresh_agent_derived_state_locked(self, agent_id: str, now: float) -> None:
        if agent_id not in self._agents:
            return
        record = self._agent_record_locked(agent_id)
        record["unread_inbox_count"] = self._unread_inbox_count_locked(
            agent_id,
            kind="agent",
        )
        record["runtime_notice_count"] = self._unread_inbox_count_locked(
            agent_id,
            kind="runtime",
        )
        record["held_lock_count"] = len(self._held_locks_locked(agent_id, now))
        record["lock_review_required"] = bool(record.get("pending_lock_review", False))
        self._agents[agent_id] = record

    def _append_runtime_message_locked(
        self,
        *,
        recipient_id: str,
        content: str,
        now: float,
    ) -> None:
        payload = {
            "id": f"msg-{uuid.uuid4().hex[:10]}",
            "kind": "runtime",
            "from": "runtime",
            "to": recipient_id,
            "content": content,
            "created_at": now,
            "read_at": 0.0,
        }
        self._inbox.append(payload)
        self._refresh_agent_derived_state_locked(recipient_id, now)
        self._log_event(
            "runtime_message",
            recipient_id=recipient_id,
            message_id=payload["id"],
            content_preview=content[:120],
        )

    def _sweep_expired_locks_locked(self, now: float) -> list[tuple[str, str]]:
        expired: list[tuple[str, str]] = []
        for path in list(self._file_locks.keys()):
            raw = self._file_locks.get(path)
            lock_entry = raw if isinstance(raw, dict) else {}
            expires_at = float(lock_entry.get("expires_at", 0.0) or 0.0)
            if expires_at <= 0 or expires_at > now:
                continue
            owner = str(lock_entry.get("owner", "")).strip()
            self._file_locks.pop(path, None)
            if owner:
                expired.append((owner, str(path)))
        for owner, path in expired:
            self._append_runtime_message_locked(
                recipient_id=owner,
                content=f"Lock expired for {path}; lease released automatically.",
                now=now,
            )
        refreshed_ids = {owner for owner, _ in expired}
        for agent_id in refreshed_ids:
            self._refresh_agent_derived_state_locked(agent_id, now)
        return expired

    def _should_inject_lock_review_locked(self, record: dict[str, Any], now: float) -> bool:
        if not bool(record.get("pending_lock_review", False)):
            return False
        agent_id = str(record.get("id", "")).strip()
        if not agent_id:
            return False
        held = self._held_locks_locked(agent_id, now)
        if not held:
            record["pending_lock_review"] = False
            record["pending_lock_review_cycle"] = 0
            self._agents[agent_id] = record
            return False
        pending_cycle = int(record.get("pending_lock_review_cycle", 0) or 0)
        cycle_count = int(record.get("cycle_count", 0) or 0)
        return cycle_count >= pending_cycle
