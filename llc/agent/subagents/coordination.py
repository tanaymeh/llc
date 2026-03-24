from __future__ import annotations

import json
import time
import uuid
from typing import Any

_WAIT_POLL_INTERVAL_S = 0.2
_MAX_BOARD_ITEMS = 500
_MAX_MESSAGE_ITEMS = 1200
_MAX_PLAN_STEPS = 24
_PEER_WAIT_SAFETY_MARGIN_MS = 5000


def _now() -> float:
    return time.time()


def _normalize_steps(raw_steps: list[str] | str) -> list[str]:
    if isinstance(raw_steps, list):
        values = [str(item).strip() for item in raw_steps if str(item).strip()]
        return values[:_MAX_PLAN_STEPS]
    value = str(raw_steps).strip()
    if not value:
        return []
    parsed: Any = None
    try:
        parsed = json.loads(value)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        values = [str(item).strip() for item in parsed if str(item).strip()]
        return values[:_MAX_PLAN_STEPS]
    parts = [part.strip() for part in value.splitlines() if part.strip()]
    normalized: list[str] = []
    for part in parts:
        if part[0].isdigit() and "." in part:
            _, _, tail = part.partition(".")
            item = tail.strip()
        else:
            item = part.strip("- ").strip()
        if item:
            normalized.append(item)
    return normalized[:_MAX_PLAN_STEPS]


class SubAgentCoordinationStore:
    def __init__(
        self,
        *,
        lock: Any,
        workers: Any,
        messages: Any,
        board: Any,
        claims: Any,
    ) -> None:
        self._lock = lock
        self._workers = workers
        self._messages = messages
        self._board = board
        self._claims = claims

    @property
    def lock(self) -> Any:
        return self._lock

    @property
    def workers(self) -> Any:
        return self._workers

    @property
    def messages(self) -> Any:
        return self._messages

    @property
    def board(self) -> Any:
        return self._board

    @property
    def claims(self) -> Any:
        return self._claims

    def register_worker(self, worker_id: str, *, name: str, goal: str) -> None:
        clean_id = worker_id.strip()
        if not clean_id:
            return
        with self._lock:
            now = _now()
            info = dict(self._workers.get(clean_id) or {})
            info["id"] = clean_id
            info["name"] = name.strip() or info.get("name", "worker")
            info["goal"] = goal.strip() or info.get("goal", "")
            info["active"] = True
            info["updated_at"] = now
            info.setdefault("created_at", now)
            info.setdefault("plan_submitted", False)
            info.setdefault("plan_step_count", 0)
            info.setdefault("plan_steps", [])
            info.setdefault("active_step_index", 0)
            info.setdefault("inbox_dirty", False)
            info.setdefault("unread_count", 0)
            info.setdefault("waiting_on", "")
            info.setdefault("peer_wait_start", 0.0)
            info.setdefault("last_peer_contact_at", 0.0)
            info.setdefault("last_inbox_checked_at", 0.0)
            self._workers[clean_id] = info

    def mark_worker_inactive(self, worker_id: str) -> None:
        clean_id = worker_id.strip()
        if not clean_id:
            return
        with self._lock:
            info = dict(self._workers.get(clean_id) or {})
            if not info:
                return
            info["active"] = False
            info["waiting_on"] = ""
            info["peer_wait_start"] = 0.0
            info["updated_at"] = _now()
            self._workers[clean_id] = info
            for scope in list(self._claims.keys()):
                claim = dict(self._claims.get(scope) or {})
                if claim.get("owner") == clean_id:
                    self._claims.pop(scope, None)

    def submit_plan(self, worker_id: str, *, steps: list[str] | str) -> dict[str, Any]:
        clean_id = worker_id.strip()
        if not clean_id:
            return {"ok": False, "error": "Worker id is required."}
        normalized_steps = _normalize_steps(steps)
        if len(normalized_steps) < 2:
            return {
                "ok": False,
                "error": "A plan must contain at least 2 non-empty steps.",
                "step_count": len(normalized_steps),
            }
        with self._lock:
            info = dict(self._workers.get(clean_id) or {})
            if not info:
                return {"ok": False, "error": f"Unknown worker: {clean_id}"}
            info["plan_submitted"] = True
            info["plan_step_count"] = len(normalized_steps)
            info["plan_steps"] = normalized_steps
            info["active_step_index"] = 1
            info["updated_at"] = _now()
            self._workers[clean_id] = info
            return {
                "ok": True,
                "step_count": len(normalized_steps),
                "steps": list(normalized_steps),
            }

    def set_waiting_on(self, worker_id: str, waiting_on: str = "") -> None:
        clean_id = worker_id.strip()
        if not clean_id:
            return
        with self._lock:
            info = dict(self._workers.get(clean_id) or {})
            if not info:
                return
            new_val = waiting_on.strip()
            old_val = str(info.get("waiting_on", "") or "").strip()
            info["waiting_on"] = new_val
            now = _now()
            if new_val and not old_val:
                info["peer_wait_start"] = now
            elif not new_val:
                info["peer_wait_start"] = 0.0
            info["updated_at"] = now
            self._workers[clean_id] = info

    def send_message(
        self,
        from_worker: str,
        to_worker: str,
        *,
        body: str,
        kind: str = "question",
        correlation_id: str = "",
        in_reply_to: str = "",
        ttl_s: int = 900,
        task_ref: str = "",
    ) -> dict[str, Any]:
        sender = from_worker.strip()
        recipient = to_worker.strip()
        content = body.strip()
        if not sender:
            return {"ok": False, "error": "from_worker is required."}
        if not recipient:
            return {"ok": False, "error": "to_worker is required."}
        if not content:
            return {"ok": False, "error": "message body cannot be empty."}
        with self._lock:
            self._prune_locked()
            sender_info = dict(self._workers.get(sender) or {})
            recipient_info = dict(self._workers.get(recipient) or {})
            if not sender_info:
                return {"ok": False, "error": f"Unknown sender worker: {sender}"}
            if not recipient_info:
                return {"ok": False, "error": f"Unknown target worker: {recipient}"}
            now = _now()
            ttl = max(int(ttl_s or 0), 30)
            message_id = f"msg-{uuid.uuid4().hex[:12]}"
            payload = {
                "id": message_id,
                "from_worker": sender,
                "to_worker": recipient,
                "kind": (kind.strip() or "question")[:40],
                "body": content,
                "task_ref": task_ref.strip(),
                "correlation_id": correlation_id.strip(),
                "in_reply_to": in_reply_to.strip(),
                "created_at": now,
                "ttl_s": ttl,
                "expires_at": now + float(ttl),
                "read": False,
                "read_at": 0.0,
            }
            self._messages.append(payload)
            self._trim_messages_locked()
            recipient_info["unread_count"] = max(
                int(recipient_info.get("unread_count", 0) or 0) + 1,
                0,
            )
            recipient_info["inbox_dirty"] = True
            recipient_info["updated_at"] = now
            sender_info["last_peer_contact_at"] = now
            recipient_info["last_peer_contact_at"] = now
            self._workers[sender] = sender_info
            self._workers[recipient] = recipient_info
            return {
                "ok": True,
                "id": message_id,
                "to_worker": recipient,
                "correlation_id": payload["correlation_id"],
                "expires_at": payload["expires_at"],
            }

    def read_inbox(
        self,
        worker_id: str,
        *,
        limit: int = 10,
        mark_read: bool = True,
        only_unread: bool = False,
    ) -> dict[str, Any]:
        clean_id = worker_id.strip()
        if not clean_id:
            return {"ok": False, "error": "worker_id is required.", "messages": []}
        with self._lock:
            self._prune_locked()
            if clean_id not in self._workers:
                return {"ok": False, "error": f"Unknown worker: {clean_id}", "messages": []}
            now = _now()
            bounded_limit = max(int(limit or 0), 1)
            selected: list[tuple[int, dict[str, Any]]] = []
            for idx, message in enumerate(list(self._messages)):
                if not isinstance(message, dict):
                    continue
                if message.get("to_worker") != clean_id:
                    continue
                is_unread = not bool(message.get("read", False))
                if only_unread and not is_unread:
                    continue
                selected.append((idx, message))
            selected.sort(key=lambda item: float(item[1].get("created_at", 0.0) or 0.0))
            selected = selected[:bounded_limit]
            payloads = [dict(item[1]) for item in selected]

            if mark_read and selected:
                selected_ids = {str(item[1].get("id", "")) for item in selected}
                for idx, message in enumerate(list(self._messages)):
                    if not isinstance(message, dict):
                        continue
                    message_id = str(message.get("id", ""))
                    if not message_id or message_id not in selected_ids:
                        continue
                    if message.get("read", False):
                        continue
                    updated = dict(message)
                    updated["read"] = True
                    updated["read_at"] = now
                    self._messages[idx] = updated

            unread_count = self._count_unread_locked(clean_id)
            info = dict(self._workers.get(clean_id) or {})
            info["unread_count"] = unread_count
            info["inbox_dirty"] = unread_count > 0
            info["last_inbox_checked_at"] = now
            info["updated_at"] = now
            if payloads:
                info["waiting_on"] = ""
                info["peer_wait_start"] = 0.0
            self._workers[clean_id] = info
            return {
                "ok": True,
                "messages": payloads,
                "unread_count": unread_count,
                "inbox_dirty": unread_count > 0,
            }

    def has_unread_messages(self, worker_id: str) -> dict[str, Any]:
        clean_id = worker_id.strip()
        if not clean_id:
            return {"ok": False, "error": "worker_id is required."}
        with self._lock:
            self._prune_locked()
            if clean_id not in self._workers:
                return {"ok": False, "error": f"Unknown worker: {clean_id}"}
            unread = self._count_unread_locked(clean_id)
            info = dict(self._workers.get(clean_id) or {})
            info["unread_count"] = unread
            info["inbox_dirty"] = unread > 0
            info["updated_at"] = _now()
            self._workers[clean_id] = info
            return {"ok": True, "has_unread": unread > 0, "unread_count": unread}

    def wait_for_message(self, worker_id: str, *, timeout_ms: int) -> dict[str, Any]:
        clean_id = worker_id.strip()
        if not clean_id:
            return {"ok": False, "error": "worker_id is required."}
        bounded_timeout_ms = max(int(timeout_ms or 0), 0)
        deadline = (
            time.monotonic() + (bounded_timeout_ms / 1000.0)
            if bounded_timeout_ms > 0
            else None
        )
        self.set_waiting_on(clean_id, "peer_message")
        while True:
            unread = self.has_unread_messages(clean_id)
            if unread.get("ok") and unread.get("has_unread"):
                self.set_waiting_on(clean_id, "")
                return {
                    "ok": True,
                    "has_message": True,
                    "unread_count": int(unread.get("unread_count", 0) or 0),
                }
            if deadline is not None and time.monotonic() >= deadline:
                self.set_waiting_on(clean_id, "")
                return {
                    "ok": True,
                    "has_message": False,
                    "unread_count": int(unread.get("unread_count", 0) or 0),
                }
            time.sleep(_WAIT_POLL_INTERVAL_S)

    def post_note(self, worker_id: str, *, channel: str, content: str) -> dict[str, Any]:
        clean_id = worker_id.strip()
        clean_channel = channel.strip() or "facts"
        clean_content = content.strip()
        if not clean_id:
            return {"ok": False, "error": "worker_id is required."}
        if not clean_content:
            return {"ok": False, "error": "content cannot be empty."}
        with self._lock:
            self._prune_locked()
            if clean_id not in self._workers:
                return {"ok": False, "error": f"Unknown worker: {clean_id}"}
            now = _now()
            entry = {
                "id": f"board-{uuid.uuid4().hex[:12]}",
                "worker_id": clean_id,
                "channel": clean_channel[:60],
                "content": clean_content,
                "created_at": now,
            }
            self._board.append(entry)
            self._trim_board_locked()
            info = dict(self._workers.get(clean_id) or {})
            info["updated_at"] = now
            self._workers[clean_id] = info
            return {"ok": True, "entry_id": entry["id"], "channel": entry["channel"]}

    def read_notes(self, *, channel: str = "", limit: int = 20) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            bounded_limit = max(int(limit or 0), 1)
            clean_channel = channel.strip()
            rows: list[dict[str, Any]] = []
            for item in list(self._board):
                if not isinstance(item, dict):
                    continue
                if clean_channel and str(item.get("channel", "")) != clean_channel:
                    continue
                rows.append(dict(item))
            rows.sort(key=lambda row: float(row.get("created_at", 0.0) or 0.0), reverse=True)
            return {"ok": True, "notes": rows[:bounded_limit]}

    def claim_scope(
        self,
        worker_id: str,
        *,
        scope: str,
        ttl_s: int,
        reason: str = "",
    ) -> dict[str, Any]:
        clean_id = worker_id.strip()
        clean_scope = scope.strip()
        if not clean_id:
            return {"ok": False, "error": "worker_id is required."}
        if not clean_scope:
            return {"ok": False, "error": "scope cannot be empty."}
        with self._lock:
            self._prune_locked()
            if clean_id not in self._workers:
                return {"ok": False, "error": f"Unknown worker: {clean_id}"}
            claim = dict(self._claims.get(clean_scope) or {})
            now = _now()
            if claim and claim.get("owner") != clean_id:
                return {
                    "ok": False,
                    "error": "Scope already claimed by another worker.",
                    "scope": clean_scope,
                    "owner": str(claim.get("owner", "")),
                    "expires_at": float(claim.get("expires_at", 0.0) or 0.0),
                }
            bounded_ttl = max(int(ttl_s or 0), 30)
            payload = {
                "scope": clean_scope,
                "owner": clean_id,
                "reason": reason.strip(),
                "acquired_at": now,
                "expires_at": now + float(bounded_ttl),
            }
            self._claims[clean_scope] = payload
            info = dict(self._workers.get(clean_id) or {})
            info["updated_at"] = now
            self._workers[clean_id] = info
            return {"ok": True, "scope": clean_scope, "expires_at": payload["expires_at"]}

    def release_scope(self, worker_id: str, *, scope: str) -> dict[str, Any]:
        clean_id = worker_id.strip()
        clean_scope = scope.strip()
        if not clean_id:
            return {"ok": False, "error": "worker_id is required."}
        if not clean_scope:
            return {"ok": False, "error": "scope cannot be empty."}
        with self._lock:
            self._prune_locked()
            claim = dict(self._claims.get(clean_scope) or {})
            if not claim:
                return {"ok": True, "scope": clean_scope, "released": False}
            owner = str(claim.get("owner", "")).strip()
            if owner != clean_id:
                return {
                    "ok": False,
                    "error": "Cannot release a scope claimed by another worker.",
                    "scope": clean_scope,
                    "owner": owner,
                }
            self._claims.pop(clean_scope, None)
            return {"ok": True, "scope": clean_scope, "released": True}

    def list_claims(self, *, worker_id: str = "") -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            target = worker_id.strip()
            rows: list[dict[str, Any]] = []
            for scope, raw_claim in list(self._claims.items()):
                claim = dict(raw_claim or {})
                claim.setdefault("scope", scope)
                if target and str(claim.get("owner", "")) != target:
                    continue
                rows.append(claim)
            rows.sort(key=lambda row: str(row.get("scope", "")))
            return {"ok": True, "claims": rows}

    def worker_state(self, worker_id: str) -> dict[str, Any]:
        clean_id = worker_id.strip()
        if not clean_id:
            return {}
        with self._lock:
            self._prune_locked()
            info = dict(self._workers.get(clean_id) or {})
            if not info:
                return {}
            unread = self._count_unread_locked(clean_id)
            claims = [
                dict(raw_claim or {})
                for raw_claim in list(self._claims.values())
                if isinstance(raw_claim, dict)
                and str(raw_claim.get("owner", "")) == clean_id
            ]
            directory = []
            for other_id, raw in list(self._workers.items()):
                if not isinstance(raw, dict):
                    continue
                directory.append(
                    {
                        "id": str(other_id),
                        "name": str(raw.get("name", "")),
                        "goal": str(raw.get("goal", "")),
                        "active": bool(raw.get("active", False)),
                        "waiting_on": str(raw.get("waiting_on", "")),
                        "plan_submitted": bool(raw.get("plan_submitted", False)),
                    }
                )
            directory.sort(key=lambda row: row["id"])
            info["unread_count"] = unread
            info["inbox_dirty"] = unread > 0
            info["updated_at"] = _now()
            self._workers[clean_id] = info
            return {
                "worker_id": clean_id,
                "active": bool(info.get("active", False)),
                "unread_count": unread,
                "inbox_dirty": bool(info.get("inbox_dirty", False)),
                "waiting_on": str(info.get("waiting_on", "")),
                "peer_wait_start": float(info.get("peer_wait_start", 0.0) or 0.0),
                "plan_submitted": bool(info.get("plan_submitted", False)),
                "plan_step_count": int(info.get("plan_step_count", 0) or 0),
                "plan_steps": list(info.get("plan_steps", []) or []),
                "active_step_index": int(info.get("active_step_index", 0) or 0),
                "last_peer_contact_at": float(info.get("last_peer_contact_at", 0.0) or 0.0),
                "last_inbox_checked_at": float(
                    info.get("last_inbox_checked_at", 0.0) or 0.0
                ),
                "claims": claims,
                "team_directory": directory,
            }

    def _prune_locked(self) -> None:
        self._prune_messages_locked()
        self._prune_claims_locked()

    def _prune_messages_locked(self) -> None:
        now = _now()
        kept: list[dict[str, Any]] = []
        changed = False
        for item in list(self._messages):
            if not isinstance(item, dict):
                changed = True
                continue
            expires_at = float(item.get("expires_at", 0.0) or 0.0)
            if expires_at > 0 and expires_at <= now:
                changed = True
                continue
            kept.append(item)
        if changed:
            self._messages[:] = kept
        self._trim_messages_locked()

    def _trim_messages_locked(self) -> None:
        size = len(self._messages)
        if size <= _MAX_MESSAGE_ITEMS:
            return
        drop = size - _MAX_MESSAGE_ITEMS
        if drop > 0:
            self._messages[:] = list(self._messages)[drop:]

    def _trim_board_locked(self) -> None:
        size = len(self._board)
        if size <= _MAX_BOARD_ITEMS:
            return
        drop = size - _MAX_BOARD_ITEMS
        if drop > 0:
            self._board[:] = list(self._board)[drop:]

    def _prune_claims_locked(self) -> None:
        now = _now()
        expired = []
        for scope, raw_claim in list(self._claims.items()):
            claim = dict(raw_claim or {})
            expires_at = float(claim.get("expires_at", 0.0) or 0.0)
            if expires_at > 0 and expires_at <= now:
                expired.append(scope)
        for scope in expired:
            self._claims.pop(scope, None)

    def _count_unread_locked(self, worker_id: str) -> int:
        total = 0
        for message in list(self._messages):
            if not isinstance(message, dict):
                continue
            if message.get("to_worker") != worker_id:
                continue
            if bool(message.get("read", False)):
                continue
            total += 1
        return total


class SubAgentCoordinationHub:
    def __init__(self, mp_context: Any) -> None:
        self._manager = mp_context.Manager()
        lock = self._manager.RLock()
        workers = self._manager.dict()
        messages = self._manager.list()
        board = self._manager.list()
        claims = self._manager.dict()
        self._store = SubAgentCoordinationStore(
            lock=lock,
            workers=workers,
            messages=messages,
            board=board,
            claims=claims,
        )

    def shutdown(self) -> None:
        try:
            self._manager.shutdown()
        except Exception:
            pass

    def register_worker(self, worker_id: str, *, name: str, goal: str) -> None:
        self._store.register_worker(worker_id, name=name, goal=goal)

    def mark_worker_inactive(self, worker_id: str) -> None:
        self._store.mark_worker_inactive(worker_id)

    def worker_state(self, worker_id: str) -> dict[str, Any]:
        return self._store.worker_state(worker_id)

    def worker_payload(
        self,
        worker_id: str,
        *,
        default_wait_timeout_ms: int,
        default_claim_ttl_s: int,
        stall_timeout_s: float,
    ) -> dict[str, Any]:
        return {
            "worker_id": worker_id,
            "default_wait_timeout_ms": max(int(default_wait_timeout_ms or 0), 0),
            "default_claim_ttl_s": max(int(default_claim_ttl_s or 0), 30),
            "stall_timeout_ms": max(int(float(stall_timeout_s or 0) * 1000), 0),
            "lock": self._store.lock,
            "workers": self._store.workers,
            "messages": self._store.messages,
            "board": self._store.board,
            "claims": self._store.claims,
        }


class SubAgentCoordinationClient:
    def __init__(
        self,
        *,
        worker_id: str,
        store: SubAgentCoordinationStore,
        default_wait_timeout_ms: int,
        default_claim_ttl_s: int,
        stall_timeout_ms: int,
    ) -> None:
        self.worker_id = worker_id
        self._store = store
        self._default_wait_timeout_ms = max(int(default_wait_timeout_ms or 0), 0)
        self._default_claim_ttl_s = max(int(default_claim_ttl_s or 0), 30)
        self._stall_timeout_ms = max(int(stall_timeout_ms or 0), 0)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SubAgentCoordinationClient | None:
        worker_id = str(payload.get("worker_id", "")).strip()
        if not worker_id:
            return None
        lock = payload.get("lock")
        workers = payload.get("workers")
        messages = payload.get("messages")
        board = payload.get("board")
        claims = payload.get("claims")
        if not all(item is not None for item in (lock, workers, messages, board, claims)):
            return None
        store = SubAgentCoordinationStore(
            lock=lock,
            workers=workers,
            messages=messages,
            board=board,
            claims=claims,
        )
        return cls(
            worker_id=worker_id,
            store=store,
            default_wait_timeout_ms=int(payload.get("default_wait_timeout_ms", 0) or 0),
            default_claim_ttl_s=int(payload.get("default_claim_ttl_s", 300) or 300),
            stall_timeout_ms=int(payload.get("stall_timeout_ms", 0) or 0),
        )

    def submit_plan(self, steps: list[str] | str) -> dict[str, Any]:
        return self._store.submit_plan(self.worker_id, steps=steps)

    def send_message(
        self,
        to_worker: str,
        *,
        body: str,
        kind: str = "question",
        correlation_id: str = "",
        in_reply_to: str = "",
        ttl_s: int = 900,
        task_ref: str = "",
    ) -> dict[str, Any]:
        return self._store.send_message(
            self.worker_id,
            to_worker,
            body=body,
            kind=kind,
            correlation_id=correlation_id,
            in_reply_to=in_reply_to,
            ttl_s=ttl_s,
            task_ref=task_ref,
        )

    def read_inbox(
        self,
        *,
        limit: int = 10,
        mark_read: bool = True,
        only_unread: bool = False,
    ) -> dict[str, Any]:
        return self._store.read_inbox(
            self.worker_id,
            limit=limit,
            mark_read=mark_read,
            only_unread=only_unread,
        )

    def has_unread_messages(self) -> dict[str, Any]:
        return self._store.has_unread_messages(self.worker_id)

    def wait_for_message(self, *, timeout_ms: int | None = None) -> dict[str, Any]:
        timeout = self._default_wait_timeout_ms
        if timeout_ms is not None:
            timeout = max(int(timeout_ms or 0), 0)
        effective_timeout = timeout
        if self._stall_timeout_ms > 0:
            bounded_max = max(self._stall_timeout_ms - _PEER_WAIT_SAFETY_MARGIN_MS, 1000)
            if effective_timeout <= 0:
                effective_timeout = bounded_max
            else:
                effective_timeout = min(effective_timeout, bounded_max)
        return self._store.wait_for_message(self.worker_id, timeout_ms=effective_timeout)

    def post_note(self, *, channel: str, content: str) -> dict[str, Any]:
        return self._store.post_note(self.worker_id, channel=channel, content=content)

    def read_notes(self, *, channel: str = "", limit: int = 20) -> dict[str, Any]:
        return self._store.read_notes(channel=channel, limit=limit)

    def claim_scope(self, *, scope: str, ttl_s: int | None = None, reason: str = "") -> dict[str, Any]:
        ttl = self._default_claim_ttl_s if ttl_s is None else max(int(ttl_s or 0), 30)
        return self._store.claim_scope(
            self.worker_id,
            scope=scope,
            ttl_s=ttl,
            reason=reason,
        )

    def release_scope(self, *, scope: str) -> dict[str, Any]:
        return self._store.release_scope(self.worker_id, scope=scope)

    def list_claims(self) -> dict[str, Any]:
        return self._store.list_claims(worker_id=self.worker_id)

    def set_waiting_on(self, waiting_on: str = "") -> None:
        self._store.set_waiting_on(self.worker_id, waiting_on)

    def get_waiting_on(self) -> str:
        state = self._store.worker_state(self.worker_id)
        return str(state.get("waiting_on", "") or "").strip() if state else ""

    def state(self) -> dict[str, Any]:
        return self._store.worker_state(self.worker_id)
