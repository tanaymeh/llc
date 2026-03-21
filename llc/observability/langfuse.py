from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from langfuse import get_client
from langfuse.langchain import CallbackHandler

_ACTIVE_TRACE = ContextVar("llc_langfuse_active_trace", default=False)
_SESSION_ID = ContextVar("llc_langfuse_session_id", default="")
_TURN_ID = ContextVar("llc_langfuse_turn_id", default="")


@dataclass(slots=True)
class TraceScope:
    enabled: bool
    observation: Any | None = None
    langchain_config: dict[str, Any] = field(default_factory=dict)


def _env_enabled() -> bool:
    raw = os.getenv("LLC_LANGFUSE_ENABLED")
    if raw is not None and raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return True


def _has_credentials() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_BASE_URL")
    )


def _get_client() -> Any | None:
    if not _env_enabled() or not _has_credentials():
        return None
    try:
        return get_client()
    except Exception:
        return None


def build_langchain_config(
    *,
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = _get_client()
    if client is None or not _ACTIVE_TRACE.get():
        return {}

    session_id = _SESSION_ID.get().strip()
    turn_id = _TURN_ID.get().strip()
    run_metadata: dict[str, Any] = {}
    if session_id:
        run_metadata["langfuse_session_id"] = session_id
        run_metadata["langfuse_user_id"] = session_id
    if tags:
        run_metadata["langfuse_tags"] = list(tags)
    if turn_id:
        run_metadata["llc_turn_id"] = turn_id
    if metadata:
        run_metadata.update(metadata)

    config: dict[str, Any] = {"callbacks": [CallbackHandler()]}
    if run_metadata:
        config["metadata"] = run_metadata
    return config


def merge_langchain_config(
    base_config: dict[str, Any],
    extra_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not extra_config:
        return dict(base_config)

    merged: dict[str, Any] = dict(base_config)
    for key, value in extra_config.items():
        if key == "callbacks":
            existing_callbacks = merged.get("callbacks", [])
            merged_callbacks: list[Any] = (
                list(existing_callbacks) if isinstance(existing_callbacks, list) else []
            )
            if isinstance(value, list):
                merged_callbacks.extend(value)
            elif value is not None:
                merged_callbacks.append(value)
            if merged_callbacks:
                merged["callbacks"] = merged_callbacks
            continue

        if key == "metadata":
            merged_metadata: dict[str, Any] = {}
            existing_metadata = merged.get("metadata")
            if isinstance(existing_metadata, dict):
                merged_metadata.update(existing_metadata)
            if isinstance(value, dict):
                merged_metadata.update(value)
            if merged_metadata:
                merged["metadata"] = merged_metadata
            continue

        merged[key] = value
    return merged


@contextmanager
def start_api_turn_trace(
    *,
    enabled: bool,
    session_id: str,
    turn_id: str,
    thread_id: str,
    model_name: str,
    sub_agent_mode_enabled: bool,
    prompt_text: str,
) -> Iterator[TraceScope]:
    client = _get_client()
    if not enabled or client is None:
        yield TraceScope(enabled=False)
        return

    session_clean = session_id.strip()
    turn_clean = turn_id.strip()
    trace_metadata = {
        "llc_runtime": "api",
        "llc_thread_id": thread_id,
        "llc_model_name": model_name,
        "llc_sub_agent_mode": sub_agent_mode_enabled,
        "llc_turn_id": turn_clean,
    }
    token_active = _ACTIVE_TRACE.set(True)
    token_session = _SESSION_ID.set(session_clean)
    token_turn = _TURN_ID.set(turn_clean)
    try:
        with client.start_as_current_observation(
            as_type="span",
            name="llc.api.turn",
            input={"prompt": prompt_text},
        ) as observation:
            update_observation(
                observation,
                tags=["llc", "api", "orchestrator"],
                metadata=trace_metadata,
                trace_name="llc.api.turn",
                user_id=session_clean or None,
                session_id=session_clean or None,
            )
            yield TraceScope(
                enabled=True,
                observation=observation,
                langchain_config=build_langchain_config(
                    tags=("llc", "api", "orchestrator"),
                    metadata={
                        "llc_thread_id": thread_id,
                        "llc_model_name": model_name,
                    },
                ),
            )
    finally:
        _TURN_ID.reset(token_turn)
        _SESSION_ID.reset(token_session)
        _ACTIVE_TRACE.reset(token_active)


def capture_parent_context() -> dict[str, str] | None:
    if not _ACTIVE_TRACE.get():
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        trace_id = str(client.get_current_trace_id() or "").strip()
        observation_id = str(client.get_current_observation_id() or "").strip()
    except Exception:
        return None
    if not trace_id or not observation_id:
        return None
    return {
        "trace_id": trace_id,
        "parent_observation_id": observation_id,
        "session_id": _SESSION_ID.get().strip(),
        "turn_id": _TURN_ID.get().strip(),
    }


@contextmanager
def start_linked_subagent_trace(
    *,
    parent_context: dict[str, str] | None,
    subagent_id: str,
    subagent_name: str,
    thread_id: str,
    task: str,
    model_name: str,
) -> Iterator[TraceScope]:
    client = _get_client()
    if client is None:
        yield TraceScope(enabled=False)
        return
    if not isinstance(parent_context, dict):
        yield TraceScope(enabled=False)
        return
    trace_id = str(parent_context.get("trace_id", "")).strip()
    parent_observation_id = str(parent_context.get("parent_observation_id", "")).strip()
    if not trace_id or not parent_observation_id:
        yield TraceScope(enabled=False)
        return

    session_id = str(parent_context.get("session_id", "")).strip()
    turn_id = str(parent_context.get("turn_id", "")).strip()
    trace_context = {
        "trace_id": trace_id,
        "parent_span_id": parent_observation_id,
    }

    token_active = _ACTIVE_TRACE.set(True)
    token_session = _SESSION_ID.set(session_id)
    token_turn = _TURN_ID.set(turn_id)
    try:
        with client.start_as_current_observation(
            as_type="span",
            name="llc.api.subagent",
            trace_context=trace_context,
            input={"task": task},
        ) as observation:
            update_observation(
                observation,
                tags=["llc", "api", "subagent"],
                metadata={
                    "llc_runtime": "api",
                    "llc_subagent_id": subagent_id,
                    "llc_subagent_name": subagent_name,
                    "llc_thread_id": thread_id,
                    "llc_model_name": model_name,
                    "llc_turn_id": turn_id,
                },
                trace_name="llc.api.subagent",
                user_id=session_id or None,
                session_id=session_id or None,
            )
            yield TraceScope(
                enabled=True,
                observation=observation,
                langchain_config=build_langchain_config(
                    tags=("llc", "api", "subagent"),
                    metadata={
                        "llc_subagent_id": subagent_id,
                        "llc_subagent_name": subagent_name,
                        "llc_thread_id": thread_id,
                        "llc_model_name": model_name,
                    },
                ),
            )
    finally:
        _TURN_ID.reset(token_turn)
        _SESSION_ID.reset(token_session)
        _ACTIVE_TRACE.reset(token_active)


@contextmanager
def start_child_span(
    name: str,
    *,
    input_payload: Any | None = None,
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    if not _ACTIVE_TRACE.get():
        yield None
        return
    client = _get_client()
    if client is None:
        yield None
        return
    try:
        if not str(client.get_current_trace_id() or "").strip():
            yield None
            return
    except Exception:
        yield None
        return

    kwargs: dict[str, Any] = {"as_type": "span", "name": name}
    if input_payload is not None:
        kwargs["input"] = input_payload

    with client.start_as_current_observation(**kwargs) as observation:
        update_payload: dict[str, Any] = {}
        session_id = _SESSION_ID.get().strip()
        if tags:
            update_payload["tags"] = list(tags)
        if metadata:
            trace_metadata = dict(metadata)
            turn_id = _TURN_ID.get().strip()
            if turn_id:
                trace_metadata.setdefault("llc_turn_id", turn_id)
            update_payload["metadata"] = trace_metadata
        if session_id:
            update_payload["user_id"] = session_id
            update_payload["session_id"] = session_id
        update_observation(observation, **update_payload)
        yield observation


def update_observation(observation: Any | None, **kwargs: Any) -> None:
    if observation is None:
        return
    payload = {key: value for key, value in kwargs.items() if value is not None}
    if not payload:
        return
    try:
        observation.update(**payload)
    except Exception:
        return


def flush() -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        return


def shutdown() -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        return
