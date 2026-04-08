from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from importlib import metadata as importlib_metadata
from typing import Any, Iterator, Literal

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

_LOGGER = logging.getLogger("llc.observability")
_CURRENT_CONTEXT = ContextVar("llc_langfuse_current_context", default=None)
_INIT_ATTEMPTED = False
_CLIENT_DISABLED_REASON = ""
_CLIENT_AVAILABLE: bool | None = None
_EXPECTED_SDK_VERSION = "4.0.6"

ObservationType = Literal[
    "generation",
    "embedding",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
]


@dataclass(slots=True, frozen=True)
class ConversationContext:
    conversation_id: str
    turn_id: str = ""
    actor_id: str = ""
    actor_kind: str = "orchestrator"
    trace_id: str = ""
    root_observation_id: str = ""

    def serialize(self) -> dict[str, str]:
        return {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "actor_id": self.actor_id,
            "actor_kind": self.actor_kind,
            "trace_id": self.trace_id,
            "root_observation_id": self.root_observation_id,
            "session_id": self.conversation_id,
        }


@dataclass(slots=True)
class TraceScope:
    enabled: bool
    observation: Any | None = None
    context: ConversationContext | None = None
    langchain_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetachedObservation:
    enabled: bool
    observation: Any | None = None
    context: ConversationContext | None = None


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


def _sdk_version() -> str:
    try:
        return importlib_metadata.version("langfuse")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _log_disabled_once(reason: str) -> None:
    global _INIT_ATTEMPTED, _CLIENT_AVAILABLE, _CLIENT_DISABLED_REASON
    if _INIT_ATTEMPTED and _CLIENT_DISABLED_REASON == reason:
        return
    _INIT_ATTEMPTED = True
    _CLIENT_AVAILABLE = False
    _CLIENT_DISABLED_REASON = reason
    _LOGGER.info("Langfuse tracing disabled: %s", reason)


def _get_client() -> Any | None:
    global _INIT_ATTEMPTED, _CLIENT_AVAILABLE, _CLIENT_DISABLED_REASON
    if _CLIENT_AVAILABLE is False:
        return None
    if not _env_enabled():
        _log_disabled_once("LLC_LANGFUSE_ENABLED is false.")
        return None
    if not _has_credentials():
        _log_disabled_once("missing LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL.")
        return None

    try:
        client = get_client()
    except Exception as exc:  # noqa: BLE001
        reason = f"client initialization failed: {exc}"
        if not _INIT_ATTEMPTED or _CLIENT_DISABLED_REASON != reason:
            _LOGGER.warning("Langfuse tracing unavailable: %s", reason)
            _INIT_ATTEMPTED = True
            _CLIENT_DISABLED_REASON = reason
        return None

    if _INIT_ATTEMPTED and _CLIENT_AVAILABLE is True:
        return client

    sdk_version = _sdk_version()
    if sdk_version != _EXPECTED_SDK_VERSION:
        _LOGGER.warning(
            "Langfuse SDK version mismatch: expected %s, got %s.",
            _EXPECTED_SDK_VERSION,
            sdk_version,
        )
    try:
        authenticated = bool(client.auth_check())
    except Exception as exc:  # noqa: BLE001
        remediation = (
            " Check LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY for this base URL. "
            "If you are using the bundled local stack, recreate it with "
            "`docker compose -f docker-compose.langfuse.yml down -v && make langfuse-up`."
        )
        reason = (
            "auth check failed for base_url="
            f"{os.getenv('LANGFUSE_BASE_URL', '').strip()}: {exc}.{remediation}"
        )
        if not _INIT_ATTEMPTED or _CLIENT_DISABLED_REASON != reason:
            _LOGGER.warning("Langfuse tracing unavailable: %s", reason)
        _INIT_ATTEMPTED = True
        _CLIENT_AVAILABLE = False
        _CLIENT_DISABLED_REASON = reason
        return None
    if authenticated:
        _CLIENT_AVAILABLE = True
        _LOGGER.info(
            "Langfuse tracing enabled (sdk=%s, base_url=%s).",
            sdk_version,
            os.getenv("LANGFUSE_BASE_URL", "").strip(),
        )
        _INIT_ATTEMPTED = True
        return client

    _INIT_ATTEMPTED = True
    _CLIENT_AVAILABLE = False
    _CLIENT_DISABLED_REASON = "auth_check returned false."
    _LOGGER.warning(
        "Langfuse tracing unavailable: auth check failed for base_url=%s.",
        os.getenv("LANGFUSE_BASE_URL", "").strip(),
    )
    return None


def current_context() -> ConversationContext | None:
    return _CURRENT_CONTEXT.get()


def current_trace_id() -> str:
    ctx = current_context()
    if ctx is not None and ctx.trace_id.strip():
        return ctx.trace_id.strip()
    return ""


def current_observation_id() -> str:
    ctx = current_context()
    if ctx is not None and ctx.root_observation_id.strip():
        return ctx.root_observation_id.strip()
    return ""


def build_langchain_config(
    *,
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    trace_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    client = _get_client()
    ctx = current_context()
    if client is None and trace_context is None:
        return {}
    if ctx is None and trace_context is None:
        return {}

    callbacks: list[Any] = []
    resolved_trace_context = trace_context or _trace_context_from_context(ctx)
    try:
        if resolved_trace_context:
            callbacks.append(CallbackHandler(trace_context=resolved_trace_context))
        else:
            callbacks.append(CallbackHandler())
    except Exception:
        callbacks = []

    run_metadata: dict[str, Any] = {}
    if ctx is not None:
        run_metadata.update(
            {
                "llc_conversation_id": ctx.conversation_id,
                "llc_turn_id": ctx.turn_id,
                "llc_actor_id": ctx.actor_id,
                "llc_actor_kind": ctx.actor_kind,
            }
        )
    if metadata:
        run_metadata.update(metadata)

    config: dict[str, Any] = {}
    if callbacks:
        config["callbacks"] = callbacks
    if run_metadata:
        config["metadata"] = run_metadata
    if tags:
        config["tags"] = list(tags)
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

        if key == "tags":
            existing_tags = merged.get("tags", [])
            merged_tags = list(existing_tags) if isinstance(existing_tags, list) else []
            if isinstance(value, list):
                merged_tags.extend(tag for tag in value if tag not in merged_tags)
            elif value is not None and value not in merged_tags:
                merged_tags.append(value)
            if merged_tags:
                merged["tags"] = merged_tags
            continue

        merged[key] = value
    return merged


@contextmanager
def start_api_turn_trace(
    *,
    enabled: bool,
    conversation_id: str,
    turn_id: str,
    thread_id: str,
    model_name: str,
    sub_agent_mode_enabled: bool,
    prompt_text: str,
) -> Iterator[TraceScope]:
    client = _get_client()
    base_context = ConversationContext(
        conversation_id=conversation_id.strip(),
        turn_id=turn_id.strip(),
        actor_id="orchestrator",
        actor_kind="orchestrator",
    )
    if not enabled or client is None or not base_context.conversation_id:
        yield TraceScope(enabled=False, context=base_context)
        return

    tags = ["llc", "api", "orchestrator"]
    metadata = {
        "llc_runtime": "api",
        "llc_thread_id": thread_id,
        "llc_model_name": model_name,
        "llc_sub_agent_mode": sub_agent_mode_enabled,
    }
    with propagate_attributes(
        session_id=base_context.conversation_id,
        tags=tags,
        trace_name="llc.orchestrator.turn",
    ):
        with client.start_as_current_observation(
            as_type="agent",
            name="llc.orchestrator.turn",
            input={"user_message": prompt_text},
            metadata=metadata,
        ) as observation:
            active_context = _observation_context(base_context, observation)
            token = _CURRENT_CONTEXT.set(active_context)
            try:
                yield TraceScope(
                    enabled=True,
                    observation=observation,
                    context=active_context,
                    langchain_config=build_langchain_config(
                        tags=tuple(tags),
                        metadata={
                            "llc_thread_id": thread_id,
                            "llc_model_name": model_name,
                        },
                    ),
                )
            finally:
                _CURRENT_CONTEXT.reset(token)


def capture_parent_context() -> dict[str, str] | None:
    ctx = current_context()
    if ctx is None:
        return None
    serialized = ctx.serialize()
    if not serialized.get("conversation_id"):
        return None
    return serialized


def create_subagent_root_observation(
    *,
    conversation_id: str,
    turn_id: str,
    subagent_id: str,
    subagent_name: str,
    thread_id: str,
    task: str,
    model_name: str,
) -> DetachedObservation:
    client = _get_client()
    base_context = ConversationContext(
        conversation_id=conversation_id.strip(),
        turn_id=turn_id.strip(),
        actor_id=subagent_id.strip(),
        actor_kind="subagent",
    )
    if client is None or not base_context.conversation_id or not base_context.actor_id:
        return DetachedObservation(enabled=False, context=base_context)

    with propagate_attributes(
        session_id=base_context.conversation_id,
        tags=["llc", "api", "subagent"],
        trace_name="llc.subagent.run",
    ):
        observation = client.start_observation(
            as_type="agent",
            name="llc.subagent.run",
            input={"task": task},
            metadata={
                "llc_runtime": "api",
                "llc_turn_id": base_context.turn_id,
                "llc_thread_id": thread_id,
                "llc_model_name": model_name,
                "llc_subagent_id": subagent_id,
                "llc_subagent_name": subagent_name,
            },
        )
    ctx = _observation_context(base_context, observation)
    flush()
    return DetachedObservation(enabled=True, observation=observation, context=ctx)


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
    base_context = (
        replace(
            _context_from_serialized(parent_context),
            actor_id=subagent_id.strip(),
            actor_kind="subagent",
        )
        if isinstance(parent_context, dict)
        else None
    )
    if client is None or base_context is None:
        yield TraceScope(enabled=False, context=base_context)
        return
    trace_id = str(parent_context.get("trace_id", "")).strip()
    parent_observation_id = str(parent_context.get("root_observation_id", "")).strip()
    if not trace_id or not parent_observation_id:
        yield TraceScope(enabled=False, context=base_context)
        return

    with propagate_attributes(
        session_id=base_context.conversation_id,
        tags=["llc", "api", "subagent", "worker"],
        trace_name="llc.subagent.run",
    ):
        with client.start_as_current_observation(
            as_type="agent",
            name="llc.subagent.worker",
            trace_context={
                "trace_id": trace_id,
                "parent_span_id": parent_observation_id,
            },
            input={"task": task},
            metadata={
                "llc_runtime": "api",
                "llc_turn_id": base_context.turn_id,
                "llc_thread_id": thread_id,
                "llc_model_name": model_name,
                "llc_subagent_id": subagent_id,
                "llc_subagent_name": subagent_name,
                "llc_parent_observation_id": parent_observation_id,
            },
        ) as observation:
            active_context = _observation_context(base_context, observation)
            token = _CURRENT_CONTEXT.set(active_context)
            try:
                yield TraceScope(
                    enabled=True,
                    observation=observation,
                    context=active_context,
                    langchain_config=build_langchain_config(
                        tags=("llc", "api", "subagent", "worker"),
                        metadata={
                            "llc_thread_id": thread_id,
                            "llc_model_name": model_name,
                            "llc_subagent_id": subagent_id,
                            "llc_subagent_name": subagent_name,
                        },
                    ),
                )
            finally:
                _CURRENT_CONTEXT.reset(token)


@contextmanager
def start_hook_trace(
    *,
    conversation_id: str,
    turn_id: str,
    hook_name: str,
    stage: str = "run",
    input_payload: Any | None = None,
) -> Iterator[TraceScope]:
    client = _get_client()
    normalized_hook_name = hook_name.strip() or "hook"
    normalized_stage = stage.strip() or "run"
    trace_name = f"llc.hook.{normalized_hook_name}.{normalized_stage}"
    base_context = ConversationContext(
        conversation_id=conversation_id.strip(),
        turn_id=turn_id.strip(),
        actor_id=normalized_hook_name,
        actor_kind="hook",
    )
    if client is None or not base_context.conversation_id:
        yield TraceScope(enabled=False, context=base_context)
        return

    with propagate_attributes(
        session_id=base_context.conversation_id,
        tags=["llc", "api", "hook"],
        trace_name=trace_name,
    ):
        observation = client.start_observation(
            trace_context={"trace_id": uuid.uuid4().hex},
            as_type="span",
            name=trace_name,
            input=input_payload,
            metadata={
                "llc_runtime": "api",
                "llc_turn_id": base_context.turn_id,
                "llc_hook_name": normalized_hook_name,
                "llc_hook_stage": normalized_stage,
            },
        )
    active_context = _observation_context(base_context, observation)
    token = _CURRENT_CONTEXT.set(active_context)
    try:
        yield TraceScope(
            enabled=True,
            observation=observation,
            context=active_context,
            langchain_config=build_langchain_config(
                tags=("llc", "api", "hook"),
                metadata={
                    "llc_hook_name": normalized_hook_name,
                    "llc_hook_stage": normalized_stage,
                },
            ),
        )
    finally:
        _CURRENT_CONTEXT.reset(token)


@contextmanager
def start_child_span(
    name: str,
    *,
    input_payload: Any | None = None,
    tags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    as_type: ObservationType = "span",
    model_name: str | None = None,
) -> Iterator[Any | None]:
    ctx = current_context()
    client = _get_client()
    if ctx is None or client is None:
        yield None
        return

    trace_metadata = dict(metadata or {})
    trace_metadata.setdefault("llc_conversation_id", ctx.conversation_id)
    if ctx.turn_id:
        trace_metadata.setdefault("llc_turn_id", ctx.turn_id)
    if ctx.actor_id:
        trace_metadata.setdefault("llc_actor_id", ctx.actor_id)
    trace_metadata.setdefault("llc_actor_kind", ctx.actor_kind)
    trace_context = _trace_context_from_context(ctx)

    with propagate_attributes(
        session_id=ctx.conversation_id,
        tags=list(tags) if tags else None,
    ):
        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": name,
            "metadata": trace_metadata,
        }
        if input_payload is not None:
            kwargs["input"] = input_payload
        if model_name:
            kwargs["model"] = model_name
        if trace_context:
            kwargs["trace_context"] = trace_context
        with client.start_as_current_observation(**kwargs) as observation:
            active_context = _observation_context(ctx, observation)
            token = _CURRENT_CONTEXT.set(active_context)
            try:
                yield observation
            finally:
                _CURRENT_CONTEXT.reset(token)


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


def set_trace_io(
    observation: Any | None,
    *,
    input: Any | None = None,
    output: Any | None = None,
) -> None:
    if observation is None:
        return
    try:
        observation.set_trace_io(input=input, output=output)
    except Exception:
        return


def end_observation(observation: Any | None) -> None:
    if observation is None:
        return
    try:
        observation.end()
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


def _context_from_serialized(payload: dict[str, str] | None) -> ConversationContext:
    payload = payload or {}
    return ConversationContext(
        conversation_id=str(payload.get("conversation_id", "")).strip(),
        turn_id=str(payload.get("turn_id", "")).strip(),
        actor_id=str(payload.get("actor_id", "")).strip(),
        actor_kind=str(payload.get("actor_kind", "")).strip() or "system",
        trace_id=str(payload.get("trace_id", "")).strip(),
        root_observation_id=str(payload.get("root_observation_id", "")).strip(),
    )


def _observation_context(base_context: ConversationContext, observation: Any) -> ConversationContext:
    trace_id = str(getattr(observation, "trace_id", "") or "").strip()
    observation_id = str(getattr(observation, "id", "") or "").strip()
    if not trace_id:
        trace_id = base_context.trace_id.strip()
    if not observation_id:
        observation_id = base_context.root_observation_id.strip()
    return ConversationContext(
        conversation_id=base_context.conversation_id,
        turn_id=base_context.turn_id,
        actor_id=base_context.actor_id,
        actor_kind=base_context.actor_kind,
        trace_id=trace_id,
        root_observation_id=observation_id,
    )


def _trace_context_from_context(
    ctx: ConversationContext | None,
) -> dict[str, str] | None:
    if ctx is None:
        return None
    trace_id = ctx.trace_id.strip()
    if not trace_id:
        return None
    payload = {"trace_id": trace_id}
    parent_span_id = ctx.root_observation_id.strip()
    if parent_span_id:
        payload["parent_span_id"] = parent_span_id
    return payload
