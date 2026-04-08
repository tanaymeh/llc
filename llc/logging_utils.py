from __future__ import annotations

import logging
import os
from typing import Any

_LOGGING_CONFIGURED = False
_SILENCED_LOGGERS = (
    "opentelemetry",
    "opentelemetry.exporter",
    "opentelemetry.sdk",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
)


def configure_logging(*, subagent_debug_logging: bool = False) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    requested_level = str(os.getenv("LLC_LOG_LEVEL", "INFO")).upper().strip()
    level = getattr(logging, requested_level, logging.INFO)
    if subagent_debug_logging and level > logging.INFO:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    for logger_name in _SILENCED_LOGGERS:
        silenced = logging.getLogger(logger_name)
        silenced.setLevel(logging.CRITICAL + 1)
        silenced.disabled = True
        silenced.propagate = False
    _LOGGING_CONFIGURED = True


def render_kv(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in payload.items():
        if value is None:
            continue
        text = str(value).replace("\n", "\\n").strip()
        if not text:
            continue
        parts.append(f"{key}={text}")
    return " ".join(parts)
