from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

_LOGGER = logging.getLogger("llc.agent.llm_retry")

_MAX_ATTEMPTS = 3
_BACKOFF_MIN_S = 2
_BACKOFF_MAX_S = 15

_RETRYABLE_ERROR_TYPES: tuple[type[BaseException], ...] = ()
try:
    from openrouter.errors import (
        BadGatewayResponseError,
        EdgeNetworkTimeoutResponseError,
        InternalServerResponseError,
        ProviderOverloadedResponseError,
        RequestTimeoutResponseError,
        ResponseValidationError,
        ServiceUnavailableResponseError,
        TooManyRequestsResponseError,
    )
    _RETRYABLE_ERROR_TYPES = (
        ResponseValidationError,
        BadGatewayResponseError,
        ServiceUnavailableResponseError,
        TooManyRequestsResponseError,
        InternalServerResponseError,
        ProviderOverloadedResponseError,
        RequestTimeoutResponseError,
        EdgeNetworkTimeoutResponseError,
    )
except ImportError:
    pass


def _is_transient_llm_error(exc: BaseException) -> bool:
    if _RETRYABLE_ERROR_TYPES and isinstance(exc, _RETRYABLE_ERROR_TYPES):
        return True
    if isinstance(exc, (ValueError, RuntimeError)):
        msg = str(exc).lower()
        return "response validation failed" in msg or "bad gateway" in msg
    return False


_async_retry = retry(
    retry=retry_if_exception(_is_transient_llm_error),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=_BACKOFF_MIN_S, max=_BACKOFF_MAX_S),
    before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
    reraise=True,
)


async def ainvoke_with_retry(
    model: Any,
    messages: list[Any],
    *,
    config: dict[str, Any] | None = None,
) -> Any:
    @_async_retry
    async def _call() -> Any:
        if config:
            return await model.ainvoke(messages, config=config)
        return await model.ainvoke(messages)

    return await _call()


def invoke_with_retry(
    model: Any,
    messages: list[Any],
) -> Any:
    @retry(
        retry=retry_if_exception(_is_transient_llm_error),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=_BACKOFF_MIN_S, max=_BACKOFF_MAX_S),
        before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
        reraise=True,
    )
    def _call() -> Any:
        return model.invoke(messages)

    return _call()
