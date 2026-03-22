from __future__ import annotations

from typing import Any


def token_usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return max(input_tokens, 0), max(output_tokens, 0)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return max(input_tokens, 0), max(output_tokens, 0)


def accumulate_usage_by_model(
    usage_by_model: dict[str, dict[str, int]],
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    in_tokens = max(int(input_tokens or 0), 0)
    out_tokens = max(int(output_tokens or 0), 0)
    if not model_name or (in_tokens == 0 and out_tokens == 0):
        return
    bucket = usage_by_model.setdefault(
        model_name,
        {"input_tokens": 0, "output_tokens": 0},
    )
    bucket["input_tokens"] += in_tokens
    bucket["output_tokens"] += out_tokens


def normalize_usage_by_model(raw_usage: Any) -> dict[str, dict[str, int]]:
    if not isinstance(raw_usage, dict):
        return {}
    normalized: dict[str, dict[str, int]] = {}
    for raw_model_name, raw_bucket in raw_usage.items():
        model_name = str(raw_model_name).strip()
        if not model_name or not isinstance(raw_bucket, dict):
            continue
        input_tokens = max(int(raw_bucket.get("input_tokens", 0) or 0), 0)
        output_tokens = max(int(raw_bucket.get("output_tokens", 0) or 0), 0)
        if input_tokens == 0 and output_tokens == 0:
            continue
        normalized[model_name] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return normalized
