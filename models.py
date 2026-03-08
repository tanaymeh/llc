from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class AvailableModel:
    id: str
    name: str


_cache: dict[tuple[str, str | None], list[AvailableModel]] = {}
_cache_lock = asyncio.Lock()


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _parse_models(payload: object) -> list[AvailableModel]:
    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    models: list[AvailableModel] = []
    seen: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        name = item.get("name")
        models.append(
            AvailableModel(id=model_id, name=name if isinstance(name, str) and name else model_id)
        )
        seen.add(model_id)

    return sorted(models, key=lambda model: model.id.lower())


async def fetch_models(
    base_url: str | None,
    api_key: str | None,
) -> list[AvailableModel]:
    if not base_url:
        return []

    cache_key = (base_url, api_key)
    async with _cache_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(_models_url(base_url), headers=headers)
            response.raise_for_status()
            models = _parse_models(response.json())
    except (httpx.HTTPError, ValueError):
        models = []

    async with _cache_lock:
        _cache[cache_key] = models

    return models
