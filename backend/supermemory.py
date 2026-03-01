"""
CATalyze Supermemory integration — store and retrieve inspection history.

Uses the Supermemory API to persist per-machine, per-component inspection
records so that Gemini can reference trends across inspections.
"""

import os
from datetime import datetime, timezone
from typing import Any

import httpx

SUPERMEMORY_BASE_URL = "https://api.supermemory.ai/v3"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('SUPERMEMORY_API_KEY', '')}",
        "Content-Type": "application/json",
    }


async def get_inspection_history(
    machine: str,
    component: str,
    inspector_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve past inspection records for a machine/component from Supermemory."""
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    filters: dict[str, str] = {
        "metadata.machine": machine,
        "metadata.component": component,
        "metadata.type": "inspection",
    }
    if inspector_id:
        filters["metadata.inspector_id"] = inspector_id

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/search",
                headers=_headers(),
                json={
                    "query": f"{machine} {component} inspection",
                    "limit": limit,
                    "filters": filters,
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            return [r.get("metadata", r) for r in data.get("results", [])]

        except httpx.HTTPError:
            return []


async def get_all_inspection_results(
    inspector_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Retrieve all inspection records for an inspector from Supermemory."""
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    filters: dict[str, str] = {
        "metadata.inspector_id": inspector_id,
        "metadata.type": "inspection",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/search",
                headers=_headers(),
                json={
                    "query": "inspection",
                    "limit": limit,
                    "filters": filters,
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            return [r.get("metadata", r) for r in data.get("results", [])]

        except httpx.HTTPError:
            return []


async def save_inspection_result(
    machine: str,
    component: str,
    result: dict[str, Any],
    inspector_id: str | None = None,
) -> bool:
    """Save an inspection result to Supermemory for future history lookups."""
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return False

    timestamp = result.get("timestamp") or datetime.now(timezone.utc).isoformat()
    content = (
        f"Inspection: {machine} — {component}\n"
        f"Inspector: {inspector_id or 'unknown'}\n"
        f"Date: {timestamp}\n"
        f"Status: {result.get('status', 'UNKNOWN')}\n"
        f"Confidence: {result.get('confidence', 0.0)}\n"
        f"Observation: {result.get('observation', '')}\n"
        f"Action: {result.get('recommended_action', '')}"
    )

    metadata: dict[str, Any] = {
        "machine": machine,
        "component": component,
        "type": "inspection",
        "date": timestamp,
        **result,
    }
    if inspector_id:
        metadata["inspector_id"] = inspector_id

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/memories",
                headers=_headers(),
                json={
                    "content": content,
                    "metadata": metadata,
                },
            )
            return resp.status_code in (200, 201)

        except httpx.HTTPError:
            return False
