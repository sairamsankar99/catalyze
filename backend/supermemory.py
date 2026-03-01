"""
CATalyze Supermemory integration — store and retrieve inspection history.

Uses the Supermemory API (v3) at https://api.supermemory.ai to persist
inspection records per inspector. Save stores full result with metadata;
search returns documents filtered by inspector_id (and optionally machine/component).
"""

import os
from datetime import datetime, timezone
from typing import Any

import httpx

SUPERMEMORY_BASE_URL = "https://api.supermemory.ai"


def _headers() -> dict[str, str]:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "x-supermemory-api-key": api_key,
        "Content-Type": "application/json",
    }


async def save_inspection_result(
    machine: str,
    component: str,
    result: dict[str, Any],
    inspector_id: str | None = None,
) -> bool:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return False

    timestamp = result.get("timestamp") or datetime.now(timezone.utc).isoformat()
    container_tag = inspector_id or "anonymous"

    content = (
        f"Inspection: {machine} — {component}. "
        f"Status: {result.get('status', 'UNKNOWN')}. "
        f"Observation: {result.get('observation', '') or 'N/A'}"
    )

    metadata: dict[str, Any] = {
        "type": "inspection",
        "machine": machine,
        "component": component,
        "status": result.get("status", "UNKNOWN"),
        "timestamp": timestamp,
        "observation": result.get("observation") or "",
        "confidence": result.get("confidence"),
        "recommended_action": result.get("recommended_action") or "",
        "maintenance_steps": result.get("maintenance_steps") or [],
    }
    if inspector_id:
        metadata["inspector_id"] = inspector_id

    payload = {
        "content": content[:10000],
        "containerTag": container_tag,
        "metadata": metadata,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v3/documents",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code in (200, 201):
                print(f"[Supermemory] save_inspection_result OK: status={resp.status_code}")
                return True
            print(f"[Supermemory] save_inspection_result HTTP error: status={resp.status_code} body={resp.text!r}")
            return False
        except Exception as e:
            print(f"[Supermemory] save_inspection_result error: {e!r}")
            return False


async def get_inspection_history(
    machine: str,
    component: str,
    inspector_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    container_tag = inspector_id or "anonymous"
    payload = {
        "q": f"{machine} {component} inspection",
        "containerTags": [container_tag],
        "limit": min(limit, 100),
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v3/search",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                print(f"[Supermemory] get_inspection_history HTTP error: status={resp.status_code} body={resp.text!r}")
                return []

            data = resp.json()
            results = data.get("results")
            if not results and isinstance(data, list):
                results = data
            results = results or []
            records = []
            for r in results:
                meta = r.get("metadata")
                if isinstance(meta, dict) and meta.get("type") == "inspection":
                    records.append(meta)
            return records
        except Exception as e:
            print(f"[Supermemory] get_inspection_history error: {e!r}")
            return []


async def get_all_inspection_results(
    inspector_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    payload = {
        "q": "inspection",
        "containerTag": inspector_id,
        "limit": min(limit, 100),
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v3/search",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                print(f"[Supermemory] get_all_inspection_results HTTP error: status={resp.status_code} body={resp.text!r}")
                return []

            data = resp.json()
            results = data.get("results")
            if not results and isinstance(data, list):
                results = data
            results = results or []
            records = []
            for r in results:
                meta = r.get("metadata")
                if isinstance(meta, dict) and meta.get("type") == "inspection":
                    records.append(meta)
            return records
        except Exception as e:
            print(f"[Supermemory] get_all_inspection_results error: {e!r}")
            return []


async def save_fleet(inspector_id: str, machine_names: list[str]) -> bool:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return False

    timestamp = datetime.now(timezone.utc).isoformat()
    content = f"Fleet: {', '.join(machine_names) or 'empty'}"
    metadata: dict[str, Any] = {
        "type": "fleet",
        "inspector_id": inspector_id,
        "machines": machine_names,
        "timestamp": timestamp,
    }

    payload = {
        "content": content[:10000],
        "containerTag": inspector_id,
        "metadata": metadata,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v3/documents",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code in (200, 201):
                print(f"[Supermemory] save_fleet OK: status={resp.status_code}")
                return True
            print(f"[Supermemory] save_fleet HTTP error: status={resp.status_code} body={resp.text!r}")
            return False
        except Exception as e:
            print(f"[Supermemory] save_fleet error: {e!r}")
            return False


async def get_fleet(inspector_id: str) -> list[str]:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    payload = {
        "q": "fleet",
        "containerTags": [inspector_id],
        "limit": 100,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v3/search",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                print(f"[Supermemory] get_fleet HTTP error: status={resp.status_code} body={resp.text!r}")
                return []

            data = resp.json()
            results = data.get("results")
            if not results and isinstance(data, list):
                results = data
            results = results or []
            records = []
            for r in results:
                meta = r.get("metadata")
                if isinstance(meta, dict) and meta.get("type") == "fleet":
                    records.append(meta)
            best: dict[str, Any] | None = None
            for meta in records:
                machines = meta.get("machines")
                if not isinstance(machines, list):
                    continue
                ts = meta.get("timestamp") or ""
                if best is None or ts > (best.get("timestamp") or ""):
                    best = meta
            return best.get("machines", []) if best else []
        except Exception as e:
            print(f"[Supermemory] get_fleet error: {e!r}")
            return []