"""
CATalyze Supermemory integration — store and retrieve inspection history.

Uses the Supermemory API (v4) at https://api.supermemory.ai to persist
inspection records per inspector. Save stores full result with metadata;
search returns memories filtered by inspector_id (and optionally machine/component).
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
        "Content-Type": "application/json",
    }


def _metadata_filter(key: str, value: str) -> dict[str, str]:
    """Build a single metadata filter for v4 search."""
    return {"filterType": "metadata", "key": key, "value": value}


async def save_inspection_result(
    machine: str,
    component: str,
    result: dict[str, Any],
    inspector_id: str | None = None,
) -> bool:
    """
    Save an inspection result to Supermemory as a memory.
    Uses POST /v4/memories with containerTag = inspector_id and full metadata.
    """
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
        "containerTag": container_tag,
        "memories": [
            {
                "content": content[:10000],
                "metadata": metadata,
                "isStatic": False,
            }
        ],
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v4/memories",
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
    """
    Retrieve past inspection records for a machine/component from Supermemory.
    Uses POST /v4/search with containerTag and metadata filters.
    Returns a list of inspection records (metadata from each result).
    """
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    container_tag = inspector_id or "anonymous"
    filters: dict[str, list[dict[str, str]]] = {
        "AND": [
            _metadata_filter("type", "inspection"),
            _metadata_filter("machine", machine),
            _metadata_filter("component", component),
        ]
    }
    if inspector_id:
        filters["AND"].append(_metadata_filter("inspector_id", inspector_id))

    payload = {
        "q": f"{machine} {component} inspection",
        "containerTag": container_tag,
        "filters": filters,
        "limit": min(limit, 100),
        "searchMode": "memories",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v4/search",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                print(f"[Supermemory] get_inspection_history HTTP error: status={resp.status_code} body={resp.text!r}")
                return []

            data = resp.json()
            results = data.get("results") or []
            records = []
            for r in results:
                meta = r.get("metadata")
                if isinstance(meta, dict):
                    records.append(meta)
                else:
                    records.append(r)
            return records
        except Exception as e:
            print(f"[Supermemory] get_inspection_history error: {e!r}")
            return []


async def get_all_inspection_results(
    inspector_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """
    Retrieve all inspection records for an inspector from Supermemory.
    Uses POST /v4/search with containerTag and type=inspection filter.
    Returns a list of inspection records (metadata from each result).
    """
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    payload = {
        "q": "inspection",
        "containerTag": inspector_id,
        "filters": {
            "AND": [
                _metadata_filter("type", "inspection"),
                _metadata_filter("inspector_id", inspector_id),
            ]
        },
        "limit": min(limit, 100),
        "searchMode": "memories",
        "threshold": 0.0,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v4/search",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                print(f"[Supermemory] get_all_inspection_results HTTP error: status={resp.status_code} body={resp.text!r}")
                return []

            data = resp.json()
            results = data.get("results") or []
            records = []
            for r in results:
                meta = r.get("metadata")
                if isinstance(meta, dict):
                    records.append(meta)
                else:
                    records.append(r)
            return records
        except Exception as e:
            print(f"[Supermemory] get_all_inspection_results error: {e!r}")
            return []


async def save_fleet(inspector_id: str, machine_names: list[str]) -> bool:
    """
    Save the user's fleet (list of machine names) to Supermemory.
    Stores as a single memory with type=fleet and metadata.machines.
    """
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
        "containerTag": inspector_id,
        "memories": [
            {
                "content": content[:10000],
                "metadata": metadata,
                "isStatic": False,
            }
        ],
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v4/memories",
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
    """
    Retrieve the user's fleet (list of machine names) from Supermemory.
    Returns the most recent fleet memory's machines list.
    """
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    payload = {
        "q": "fleet",
        "containerTag": inspector_id,
        "filters": {
            "AND": [
                _metadata_filter("type", "fleet"),
                _metadata_filter("inspector_id", inspector_id),
            ]
        },
        "limit": 20,
        "searchMode": "memories",
        "threshold": 0.0,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v4/search",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                print(f"[Supermemory] get_fleet HTTP error: status={resp.status_code} body={resp.text!r}")
                return []

            data = resp.json()
            results = data.get("results") or []
            best: dict[str, Any] | None = None
            for r in results:
                meta = r.get("metadata")
                if not isinstance(meta, dict):
                    continue
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
