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


def _sanitize_tag(inspector_id: str) -> str:
    """Convert email to valid containerTag (alphanumeric, hyphens, underscores only)."""
    return inspector_id.replace("@", "_at_").replace(".", "_")


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
    container_tag = _sanitize_tag(inspector_id) if inspector_id else "anonymous"

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

    container_tag = _sanitize_tag(inspector_id) if inspector_id else "anonymous"
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


async def delete_inspection_results_for_machine(
    inspector_id: str,
    machine: str,
) -> tuple[bool, int]:
    """Delete all Supermemory inspection documents for the given inspector + machine.
    Returns (success, deleted_count)."""
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        print("[Supermemory] delete_inspection_results_for_machine: no API key")
        return False, 0

    container_tag = _sanitize_tag(inspector_id)
    payload = {
        "q": "inspection",
        "containerTags": [container_tag],
        "limit": 500,
        "filters": {
            "AND": [
                {"filterType": "metadata", "key": "type", "value": "inspection"},
                {"filterType": "metadata", "key": "machine", "value": machine},
            ]
        },
    }

    doc_ids: list[str] = []
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{SUPERMEMORY_BASE_URL}/v3/search",
                headers=_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                print(
                    f"[Supermemory] delete search HTTP error: status={resp.status_code} body={resp.text!r}"
                )
                return False, 0

            data = resp.json()
            results = data.get("results")
            if not results and isinstance(data, list):
                results = data
            results = results or []
            seen: set[str] = set()
            for r in results:
                meta = r.get("metadata")
                if not isinstance(meta, dict) or meta.get("type") != "inspection":
                    continue
                if (meta.get("machine") or "") != machine:
                    continue
                doc_id = r.get("documentId") or r.get("id") or r.get("docId")
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    doc_ids.append(doc_id)
        except Exception as e:
            print(f"[Supermemory] delete_inspection_results_for_machine search error: {e!r}")
            return False, 0

    if not doc_ids:
        return True, 0

    total_deleted = 0
    async with httpx.AsyncClient(timeout=15) as client:
        for i in range(0, len(doc_ids), 100):
            chunk = doc_ids[i : i + 100]
            try:
                del_resp = await client.delete(
                    f"{SUPERMEMORY_BASE_URL}/v3/documents/bulk",
                    headers=_headers(),
                    json={"ids": chunk},
                )
                if del_resp.status_code == 200:
                    body = del_resp.json()
                    total_deleted += body.get("deletedCount", 0)
                else:
                    print(
                        f"[Supermemory] bulk delete HTTP error: status={del_resp.status_code} body={del_resp.text!r}"
                    )
            except Exception as e:
                print(f"[Supermemory] bulk delete error: {e!r}")
                return False, total_deleted

    print(f"[Supermemory] delete_inspection_results_for_machine: deleted {total_deleted} docs for machine={machine!r}")
    return True, total_deleted


async def get_all_inspection_results(
    inspector_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
    if not api_key:
        return []

    payload = {
        "q": "inspection",
        "containerTags": [_sanitize_tag(inspector_id)],
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
        "containerTag": _sanitize_tag(inspector_id),
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
        "containerTags": [_sanitize_tag(inspector_id)],
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