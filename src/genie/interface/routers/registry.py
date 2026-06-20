"""Registry proxy endpoint — exposes discovered agents for the trace UI."""
import asyncio

from fastapi import APIRouter

from genie.registry.registry_client import RegistryUnavailable, get_registry_client

router = APIRouter()


@router.get("/registry")
async def registry_dump():
    """Expose discovered agents so the trace UI can show live agent discovery.

    Proxies the Registry Service (same data the Planner sees). The shape is
    backward-compatible with the previous in-process dump and additionally
    carries liveness fields (endpoint, last_heartbeat) so the UI can mark each
    agent as live.
    """
    try:
        metas = await asyncio.to_thread(get_registry_client().list_active)
    except RegistryUnavailable as e:
        return {"agents": [], "error": str(e)}
    return {
        "agents": [
            {
                "agent_id": m.agent_id,
                "version": m.version,
                "capability_tags": m.capability_tags,
                "description": m.description,
                "input_schema": {k: v.model_dump() for k, v in m.input_schema.items()},
                "output_schema": {k: v.model_dump() for k, v in m.output_schema.items()},
                "sla_ms": m.sla_ms,
                "transport": m.transport,
                "status": m.status,
                "endpoint": m.endpoint,
                "instance_id": m.instance_id,
                "last_heartbeat": m.last_heartbeat.isoformat() if m.last_heartbeat else None,
            }
            for m in metas
        ]
    }
