"""Run-status endpoint for asynchronously suspended chat turns.

When a run suspends on a bus task, ``/chat`` answers ``202 {status: "pending"}``
and the frontend polls here until the reply-router has resumed and finished the
run. Reads the durable graph checkpoint, so it works from any gateway instance
and across restarts.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from genie.application.checkpointer import get_thread_config
from genie.application.graph import get_graph

router = APIRouter()


@router.get("/runs/{thread_id}/{run_id}")
async def get_run(thread_id: str, run_id: str) -> dict:
    """Status of one run: ``pending`` (still suspended/working), ``completed``, or ``unknown``."""
    graph = get_graph()
    snap = graph.get_state(get_thread_config(thread_id))
    values = snap.values if snap and snap.values else None
    if not values or values.get("run_id") != run_id:
        return {"status": "unknown", "thread_id": thread_id, "run_id": run_id}
    if snap.next:  # a pending interrupt / unfinished node remains
        return {"status": "pending", "thread_id": thread_id, "run_id": run_id}
    return {
        "status": "completed",
        "thread_id": thread_id,
        "run_id": run_id,
        "response": values.get("final_output") or values.get("error") or "",
        "view": values.get("view"),
        "partial": bool(values.get("partial")),
    }


@router.post("/runs/{thread_id}/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str, request: Request) -> dict:
    """Cancel a pending run's outstanding bus waits (user-facing cancel).

    Each pending wait resolves as cancelled with an error result; the run
    unblocks immediately and the Gate/Synthesizer produce a partial answer.
    Only meaningful in async mode — 409 when the reply-router isn't running.
    """
    reply_router = getattr(request.app.state, "reply_router", None)
    if reply_router is None:
        raise HTTPException(status_code=409, detail="async A2A transport is not enabled on this gateway")
    cancelled = await reply_router.cancel_run(thread_id, run_id)
    return {"status": "cancelled" if cancelled else "nothing_to_cancel",
            "thread_id": thread_id, "run_id": run_id, "waits_cancelled": cancelled}
