"""Liveness endpoint."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness probe — returns ``{"status": "ok"}`` if the process is up."""
    return {"status": "ok"}
