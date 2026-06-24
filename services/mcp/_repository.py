"""Data-access seam for the sample MCP tools.

Today this reads the bundled ``Data.Json`` sample report; the real system will
fetch the same records from a database. Keeping every data access behind these
functions means the MCP tool surface (``genie_mcp_server``) never changes when
the source does — swap the bodies here for DB queries and the tools, their
output schemas, and the agents stay untouched.
"""
from pathlib import Path
from typing import Any
import json

# services/mcp/_repository.py -> repo root (where Data.Json lives) is 3 levels up.
_OUTAGE_DATA_PATH = Path(__file__).resolve().parents[2] / "Data.Json"

_CACHE: dict[str, Any] | None = None
_INDEX: dict[int, dict[str, Any]] = {}


def _report() -> dict[str, Any]:
    """Load and cache the outage report, indexing outages by id on first load."""
    global _CACHE
    if _CACHE is None:
        with _OUTAGE_DATA_PATH.open(encoding="utf-8") as f:
            _CACHE = json.load(f)
        for item in _CACHE.get("outagewise_analysis", []):
            _INDEX[int(item["id"])] = item
    return _CACHE


def report() -> dict[str, Any]:
    """Return the full report record (caller projects the fields it needs)."""
    return _report()


def list_outages() -> list[dict[str, Any]]:
    """Return all per-outage analysis records, in report order."""
    return _report().get("outagewise_analysis", [])


def get_outage(outage_id: int) -> dict[str, Any] | None:
    """Return a single outage record by id, or ``None`` if there is no such outage."""
    _report()
    return _INDEX.get(int(outage_id))


def linked_outages() -> list[dict[str, Any]]:
    """Return the linked-outage detections from the report."""
    return _report().get("linked_outages_detected", [])
