"""Standalone MCP server exposing static travel data via SSE.

Run: python -m services.mcp.weather_server
Endpoint: http://127.0.0.1:8001/sse
"""
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from genie.platform.config import get_settings
from services.mcp.rag_index import get_index

WEATHER_DATA: dict[str, str] = {
    "london": "Cloudy, 14°C, light rain expected",
    "paris": "Sunny, 22°C, clear skies",
    "new york": "Partly cloudy, 18°C, mild winds",
    "tokyo": "Humid, 28°C, chance of thunderstorm",
    "dubai": "Hot and sunny, 41°C, no cloud cover",
    "minneapolis": "warm and humid, 28-30°C, small chance of showers or thunderstorms",
    "bloomington": "warm and mostly cloudy, 28-30°C, scattered thunderstorms or showers possible",
}

# services/mcp/weather_server.py -> repo root (where Data.Json lives) is 3 levels up.
OUTAGE_DATA_PATH = Path(__file__).resolve().parents[2] / "Data.Json"
_OUTAGE_CACHE: dict[str, Any] | None = None
_OUTAGE_INDEX: dict[int, dict[str, Any]] = {}


def _load_outage_data() -> dict[str, Any]:
    """Load and cache the outage report, building an id -> outage lookup index once."""
    global _OUTAGE_CACHE
    if _OUTAGE_CACHE is None:
        with OUTAGE_DATA_PATH.open(encoding="utf-8") as f:
            _OUTAGE_CACHE = json.load(f)
        for item in _OUTAGE_CACHE.get("outagewise_analysis", []):
            _OUTAGE_INDEX[int(item["id"])] = item
    return _OUTAGE_CACHE


# Bind to the host/port advertised by mcp_server_url so the server's bind address
# and the URL clients connect to share one source of truth and can never drift.
# Falls back to 127.0.0.1:8001 when the setting is unset or carries no port.
_mcp_url = urlparse(get_settings().mcp_server_url or "")
mcp = FastMCP(
    "weather-server",
    host=_mcp_url.hostname or "127.0.0.1",
    port=_mcp_url.port or 8001,
)


@mcp.tool()
def get_weather(city: str) -> str:
    """Return the current weather report for the given city.

    Supported cities (case-insensitive): london, paris, new york, tokyo, dubai.
    Returns a short human-readable weather summary, or a not-found message.
    """
    key = (city or "").strip().lower()
    report = WEATHER_DATA.get(key)
    if report is None:
        return f"No weather data available for '{city}'."
    return report


@mcp.tool()
def get_outage_report_summary() -> dict[str, Any]:
    """Return top-level summary of the outage analysis report.

    Includes report id/name, time period, status, and aggregate counts
    (total outages, significant outages, report inconsistencies, keywords).
    """
    data = _load_outage_data()
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "status": data.get("status"),
        "created_dt": data.get("created_dt"),
        "time_period": data.get("time_period"),
        "total_outages": data.get("total_outages"),
        "total_significant_outages": data.get("total_significant_outages"),
        "total_report_inconsistencies": data.get("total_report_inconsistencies"),
        "total_keywords_detected": data.get("total_keywords_detected"),
        "linked_outages_count": len(data.get("linked_outages_detected", [])),
    }


TOP_OUTAGES_LIMIT = 5


@mcp.tool()
def list_outage_ids() -> dict[str, Any]:
    """Return the top 5 outages from the report with short descriptions."""
    data = _load_outage_data()
    items = data.get("outagewise_analysis", [])
    top = [
        {
            "id": item.get("id"),
            "short_description": item.get("metadata", {}).get("short_description"),
            "outage_type": item.get("metadata", {}).get("outage_type"),
            "participant": item.get("metadata", {}).get("participant"),
            "status": item.get("metadata", {}).get("status"),
            "is_significant": item.get("analysis", {}).get("is_significant"),
        }
        for item in items[:TOP_OUTAGES_LIMIT]
    ]
    return {"total": len(items), "returned": len(top), "items": top}


@mcp.tool()
def get_outage_metadata(outage_id: int) -> dict[str, Any]:
    """Return the metadata block for a specific outage id."""
    _load_outage_data()
    item = _OUTAGE_INDEX.get(int(outage_id))
    if item is None:
        return {"error": f"No outage found with id {outage_id}."}
    return item.get("metadata", {})


@mcp.tool()
def get_outage_analysis_summary(outage_id: int) -> dict[str, Any]:
    """Return the analysis summary for a specific outage id.

    Includes total inconsistencies, the markdown summary, significance flag,
    and the list of attributewise inconsistencies (without the full
    long-form analysis blob).
    """
    _load_outage_data()
    item = _OUTAGE_INDEX.get(int(outage_id))
    if item is None:
        return {"error": f"No outage found with id {outage_id}."}
    analysis = item.get("analysis", {})
    return {
        "id": item.get("id"),
        "total_outage_inconsistencies": analysis.get("total_outage_inconsistencies"),
        "is_significant": analysis.get("is_significant"),
        "summary": analysis.get("summary"),
        "critical_criteria": analysis.get("critical_criteria"),
        "attributewise_inconsistencies": [
            {
                "attribute": a.get("attribute"),
                "is_inconsistent": a.get("is_inconsistent"),
            }
            for a in analysis.get("attributewise_analysis", [])
        ],
    }


@mcp.tool()
def get_outage_attribute_analysis(outage_id: int, attribute: str) -> dict[str, Any]:
    """Return the full analysis text for one attribute of a specific outage.

    Use list_outage_ids and get_outage_analysis_summary first to discover
    which attributes have inconsistencies.
    """
    _load_outage_data()
    item = _OUTAGE_INDEX.get(int(outage_id))
    if item is None:
        return {"error": f"No outage found with id {outage_id}."}
    target = (attribute or "").strip().lower()
    for a in item.get("analysis", {}).get("attributewise_analysis", []):
        if (a.get("attribute") or "").lower() == target:
            return a
    return {"error": f"No attribute '{attribute}' found for outage {outage_id}."}


@mcp.tool()
def get_linked_outages() -> list[dict[str, Any]]:
    """Return the list of linked-outage detections from the report."""
    data = _load_outage_data()
    return data.get("linked_outages_detected", [])


@mcp.tool()
def search_docs(query: str, k: int = 4) -> dict[str, Any]:
    """Retrieve the top-k most relevant documentation chunks for a query.

    Backs the RAG agent: searches this framework's own markdown docs
    (README, WORKFLOW, docs/*) with BM25 ranking and returns the matching
    chunks with their source path and relevance score. Use to answer
    'what is', 'how does', 'explain', 'why' questions about the system
    (A2A, router, registry, planner, blackboard, synthesizer, ...).
    """
    chunks = get_index().search(query or "", k=max(1, min(int(k or 4), 10)))
    return {"query": query, "returned": len(chunks), "chunks": chunks}


if __name__ == "__main__":
    mcp.run(transport="sse")
