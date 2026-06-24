"""Standalone MCP server exposing sample outage/weather/docs tools via SSE.

Run: python -m services.mcp.genie_mcp_server
Endpoint: http://127.0.0.1:8001/sse

Each tool declares a typed (Pydantic) return model, so FastMCP advertises an
``outputSchema`` and populates ``structuredContent`` on the wire — clients get
back machine-readable objects, not stringified JSON. Tools are read-only
(``readOnlyHint``) and signal failures by raising ``ToolError`` (MCP renders
that as ``CallToolResult(isError=true)``) rather than returning error payloads.
Data access goes through :mod:`services.mcp._repository` so the eventual
database-backed source can be swapped in without touching this tool surface.
"""
from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from genie.platform.config import get_settings
from services.mcp import _repository as repo
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

TOP_OUTAGES_LIMIT = 5

# All tools here are read-only queries; advertise that to clients.
_READ_ONLY = ToolAnnotations(readOnlyHint=True)


# ---------------------------------------------------------------------------
# Output models — drive FastMCP's outputSchema + structuredContent. Models that
# wrap rich, evolving report blocks allow extra keys so new source fields flow
# through without a schema change here.
# ---------------------------------------------------------------------------
class WeatherReport(BaseModel):
    """Current weather for a city."""
    city: str
    report: str


class OutageReportSummary(BaseModel):
    """Top-level summary of the outage analysis report."""
    id: int | None = None
    name: str | None = None
    status: str | None = None
    created_dt: str | None = None
    time_period: Any = None
    total_outages: int | None = None
    total_significant_outages: int | None = None
    total_report_inconsistencies: int | None = None
    total_keywords_detected: int | None = None
    linked_outages_count: int = 0


class OutageListItem(BaseModel):
    """One row in the top-N outage list."""
    id: int | None = None
    short_description: str | None = None
    outage_type: str | None = None
    participant: str | None = None
    status: str | None = None
    is_significant: bool | None = None


class OutageList(BaseModel):
    """The top-N outage list plus totals."""
    total: int
    returned: int
    items: list[OutageListItem]


class OutageMetadata(BaseModel):
    """Metadata block for one outage (known fields typed; extras passed through)."""
    model_config = ConfigDict(extra="allow")
    outage_type: str | None = None
    participant: str | None = None
    status: str | None = None
    short_description: str | None = None
    nature_of_work: str | None = None
    planned_duration: Any = None


class AttributewiseInconsistency(BaseModel):
    """Whether one analyzed attribute was found inconsistent."""
    attribute: str | None = None
    is_inconsistent: bool | None = None


class OutageAnalysisSummary(BaseModel):
    """Analysis summary for one outage (omits the long-form per-attribute blobs)."""
    id: int | None = None
    total_outage_inconsistencies: int | None = None
    is_significant: bool | None = None
    summary: str | None = None
    critical_criteria: Any = None
    attributewise_inconsistencies: list[AttributewiseInconsistency] = Field(default_factory=list)


class OutageAttributeAnalysis(BaseModel):
    """Full analysis text for one attribute of an outage."""
    model_config = ConfigDict(extra="allow")
    attribute: str | None = None
    is_inconsistent: bool | None = None
    analysis: Any = None


class LinkedOutage(BaseModel):
    """One linked-outage detection (a group of related outages plus its analysis)."""
    model_config = ConfigDict(extra="allow")
    linked_outages: Any = None
    analysis: Any = None


class LinkedOutages(BaseModel):
    """Wrapper so the linked-outage list is an object (clean structuredContent)."""
    total: int
    items: list[LinkedOutage]


class DocChunk(BaseModel):
    """One retrieved documentation chunk."""
    model_config = ConfigDict(extra="allow")
    source: str | None = None
    text: str | None = None
    score: float | None = None


class DocSearchResult(BaseModel):
    """Result of a documentation search."""
    query: str
    returned: int
    chunks: list[DocChunk]


# Bind to the host/port advertised by mcp_server_url so the server's bind address
# and the URL clients connect to share one source of truth and can never drift.
# Falls back to 127.0.0.1:8001 when the setting is unset or carries no port.
_mcp_url = urlparse(get_settings().mcp_server_url or "")
mcp = FastMCP(
    "genie-mcp-server",
    host=_mcp_url.hostname or "127.0.0.1",
    port=_mcp_url.port or 8001,
)


@mcp.tool(annotations=_READ_ONLY)
def get_weather(city: str) -> WeatherReport:
    """Return the current weather report for the given city.

    Supported cities (case-insensitive): london, paris, new york, tokyo, dubai,
    minneapolis, bloomington. Raises if the city is not known.
    """
    key = (city or "").strip().lower()
    report = WEATHER_DATA.get(key)
    if report is None:
        raise ToolError(f"No weather data available for '{city}'.")
    return WeatherReport(city=key, report=report)


@mcp.tool(annotations=_READ_ONLY)
def get_outage_report_summary() -> OutageReportSummary:
    """Return the top-level summary of the outage analysis report.

    Includes report id/name, time period, status, and aggregate counts
    (total outages, significant outages, report inconsistencies, keywords).
    """
    data = repo.report()
    return OutageReportSummary(
        id=data.get("id"),
        name=data.get("name"),
        status=data.get("status"),
        created_dt=data.get("created_dt"),
        time_period=data.get("time_period"),
        total_outages=data.get("total_outages"),
        total_significant_outages=data.get("total_significant_outages"),
        total_report_inconsistencies=data.get("total_report_inconsistencies"),
        total_keywords_detected=data.get("total_keywords_detected"),
        linked_outages_count=len(data.get("linked_outages_detected", [])),
    )


@mcp.tool(annotations=_READ_ONLY)
def list_outage_ids() -> OutageList:
    """Return the top 5 outages from the report with short descriptions."""
    items = repo.list_outages()
    top = [
        OutageListItem(
            id=item.get("id"),
            short_description=item.get("metadata", {}).get("short_description"),
            outage_type=item.get("metadata", {}).get("outage_type"),
            participant=item.get("metadata", {}).get("participant"),
            status=item.get("metadata", {}).get("status"),
            is_significant=item.get("analysis", {}).get("is_significant"),
        )
        for item in items[:TOP_OUTAGES_LIMIT]
    ]
    return OutageList(total=len(items), returned=len(top), items=top)


@mcp.tool(annotations=_READ_ONLY)
def get_outage_metadata(outage_id: int) -> OutageMetadata:
    """Return the metadata block for a specific outage id.

    Raises if there is no outage with that id.
    """
    item = repo.get_outage(outage_id)
    if item is None:
        raise ToolError(f"No outage found with id {outage_id}.")
    return OutageMetadata(**item.get("metadata", {}))


@mcp.tool(annotations=_READ_ONLY)
def get_outage_analysis_summary(outage_id: int) -> OutageAnalysisSummary:
    """Return the analysis summary for a specific outage id.

    Includes total inconsistencies, the markdown summary, significance flag,
    and the list of attributewise inconsistencies (without the full long-form
    analysis blob). Raises if there is no outage with that id.
    """
    item = repo.get_outage(outage_id)
    if item is None:
        raise ToolError(f"No outage found with id {outage_id}.")
    analysis = item.get("analysis", {})
    return OutageAnalysisSummary(
        id=item.get("id"),
        total_outage_inconsistencies=analysis.get("total_outage_inconsistencies"),
        is_significant=analysis.get("is_significant"),
        summary=analysis.get("summary"),
        critical_criteria=analysis.get("critical_criteria"),
        attributewise_inconsistencies=[
            AttributewiseInconsistency(
                attribute=a.get("attribute"),
                is_inconsistent=a.get("is_inconsistent"),
            )
            for a in analysis.get("attributewise_analysis", [])
        ],
    )


@mcp.tool(annotations=_READ_ONLY)
def get_outage_attribute_analysis(outage_id: int, attribute: str) -> OutageAttributeAnalysis:
    """Return the full analysis text for one attribute of a specific outage.

    Use list_outage_ids and get_outage_analysis_summary first to discover which
    attributes have inconsistencies. Raises if the outage or attribute is absent.
    """
    item = repo.get_outage(outage_id)
    if item is None:
        raise ToolError(f"No outage found with id {outage_id}.")
    target = (attribute or "").strip().lower()
    for a in item.get("analysis", {}).get("attributewise_analysis", []):
        if (a.get("attribute") or "").lower() == target:
            return OutageAttributeAnalysis(**a)
    raise ToolError(f"No attribute '{attribute}' found for outage {outage_id}.")


@mcp.tool(annotations=_READ_ONLY)
def get_linked_outages() -> LinkedOutages:
    """Return the list of linked-outage detections from the report."""
    items = repo.linked_outages()
    return LinkedOutages(
        total=len(items),
        items=[LinkedOutage(**item) for item in items],
    )


@mcp.tool(annotations=_READ_ONLY)
def search_docs(query: str, k: int = 4) -> DocSearchResult:
    """Retrieve the top-k most relevant documentation chunks for a query.

    Backs the RAG agent: searches this framework's own markdown docs
    (README, SETUP, docs/*) with BM25 ranking and returns the matching
    chunks with their source path and relevance score. Use to answer
    'what is', 'how does', 'explain', 'why' questions about the system
    (A2A, router, registry, planner, blackboard, synthesizer, ...).
    """
    chunks = get_index().search(query or "", k=max(1, min(int(k or 4), 10)))
    return DocSearchResult(
        query=query,
        returned=len(chunks),
        chunks=[DocChunk(**c) for c in chunks],
    )


if __name__ == "__main__":
    mcp.run(transport="sse")
