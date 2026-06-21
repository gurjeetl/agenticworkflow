"""Outage demo agent: lists grid outages or describes a single one.

Routes between two MCP-backed views: a top-N outage list (no args) and a
per-outage detail view (when ``outage_id`` is present). ``run(state)`` is the
entry point the graph executor calls; the ``__main__`` block runs it standalone
as a self-registering A2A service.
"""
import json

from genie.agents.base import BaseAgent
from genie.registry import AgentMeta, FieldSpec, Skill
from genie.application.state import AgentState


class OutageAgent(BaseAgent):
    """Lists current grid outages or returns a structured detail view for one.

    Reads from the outage MCP tools and produces both a short headline ``text``
    and a structured ``view`` (``outage_list`` or ``outage_detail``) that a UI —
    or a downstream chained step — can consume.
    """

    system_prompt = "You are a grid-outage analyst summarizing outage reports."
    tool_names: list[str] = [
        "list_outage_ids",
        "get_outage_metadata",
        "get_outage_analysis_summary",
    ]

    @staticmethod
    def _parse_json(s: str) -> dict:
        """Best-effort JSON decode of an MCP tool result; return {} on bad/empty input."""
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _list_view(self) -> tuple[str, dict] | str:
        """Build the top-N outage list view, or a plain message when none exist."""
        data = self._parse_json(self.call_mcp_tool("list_outage_ids", {}))
        items = data.get("items", [])
        total = data.get("total")
        if not items:
            return "No outages found in the current report."
        text = f"Top {len(items)} outages (of {total} total)."
        view = {"type": "outage_list", "total": total, "items": items}
        return text, view

    def _detail_view(self, outage_id: int) -> tuple[str, dict] | str:
        """Fetch metadata + analysis for one outage; return an error message if missing."""
        metadata = self._parse_json(
            self.call_mcp_tool("get_outage_metadata", {"outage_id": outage_id})
        )
        if metadata.get("error"):
            return f"Could not find outage {outage_id}: {metadata['error']}"

        analysis = self._parse_json(
            self.call_mcp_tool("get_outage_analysis_summary", {"outage_id": outage_id})
        )
        if analysis.get("error"):
            return f"Could not load analysis for outage {outage_id}: {analysis['error']}"

        text = f"Outage {outage_id}: {metadata.get('short_description') or '(no description)'}"
        view = {
            "type": "outage_detail",
            "outage_id": outage_id,
            "metadata": metadata,
            "analysis": analysis,
        }
        return text, view

    def run(self, state: AgentState) -> AgentState:
        """Dispatch to the detail view when ``outage_id`` is given, else the list view."""
        outage_id = state.get("outage_id")
        if outage_id is not None:
            oid = int(outage_id)
            return self.answer_with(
                state, lambda: self._detail_view(oid),
                source="mcp:outage_detail", outage_id=oid,
            )
        return self.answer_with(state, self._list_view, source="mcp:outage_list")


META = AgentMeta(
    agent_id="outage",
    version="1.0.0",
    capability_tags=[
        "outage", "outages", "grid", "power", "report",
        "list", "top", "summary", "outage_detail", "outage_list",
    ],
    description=(
        "Lists or describes grid outages. Call with no args to get the top-N outage "
        "list (covers 'show me outages', 'top 5 outages', 'recent outages'). "
        "Call with outage_id to get a structured detail view for one specific outage."
    ),
    # Two genuinely distinct A2A skills — the list view and the per-outage detail
    # view — advertised separately so an A2A client can discover each capability.
    skills=[
        Skill(
            id="list_outages",
            name="List grid outages",
            description=(
                "Returns the top-N current grid outages (id, short description, type, "
                "participant, status, significance) from the latest report. Call with no arguments."
            ),
            tags=["outage", "outages", "grid", "power", "report", "list", "top", "summary"],
            examples=[
                "show me outages",
                "top 5 outages",
                "recent grid outages",
                "list current power outages",
            ],
        ),
        Skill(
            id="outage_detail",
            name="Outage detail",
            description=(
                "Returns a structured detail view (metadata + analysis summary) for one "
                "outage identified by its numeric ID."
            ),
            tags=["outage", "outage_detail", "analysis", "grid", "power"],
            examples=[
                "details for outage 18553223",
                "explain outage 16515354",
                "analysis for outage 18562435",
            ],
        ),
    ],
    input_schema={
        "outage_id": FieldSpec(
            type="integer",
            required=False,
            description="Specific outage ID. Omit to get the top-N list.",
        ),
    },
    output_schema={
        "text": FieldSpec(type="string", description="Short headline for the result.", persist=True),
        "view": FieldSpec(
            type="object",
            persist=True,
            description=(
                "outage_list = {total, items:[{id, short_description, outage_type, participant, status, is_significant}]}; "
                "outage_detail = {outage_id, metadata, analysis}. "
                "To chain, reference a field by path, e.g. ${<id>.view.items.0.id} = first listed outage's id."
            ),
        ),
    },
    sla_ms=6000,
)


if __name__ == "__main__":
    # Run this agent as an independent service that self-registers with the
    # Registry Service and exposes the A2A endpoint POST /a2a. Set AGENT_PORT (e.g. 8011).
    from genie.agents.server import run_agent

    run_agent(OutageAgent, META)
