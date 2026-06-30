"""Outage demo agent: lists grid outages or describes a single one.

Routes between two MCP-backed views: a top-N outage list (no args) and a
per-outage detail view (when ``outage_id`` is present). ``run(state)`` is the
entry point the graph executor calls; the ``__main__`` block runs it standalone
as a self-registering A2A service.
"""
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

    def _list_view(self) -> tuple[str, dict] | str:
        """Build the top-N outage list view, or a plain message when none exist."""
        data = self.call_mcp_tool_structured("list_outage_ids", {}).structured or {}
        items = data.get("items", [])
        total = data.get("total")
        if not items:
            return "No outages found in the current report."
        text = f"Top {len(items)} outages (of {total} total)."
        view = {"type": "outage_list", "total": total, "items": items}
        return text, view

    def _detail_view(self, outage_id: int) -> tuple[str, dict] | str:
        """Fetch metadata + analysis for one outage as a structured detail view.

        A missing outage makes the MCP tool raise (``isError``), which surfaces as
        a ``LookupError`` that ``answer_with`` turns into a terminal agent error.
        """
        metadata = self.call_mcp_tool_structured(
            "get_outage_metadata", {"outage_id": outage_id}
        ).structured or {}
        analysis = self.call_mcp_tool_structured(
            "get_outage_analysis_summary", {"outage_id": outage_id}
        ).structured or {}

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
    # Registry Service and exposes the A2A endpoint POST /a2a. 8011 is a stable
    # default for manual testing; AGENT_PORT (env) or agent_port (YAML) overrides
    # it, and unsetting both binds an ephemeral port advertised via the registry.
    from genie.agents.server import run_agent

    run_agent(OutageAgent, META, port=8011)
