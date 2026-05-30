import json

from baseagent.base_agent import BaseAgent
from registry import AgentMeta, FieldSpec, register
from state import AgentState


class OutageAgent(BaseAgent):
    system_prompt = "You are a grid-outage analyst summarizing outage reports."
    tool_names: list[str] = [
        "list_outage_ids",
        "get_outage_metadata",
        "get_outage_analysis_summary",
    ]

    @staticmethod
    def _parse_json(s: str) -> dict:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _list_view(self) -> tuple[str, dict] | str:
        data = self._parse_json(self.call_mcp_tool("list_outage_ids", {}))
        items = data.get("items", [])
        total = data.get("total")
        if not items:
            return "No outages found in the current report."
        text = f"Top {len(items)} outages (of {total} total)."
        view = {"type": "outage_list", "total": total, "items": items}
        return text, view

    def _detail_view(self, outage_id: int) -> tuple[str, dict] | str:
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
    input_schema={
        "outage_id": FieldSpec(
            type="integer",
            required=False,
            description="Specific outage ID. Omit to get the top-N list.",
        ),
    },
    output_schema={
        "text": FieldSpec(type="string", description="Short headline for the result."),
        "view": FieldSpec(type="object", description="Structured outage_list or outage_detail dict."),
    },
    sla_ms=6000,
)
register(META, OutageAgent)
