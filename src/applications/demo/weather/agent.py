"""Weather demo agent: reports current conditions for a named city.

Thin wrapper over the single ``get_weather`` MCP tool. ``run(state)`` is the
entry point the graph executor calls; the ``__main__`` block runs it standalone
as a self-registering A2A service.
"""
from genie.agents.base import BaseAgent
from genie.registry import AgentMeta, FieldSpec, Skill
from genie.application.state import AgentState


class WeatherAgent(BaseAgent):
    """Reports current weather for a city via the ``get_weather`` MCP tool.

    A minimal single-tool agent: it takes the ``location`` from state, calls the
    MCP tool, and formats the raw report into a friendly one-line answer.
    """

    system_prompt = "You are a helpful weather reporter for a travel assistant."
    tool_names: list[str] = ["get_weather"]

    def run(self, state: AgentState) -> AgentState:
        """Look up the weather for ``state['location']`` and answer in plain language."""
        city = (state.get("location") or "").lower().strip()
        return self.answer_with_tool(
            state,
            tool_name="get_weather",
            args={"city": city},
            format_text=lambda res: (
                f"Here's the current weather for {city.title()}: "
                f"{(res.structured or {}).get('report', res.text)}"
            ),
            city=city,
        )


META = AgentMeta(
    agent_id="weather",
    version="1.0.0",
    capability_tags=["weather", "forecast", "city"],
    description="Reports current weather conditions for a named city.",
    # Explicit A2A skills (served verbatim in the Agent Card). This agent does one
    # thing, so it advertises a single, well-described skill rather than the
    # auto-derived mirror of capability_tags.
    skills=[
        Skill(
            id="get_current_weather",
            name="Current weather report",
            description="Reports current weather conditions (temperature, sky, precipitation) for a named city.",
            tags=["weather", "forecast", "city"],
            examples=[
                "What's the weather in Paris?",
                "Weather in Tokyo",
                "Is it raining in London right now?",
            ],
        ),
    ],
    input_schema={
        "location": FieldSpec(type="string", required=True, description="City name."),
    },
    output_schema={
        "text": FieldSpec(type="string", description="Plain-language weather report.", persist=True),
    },
    sla_ms=4000,
)


if __name__ == "__main__":
    # Run this agent as an independent service that self-registers with the
    # Registry Service and exposes the A2A endpoint POST /a2a. 8010 is a stable
    # default for manual testing; AGENT_PORT (env) or agent_port (YAML) overrides
    # it, and unsetting both binds an ephemeral port advertised via the registry.
    from genie.agents.server import run_agent

    run_agent(WeatherAgent, META, port=8010)
