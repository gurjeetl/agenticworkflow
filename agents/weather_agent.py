from baseagent.base_agent import BaseAgent
from registry import AgentMeta, FieldSpec, register
from state import AgentState


class WeatherAgent(BaseAgent):
    system_prompt = "You are a helpful weather reporter for a travel assistant."
    tool_names: list[str] = ["get_weather"]

    def run(self, state: AgentState) -> AgentState:
        city = (state.get("location") or "").lower().strip()
        return self.answer_with_tool(
            state,
            tool_name="get_weather",
            args={"city": city},
            format_text=lambda report: f"Here's the current weather for {city.title()}: {report}",
            city=city,
        )


META = AgentMeta(
    agent_id="weather",
    version="1.0.0",
    capability_tags=["weather", "forecast", "city"],
    description="Reports current weather conditions for a named city.",
    input_schema={
        "location": FieldSpec(type="string", required=True, description="City name."),
    },
    output_schema={
        "text": FieldSpec(type="string", description="Plain-language weather report."),
    },
    sla_ms=4000,
)
register(META, WeatherAgent)
