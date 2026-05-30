import json
import re

from baseagent.base_agent import BaseAgent
from state import AgentState


class RouterAgent(BaseAgent):
    tool_names: list[str] = []  # pure JSON routing — tools would corrupt the output format
    system_prompt = (
        "You are an assistant router. Look at the FULL conversation below "
        "(prior turns + the newest user message) and extract three things:\n"
        "1. location: the city name (lowercase). If the newest message does not name a city, "
        "carry forward the most recent city mentioned earlier in the conversation. "
        "Return null only if no city has ever been mentioned. Location is not required "
        "for the 'outages' intent.\n"
        "2. intent: one of 'weather', 'outages', or 'unknown'. Use 'outages' when "
        "the user asks about grid/power outages, outage reports, outage analysis, or "
        "outage inconsistencies. Take this from the newest user message — do not carry it "
        "forward from earlier turns.\n"
        "3. outage_id: a numeric outage ID (integer) the user explicitly references in the "
        "newest message — e.g. 'tell me about outage 16515354', 'details for 18106219', "
        "or just '16515354'. Return null if the newest message does not mention a numeric "
        "outage ID. Do not carry forward from earlier turns. Only applies when intent is "
        "'outages'.\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        "{\"location\": \"<city or null>\", \"intent\": \"<weather|outages|unknown>\", "
        "\"outage_id\": <integer or null>}\n"
        "No extra text, no explanation — just the JSON."
    )

    def _parse(self, raw: str) -> tuple[str | None, str, int | None]:
        location = None
        intent = "unknown"
        outage_id: int | None = None
        try:
            json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                raw_location = parsed.get("location")
                location = raw_location.lower().strip() if raw_location and raw_location != "null" else None
                intent = parsed.get("intent", "unknown")
                raw_outage_id = parsed.get("outage_id")
                if raw_outage_id not in (None, "null", ""):
                    try:
                        outage_id = int(raw_outage_id)
                    except (TypeError, ValueError):
                        outage_id = None
        except (json.JSONDecodeError, AttributeError) as e:
            self.log("warning", "router.parse_failed", raw=raw[:500], error=str(e))
            self.log_event("parse.failure", raw_excerpt=(raw or "")[:200], error=str(e))
        return location, intent, outage_id

    def run(self, state: AgentState) -> AgentState:
        updated = self._increment(state)
        messages = self.format_messages(state)
        raw = self.call_llm(messages)

        updated["agent_scratchpad"] = raw

        location, intent, outage_id = self._parse(raw)
        if location is None:
            location = state.get("location")
        updated["location"] = location
        updated["intent"] = intent
        updated["outage_id"] = outage_id

        if intent == "outages":
            if outage_id is not None:
                updated["current_task"] = f"Get details for outage {outage_id}"
                updated["delegated_task"] = f"Provide details for outage {outage_id}"
            else:
                updated["current_task"] = "Get top outages from the outage report"
                updated["delegated_task"] = "Provide top 5 outages from the report"
            updated["active_agent"] = "outage_agent"
            updated["next_action"] = "delegate_to_outage_agent"
            updated = self._append_trace(updated, location=location, intent=intent, outage_id=outage_id, action=updated["next_action"])
        elif location and intent == "weather":
            updated["current_task"] = f"Get weather information for {location}"
            updated["active_agent"] = "weather_agent"
            updated["next_action"] = "delegate_to_weather_agent"
            updated["delegated_task"] = f"Provide weather info for {location}"
            updated = self._append_trace(updated, location=location, intent=intent, action=updated["next_action"])
        else:
            updated["current_task"] = "Clarify user intent"
            updated["active_agent"] = "router_agent"
            updated["next_action"] = "ask_clarification"
            updated["delegated_task"] = None
            updated = self._append_trace(updated, location=location, intent=intent, action="ask_clarification")
            clarifying = "I can help with weather or outage reports. What would you like to know?"
            updated = self.set_final_output(updated, clarifying)

        return updated
