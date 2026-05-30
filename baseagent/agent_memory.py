from langchain_core.messages import BaseMessage

from memory.mongo_store import get_mongo_store
from state import AgentState


class AgentMemory:
    """Short-term message-window trimming + long-term fact persistence."""

    def __init__(self, max_window: int = 15) -> None:
        self.max_window = max_window

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if len(messages) <= self.max_window:
            return messages
        return messages[-self.max_window:]

    @staticmethod
    def facts_block(facts: list[str]) -> str:
        return "\n".join(f"- {f}" for f in facts)

    async def save_fact(self, state: AgentState, key: str, value: str) -> None:
        thread_id = state.get("thread_id", "")
        if not thread_id:
            return
        await get_mongo_store().upsert_fact(thread_id, key, value)
