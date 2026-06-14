from langchain_core.messages import BaseMessage


class AgentMemory:
    """Short-term message-window trimming + long-term fact rendering.

    Facts are written by the Synthesizer (memory.facts_store) and read at request
    start (memory.mongo_store.get_facts); this class only formats them for prompts.
    """

    def __init__(self, max_window: int = 15) -> None:
        self.max_window = max_window

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if len(messages) <= self.max_window:
            return messages
        return messages[-self.max_window:]

    @staticmethod
    def facts_block(facts: list[str]) -> str:
        return "\n".join(f"- {f}" for f in facts)
