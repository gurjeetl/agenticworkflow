"""RAG demo agent: grounded documentation Q&A with citations.

Retrieves relevant doc chunks via the ``search_docs`` MCP tool and has the LLM
compose an answer from only that context, citing sources as [n]. ``run(state)``
is the entry point the graph executor calls; the ``__main__`` block runs it
standalone as a self-registering A2A service.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from genie.agents.base import BaseAgent
from genie.registry import AgentMeta, FieldSpec, Skill
from genie.application.state import AgentState


class RagAgent(BaseAgent):
    """Retrieval-augmented Q&A over the framework's own documentation.

    Retrieves the most relevant doc chunks via the ``search_docs`` MCP tool, then
    composes a grounded answer with the LLM — answering only from the retrieved
    context and citing sources. This is the single-knowledge-agent the Router
    fast-paths to for 'what is / how does / explain' questions.
    """

    system_prompt = (
        "You are a documentation assistant for an agentic-workflow framework. "
        "Answer the user's question USING ONLY the provided context chunks. Cite "
        "the chunks you use inline as [n]. If the context does not contain the "
        "answer, say so plainly instead of guessing. Be concise and concrete."
    )
    tool_names: list[str] = ["search_docs"]

    @staticmethod
    def _parse_json(s: str) -> dict:
        """Best-effort JSON decode of an MCP tool result; return {} on bad/empty input."""
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _answer(self, query: str) -> tuple[str, dict] | str:
        """Retrieve doc chunks, ask the LLM for a grounded answer, and attach a sources view.

        Returns a plain message when retrieval finds nothing, otherwise
        ``(answer, view)`` where ``view`` lists the cited chunk sources/scores.
        """
        data = self._parse_json(self.call_mcp_tool("search_docs", {"query": query, "k": 4}))
        chunks = data.get("chunks", [])
        if not chunks:
            return "I couldn't find anything about that in the documentation."

        context = "\n\n".join(
            f"[{i + 1}] (source: {c.get('source')})\n{c.get('text')}"
            for i, c in enumerate(chunks)
        )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"QUESTION:\n{query}\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    "Answer using only the context above, citing sources as [n]."
                )
            ),
        ]
        answer = self.call_llm(messages)
        view = {
            "type": "rag",
            "query": query,
            "sources": [
                {"n": i + 1, "source": c.get("source"), "score": c.get("score")}
                for i, c in enumerate(chunks)
            ],
        }
        return answer, view

    def run(self, state: AgentState) -> AgentState:
        """Answer the user's question from the docs, or prompt for one if absent."""
        query = (state.get("query") or state.get("user_input") or "").strip()
        if not query:
            return self.answer_with(
                state,
                lambda: "What would you like to know about the system?",
                source="rag:empty",
            )
        return self.answer_with(state, lambda: self._answer(query), source="rag", query=query[:120])


META = AgentMeta(
    agent_id="rag",
    version="1.0.0",
    capability_tags=[
        "rag", "docs", "documentation", "knowledge", "faq", "help",
        "explain", "architecture", "question", "what", "how", "why",
        "retrieval", "search",
    ],
    description=(
        "Answers questions about THIS framework/system from its documentation using "
        "retrieval-augmented generation (RAG). Use for 'what is X', 'how does X work', "
        "'explain X', 'why X' questions about the architecture — e.g. the A2A protocol, "
        "router, registry/discovery, planner, orchestrator, blackboard, synthesizer, gate."
    ),
    # Single A2A skill: grounded documentation Q&A with citations.
    skills=[
        Skill(
            id="documentation_qa",
            name="Documentation Q&A",
            description=(
                "Answers natural-language questions about this framework/system from its "
                "documentation using retrieval-augmented generation (RAG), citing the "
                "sources it used inline as [n]."
            ),
            tags=[
                "rag", "docs", "documentation", "knowledge", "faq", "help",
                "explain", "architecture", "retrieval", "search",
            ],
            examples=[
                "What is the A2A protocol?",
                "How does the router decide a route?",
                "Explain the blackboard",
                "Why is there a completion gate?",
            ],
        ),
    ],
    input_schema={
        "query": FieldSpec(
            type="string",
            required=True,
            description="The natural-language question to answer from the docs.",
        ),
    },
    output_schema={
        "text": FieldSpec(type="string", description="Grounded answer with [n] citations.", persist=True),
        "view": FieldSpec(type="object", description="Structured 'rag' view listing the cited sources."),
    },
    sla_ms=12000,
)


if __name__ == "__main__":
    # Run this agent as an independent service that self-registers with the
    # Registry Service and exposes the A2A endpoint POST /a2a. Set AGENT_PORT (e.g. 8012).
    from genie.agents.server import run_agent

    run_agent(RagAgent, META)
