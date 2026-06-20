"""Launch manifest for the demo application's agents.

Unlike the reference platform's in-process provider injection, agents in this
framework run as their own A2A services (each ``agent.py`` has a ``__main__`` that
calls ``genie.agents.server.run_agent`` and self-registers with the Registry).
This manifest is therefore a *catalog* for launch tooling, not something wired
into ``create_app``. Each entry is ``(AgentClass, META, default_port)``.
"""
from __future__ import annotations

from applications.demo.outage.agent import META as OUTAGE_META, OutageAgent
from applications.demo.rag.agent import META as RAG_META, RagAgent
from applications.demo.weather.agent import META as WEATHER_META, WeatherAgent

AGENT_PROVIDERS = [
    (WeatherAgent, WEATHER_META, 8010),
    (OutageAgent, OUTAGE_META, 8011),
    (RagAgent, RAG_META, 8012),
]

__all__ = ["AGENT_PROVIDERS"]
