from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FieldSpec(BaseModel):
    """One field in an agent's input or output schema."""

    type: Literal["string", "integer", "number", "boolean", "object", "array"] = "string"
    required: bool = False
    description: str = ""
    persist: bool = False  # Synthesizer commits to Postgres when True


class AgentMeta(BaseModel):
    """Registry record for one agent.

    Mirrors the registry contract in docs/PLAN_PLANNER_ORCHESTRATOR.md.
    """

    agent_id: str
    version: str = "1.0.0"
    capability_tags: list[str] = Field(default_factory=list)
    description: str = ""
    input_schema: dict[str, FieldSpec] = Field(default_factory=dict)
    output_schema: dict[str, FieldSpec] = Field(default_factory=dict)
    sla_ms: int = 10000
    transport: Literal["json-rpc", "kafka", "both"] = "json-rpc"
    status: Literal["active", "deprecated"] = "active"
    changelog_url: str | None = None

    def validate_args(self, args: dict) -> tuple[bool, str]:
        """Lightweight required-field check. Type coercion is intentionally lenient."""
        for name, spec in self.input_schema.items():
            if spec.required and (name not in args or args[name] in (None, "")):
                return False, f"missing required input '{name}'"
        return True, ""
