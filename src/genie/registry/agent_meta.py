"""Registry data models: the ``AgentMeta`` record an agent advertises.

Defines the schema an agent publishes to the Registry (capabilities, A2A skills,
input/output field specs, liveness fields) plus the supporting ``FieldSpec`` and
``Skill`` models. Skills are auto-derived from ``capability_tags`` when an agent
doesn't set them explicitly, keeping the registry record and the A2A Agent Card
in lockstep.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FieldSpec(BaseModel):
    """One field in an agent's input or output schema."""

    type: Literal["string", "integer", "number", "boolean", "object", "array"] = "string"
    required: bool = False
    description: str = ""
    persist: bool = False  # Synthesizer commits this field to MongoDB (agent_commits) when True


class Skill(BaseModel):
    """One capability an agent advertises, aligned with the A2A ``AgentSkill``.

    Carried on the registry record so the stored agent entry matches the A2A
    Agent Card served at ``/.well-known/agent.json`` (see a2a/agent_card.py).
    """

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] | None = None


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

    # A2A ``capabilities.streaming``. The harness serves ``message/stream`` for
    # every agent, so this defaults True; an agent can set it False to hide the
    # streaming endpoint — the card advertises the flag and the harness gates the
    # endpoint on it, so the two never disagree.
    supports_streaming: bool = True

    # A2A-aligned skills. Left empty by an agent, they are auto-derived from
    # capability_tags + description + input_schema (see ``_ensure_skills``) so the
    # registry record always carries the same skills the Agent Card exposes. An
    # agent may set them explicitly for a richer, multi-skill advertisement.
    skills: list[Skill] = Field(default_factory=list)

    # --- Remote operation / discovery (populated when the agent runs as a service) ---
    # Base URL the Executor POSTs to; the A2A "/a2a" path is appended by convention.
    endpoint: str | None = None
    # Unique per agent process. The registry service assigns a uuid4 if absent.
    instance_id: str | None = None
    # Server-owned liveness fields — stamped by the registry service, not the agent.
    last_heartbeat: datetime | None = None
    registered_at: datetime | None = None

    @model_validator(mode="after")
    def _ensure_skills(self) -> "AgentMeta":
        """Guarantee at least one skill, derived to match the Agent Card.

        Runs on every construction — including when the registry deserializes a
        stored doc — so even records registered before ``skills`` existed surface
        a skill on read. Explicitly-provided skills are kept as-is.
        """
        if not self.skills:
            self.skills = [self._derived_skill()]
        return self

    def _derived_skill(self) -> Skill:
        """A single skill summarizing this agent: tags + an input-shape note.

        Mirrors the projection a2a/agent_card.py applies, so the registry record
        and the served Agent Card advertise identical skills.
        """
        required = [name for name, spec in self.input_schema.items() if spec.required]
        optional = [name for name, spec in self.input_schema.items() if not spec.required]
        parts = []
        if required:
            parts.append("requires " + ", ".join(required))
        if optional:
            parts.append("optional " + ", ".join(optional))
        io_note = "; ".join(parts) if parts else "no inputs"
        return Skill(
            id=self.agent_id,
            name=self.agent_id,
            description=f"{self.description or '(no description)'} ({io_note})".strip(),
            tags=list(self.capability_tags),
        )

    def validate_args(self, args: dict) -> tuple[bool, str]:
        """Lightweight required-field check. Type coercion is intentionally lenient."""
        for name, spec in self.input_schema.items():
            if spec.required and (name not in args or args[name] in (None, "")):
                return False, f"missing required input '{name}'"
        return True, ""
