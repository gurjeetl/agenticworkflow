"""Request/response models for the Registry/Discovery Service.

These define the wire contract between agent processes (which register and
heartbeat) and the Planner/Executor (which discover agents). The agent payload
itself is the shared :class:`AgentMeta` pydantic model so there is no schema
drift between what an agent advertises and what consumers deserialize.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from genie.registry.agent_meta import AgentMeta


class RegisterRequest(BaseModel):
    """Agent → registry: register (or refresh) one instance via its AgentMeta."""

    meta: AgentMeta


class RegisterResponse(BaseModel):
    """Registry → agent: assigned instance_id plus the liveness/heartbeat cadence."""

    instance_id: str
    ttl_seconds: int
    heartbeat_interval_seconds: int


class HeartbeatRequest(BaseModel):
    """Agent → registry: keep one instance alive, optionally updating its status."""

    instance_id: str
    status: Literal["active", "deprecated"] | None = None


class HeartbeatResponse(BaseModel):
    """Registry → agent: heartbeat result; ``known`` is False if the instance is gone."""

    ok: bool
    # False when the registry has no record for this instance_id — the agent
    # harness treats this as a signal to re-register (e.g. after a TTL sweep).
    known: bool


class DeregisterRequest(BaseModel):
    """Agent → registry: remove one instance (graceful shutdown)."""

    instance_id: str


class ListResponse(BaseModel):
    """Registry → consumer: the set of currently live + active agents."""

    agents: list[AgentMeta] = Field(default_factory=list)
