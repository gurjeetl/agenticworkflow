"""Minimal, spec-aligned A2A protocol types.

A pragmatic subset of the Agent2Agent (A2A) protocol at **v1.2** — enough for
synchronous JSON-RPC ``message/send``, the ``Task`` lifecycle, ``message/stream``
(SSE) and Agent Card discovery. Field names mirror the A2A spec (``kind``,
``parts``, ``role``, ``messageId``, ``jsonrpc``, ``method``, ``params``,
``result``, ``error``, ``status``, ``artifacts``) so the wire format is
interoperable with other A2A implementations (e.g. the A2A Inspector).

Deliberately omitted (not in scope): signed Agent Cards (JWS ``signatures``),
gRPC transport, multi-tenant hosting, push notifications, and file parts.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

# --- Protocol version -------------------------------------------------------
# Single source of truth for the advertised A2A protocol version.
PROTOCOL_VERSION = "1.2"

# --- JSON-RPC method names --------------------------------------------------
METHOD_MESSAGE_SEND = "message/send"
METHOD_MESSAGE_STREAM = "message/stream"
METHOD_TASKS_GET = "tasks/get"
METHOD_TASKS_CANCEL = "tasks/cancel"

# --- JSON-RPC error codes ---------------------------------------------------
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_AGENT_EXECUTION = -32001  # custom: the agent ran but returned an error
ERR_TASK_NOT_FOUND = -32002   # A2A: referenced task id is unknown


# --- Message parts ----------------------------------------------------------
class TextPart(BaseModel):
    """A message part carrying free-text content."""

    kind: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    """A message part carrying a structured JSON object (e.g. args or a view)."""

    kind: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None


Part = Annotated[Union[TextPart, DataPart], Field(discriminator="kind")]


class Message(BaseModel):
    """An A2A message: an ordered list of parts with a role and free metadata.

    We carry invocation context (run_id, task_id, blackboard, sla_ms, ...) in
    ``metadata`` and structured args / views in :class:`DataPart`s.
    """

    kind: Literal["message"] = "message"
    role: Literal["user", "agent"]
    parts: list[Part] = Field(default_factory=list)
    messageId: str
    taskId: str | None = None
    contextId: str | None = None
    metadata: dict[str, Any] | None = None


# --- Agent Card (discovery) -------------------------------------------------
class AgentCapabilities(BaseModel):
    """Optional A2A protocol features an agent supports.

    ``streaming`` (``message/stream`` SSE) is supported by the harness;
    ``pushNotifications`` is not. ``stateTransitionHistory``/``extensions`` are
    1.x fields kept at their conservative defaults.
    """

    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False
    extensions: list[dict[str, Any]] | None = None


class AgentSkill(BaseModel):
    """One advertised capability on an Agent Card (A2A ``AgentSkill``)."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] | None = None
    inputModes: list[str] | None = None
    outputModes: list[str] | None = None


class AgentProvider(BaseModel):
    """The organization that publishes an agent (A2A ``AgentProvider``)."""

    organization: str
    url: str = ""


class AgentInterface(BaseModel):
    """One transport interface an agent exposes (A2A ``AgentInterface``).

    ``additionalInterfaces`` lists these so a client can pick a transport; we
    advertise only ``JSONRPC`` today, but the shape lets a gRPC/HTTP+JSON
    interface be appended later without a breaking change.
    """

    url: str
    transport: str = "JSONRPC"


class AgentCard(BaseModel):
    """A2A v1.2 discovery document served at ``/.well-known/agent-card.json``."""

    name: str
    description: str = ""
    url: str
    version: str = "1.0.0"
    protocolVersion: str = PROTOCOL_VERSION
    preferredTransport: str = "JSONRPC"
    additionalInterfaces: list[AgentInterface] = Field(default_factory=list)
    provider: AgentProvider | None = None
    iconUrl: str | None = None
    documentationUrl: str | None = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    # Auth advertisement — populated only when the agent enforces a token, so
    # unauthenticated agents keep serving a card with no security section.
    securitySchemes: dict[str, dict[str, Any]] | None = None
    security: list[dict[str, list[str]]] | None = None
    supportsAuthenticatedExtendedCard: bool = False
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text", "data"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text", "data"])
    skills: list[AgentSkill] = Field(default_factory=list)


# --- Task lifecycle ---------------------------------------------------------
class TaskState(str, Enum):
    """A2A task lifecycle states."""

    submitted = "submitted"
    working = "working"
    input_required = "input-required"
    completed = "completed"
    canceled = "canceled"
    failed = "failed"
    rejected = "rejected"
    unknown = "unknown"


class TaskStatus(BaseModel):
    """The current state of a Task, with an optional agent message + timestamp."""

    state: TaskState
    message: Message | None = None
    timestamp: str | None = None


class Artifact(BaseModel):
    """A structured output produced by a Task (A2A ``Artifact``)."""

    artifactId: str
    name: str | None = None
    description: str | None = None
    parts: list[Part] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class Task(BaseModel):
    """An A2A Task: the stateful unit ``message/send``/``message/stream`` return."""

    kind: Literal["task"] = "task"
    id: str
    contextId: str | None = None
    status: TaskStatus
    history: list[Message] | None = None
    artifacts: list[Artifact] | None = None
    metadata: dict[str, Any] | None = None


class TaskStatusUpdateEvent(BaseModel):
    """SSE event: a Task changed state (``final`` marks the terminal event)."""

    kind: Literal["status-update"] = "status-update"
    taskId: str
    contextId: str | None = None
    status: TaskStatus
    final: bool = False
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(BaseModel):
    """SSE event: a Task emitted (or appended to) an artifact."""

    kind: Literal["artifact-update"] = "artifact-update"
    taskId: str
    contextId: str | None = None
    artifact: Artifact
    append: bool = False
    lastChunk: bool = False
    metadata: dict[str, Any] | None = None


# --- JSON-RPC envelopes -----------------------------------------------------
class JsonRpcError(BaseModel):
    """The ``error`` member of a JSON-RPC response."""

    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcRequest(BaseModel):
    """A JSON-RPC 2.0 request envelope (e.g. ``method='message/send'``)."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """A JSON-RPC 2.0 response envelope carrying exactly one of result/error."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


# --- Helpers ----------------------------------------------------------------
def text_part(text: str) -> TextPart:
    """Convenience constructor for a :class:`TextPart`."""
    return TextPart(text=text)


def data_part(data: dict[str, Any]) -> DataPart:
    """Convenience constructor for a :class:`DataPart`."""
    return DataPart(data=data)


def get_text(message: Message) -> str:
    """Concatenate the text of every TextPart in the message."""
    chunks = [p.text for p in message.parts if isinstance(p, TextPart)]
    return "\n".join(c for c in chunks if c)


def get_data(message: Message) -> dict[str, Any]:
    """Merge the ``data`` of every DataPart in the message (later parts win)."""
    merged: dict[str, Any] = {}
    for p in message.parts:
        if isinstance(p, DataPart):
            merged.update(p.data)
    return merged


def task_final_message(task: Task) -> Message:
    """Extract the agent's reply Message from a completed :class:`Task`.

    Prefers ``status.message``; falls back to assembling a Message from the
    task's artifact parts so a caller (e.g. the internal :class:`A2AClient`)
    always gets a :class:`Message` regardless of how the agent packaged its
    output. Raises ``ValueError`` if the task carries no recoverable content.
    """
    if task.status.message is not None:
        return task.status.message
    parts: list[Part] = []
    for artifact in task.artifacts or []:
        parts.extend(artifact.parts)
    if not parts:
        raise ValueError(f"task '{task.id}' has no message or artifact content")
    return Message(role="agent", messageId=task.id, parts=parts)
