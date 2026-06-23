"""Registry-aware A2A client.

A single client used by **both** the Executor and (via ``BaseAgent.call_peer``)
peer agents: it resolves a target agent through the central **Registry**, then
sends it a JSON-RPC ``message/send`` over HTTP. This is the "hybrid" in A2A
Hybrid — formal A2A messaging on top of centralized registry discovery.

Transport is synchronous JSON-RPC only for now. ``AgentMeta.transport`` is left
intact so an async (e.g. Kafka) transport can be selected here later.
"""
from __future__ import annotations

import uuid
from typing import Any

import httpx

from genie.a2a.agent_card import a2a_url
from genie.platform.config import get_settings
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    data_part,
)
from genie.registry.registry_client import RegistryClient, get_registry_client


class A2AError(RuntimeError):
    """Raised on transport failure or a JSON-RPC/agent error response."""

    def __init__(self, message: str, code: int | None = None) -> None:
        """Store the human-readable message plus an optional JSON-RPC error ``code``."""
        super().__init__(message)
        self.code = code


class A2AClient:
    """Resolve an agent via the Registry and send it an A2A ``message/send``."""

    def __init__(self, registry: RegistryClient | None = None) -> None:
        """Use the given Registry client, or the process-wide one when omitted."""
        self._registry = registry or get_registry_client()

    # ------------------------------------------------------------------
    def _resolve_url(self, agent_id: str) -> str:
        """Discover the target's A2A URL via the Registry (one refresh on miss)."""
        meta = self._registry.get(agent_id)
        if meta is None:
            self._registry.invalidate()
            meta = self._registry.get(agent_id)
        if meta is None:
            raise A2AError(f"agent_id '{agent_id}' not in registry")
        if not meta.endpoint:
            raise A2AError(f"agent '{agent_id}' has no endpoint registered")
        return a2a_url(meta.endpoint)

    @staticmethod
    def _headers() -> dict:
        """Bearer auth header when AGENT_INVOKE_TOKEN is set, else no auth."""
        token = get_settings().agent_invoke_token
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _build_request(
        self, agent_id: str, args: dict | None, context: dict, sla_ms: int
    ) -> JsonRpcRequest:
        """Wrap args + invocation context into a JSON-RPC ``message/send`` request."""
        ctx = dict(context or {})
        message = Message(
            role="user",
            messageId=uuid.uuid4().hex,
            taskId=ctx.get("task_id"),
            contextId=ctx.get("thread_id"),
            parts=[data_part({"args": args or {}})],
            metadata={
                "agent_id": agent_id,
                "task_id": ctx.get("task_id"),
                "run_id": ctx.get("run_id"),
                "thread_id": ctx.get("thread_id"),
                "correlation_id": ctx.get("correlation_id") or uuid.uuid4().hex,
                "blackboard": ctx.get("blackboard") or {},
                "sla_ms": sla_ms,
            },
        )
        return JsonRpcRequest(
            id=ctx.get("task_id") or uuid.uuid4().hex,
            method=METHOD_MESSAGE_SEND,
            params={"message": message.model_dump(mode="json")},
        )

    @staticmethod
    def _parse_response(data: Any) -> Message:
        """Unwrap a JSON-RPC response to its Message result, raising on any error."""
        rpc = JsonRpcResponse.model_validate(data)
        if rpc.error is not None:
            raise A2AError(rpc.error.message, code=rpc.error.code)
        if not rpc.result:
            raise A2AError("A2A response had neither result nor error")
        return Message.model_validate(rpc.result)

    # ------------------------------------------------------------------
    async def send(
        self,
        agent_id: str,
        args: dict | None,
        context: dict,
        *,
        sla_ms: int,
        http: httpx.AsyncClient | None = None,
    ) -> Message:
        """Send a JSON-RPC ``message/send`` to ``agent_id`` and return its reply.

        Raises :class:`A2AError` on any transport, JSON-RPC, or agent error so the
        caller (Executor / peer agent) can decide how to record the failure.
        """
        url = self._resolve_url(agent_id)
        req = self._build_request(agent_id, args, context, sla_ms)
        payload = req.model_dump(mode="json")
        timeout = httpx.Timeout(sla_ms / 1000.0)

        async def _post(client: httpx.AsyncClient) -> Message:
            """POST the JSON-RPC payload on ``client`` and parse the reply into a Message."""
            resp = await client.post(url, json=payload, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            return self._parse_response(resp.json())

        if http is not None:
            return await _post(http)
        async with httpx.AsyncClient() as client:
            return await _post(client)
