"""Registry-aware A2A client.

A single client used by **both** the Executor and (via ``BaseAgent.call_peer``)
peer agents: it resolves a target agent through the central **Registry**, then
sends it a JSON-RPC ``message/send`` over HTTP. This is the "hybrid" in A2A
Hybrid — formal A2A messaging on top of centralized registry discovery.

Two transports:

* :meth:`A2AClient.send` — synchronous JSON-RPC over HTTP. Used by the Executor
  for ``transport="json-rpc"`` agents and **always** by ``BaseAgent.call_peer``
  (an agent is not a graph and cannot suspend, so peer calls stay blocking).
* :meth:`A2AClient.send_via_bus` — fire-and-forget Kafka produce for
  ``transport="kafka"`` agents. Only the Executor uses it: it returns the
  correlation id immediately and the run suspends until the gateway's
  reply-router resumes it with the agent's reply.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from genie.a2a.agent_card import a2a_url
from genie.platform.config import get_settings
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    METHOD_MESSAGE_STREAM,
    Message,
    Role,
    Task,
    TaskState,
    data_part,
    get_text,
    task_final_message,
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

    @staticmethod
    def _build_message(agent_id: str, args: dict | None, ctx: dict, sla_ms: int) -> Message:
        """The a2a-sdk request Message shared by both transports (HTTP + bus)."""
        return Message(
            role=Role.user,
            message_id=uuid.uuid4().hex,
            task_id=ctx.get("task_id"),
            context_id=ctx.get("thread_id"),
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

    def _build_request(
        self,
        agent_id: str,
        args: dict | None,
        context: dict,
        sla_ms: int,
        *,
        method: str = METHOD_MESSAGE_SEND,
    ) -> dict:
        """Wrap args + invocation context into a JSON-RPC request (``message/send`` or ``message/stream``).

        The a2a-sdk Message serializes to the camelCase wire form via ``by_alias``.
        """
        ctx = dict(context or {})
        message = self._build_message(agent_id, args, ctx, sla_ms)
        return {
            "jsonrpc": "2.0",
            "id": ctx.get("task_id") or uuid.uuid4().hex,
            "method": method,
            "params": {"message": message.model_dump(mode="json", by_alias=True, exclude_none=True)},
        }

    @staticmethod
    def _parse_response(data: Any) -> Message:
        """Unwrap a JSON-RPC response to its reply Message, raising on any error.

        A2A ``message/send`` returns either a ``Message`` or a ``Task``. A completed
        Task is unwrapped to its final Message (via :func:`task_final_message`); a
        ``failed``/``canceled``/``rejected`` Task is surfaced as an :class:`A2AError`
        so the caller's existing error/retry handling (Executor, ``call_peer``) is
        preserved exactly.
        """
        error = data.get("error")
        if error:
            raise A2AError(error.get("message", "A2A error"), code=error.get("code"))
        result = data.get("result")
        if not result:
            raise A2AError("A2A response had neither result nor error")
        if result.get("kind") == "task":
            task = Task.model_validate(result)
            if task.status.state in (TaskState.failed, TaskState.canceled, TaskState.rejected):
                detail = get_text(task.status.message) if task.status.message else task.status.state.value
                raise A2AError(f"agent task {task.status.state.value}: {detail}")
            return task_final_message(task)
        return Message.model_validate(result)

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
        payload = self._build_request(agent_id, args, context, sla_ms)
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

    # ------------------------------------------------------------------
    async def send_via_bus(
        self,
        agent_id: str,
        args: dict | None,
        context: dict,
        *,
        attempt: int = 1,
        deadline_ms: int | None = None,
        group_id: str | None = None,
        broker=None,
    ) -> str:
        """Produce a request to ``agent_id``'s Kafka inbox; return its correlation id.

        Fire-and-forget: the caller (the Executor — nothing else should use this)
        suspends the run and the gateway's reply-router resumes it when every
        reply of the dispatch **group** has landed. The correlation id is
        **deterministic** (``uuid5(run_id:task_id:attempt)``), so the re-produce
        that happens when LangGraph re-executes the interrupted node is a
        duplicate the agent's inbox dedup drops — never a second execution.

        This method also owns the durable ``a2a_awaiting`` record — written
        (idempotent upsert, **before** the produce so a fast reply can never
        race an unrecorded wait) with the serialized request + inbox topic, which
        is what lets the Supervisor retry and dead-letter without re-reading
        Kafka. Produced ``oneshot`` because the Executor runs on transient loops.
        """
        from datetime import datetime, timedelta, timezone

        from genie.messaging import envelope, get_awaiting_store, get_broker

        meta = self._registry.get(agent_id)
        if meta is None:
            self._registry.invalidate()
            meta = self._registry.get(agent_id)
        if meta is None:
            raise A2AError(f"agent_id '{agent_id}' not in registry")

        settings = get_settings()
        ctx = dict(context or {})
        deadline_ms = deadline_ms or settings.a2a_default_deadline_ms
        cid = ctx.get("correlation_id") or envelope.correlation_id(
            ctx.get("run_id") or "", ctx.get("task_id") or "", attempt
        )
        ctx["correlation_id"] = cid
        group_id = group_id or cid  # a lone dispatch is a group of one
        message = self._build_message(agent_id, args, ctx, deadline_ms)
        value = message.model_dump_json(by_alias=True, exclude_none=True)
        headers = {
            envelope.HDR_KIND: envelope.KIND_REQUEST,
            envelope.HDR_CORRELATION_ID: cid,
            envelope.HDR_ATTEMPT: str(attempt),
            envelope.HDR_GROUP_ID: group_id,
            envelope.HDR_FROM: "executor",
            envelope.HDR_TO: agent_id,
            envelope.HDR_REPLY_TO: envelope.reply_topic(),
            envelope.HDR_THREAD_ID: ctx.get("thread_id") or "",
            envelope.HDR_RUN_ID: ctx.get("run_id") or "",
            envelope.HDR_TASK_ID: ctx.get("task_id") or "",
            envelope.HDR_TRACE_ID: ctx.get("trace_id") or ctx.get("run_id") or "",
            envelope.HDR_TENANT_ID: ctx.get("tenant_id") or "",
            envelope.HDR_DEADLINE: envelope.deadline_from_now(deadline_ms),
        }
        topic = meta.inbox_topic or envelope.inbox_topic(agent_id)

        get_awaiting_store().put(
            cid,
            thread_id=ctx.get("thread_id") or "",
            run_id=ctx.get("run_id") or "",
            task_id=ctx.get("task_id") or "",
            agent_id=agent_id,
            deadline=datetime.now(timezone.utc) + timedelta(milliseconds=deadline_ms),
            group_id=group_id,
            attempt=attempt,
            request=value,
            inbox_topic=topic,
            tenant_id=ctx.get("tenant_id") or None,
        )
        await (broker or get_broker()).produce(
            topic,
            value=value.encode("utf-8"),
            key=ctx.get("thread_id") or None,
            headers=headers,
            oneshot=True,
        )
        return cid

    # ------------------------------------------------------------------
    async def stream(
        self,
        agent_id: str,
        args: dict | None,
        context: dict,
        *,
        sla_ms: int,
    ) -> AsyncIterator[dict]:
        """Open an A2A ``message/stream`` (SSE) and yield each event's ``result``.

        Yields the parsed JSON-RPC ``result`` object of every server-sent frame
        (a ``Task`` then ``status-update``/``artifact-update`` events, ending on
        a ``final`` status). Provided for external/streaming consumers; the
        platform graph uses :meth:`send` and is unaffected. Raises
        :class:`A2AError` on transport failure or a JSON-RPC error frame.
        """
        url = self._resolve_url(agent_id)
        payload = self._build_request(agent_id, args, context, sla_ms, method=METHOD_MESSAGE_STREAM)
        headers = {**self._headers(), "Accept": "text/event-stream"}
        timeout = httpx.Timeout(sla_ms / 1000.0)
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    frame = json.loads(line[len("data:"):].strip())
                    if frame.get("error"):
                        raise A2AError(frame["error"].get("message", "stream error"), code=frame["error"].get("code"))
                    if frame.get("result") is not None:
                        yield frame["result"]
