"""Executor node: runs the Orchestrator's waves by invoking agents over A2A.

Executes each dependency wave concurrently, resolving ``${task.path}`` arg
references against upstream results, and writes every task's outcome (success or
error) to the shared blackboard for the Completion Gate to inspect.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
import mlflow
from langgraph.types import interrupt
from mlflow.entities import SpanType

from genie.a2a.client import A2AClient, A2AError
from genie.a2a.types import Message, Task, TaskState, get_data, get_text, task_final_message
from genie.agents.base import patch
from genie.memory.redis_store import get_redis_store
from genie.messaging import envelope
from genie.observability import Observable
from genie.platform.config import get_settings
from genie.application.blackboard import Blackboard
from genie.application.nodes._planner_dag import Plan, Subtask
from genie.registry.registry_client import RegistryUnavailable, get_registry_client
from genie.application.state import AgentState

# Matches an upstream-output reference like ${t1.text} or ${t1.view.items.0.id}.
_REF_RE = re.compile(r"\$\{([^}]+)\}")


class Executor(Observable):
    """Runs the Orchestrator's wave decomposition against discovered agents.

    Consumes ``state["waves"]`` (task ids per wave, produced by the
    Orchestrator) and runs each wave concurrently via ``asyncio.gather``;
    the next wave starts only after the current one's tasks have written to
    the shared blackboard. Each task is invoked by sending the agent an A2A
    JSON-RPC ``message/send`` (endpoint looked up via the Registry Service).
    Failures are captured per task — the gate decides whether to re-plan.
    """

    _traced_methods: tuple[str, ...] = ("run",)
    _component_kind: str = "executor"
    _span_type: str = SpanType.CHAIN

    def __init__(self) -> None:
        """Wire up the Registry client and the A2A client used to dispatch tasks to agents."""
        super().__init__()
        self._registry = get_registry_client()
        self._a2a = A2AClient(registry=self._registry)

    def run(self, state: AgentState) -> AgentState:
        """Execute ONE wave per invocation (tasks within the wave concurrently).

        Wave-per-invocation is what makes durable async dispatch safe: the node
        loops back on itself (see ``route_after_executor``) while waves remain,
        so every finished wave's blackboard results are committed to the
        checkpoint before the next wave starts. A bus task suspends the run via
        ``interrupt()`` — LangGraph then re-executes *this wave only* on resume,
        and the re-run dispatch is idempotent (deterministic correlation id +
        consumer dedup). Bus tasks dispatch **before** sync tasks so that on
        resume the interrupt returns immediately and sync tasks run exactly once.
        """
        plan = Plan(**(state.get("plan") or {}))
        by_id = plan.by_id()
        bb = Blackboard(
            thread_id=state.get("thread_id", ""),
            run_id=state.get("run_id", ""),
            tenant_id=state.get("tenant_id"),
        )
        wave_ids = state.get("waves") or []
        cursor = int(state.get("wave_cursor") or 0)

        if cursor == 0 and not state.get("bus_fallback"):
            # First wave of this (re)plan: seed from the previous attempt's
            # snapshot so successful tasks don't re-run on re-plan loops.
            snapshot = state.get("blackboard_snapshot") or {}
            for tid, entry in snapshot.items():
                if isinstance(entry, dict) and "error" not in entry:
                    bb._mem[tid] = entry
        else:
            # Later waves (or a fast-path fallback re-entry of the same wave):
            # carry forward everything earlier invocations committed.
            for tid, entry in (state.get("blackboard") or {}).items():
                if isinstance(entry, dict):
                    bb._mem[tid] = entry

        # Orchestrator could not decompose the DAG — surface the error on the
        # blackboard so the gate routes to a re-plan.
        plan_error = state.get("plan_error")
        if plan_error:
            return patch(state, blackboard={"_plan_error": {"error": plan_error}}, wave_cursor=len(wave_ids))

        if not plan.subtasks or not wave_ids or cursor >= len(wave_ids):
            self.log_event("executor.nothing_to_run")
            return patch(state, blackboard=bb.snapshot(), db_ops=self._redis_ops(bb), wave_cursor=len(wave_ids))

        wave = [by_id[tid] for tid in wave_ids[cursor] if tid in by_id]
        # Skip tasks already satisfied by snapshot / earlier attempt.
        todo = [t for t in wave if bb.get(t.id) is None or "error" in (bb.get(t.id) or {})]
        bus_todo = [t for t in todo if self._uses_bus(t, state)]
        sync_todo = [t for t in todo if t not in bus_todo]

        if bus_todo:
            # Parallel fan-out: produce EVERY bus request first, then suspend
            # ONCE for the whole group — the reply-router resumes when all of
            # the group's results (replies/timeouts/cancels) are in.
            group_id = f"{bb.run_id}:{cursor}"
            awaited = self._dispatch_bus_wave(bus_todo, cursor, bb, state, group_id)
            if awaited:
                results = interrupt({"group": group_id, "awaiting": awaited})
                for task in bus_todo:
                    if task.id in awaited:
                        self._apply_bus_result(task, bb, (results or {}).get(task.id))
        if sync_todo:
            self._run_async(self._run_wave(sync_todo, cursor, bb))

        # Fast path (transport="both"): a failed direct call falls back to the
        # bus by re-entering this SAME wave with those tasks forced onto it.
        # Never re-fallback a task that already came through the bus.
        prior_fallback = set(state.get("bus_fallback") or [])
        fallback = [
            t.id for t in sync_todo
            if t.id not in prior_fallback
            and "error" in (bb.get(t.id) or {})
            and self._bus_fallback_available(t)
        ]
        if fallback:
            self.log_event("executor.fast_path_fallback", tasks=fallback, wave=cursor)
            return patch(
                state, blackboard=bb.snapshot(), db_ops=self._redis_ops(bb),
                wave_cursor=cursor, bus_fallback=fallback,
            )

        return patch(
            state, blackboard=bb.snapshot(), db_ops=self._redis_ops(bb),
            wave_cursor=cursor + 1, bus_fallback=None,
        )

    # ------------------------------------------------------------------
    # Async (bus) dispatch
    # ------------------------------------------------------------------
    def _uses_bus(self, task: Subtask, state: AgentState) -> bool:
        """True when this subtask's agent is invoked over Kafka instead of HTTP.

        Requires the platform switch (``kafka_enabled``) AND either the agent
        opting in with ``transport="kafka"`` or this task being forced onto the
        bus by a fast-path fallback (``transport="both"``, direct call failed).
        Registry misses fall through to the sync path, which already records
        them as blackboard errors.
        """
        if not get_settings().kafka_enabled:
            return False
        if task.id in (state.get("bus_fallback") or []):
            return True
        try:
            meta = self._resolve_meta(task.agent_id)
        except RegistryUnavailable:
            return False
        return meta is not None and meta.transport == "kafka"

    def _bus_fallback_available(self, task: Subtask) -> bool:
        """Fast-path contract: a failed direct call falls back to the bus for ``both``."""
        if not get_settings().kafka_enabled:
            return False
        try:
            meta = self._resolve_meta(task.agent_id)
        except RegistryUnavailable:
            return False
        return meta is not None and meta.transport == "both"

    def _dispatch_bus_wave(
        self, tasks: list[Subtask], wave_idx: int, bb: Blackboard, state: AgentState, group_id: str
    ) -> dict[str, str]:
        """Produce every bus request of this wave; return ``{task_id: cid}`` awaited.

        Order is load-bearing per task: the client records the wait in
        ``a2a_awaiting`` **before** producing, and the correlation id is
        deterministic — so when LangGraph re-executes this after the interrupt,
        the upserts are no-ops and the duplicate produces are dropped by the
        agents' dedup. Tasks that can't resolve a registry entry become
        blackboard errors and are excluded from the awaited group.
        """
        awaited: dict[str, str] = {}
        for task in tasks:
            try:
                meta = self._resolve_meta(task.agent_id)
            except RegistryUnavailable as e:
                self._write_error(bb, task.id, f"registry unavailable: {e}")
                continue
            if meta is None:
                self._write_error(bb, task.id, f"agent_id '{task.agent_id}' not in registry")
                continue

            attempt = 1  # the retry ladder (extend → retry → DLQ) is the Supervisor's job
            cid = envelope.correlation_id(bb.run_id, task.id, attempt)
            context = {
                "task_id": task.id,
                "thread_id": bb.thread_id,
                "run_id": bb.run_id,
                "blackboard": bb.snapshot(),
                "correlation_id": cid,
                "tenant_id": state.get("tenant_id"),
                "trace_id": bb.run_id,
            }
            resolved_args = self._resolve_args(task.args or {}, bb)
            self._run_async(
                self._a2a.send_via_bus(
                    task.agent_id, resolved_args, context, attempt=attempt, group_id=group_id
                )
            )
            awaited[task.id] = cid
            self.log_event(
                "executor.bus_dispatched",
                task=task.id, agent=task.agent_id, cid=cid, wave=wave_idx, group=group_id,
            )
        return awaited

    def _apply_bus_result(self, task: Subtask, bb: Blackboard, payload: Any) -> None:
        """Write one task's bus result (from the group resume payload) to the blackboard.

        ``{"task": <terminal Task dump>}`` from the reply-router, or
        ``{"error": <reason>}`` from the deadline sweep / Supervisor / a cancel.
        Failed or malformed replies become blackboard errors — the Gate then
        decides whether to re-plan, identical to the sync path's failure contract.
        """
        if payload is None:
            self._write_error(bb, task.id, "bus group resumed without a result for this task")
            return
        if not isinstance(payload, dict) or not (payload.get("task") or payload.get("error")):
            self._write_error(bb, task.id, f"malformed bus resume payload: {payload!r}")
            return
        if payload.get("error"):
            self._write_error(bb, task.id, str(payload["error"]))
            return
        try:
            t = Task.model_validate(payload["task"])
            if t.status.state in (TaskState.failed, TaskState.canceled, TaskState.rejected):
                detail = get_text(t.status.message) if t.status.message else t.status.state.value
                self._write_error(bb, task.id, f"agent task {t.status.state.value}: {detail}")
                return
            reply = task_final_message(t)
        except Exception as e:
            self._write_error(bb, task.id, f"invalid bus reply: {e}")
            return
        self._run_async(bb.write(task.id, self._map_response(task, reply)))
        self.log_event("executor.bus_completed", task=task.id, agent=task.agent_id)

    def _write_error(self, bb: Blackboard, task_id: str, message: str) -> None:
        """Record a task failure on the blackboard from this synchronous node."""
        self.log("warning", "executor.task_error", task=task_id, error=message)
        self._run_async(bb.write_error(task_id, message))

    # ------------------------------------------------------------------
    @staticmethod
    def _redis_ops(bb: Blackboard) -> list[dict]:
        """Tracer op records for the real blackboard → Redis mirror writes."""
        enabled = get_redis_store().enabled
        ops: list[dict] = []
        for tid, entry in bb.snapshot().items():
            key = bb.key(tid)
            is_err = isinstance(entry, dict) and "error" in entry
            ops.append({
                "store": "redis",
                "op": "write",
                "keys": [key],
                "detail": f"{tid} {'error' if is_err else 'summary'}",
                "code": f"SETEX {key} 3600 {{...}}",
                "enabled": enabled,
            })
        if not ops:
            ops.append({"store": "redis", "op": "write", "detail": "no tasks written", "enabled": enabled})
        return ops

    # ------------------------------------------------------------------
    async def _run_wave(self, tasks: list[Subtask], wave_idx: int, bb: Blackboard) -> None:
        """Run one wave's tasks concurrently over a shared HTTP client."""
        self.log_event("executor.wave_start", wave=wave_idx, count=len(tasks))
        async with httpx.AsyncClient() as http:
            await asyncio.gather(
                *(self._run_task(t, wave_idx, bb, http) for t in tasks),
                return_exceptions=False,
            )
        self.log_event("executor.wave_done", wave=wave_idx)

    def _resolve_meta(self, agent_id: str):
        """Look up an agent's live record, refreshing the cache once on a miss."""
        meta = self._registry.get(agent_id)
        if meta is None:
            self._registry.invalidate()
            meta = self._registry.get(agent_id)
        return meta

    # ------------------------------------------------------------------
    # Runtime data-passing: resolve ${task_id.path} arg references against the
    # blackboard so a dependent task consumes an upstream task's output.
    # ------------------------------------------------------------------
    def _lookup_ref(self, ref: str, bb: Blackboard) -> Any:
        """Resolve a dotted ref like 't1.text' or 't1.view.items.0.id' from the blackboard."""
        parts = ref.strip().split(".")
        cur: Any = bb.get(parts[0])  # the upstream entry: {agent_id, text, view?}
        for p in parts[1:]:
            if isinstance(cur, dict):
                cur = cur.get(p)
            elif isinstance(cur, list):
                try:
                    cur = cur[int(p)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return cur

    def _resolve_args(self, value: Any, bb: Blackboard) -> Any:
        """Recursively resolve ${...} references in a task's args.

        A string that is exactly one ${ref} yields the raw typed value (an id
        stays an int); embedded refs are string-interpolated. Unresolved refs
        are left as the literal ${...} and logged, so a missing/errored upstream
        produces a visibly-unresolved arg rather than crashing the wave.
        """
        if isinstance(value, dict):
            return {k: self._resolve_args(v, bb) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_args(v, bb) for v in value]
        if not isinstance(value, str):
            return value

        full = _REF_RE.fullmatch(value.strip())
        if full:
            resolved = self._lookup_ref(full.group(1), bb)
            if resolved is None:
                self.log("warning", "executor.ref_unresolved", ref=full.group(1))
                return value
            return resolved

        def _sub(m: "re.Match") -> str:
            """Replace one ``${...}`` reference with its blackboard value; leave it intact (with a warning) if unresolved."""
            resolved = self._lookup_ref(m.group(1), bb)
            if resolved is None:
                self.log("warning", "executor.ref_unresolved", ref=m.group(1))
                return m.group(0)
            return str(resolved)

        return _REF_RE.sub(_sub, value)

    async def _run_task(
        self, task: Subtask, wave_idx: int, bb: Blackboard, http: httpx.AsyncClient
    ) -> None:
        """Dispatch one subtask via A2A (one retry), writing its result or error to the blackboard."""
        try:
            meta = self._resolve_meta(task.agent_id)
        except RegistryUnavailable as e:
            await bb.write_error(task.id, f"registry unavailable: {e}")
            return
        if meta is None:
            await bb.write_error(task.id, f"agent_id '{task.agent_id}' not in registry")
            return

        # Resolve ${tN.path} references against upstream results before dispatch.
        resolved_args = self._resolve_args(task.args or {}, bb)

        context = {
            "task_id": task.id,
            "thread_id": bb.thread_id,
            "run_id": bb.run_id,
            "blackboard": bb.snapshot(),
        }

        with mlflow.start_span(name=f"agent.{task.agent_id}.{task.id}", span_type=SpanType.AGENT) as span:
            span.set_attribute("agent.id", task.agent_id)
            span.set_attribute("agent.version", meta.version)
            span.set_attribute("agent.endpoint", str(meta.endpoint))
            span.set_attribute("agent.transport", "a2a/json-rpc")
            span.set_attribute("a2a.protocol", "1.2")
            span.set_attribute("exec.wave", wave_idx)
            span.set_attribute("task.id", task.id)
            span.set_inputs({"args": resolved_args, "args_template": task.args})

            last_error: str | None = None
            for attempt in range(2):  # 1 retry for MVP
                try:
                    reply = await self._a2a.send(
                        task.agent_id, resolved_args, context, sla_ms=meta.sla_ms, http=http
                    )
                    payload = self._map_response(task, reply)
                    await bb.write(task.id, payload)
                    span.set_attribute("retry.count", attempt)
                    span.set_outputs({"payload_keys": list(payload.keys())})
                    return
                except httpx.TimeoutException:
                    last_error = f"agent timed out after {meta.sla_ms}ms"
                    self.log("warning", "executor.task_retry", task=task.id, attempt=attempt, error=last_error)
                except A2AError as e:
                    last_error = f"a2a error: {e}"
                    self.log("warning", "executor.task_retry", task=task.id, attempt=attempt, error=last_error)
                except Exception as e:
                    last_error = str(e)
                    self.log("warning", "executor.task_retry", task=task.id, attempt=attempt, error=last_error)
            span.set_attribute("retry.count", 1)
            span.set_attribute("status", "error")
            await bb.write_error(task.id, last_error or "unknown failure")

    # ------------------------------------------------------------------
    @staticmethod
    def _map_response(task: Subtask, reply: Message) -> dict[str, Any]:
        """Map an agent's A2A reply Message to the blackboard payload contract.

        The Executor re-adds ``agent_id`` so the blackboard shape the
        gate/synthesizer read is identical to the pre-A2A contract:
        ``{agent_id, text, view?}``. ``text`` comes from the reply's TextPart(s);
        an optional structured ``view`` rides in a DataPart.
        """
        payload: dict[str, Any] = {"agent_id": task.agent_id, "text": get_text(reply)}
        view = (get_data(reply) or {}).get("view")
        if view:
            payload["view"] = view
        return payload
