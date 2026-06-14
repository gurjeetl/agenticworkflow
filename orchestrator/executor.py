from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
import mlflow
from mlflow.entities import SpanType

from a2a.client import A2AClient, A2AError
from a2a.types import Message, get_data, get_text
from baseagent.base_agent import patch
from memory.redis_store import get_redis_store
from observability import Observable
from orchestrator.blackboard import Blackboard
from planner.dag import Plan, Subtask
from registry.registry_client import RegistryUnavailable, get_registry_client
from state import AgentState

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
        super().__init__()
        self._registry = get_registry_client()
        self._a2a = A2AClient(registry=self._registry)

    def run(self, state: AgentState) -> AgentState:
        plan = Plan(**(state.get("plan") or {}))
        by_id = plan.by_id()
        bb = Blackboard(thread_id=state.get("thread_id", ""), run_id=state.get("run_id", ""))

        # Seed blackboard from previous attempt's snapshot so successful tasks
        # don't re-run on re-plan loops.
        snapshot = state.get("blackboard_snapshot") or {}
        for tid, entry in snapshot.items():
            if isinstance(entry, dict) and "error" not in entry:
                bb._mem[tid] = entry

        # Orchestrator could not decompose the DAG — surface the error on the
        # blackboard so the gate routes to a re-plan.
        plan_error = state.get("plan_error")
        if plan_error:
            return patch(state, blackboard={"_plan_error": {"error": plan_error}})

        wave_ids = state.get("waves") or []
        if not plan.subtasks or not wave_ids:
            self.log_event("executor.nothing_to_run")
            return patch(state, blackboard=bb.snapshot(), db_ops=self._redis_ops(bb))

        for wave_idx, ids in enumerate(wave_ids):
            wave = [by_id[tid] for tid in ids if tid in by_id]
            # Skip tasks already satisfied by snapshot.
            todo = [t for t in wave if bb.get(t.id) is None or "error" in (bb.get(t.id) or {})]
            if not todo:
                continue
            self._run_async(self._run_wave(todo, wave_idx, bb))

        return patch(state, blackboard=bb.snapshot(), db_ops=self._redis_ops(bb))

    # ------------------------------------------------------------------
    @staticmethod
    def _redis_ops(bb: Blackboard) -> list[dict]:
        """Tracer op records for the real blackboard → Redis mirror writes."""
        enabled = get_redis_store().enabled
        ops: list[dict] = []
        for tid, entry in bb.snapshot().items():
            key = f"bb:{bb.thread_id}:{bb.run_id}:{tid}"
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
            resolved = self._lookup_ref(m.group(1), bb)
            if resolved is None:
                self.log("warning", "executor.ref_unresolved", ref=m.group(1))
                return m.group(0)
            return str(resolved)

        return _REF_RE.sub(_sub, value)

    async def _run_task(
        self, task: Subtask, wave_idx: int, bb: Blackboard, http: httpx.AsyncClient
    ) -> None:
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
