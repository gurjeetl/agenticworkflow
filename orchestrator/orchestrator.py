from __future__ import annotations

import asyncio
from typing import Any

import mlflow
from mlflow.entities import SpanType

from baseagent.base_agent import patch
from observability import Observable
from orchestrator.blackboard import Blackboard
from planner.dag import Plan, Subtask
from registry import AGENT_REGISTRY
from state import AgentState


class Orchestrator(Observable):
    """Runs the plan's DAG wave by wave.

    All tasks in a wave fan out concurrently via asyncio.gather; the next wave
    starts only after the current one's tasks have written to the blackboard.
    Failures are captured per task — the gate decides whether to re-plan.
    """

    _traced_methods: tuple[str, ...] = ("run",)
    _component_kind: str = "orchestrator"
    _span_type: str = SpanType.CHAIN

    # Cache of instantiated agent objects — each agent has its own LLM/MCP cost.
    _agent_cache: dict[str, object] = {}

    def run(self, state: AgentState) -> AgentState:
        plan_dict = state.get("plan") or {}
        plan = Plan(**plan_dict)
        bb = Blackboard(thread_id=state.get("thread_id", ""), run_id=state.get("run_id", ""))

        # Seed blackboard from previous attempt's snapshot so successful tasks
        # don't re-run on re-plan loops.
        snapshot = state.get("blackboard_snapshot") or {}
        for tid, entry in snapshot.items():
            if isinstance(entry, dict) and "error" not in entry:
                bb._mem[tid] = entry

        if not plan.subtasks:
            self.log_event("orchestrator.empty_plan")
            return patch(state, blackboard=bb.snapshot())

        try:
            waves = plan.waves()
        except Exception as e:
            self.log("error", "orchestrator.dag_invalid", error=str(e))
            return patch(state, blackboard={"_plan_error": {"error": str(e)}})

        for wave_idx, wave in enumerate(waves):
            # Skip tasks already satisfied by snapshot.
            todo = [t for t in wave if bb.get(t.id) is None or "error" in (bb.get(t.id) or {})]
            if not todo:
                continue
            self._run_async(self._run_wave(todo, wave_idx, bb))

        return patch(state, blackboard=bb.snapshot())

    # ------------------------------------------------------------------
    async def _run_wave(self, tasks: list[Subtask], wave_idx: int, bb: Blackboard) -> None:
        self.log_event("orchestrator.wave_start", wave=wave_idx, count=len(tasks))
        await asyncio.gather(*(self._run_task(t, wave_idx, bb) for t in tasks), return_exceptions=False)
        self.log_event("orchestrator.wave_done", wave=wave_idx)

    async def _run_task(self, task: Subtask, wave_idx: int, bb: Blackboard) -> None:
        entry = AGENT_REGISTRY.get(task.agent_id)
        if entry is None:
            await bb.write_error(task.id, f"agent_id '{task.agent_id}' not in registry")
            return
        meta, cls = entry
        agent = self._agent_cache.get(task.agent_id) or cls()
        self._agent_cache[task.agent_id] = agent

        with mlflow.start_span(name=f"agent.{task.agent_id}.{task.id}", span_type=SpanType.AGENT) as span:
            span.set_attribute("agent.id", task.agent_id)
            span.set_attribute("agent.version", meta.version)
            span.set_attribute("orch.wave", wave_idx)
            span.set_attribute("task.id", task.id)
            span.set_inputs({"args": task.args})

            last_error: str | None = None
            for attempt in range(2):  # 1 retry for MVP
                try:
                    result_state = await asyncio.to_thread(
                        self._invoke_agent, agent, task, bb, meta.sla_ms
                    )
                    payload = self._extract_payload(task, result_state)
                    await bb.write(task.id, payload)
                    span.set_attribute("retry.count", attempt)
                    span.set_outputs({"payload_keys": list(payload.keys())})
                    return
                except Exception as e:
                    last_error = str(e)
                    self.log("warning", "orchestrator.task_retry", task=task.id, attempt=attempt, error=last_error)
            span.set_attribute("retry.count", 1)
            span.set_attribute("status", "error")
            await bb.write_error(task.id, last_error or "unknown failure")

    # ------------------------------------------------------------------
    def _invoke_agent(self, agent, task: Subtask, bb: Blackboard, sla_ms: int) -> AgentState:
        # Narrow state passed to the agent: spread args as top-level keys
        # (existing agents read state["location"], state["outage_id"], etc.) plus
        # blackboard visibility for cross-task reads.
        task_state: AgentState = {
            "user_input": "",
            "current_task": task.id,
            "thread_id": bb.thread_id,
            "run_id": bb.run_id,
            "messages": [],
            "agent_scratchpad": "",
            "iteration_count": 0,
            "max_iterations": 5,
            "tool_calls": [],
            "tool_results": [],
            "short_term_memory": [],
            "long_term_memory_keys": [],
            "active_agent": task.agent_id,
            "next_action": "",
            "delegated_task": None,
            "location": None,
            "intent": None,
            "outage_id": None,
            "plan": None,
            "agent_versions": {},
            "blackboard": bb.snapshot(),
            "blackboard_snapshot": None,
            "replan_count": 0,
            "max_replans": 0,
            "replan_reason": None,
            "partial": False,
            "final_output": None,
            "view": None,
            "is_complete": False,
            "error": None,
        }
        for k, v in (task.args or {}).items():
            task_state[k] = v  # type: ignore[literal-required]
        return agent.run(task_state)

    @staticmethod
    def _extract_payload(task: Subtask, result_state: AgentState) -> dict[str, Any]:
        if result_state.get("error"):
            return {"error": str(result_state["error"])}
        payload: dict[str, Any] = {
            "agent_id": task.agent_id,
            "text": result_state.get("final_output"),
        }
        view = result_state.get("view")
        if view:
            payload["view"] = view
        return payload
