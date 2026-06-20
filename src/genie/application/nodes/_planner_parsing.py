"""Parsing/menu helpers shared by the Planner and the Router.

Both turn the registry's live ``AgentMeta`` list into a prompt menu and both must
tolerantly parse an LLM's JSON and resolve the agent id it picked. Keeping these
in one place makes the Router a *fast mirror* of the Planner's routing logic
rather than a second implementation that can drift.
"""
from __future__ import annotations

import json
import re

from genie.registry.agent_meta import AgentMeta


def render_capability_menu(metas: list[AgentMeta]) -> str:
    """Format live agents into the menu both the Planner and Router prompt with."""
    lines = []
    for meta in metas:
        inputs = ", ".join(
            f"{name}{'*' if spec.required else ''}:{spec.type}"
            for name, spec in meta.input_schema.items()
        ) or "(none)"
        outputs = ", ".join(
            f"{name}:{spec.type}" + (f" ({spec.description})" if spec.description else "")
            for name, spec in meta.output_schema.items()
        ) or "(none)"
        tags = ", ".join(meta.capability_tags) or "(none)"
        lines.append(
            f'- agent_id: "{meta.agent_id}"   (use this exact string; the version below is INFO ONLY, do NOT include it)\n'
            f"    version: {meta.version}\n"
            f"    capability: {meta.description or '(no description)'}\n"
            f"    tags: {tags}\n"
            f"    inputs: {inputs}\n"
            f"    outputs: {outputs}\n"
            f"    sla_ms: {meta.sla_ms}"
        )
    return "\n".join(lines) if lines else "(no agents registered)"


def extract_json(raw: str) -> dict | None:
    """Find the first balanced JSON object in ``raw`` and parse it.

    Tolerant of LLM tics like trailing junk, an extra closing brace, or a
    markdown code fence — we walk the string tracking brace depth and string
    state so we stop exactly at the matching closer of the first object.
    """
    if not raw:
        return None
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    # Unbalanced — fall back to greedy parse as a last resort.
    try:
        return json.loads(raw[start:])
    except json.JSONDecodeError:
        return None


def normalize_agent_id(raw_id: str | None, known_ids: set[str]) -> str | None:
    """Resolve common LLM stumbles to a real discovered agent id.

    Handles: trailing version (`` v1.0.0``), accidental quotes/whitespace, case
    differences. Returns the canonical agent_id if a match is found, else None.
    """
    if not raw_id or not isinstance(raw_id, str):
        return None
    cleaned = raw_id.strip().strip('"').strip("'").strip()
    # Drop trailing " v1.2.3" or "@1.2.3" version suffixes.
    cleaned = re.sub(r"[\s@]+v?\d+(?:\.\d+){0,3}\s*$", "", cleaned).strip()
    if cleaned in known_ids:
        return cleaned
    # Case-insensitive fallback.
    lower_map = {k.lower(): k for k in known_ids}
    return lower_map.get(cleaned.lower())
