# BaseAgentFramework — Overview

> **Navigation aid.** This article shows WHERE things live (routes, models, files). Read actual source files before implementing new features or making changes.

**BaseAgentFramework** is a python project built with fastapi.

## Scale

3 API routes · 19 library files · 12 environment variables

## Subsystems

- **[Chat](./chat.md)** — 1 routes
- **[State](./state.md)** — 1 routes
- **[Infra](./infra.md)** — 1 routes

**Libraries:** 19 files — see [libraries.md](./libraries.md)

## Required Environment Variables

- `MCP_AUTH_TOKEN` — `.env.example`
- `OPENAI_BASE_URL` — `baseagent\base_agent.py`

---
_Back to [index.md](./index.md) · Generated 2026-05-30_