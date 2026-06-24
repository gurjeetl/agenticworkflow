# Genie Platform — Setup & Running

Everything you need to install, configure, and run the platform locally or in
production. For what the platform *is* and how it works, see **[README.md](README.md)**.

---

## 1. Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — the package/venv/runner this project uses. Install it
  first; it also manages the Python toolchain.
  ```powershell
  # Windows (PowerShell)
  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  # macOS / Linux
  # curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  (Alternatives: `pipx install uv`, `winget install astral-sh.uv`, `brew install uv`.)
- **Python 3.13+** — optional to install yourself; `uv` fetches a compatible interpreter on
  first sync (`uv python install 3.13`).
- **MongoDB** — **required** (default `mongodb://localhost:27017`). Primary datastore; also
  backs the registry.
- **An OpenAI API key** or any OpenAI-compatible endpoint (set `OPENAI_BASE_URL`).
- **PostgreSQL** — needed only if you use the `run-*.ps1` launchers' MLflow tracking server
  (they start MLflow against `mlflow_backend_store_uri`). Not needed if you disable/redirect
  MLflow (see [Troubleshooting](#7-troubleshooting)).
- **Optional backends** (each activates only when its DSN/URL is set; the platform degrades
  gracefully otherwise):
  - **Redis** — blackboard hot mirror.
  - **Milvus** — semantic long-term memory (a local Milvus Lite file needs no server).
  - **PostgreSQL / SQL Server** — relational stores for your own tools/agents. SQL Server
    additionally requires the OS **"ODBC Driver 18 for SQL Server"**.

---

## 2. Install

From the repo root:

```powershell
uv sync                 # create .venv + install the platform, applications, and the
                        # editable genie-rag-contracts package
uv sync --extra dev     # adds the test/lint toolchain: pytest, pytest-asyncio, ruff, import-linter
```

`uv sync` resolves and installs everything in `pyproject.toml` (including the local
`packages/genie-rag-contracts` editable source) into a project `.venv` — no manual
`venv`/activation needed.

> ⚠️ **Use `uv sync --extra dev` if you'll run tests or linters.** A plain `uv sync` *removes*
> the dev toolchain (pytest, ruff, import-linter) because they live in the `dev` extra.

Run any command inside the managed environment by prefixing it with `uv run` (e.g.
`uv run python -m ...`, `uv run uvicorn ...`, `uv run pytest`). `uv` keeps the env up to date
before each run. If you prefer a classic activated shell, `.venv\Scripts\Activate.ps1`
(PowerShell) or `source .venv/bin/activate` (bash) still works.

---

## 3. Configure

```powershell
Copy-Item config\local.yaml.example config\local.yaml
# Edit config/local.yaml: set openai_api_key at minimum, plus mlflow_backend_store_uri
# (a PostgreSQL DSN) if you use the launchers' MLflow server.
```

Configuration is centralized in `genie.platform.config.Settings` (pydantic-settings) and
resolved in priority order (**first wins**):

1. `config/local.yaml` (gitignored — secrets & machine overrides)
2. `config/default.yaml` (committed canonical config; or `$GENIE_CONFIG_FILE`)
3. environment variables / `.env`
4. built-in field defaults

YAML is authoritative over the environment; `env_prefix=""`, so a field maps to its UPPER_SNAKE
name (`mongodb_uri` ← `MONGODB_URI`). Nested blocks (`mcp_services`, `llm_services`) live only
in YAML. The full settings list and key-vars table are in
[README → Configuration](README.md#configuration).

**Relational backends** (optional) — set the connection string in `config/local.yaml` or via
env:

```yaml
# config/local.yaml
postgres_dsn: postgresql://user:pass@localhost:5432/genie
sqlserver_dsn: "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost,1433;DATABASE=genie;UID=sa;PWD=changeme;TrustServerCertificate=yes"
```
(env fallbacks: `POSTGRES_DSN`, `SQLSERVER_DSN`.)

---

## 4. Run

The system is multi-process; each piece runs on its own port:

| Service | Port | Module / command |
| --- | --- | --- |
| Gateway (FastAPI) | 8000 | `uvicorn app:app` |
| MLflow tracking server | 5000 | `python -m mlflow server --backend-store-uri <dsn> ...` |
| MCP tool server | 8001 | `python -m services.mcp.genie_mcp_server` |
| Registry / discovery | 8002 | `python -m services.registry.server` |
| RAG service | 8003 | `python -m services.rag.server` |
| Weather agent (sample) | 8010 | `python -m applications.demo.weather.agent` |
| Outage agent (sample) | 8011 | `python -m applications.demo.outage.agent` |
| RAG agent (sample) | 8012 | `python -m applications.demo.rag.agent` |

### Launcher scripts (`scripts/`)

Each script starts services in the correct order (MLflow → MCP/Registry/RAG → agents →
gateway), opens each in its own titled window, optionally auto-starts a portable Redis, and is
idempotent (re-running closes the previous windows first).

| Script | What it runs | Use when |
| --- | --- | --- |
| **`run-platform.ps1`** | Core platform **without** the sample agents (MLflow, MCP, Registry, RAG service, Gateway) | You want the platform running so **your own** agents (started separately) can register against it. |
| **`run-full.ps1`** | The **whole** stack **including** the sample agents (weather/outage/rag) | You want an end-to-end demo with the bundled agents. |
| `run-dev.ps1` | Same as `run-full` but gateway runs with `--reload` (hot-reload) | Inner-loop development on the kernel. |
| `run-all.ps1` | Legacy full-stack launcher (raw venv python, no `uv run`) | Back-compat. |
| `run-prod.ps1` | Production: multiple Uvicorn workers, detached + logged, external infra assumed | Single-host production deploy. |

```powershell
# Platform only — bring your own agents
powershell -ExecutionPolicy Bypass -File scripts\run-platform.ps1

# Whole stack including the sample agents
powershell -ExecutionPolicy Bypass -File scripts\run-full.ps1

# Development with gateway hot-reload
powershell -ExecutionPolicy Bypass -File scripts\run-dev.ps1

# Production (8 workers, custom config); stop with -Stop
powershell -ExecutionPolicy Bypass -File scripts\run-prod.ps1 -Workers 8 -ConfigFile config\prod.yaml
powershell -ExecutionPolicy Bypass -File scripts\run-prod.ps1 -Stop
```

`run-platform.ps1`, `run-full.ps1`, and `run-dev.ps1` read `mlflow_backend_store_uri` from
`config/local.yaml` and exit early if it isn't set — configure it first (or see
[Troubleshooting](#7-troubleshooting) to skip Postgres).

Then open:

- <http://127.0.0.1:8000> — chat UI
- <http://127.0.0.1:8000/trace.html> — execution tracer (recommended starting point)

### Manual (one terminal per process)

```powershell
$env:PYTHONPATH = "src"
uv run python -m services.mcp.genie_mcp_server     # :8001
uv run python -m services.registry.server          # :8002
uv run python -m services.rag.server               # :8003
$env:AGENT_PORT="8010"; uv run python -m applications.demo.weather.agent
uv run uvicorn app:app --host 0.0.0.0 --port 8000  # :8000
```

---

## 5. Running your own agents

`run-platform.ps1` starts the platform without any agents — the registry comes up empty. To add
an agent, run its module on a free port; it self-registers and the Planner discovers it on the
next request:

```powershell
$env:PYTHONPATH = "src"
$env:AGENT_PORT = "8020"
uv run python -m applications.<your_app>.<your_agent>.agent
```

See [README → Building an agent](README.md#building-an-agent) for the agent class + `AgentMeta`.

---

## 6. Tests & boundaries

```powershell
uv sync --extra dev
uv run pytest               # unit / integration / e2e
uv run lint-imports         # enforces: genie.* must not import applications.*
```

---

## 7. Troubleshooting

- **MLflow without PostgreSQL.** The launchers start a Postgres-backed MLflow server. To skip
  Postgres, don't rely on the launcher's server — point the app straight at a local store with
  `mlflow_tracking_uri: sqlite:///mlflow_local.db` in `config/local.yaml`, or leave
  `mlflow_tracking_uri` unset to disable tracing (it degrades to a no-op, not a crash).
  Otherwise every process blocks on connection retries at startup.
- **`pytest` / `ruff` / `lint-imports` not found.** You ran a plain `uv sync`, which drops the
  dev extra. Re-run `uv sync --extra dev`.
- **SQL Server connection fails.** Ensure the OS has the **"ODBC Driver 18 for SQL Server"**
  installed; `pyodbc`/`aioodbc` are only the Python bindings.
- **No agents show up** in `/registry` or the tracer's discovery panel. With `run-platform.ps1`
  that's expected — start your agents (or use `run-full.ps1` for the samples). Otherwise check
  the registry is up on :8002 and the agents' `AGENT_PORT`s don't collide.
- **Redis disabled** message at startup is fine — the blackboard hot mirror is optional.
