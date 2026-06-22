<#
.SYNOPSIS
    Launch the full Genie platform stack for LOCAL DEVELOPMENT.

.DESCRIPTION
    Starts every process the platform needs, each in its OWN titled console
    window so you can watch its logs live. Optimised for an inner-loop dev
    experience:

      * Gateway runs under Uvicorn with --reload (hot-reload on code changes).
      * Everything binds to 127.0.0.1 (loopback only — not exposed off-box).
      * A self-contained MLflow server (SQLite backend) is started locally, so
        tracing works with no external Postgres.
      * Optional Redis (blackboard hot mirror) is auto-started if a portable
        build is present; the app degrades gracefully when it is absent.

    Ports / processes (see also scripts\run-all.ps1):
      MLflow            : 5000   tracing backend
      MCP server        : 8001   services.mcp.weather_server
      Registry          : 8002   services.registry.server
      RAG service       : 8003   services.rag.server
      Weather agent     : 8010   applications.demo.weather.agent
      Outage agent      : 8011   applications.demo.outage.agent
      RAG agent         : 8012   applications.demo.rag.agent
      Gateway (FastAPI) : 8000   uvicorn app:app --reload

    Start-up ORDER matters: MLflow first (agents/app log spans to it), then the
    registry (agents self-register against it), then the agents (so the
    Planner's first menu query finds them), then the gateway.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run-dev.ps1
#>

# Fail fast on any unhandled error during launch.
$ErrorActionPreference = "Stop"

# Repo root = the parent of this script's folder. Everything is resolved
# relative to it so the script works regardless of the caller's CWD.
$root = Split-Path -Parent $PSScriptRoot

# Importable code lives under src/ (the `genie` + `applications` packages),
# while `services.*` and the `app` entry resolve from the repo root. Putting
# src/ on PYTHONPATH makes every process importable even without `uv sync`
# having produced an editable install path entry for `app`/`services`.
$src = Join-Path $root "src"

# --------------------------------------------------------------------------
# Helper: open a service in its own titled PowerShell window.
#   $title  - window title (also used as a kill-marker on re-run, below)
#   $cmd    - the command line to execute (we prepend `uv run`)
#   $svcEnv - hashtable of per-process env vars (e.g. AGENT_PORT)
# `uv run` guarantees the command executes inside the project's managed venv.
# --------------------------------------------------------------------------
function Start-Svc($title, $cmd, $svcEnv = @{}) {
    # Always put src/ on PYTHONPATH; merge any caller-supplied vars on top.
    $svcEnv["PYTHONPATH"] = $src
    $pre = ($svcEnv.GetEnumerator() | ForEach-Object { "`$env:$($_.Key)='$($_.Value)';" }) -join " "
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$Host.UI.RawUI.WindowTitle='$title'; Set-Location '$root'; $pre uv run $cmd"
    )
}

# --------------------------------------------------------------------------
# Idempotent re-run: close service windows left over from a previous launch.
# Our windows are started with -NoExit (logs stay readable), so they survive a
# plain process kill. We match the launch command line, which carries a unique
# `RawUI.WindowTitle='...'` marker that only our service windows have — this
# never matches the launcher / editor / plain shells.
# --------------------------------------------------------------------------
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'RawUI\.WindowTitle' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# --------------------------------------------------------------------------
# Optional Redis (blackboard hot mirror). Start the portable build if present
# and not already listening. The platform runs fine without it.
# --------------------------------------------------------------------------
$redisExe = Join-Path $env:LOCALAPPDATA "redis-portable\redis-server.exe"
if (Test-Path $redisExe) {
    if (-not (Get-NetTCPConnection -LocalPort 6379 -State Listen -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $redisExe -ArgumentList "--port", "6379" -WindowStyle Hidden
        Write-Host "Started Redis on :6379"
    } else { Write-Host "Redis already running on :6379" }
} else { Write-Host "Redis not found - blackboard mirror disabled (this is fine for dev)" }

# --------------------------------------------------------------------------
# MLflow tracking server with a self-contained SQLite backend — no Postgres
# needed for dev. Must come up before the app/agents so their spans land
# somewhere. Artifacts go to ./mlartifacts.
# --------------------------------------------------------------------------
Start-Svc "MLflow :5000" "python -m mlflow server --backend-store-uri sqlite:///mlflow_dev.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000"
Start-Sleep -Seconds 5

# Backend services (MCP tools, registry, RAG retrieval).
Start-Svc "MCP :8001"         "python -m services.mcp.weather_server"
Start-Svc "Registry :8002"    "python -m services.registry.server"
Start-Svc "RAG Service :8003" "python -m services.rag.server"
Start-Sleep -Seconds 2

# Demo agents. AGENT_PORT is the one per-agent value YAML can't carry (each
# agent needs a distinct port); every other setting comes from config/*.yaml.
Start-Svc "Weather :8010" "python -m applications.demo.weather.agent" @{ AGENT_PORT = "8010" }
Start-Svc "Outage :8011"  "python -m applications.demo.outage.agent"  @{ AGENT_PORT = "8011" }
Start-Svc "RAG :8012"     "python -m applications.demo.rag.agent"     @{ AGENT_PORT = "8012" }
Start-Sleep -Seconds 2

# Gateway with hot-reload — the inner-loop win for dev. Loopback bind only.
Start-Svc "App :8000" "uvicorn app:app --host 127.0.0.1 --port 8000 --reload --reload-dir src"

Write-Host ""
Write-Host "Dev stack launched (each service has its own window)." -ForegroundColor Green
Write-Host "  Chat UI  : http://127.0.0.1:8000"
Write-Host "  Trace UI : http://127.0.0.1:8000/trace.html"
Write-Host "  Registry : http://127.0.0.1:8002/agents"
Write-Host "  MLflow   : http://127.0.0.1:5000"
Write-Host ""
Write-Host "Re-run this script any time; it closes the old service windows first."
