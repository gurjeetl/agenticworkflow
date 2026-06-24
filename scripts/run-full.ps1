<#
.SYNOPSIS
    Launch the WHOLE Genie stack — every core service PLUS the bundled sample agents.

.DESCRIPTION
    Starts the full end-to-end stack: the platform services and the demo agents
    (weather/outage/rag), so the Planner has agents to discover and orchestrate out
    of the box.

    Each process opens in its own titled console window (live logs). Optionally
    auto-starts a portable Redis. Re-running is idempotent: it closes the previous
    run's service windows first. The gateway runs WITHOUT --reload (use run-dev.ps1
    for hot-reload).

    Start-up ORDER matters: MLflow first (agents/app log spans to it), then the
    registry (agents self-register against it), then the agents (so the Planner's
    first menu query finds them), then the gateway.

    Ports / processes started:
      MLflow            : 5000   tracing backend
      MCP server        : 8001   services.mcp.genie_mcp_server
      Registry          : 8002   services.registry.server
      RAG service       : 8003   services.rag.server
      Weather agent     : 8010   applications.demo.weather.agent
      Outage agent      : 8011   applications.demo.outage.agent
      RAG agent         : 8012   applications.demo.rag.agent
      Gateway (FastAPI) : 8000   uvicorn app:app

    To run the platform WITHOUT the sample agents, use scripts\run-platform.ps1.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run-full.ps1
#>

# Fail fast on any unhandled error during launch.
$ErrorActionPreference = "Stop"

# Repo root = the parent of this script's folder; everything resolves relative to it.
$root = Split-Path -Parent $PSScriptRoot

# Importable code lives under src/; putting it on PYTHONPATH makes every process importable.
$src = Join-Path $root "src"

# --------------------------------------------------------------------------
# Helper: open a service in its own titled PowerShell window (via `uv run`).
# --------------------------------------------------------------------------
function Start-Svc($title, $cmd, $svcEnv = @{}) {
    $svcEnv["PYTHONPATH"] = $src
    $pre = ($svcEnv.GetEnumerator() | ForEach-Object { "`$env:$($_.Key)='$($_.Value)';" }) -join " "
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$Host.UI.RawUI.WindowTitle='$title'; Set-Location '$root'; $pre uv run $cmd"
    )
}

# --------------------------------------------------------------------------
# Read a flat scalar key from the YAML config (local.yaml wins over default.yaml),
# falling back to the legacy UPPER_SNAKE name in .env.
# --------------------------------------------------------------------------
function Get-ConfigValue($key) {
    foreach ($f in @((Join-Path $root "config\local.yaml"), (Join-Path $root "config\default.yaml"))) {
        if (Test-Path $f) {
            $line = Get-Content $f | Where-Object { $_ -match "^\s*$key\s*:" -and $_ -notmatch '^\s*#' } | Select-Object -First 1
            if ($line) { return ($line -replace "^\s*$key\s*:\s*", '').Trim().Trim('"').Trim("'") }
        }
    }
    $envFile = Join-Path $root ".env"
    if (Test-Path $envFile) {
        $envKey = $key.ToUpper()
        $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$envKey\s*=" } | Select-Object -First 1
        if ($line) { return ($line -replace "^\s*$envKey\s*=\s*", '').Trim() }
    }
    return $null
}

# --------------------------------------------------------------------------
# Idempotent re-run: close service windows left over from a previous launch.
# --------------------------------------------------------------------------
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'RawUI\.WindowTitle' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# --------------------------------------------------------------------------
# Optional Redis (blackboard hot mirror). The platform runs fine without it.
# --------------------------------------------------------------------------
$redisExe = Join-Path $env:LOCALAPPDATA "redis-portable\redis-server.exe"
if (Test-Path $redisExe) {
    if (-not (Get-NetTCPConnection -LocalPort 6379 -State Listen -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $redisExe -ArgumentList "--port", "6379" -WindowStyle Hidden
        Write-Host "Started Redis on :6379"
    } else { Write-Host "Redis already running on :6379" }
} else { Write-Host "Redis not found - blackboard mirror disabled (this is fine)" }

# --------------------------------------------------------------------------
# MLflow tracking server backed by PostgreSQL (DSN from config/local.yaml).
# Must come up before the app/agents so their spans land somewhere.
# --------------------------------------------------------------------------
$mlflowBackend = Get-ConfigValue 'mlflow_backend_store_uri'
if (-not $mlflowBackend) {
    Write-Host "ERROR: mlflow_backend_store_uri is not set in config/local.yaml (PostgreSQL DSN required)." -ForegroundColor Red
    Write-Host "       Copy config/local.yaml.example to config/local.yaml and fill it in." -ForegroundColor Red
    exit 1
}
$mlflowUri = Get-ConfigValue 'mlflow_tracking_uri'; if (-not $mlflowUri) { $mlflowUri = "http://127.0.0.1:5000" }
$mlflowParsed = [System.Uri]$mlflowUri
Start-Svc "MLflow :$($mlflowParsed.Port)" "python -m mlflow server --backend-store-uri $mlflowBackend --default-artifact-root ./mlartifacts --host $($mlflowParsed.Host) --port $($mlflowParsed.Port)"
Start-Sleep -Seconds 5

# Core backend services (MCP tools, registry, RAG retrieval).
Start-Svc "MCP :8001"         "python -m services.mcp.genie_mcp_server"
Start-Svc "Registry :8002"    "python -m services.registry.server"
Start-Svc "RAG Service :8003" "python -m services.rag.server"
Start-Sleep -Seconds 2

# Sample agents. AGENT_PORT is the one per-agent value YAML can't carry (each agent
# needs a distinct port); every other setting comes from config/*.yaml.
Start-Svc "Weather :8010" "python -m applications.demo.weather.agent" @{ AGENT_PORT = "8010" }
Start-Svc "Outage :8011"  "python -m applications.demo.outage.agent"  @{ AGENT_PORT = "8011" }
Start-Svc "RAG :8012"     "python -m applications.demo.rag.agent"     @{ AGENT_PORT = "8012" }
Start-Sleep -Seconds 2

# Gateway. Loopback bind for local use.
Start-Svc "App :8000" "uvicorn app:app --host 127.0.0.1 --port 8000"

Write-Host ""
Write-Host "Full stack launched WITH sample agents (each service has its own window)." -ForegroundColor Green
Write-Host "  Chat UI  : http://127.0.0.1:8000"
Write-Host "  Trace UI : http://127.0.0.1:8000/trace.html"
Write-Host "  Registry : http://127.0.0.1:8002/agents"
Write-Host "  MLflow   : $mlflowUri"
Write-Host ""
Write-Host "Re-run this script any time; it closes the old service windows first."
