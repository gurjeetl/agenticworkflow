<#
.SYNOPSIS
    Launch the Genie platform stack for PRODUCTION (single-host deployment).

.DESCRIPTION
    Starts the platform as long-running BACKGROUND processes (no interactive
    windows), with logs redirected to .\logs\*.log and every PID recorded in
    .\run\pids.txt so the stack can be stopped cleanly with -Stop.

    Production differs from scripts\run-dev.ps1 in the ways that matter for a
    real deployment:

      * The gateway runs WITHOUT --reload and with multiple Uvicorn workers
        (-Workers), and binds to 0.0.0.0 so it is reachable off-box. The
        gateway is stateless (all state lives in Mongo/Redis), so workers scale
        horizontally on one host.
      * Backend services and agents stay on 127.0.0.1 (internal A2A traffic);
        only the gateway is exposed. For a MULTI-host layout, set
        AGENT_ADVERTISE_HOST/PORT in config so agents register a routable
        endpoint instead.
      * Infra (MLflow, Redis, MongoDB) is assumed to be MANAGED EXTERNALLY.
        This script does NOT start them — it only checks reachability and
        warns. Point config at them via config/local.yaml (or a prod config
        passed with -ConfigFile).
      * Processes are detached and logged to files, not consoles.

    Ports / processes:
      MCP server        : 8001   services.mcp.genie_mcp_server
      Registry          : 8002   services.registry.server
      RAG service       : 8003   services.rag.server
      Weather agent     : 8010   applications.demo.weather.agent
      Outage agent      : 8011   applications.demo.outage.agent
      RAG agent         : 8012   applications.demo.rag.agent
      Gateway (FastAPI) : 8000   uvicorn app:app --workers N

.PARAMETER GatewayHost
    Interface the gateway binds to. Default 0.0.0.0 (all interfaces).

.PARAMETER Port
    Gateway port. Default 8000.

.PARAMETER Workers
    Number of Uvicorn worker processes for the gateway. Default 4.

.PARAMETER ConfigFile
    Optional path to a production YAML config. When set, it is exported as
    GENIE_CONFIG_FILE so every process reads it instead of config/default.yaml.

.PARAMETER Stop
    Stop a previously launched stack: kills every PID listed in .\run\pids.txt
    and clears the file. Does not start anything.

.EXAMPLE
    # Start with 8 workers, behind a config/prod.yaml.
    powershell -ExecutionPolicy Bypass -File scripts\run-prod.ps1 -Workers 8 -ConfigFile config\prod.yaml

.EXAMPLE
    # Gracefully stop everything this script started.
    powershell -ExecutionPolicy Bypass -File scripts\run-prod.ps1 -Stop
#>

param(
    [string]$GatewayHost = "0.0.0.0",
    [int]$Port           = 8000,
    [int]$Workers        = 4,
    [string]$ConfigFile  = "",
    [switch]$Stop
)

# Fail fast — a half-started production stack is worse than a clean abort.
$ErrorActionPreference = "Stop"

# Resolve paths relative to the repo root (parent of this script's folder).
$root    = Split-Path -Parent $PSScriptRoot
$src     = Join-Path $root "src"
$logsDir = Join-Path $root "logs"
$runDir  = Join-Path $root "run"
$pidFile = Join-Path $runDir "pids.txt"

# --------------------------------------------------------------------------
# -Stop: tear down a running stack and exit. We kill by recorded PID (precise),
# then remove the PID file so a later start is clean.
# --------------------------------------------------------------------------
if ($Stop) {
    if (Test-Path $pidFile) {
        Get-Content $pidFile | Where-Object { $_ } | ForEach-Object {
            try {
                Stop-Process -Id ([int]$_) -Force -ErrorAction Stop
                Write-Host "Stopped PID $_"
            } catch {
                Write-Host "PID $_ not running (skipped)"
            }
        }
        Remove-Item $pidFile -Force
        Write-Host "Production stack stopped." -ForegroundColor Green
    } else {
        Write-Host "No PID file at $pidFile - nothing to stop."
    }
    return
}

# Ensure the logs/ and run/ directories exist (created on first start).
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $runDir  | Out-Null

# Refuse to start on top of an existing stack — avoids orphaned, untracked
# duplicate processes. Operator must -Stop first.
if (Test-Path $pidFile) {
    Write-Host "ERROR: $pidFile exists - a stack may already be running." -ForegroundColor Red
    Write-Host "       Run with -Stop first, or delete the file if it is stale." -ForegroundColor Red
    exit 1
}

# Base environment shared by every child process:
#   PYTHONPATH       - makes `genie`, `applications`, `services`, `app` importable.
#   GENIE_CONFIG_FILE - optional prod config override (see -ConfigFile).
$baseEnv = @{ PYTHONPATH = $src }
if ($ConfigFile) {
    $cfgPath = if ([System.IO.Path]::IsPathRooted($ConfigFile)) { $ConfigFile } else { Join-Path $root $ConfigFile }
    if (-not (Test-Path $cfgPath)) {
        Write-Host "ERROR: -ConfigFile '$cfgPath' not found." -ForegroundColor Red
        exit 1
    }
    $baseEnv["GENIE_CONFIG_FILE"] = $cfgPath
    Write-Host "Using config: $cfgPath"
}

# --------------------------------------------------------------------------
# Helper: check an infra dependency is listening; warn (don't fail) if not.
# The platform degrades gracefully (MLflow/Redis are best-effort), so a missing
# dependency should not block a deploy — but the operator should know.
# --------------------------------------------------------------------------
function Test-Infra($name, $listenPort) {
    if (Get-NetTCPConnection -LocalPort $listenPort -State Listen -ErrorAction SilentlyContinue) {
        Write-Host "  [ok]   $name reachable on :$listenPort"
    } else {
        Write-Host "  [warn] $name NOT reachable on :$listenPort - is it running?" -ForegroundColor Yellow
    }
}

Write-Host "Checking externally-managed infrastructure..."
Test-Infra "MongoDB" 27017   # primary datastore (required at runtime)
Test-Infra "MLflow"  5000    # tracing backend (best-effort)
Test-Infra "Redis"   6379    # blackboard hot mirror (optional)

# --------------------------------------------------------------------------
# Helper: start one detached background process via `uv run`, redirecting both
# stdout and stderr to a per-service log file, and record its PID.
#   $name   - short service name (used for the log filename)
#   $argline - arguments after `uv run` (e.g. "python -m services.registry.server")
#   $svcEnv - per-process env vars merged over $baseEnv
# Env vars are scoped to this function call: we set them, launch, then restore,
# so each service gets only its intended overrides (e.g. its own AGENT_PORT).
# --------------------------------------------------------------------------
function Start-Bg($name, $argline, $svcEnv = @{}) {
    $merged = $baseEnv.Clone()
    foreach ($k in $svcEnv.Keys) { $merged[$k] = $svcEnv[$k] }

    # Apply env vars for the child, remembering prior values to restore after.
    $saved = @{}
    foreach ($k in $merged.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        Set-Item -Path "Env:$k" -Value $merged[$k]
    }

    $out = Join-Path $logsDir "$name.log"
    $err = Join-Path $logsDir "$name.err.log"
    # Split argline into the exe ("uv") and its arguments ("run python -m ...").
    $proc = Start-Process -FilePath "uv" `
        -ArgumentList ("run " + $argline) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err `
        -WindowStyle Hidden -PassThru

    # Record the PID for -Stop.
    Add-Content -Path $pidFile -Value $proc.Id

    # Restore the launcher's environment.
    foreach ($k in $saved.Keys) {
        if ($null -eq $saved[$k]) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
        else { Set-Item -Path "Env:$k" -Value $saved[$k] }
    }

    Write-Host "  started $name (PID $($proc.Id)) -> logs\$name.log"
}

Write-Host "Starting backend services..."
# Backend services first (registry must be up before agents self-register).
Start-Bg "mcp"      "python -m services.mcp.genie_mcp_server"
Start-Bg "registry" "python -m services.registry.server"
Start-Bg "rag"      "python -m services.rag.server"
Start-Sleep -Seconds 3

Write-Host "Starting agents..."
# Agents stay on loopback for internal A2A; only AGENT_PORT differs per agent.
Start-Bg "weather" "python -m applications.demo.weather.agent" @{ AGENT_PORT = "8010" }
Start-Bg "outage"  "python -m applications.demo.outage.agent"  @{ AGENT_PORT = "8011" }
Start-Bg "rag-agent" "python -m applications.demo.rag.agent"   @{ AGENT_PORT = "8012" }
Start-Sleep -Seconds 3

Write-Host "Starting gateway ($Workers workers on ${GatewayHost}:${Port})..."
# Production gateway: no reload, multiple workers, exposed bind. Uvicorn needs
# the import-string form ("app:app") to spawn workers.
Start-Bg "gateway" "uvicorn app:app --host $GatewayHost --port $Port --workers $Workers --no-access-log"

Write-Host ""
Write-Host "Production stack started." -ForegroundColor Green
Write-Host "  Gateway : http://${GatewayHost}:${Port}  (chat UI + /trace.html)"
Write-Host "  Logs    : $logsDir"
Write-Host "  PIDs    : $pidFile"
Write-Host ""
Write-Host "Stop everything with:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run-prod.ps1 -Stop"
