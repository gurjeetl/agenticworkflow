# Launch the full multi-process stack for local development.
#
#   MLflow server   : 5000   (python -m mlflow server ...)   <- tracing backend
#   MCP server      : 8001   (python -m services.mcp.weather_server)
#   Registry service: 8002   (python -m services.registry.server)
#   RAG service     : 8003   (python -m services.rag.server)
#   Weather agent   : 8010   (python -m applications.demo.weather.agent)
#   Outage agent    : 8011   (python -m applications.demo.outage.agent)
#   RAG agent       : 8012   (python -m applications.demo.rag.agent)
#   Main app        : 8000   (python src\app.py)
#
# Each service opens in its own titled window. Order matters: MLflow comes up
# first (the app + agents log traces to it), the registry must be up before
# agents self-register, and agents should be registering before the app's
# Planner queries the menu (the client's TTL cache + the Planner's empty-plan
# tolerance cover the brief startup race).
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run-all.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# Prefer a project venv python if present, else fall back to PATH python.
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $venvPy) { $venvPy } else { "python" }

# After the genie-platform restructure, importable code lives under src/ (the
# `genie` and `applications` packages) while `services.*` and the `app` entry
# resolve from the repo root (the working dir). Putting src/ on PYTHONPATH makes
# every service importable even without `pip install -e .`.
$src = Join-Path $root "src"

# Configuration is YAML-driven (config/default.yaml + the gitignored
# config/local.yaml). The app/agents/services read every setting from there via
# genie.platform.config, so this launcher injects ONLY the things YAML can't carry:
# PYTHONPATH (to import the packages) and AGENT_PORT (a distinct port per agent).
function Start-Svc($title, $cmd, $svcEnv = @{}) {
    $pre = ($svcEnv.GetEnumerator() | ForEach-Object { "`$env:$($_.Key)='$($_.Value)';" }) -join " "
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$Host.UI.RawUI.WindowTitle='$title'; Set-Location '$root'; $pre $cmd"
    )
}

# Read a flat scalar key from the YAML config (local.yaml wins over default.yaml),
# falling back to the legacy UPPER_SNAKE name in .env for back-compat.
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

$mlflow = "http://127.0.0.1:5000"

# Close any service windows left over from a previous run. The service windows
# use -NoExit (so their logs stay readable), which means a `taskkill python.exe`
# leaves the empty host window behind. Under Windows Terminal a single process
# owns every tab, so MainWindowTitle can't identify our hosts -- instead match the
# launch command line, which carries a `RawUI.WindowTitle='<service>'` marker that
# only our service windows have. This makes re-running idempotent: it never stacks
# up duplicate windows, and the launcher/cmd/editor shells are never a match.
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'RawUI\.WindowTitle' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Redis (blackboard hot mirror) — start the local portable server if present and
# not already running. Optional: the app degrades gracefully when Redis is down.
$redisExe = Join-Path $env:LOCALAPPDATA "redis-portable\redis-server.exe"
if (Test-Path $redisExe) {
    if (-not (Get-NetTCPConnection -LocalPort 6379 -State Listen -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $redisExe -ArgumentList "--port", "6379" -WindowStyle Hidden
        Write-Host "Started Redis on :6379"
    } else { Write-Host "Redis already running on :6379" }
} else { Write-Host "Redis not found at $redisExe - blackboard mirror disabled" }

# MLflow backend store: PostgreSQL DSN from config/local.yaml
# (mlflow_backend_store_uri). Required — no SQLite fallback.
$mlflowBackend = Get-ConfigValue 'mlflow_backend_store_uri'
if (-not $mlflowBackend) {
    Write-Host "ERROR: mlflow_backend_store_uri is not set in config/local.yaml (PostgreSQL DSN required)." -ForegroundColor Red
    Write-Host "       Copy config/local.yaml.example to config/local.yaml and fill it in." -ForegroundColor Red
    exit 1
}

# MLflow tracking server (PostgreSQL backend). Must be up before the app/agents
# so their trace spans have somewhere to go.
Start-Svc "MLflow :5000" "$py -m mlflow server --backend-store-uri $mlflowBackend --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000"
Start-Sleep -Seconds 5

Start-Svc "MCP :8001"      "$py -m services.mcp.weather_server" @{ PYTHONPATH = $src }
Start-Svc "Registry :8002" "$py -m services.registry.server" @{ PYTHONPATH = $src }
Start-Svc "RAG Service :8003" "$py -m services.rag.server" @{ PYTHONPATH = $src }
Start-Sleep -Seconds 2

# AGENT_PORT is the one per-agent setting YAML can't carry (each needs a distinct
# port); everything else (registry URL, MLflow URI, ...) comes from the YAML config.
Start-Svc "Weather :8010" "$py -m applications.demo.weather.agent" @{ AGENT_PORT = "8010"; PYTHONPATH = $src }
Start-Svc "Outage :8011"  "$py -m applications.demo.outage.agent"  @{ AGENT_PORT = "8011"; PYTHONPATH = $src }
Start-Svc "RAG :8012"     "$py -m applications.demo.rag.agent"     @{ AGENT_PORT = "8012"; PYTHONPATH = $src }
Start-Sleep -Seconds 2

Start-Svc "App :8000" "$py src\app.py" @{ PYTHONPATH = $src }

$registry = (Get-ConfigValue 'registry_url'); if (-not $registry) { $registry = "http://127.0.0.1:8002" }
Write-Host "All services launched."
Write-Host "  Chat UI  : http://127.0.0.1:8000"
Write-Host "  Trace UI : http://127.0.0.1:8000/trace.html"
Write-Host "  Registry : $registry/agents"
Write-Host "  MLflow   : $mlflow"
