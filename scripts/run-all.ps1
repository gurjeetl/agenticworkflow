# Launch the full multi-process stack for local development.
#
#   MLflow server   : 5000   (python -m mlflow server ...)   <- tracing backend
#   MCP server      : 8001   (python -m mcp_server.weather_server)
#   Registry service: 8002   (python -m registry.service)
#   Weather agent   : 8010   (python -m agents.weather_agent)
#   Outage agent    : 8011   (python -m agents.outage_agent)
#   RAG agent       : 8012   (python -m agents.rag_agent)
#   Main app        : 8000   (python main.py)
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

function Start-Svc($title, $cmd, $svcEnv = @{}) {
    $pre = ($svcEnv.GetEnumerator() | ForEach-Object { "`$env:$($_.Key)='$($_.Value)';" }) -join " "
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$Host.UI.RawUI.WindowTitle='$title'; Set-Location '$root'; $pre $cmd"
    )
}

$registry = "http://127.0.0.1:8002"
$mlflow   = "http://127.0.0.1:5000"

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

# MLflow backend store: PostgreSQL DSN from .env (MLFLOW_BACKEND_STORE_URI).
# Required — no SQLite fallback.
$mlflowBackend = ""
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    $line = Get-Content $envFile | Where-Object { $_ -match '^\s*MLFLOW_BACKEND_STORE_URI\s*=' } | Select-Object -First 1
    if ($line) { $mlflowBackend = ($line -replace '^\s*MLFLOW_BACKEND_STORE_URI\s*=\s*', '').Trim() }
}
if (-not $mlflowBackend) {
    Write-Host "ERROR: MLFLOW_BACKEND_STORE_URI is not set in .env (PostgreSQL DSN required)." -ForegroundColor Red
    exit 1
}

# MLflow tracking server (PostgreSQL backend). Must be up before the app/agents
# so their trace spans have somewhere to go.
Start-Svc "MLflow :5000" "$py -m mlflow server --backend-store-uri $mlflowBackend --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000"
Start-Sleep -Seconds 5

Start-Svc "MCP :8001"      "$py -m mcp_server.weather_server"
Start-Svc "Registry :8002" "$py -m registry.service" @{ REGISTRY_PORT = "8002" }
Start-Sleep -Seconds 2

Start-Svc "Weather :8010" "$py -m agents.weather_agent" @{ AGENT_PORT = "8010"; REGISTRY_URL = $registry; MLFLOW_TRACKING_URI = $mlflow }
Start-Svc "Outage :8011"  "$py -m agents.outage_agent"  @{ AGENT_PORT = "8011"; REGISTRY_URL = $registry; MLFLOW_TRACKING_URI = $mlflow }
Start-Svc "RAG :8012"     "$py -m agents.rag_agent"     @{ AGENT_PORT = "8012"; REGISTRY_URL = $registry; MLFLOW_TRACKING_URI = $mlflow }
Start-Sleep -Seconds 2

Start-Svc "App :8000" "$py main.py" @{ REGISTRY_URL = $registry; MLFLOW_TRACKING_URI = $mlflow }

Write-Host "All services launched."
Write-Host "  Chat UI  : http://127.0.0.1:8000"
Write-Host "  Trace UI : http://127.0.0.1:8000/trace.html"
Write-Host "  Registry : $registry/agents"
Write-Host "  MLflow   : $mlflow"
