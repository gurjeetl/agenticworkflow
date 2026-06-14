@echo off
REM Hard refresh: kill the stack, wipe ephemeral state, relaunch everything.
REM Run from the project root:  scripts\hard-refresh.bat
REM Safe to run inside cmd.exe -- batch files read line-by-line, so the delay
REM below never swallows the launch command (the paste-into-cmd gotcha).

setlocal
cd /d "%~dp0\.."

echo [hard-refresh] Killing python / milvus...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM milvus-lite.exe >nul 2>&1

echo [hard-refresh] Waiting for ports to release...
timeout /t 2 /nobreak >nul

echo [hard-refresh] Dropping Mongo 'agent_memory'...
.venv\Scripts\python.exe -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017').drop_database('agent_memory'); print('mongo: dropped agent_memory')"

echo [hard-refresh] Removing Milvus stores...
rmdir /S /Q milvus_store.db >nul 2>&1
rmdir /S /Q milvus_local.db >nul 2>&1

echo [hard-refresh] Launching stack...
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-all.ps1

endlocal
