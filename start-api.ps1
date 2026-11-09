$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project virtual environment not found. Create it with: python -m venv .venv"
}

Set-Location $projectRoot
& $python -m uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
