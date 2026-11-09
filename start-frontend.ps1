$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $projectRoot "frontend")

if (-not (Test-Path (Join-Path (Get-Location) "node_modules"))) {
    npm install
}

npm run dev -- --host 127.0.0.1
