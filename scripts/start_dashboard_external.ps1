$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$logPath = Join-Path $repoRoot "data\live\dashboard_external.log"
$errPath = Join-Path $repoRoot "data\live\dashboard_external.err"
$pidPath = Join-Path $repoRoot "data\live\dashboard_external.pid"
$port = 8787

$existing = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*serve_dashboard.py*' -and $_.CommandLine -like "*--port $port*" } |
  Select-Object -First 1

if ($existing) {
  Set-Content -Path $pidPath -Value $existing.ProcessId
  Write-Output "Dashboard server already running (PID=$($existing.ProcessId), port=$port)."
  Write-Output "LOCAL_URL=http://127.0.0.1:$port/dashboard/live/"
  exit 0
}

if (Test-Path $logPath) { Remove-Item $logPath -Force -ErrorAction SilentlyContinue }
if (Test-Path $errPath) { Remove-Item $errPath -Force -ErrorAction SilentlyContinue }
if (Test-Path $pidPath) { Remove-Item $pidPath -Force -ErrorAction SilentlyContinue }

$proc = Start-Process `
  -FilePath python `
  -ArgumentList "scripts\serve_dashboard.py --host 0.0.0.0 --port $port" `
  -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $logPath `
  -RedirectStandardError $errPath `
  -PassThru

Set-Content -Path $pidPath -Value $proc.Id

for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Milliseconds 500
  try {
    $resp = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/dashboard/live/" -TimeoutSec 2
    if ($resp.StatusCode -eq 200) {
      Write-Output "Dashboard server started (PID=$($proc.Id), port=$port)."
      Write-Output "LOCAL_URL=http://127.0.0.1:$port/dashboard/live/"
      exit 0
    }
  } catch {
  }
}

Write-Output "Dashboard server started, but readiness check did not complete yet."
Write-Output "Check $logPath"
