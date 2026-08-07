$ErrorActionPreference = "Stop"

$dashboardRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $dashboardRoot
$pidPath = Join-Path $repoRoot "data\live\dashboard_external.pid"
$targets = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*dashboard\\server.py*' -and $_.CommandLine -like '*--port 8787*' } |
  Select-Object ProcessId

foreach ($target in $targets) {
  Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
  Write-Output "Stopped dashboard server PID=$($target.ProcessId)"
}

if (Test-Path $pidPath) {
  Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
} elseif (-not $targets) {
  Write-Output "No external dashboard server was running."
}
