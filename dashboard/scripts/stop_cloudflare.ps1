$ErrorActionPreference = "Stop"

$dashboardRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $dashboardRoot
$pidPath = Join-Path $repoRoot "data\live\cloudflare_tunnel.pid"
$procs = Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like '*dashboard\\scripts\\cloudflare_quick_tunnel.py*' -or
    ($_.CommandLine -like '*dashboard\\server.py*' -and $_.CommandLine -like '*8788*')
  } |
  Select-Object ProcessId

foreach ($p in $procs) {
  Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  Write-Output "Stopped process PID=$($p.ProcessId)"
}

if (Test-Path $pidPath) {
  Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
} elseif (-not $procs) {
  Write-Output "No tunnel PID file found."
}
