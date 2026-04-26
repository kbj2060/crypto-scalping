$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $repoRoot "data\live\cloudflare_tunnel.pid"
$procs = Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like '*cloudflare_quick_tunnel.py*' -or
    ($_.CommandLine -like '*serve_dashboard.py*' -and $_.CommandLine -like '*8788*')
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
