$ErrorActionPreference = "Stop"

$dashboardRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $dashboardRoot
$logPath = Join-Path $repoRoot "data\live\cloudflare_tunnel.log"
$errPath = Join-Path $repoRoot "data\live\cloudflare_tunnel.err"
$pidPath = Join-Path $repoRoot "data\live\cloudflare_tunnel.pid"
$existingTunnel = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*dashboard\\scripts\\cloudflare_quick_tunnel.py*' } |
  Select-Object -First 1
$strayServers = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*dashboard\\server.py*' -and $_.CommandLine -like '*8788*' } |
  Select-Object ProcessId

if ($existingTunnel) {
  $existingPid = $existingTunnel.ProcessId
  Set-Content -Path $pidPath -Value $existingPid
  Write-Output "Cloudflare tunnel already running (PID=$existingPid)."
  if (Test-Path $logPath) {
    Get-Content $logPath -Tail 20
  }
  exit 0
}

if (Test-Path $pidPath) {
  Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
}

foreach ($p in $strayServers) {
  Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

if ((Test-Path $logPath) -and -not $existingTunnel) { Remove-Item $logPath -Force -ErrorAction SilentlyContinue }
if ((Test-Path $errPath) -and -not $existingTunnel) { Remove-Item $errPath -Force -ErrorAction SilentlyContinue }

$proc = Start-Process `
  -FilePath python `
  -ArgumentList 'dashboard\scripts\cloudflare_quick_tunnel.py --host 127.0.0.1 --port 8788' `
  -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $logPath `
  -RedirectStandardError $errPath `
  -PassThru

Set-Content -Path $pidPath -Value $proc.Id

for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 1
  if (Test-Path $logPath) {
    $match = Select-String -Path $logPath -Pattern 'PUBLIC_URL=(https://\S+)' -AllMatches -ErrorAction SilentlyContinue
    if ($match) {
      $url = $match.Matches[-1].Groups[1].Value
      Write-Output "PUBLIC_URL=$url"
      exit 0
    }
  }
}

Write-Output "Tunnel started but URL not found yet. Check $logPath"
