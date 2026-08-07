param(
  [string]$WslIp = ""
)

$ErrorActionPreference = "Stop"

$port = 8787
$ruleName = "CryptoDashboard8787"
$legacyRuleName = "CryptoScalping Dashboard 8787"
$cmd = Join-Path $env:SystemRoot "System32\cmd.exe"
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
  throw "Run this script from an Administrator PowerShell. netsh portproxy and firewall changes require admin rights."
}

if (-not $WslIp) {
  $wslIpRaw = & wsl.exe hostname -I
  $WslIp = ($wslIpRaw -split "\s+" | Where-Object { $_ -match "^\d+\.\d+\.\d+\.\d+$" } | Select-Object -First 1)
}
if (-not $WslIp) {
  throw "Could not detect WSL IPv4 address from 'wsl.exe hostname -I'."
}

& $cmd /c "netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$port" | Out-Null
& $cmd /c "netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$port connectaddress=$WslIp connectport=$port" | Out-Null

Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
Get-NetFirewallRule -DisplayName $legacyRuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule

New-NetFirewallRule `
  -DisplayName $ruleName `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort $port `
  -Profile Any | Out-Null

$lanIp = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null } | ForEach-Object { $_.IPv4Address.IPAddress } | Select-Object -First 1)

Write-Output "Dashboard external access refreshed."
Write-Output "WSL_TARGET=$WslIp`:$port"
Write-Output "WINDOWS_LAN_URL=http://$lanIp`:$port/dashboard/live/"
Write-Output "FIREWALL_RULE=$ruleName"
