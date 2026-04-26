$ErrorActionPreference = "Stop"

$ruleName = "CryptoScalping Dashboard 8787"
$port = 8787
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

if ($existing) {
  Write-Output "Firewall rule already exists: $ruleName"
  exit 0
}

New-NetFirewallRule `
  -DisplayName $ruleName `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort $port | Out-Null

Write-Output "Opened Windows Firewall TCP port $port with rule '$ruleName'"
