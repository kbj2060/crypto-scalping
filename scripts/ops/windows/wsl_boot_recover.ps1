# Server boot recovery (2026-10-22: a Windows Update reboot left WSL off, the LAN profile Public and the
# 2222 portproxy pointing at the old WSL IP -> dashboard 1033 and no SSH until a human fixed it by hand).
# Run by the scheduled task "CryptoWSLBootRecover" (install_boot_task.ps1). Idempotent; safe to run any time.
#   1) LAN profile Public -> Private   2) start WSL and keep it alive   3) portproxy 0.0.0.0:2222 -> <WSL IP>:22
#   then re-check every 60 s (WSL IP changes on every WSL start).
# ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.
param([string]$Distro = "Ubuntu", [int]$ListenPort = 2222, [int]$WslSshPort = 22)
$ErrorActionPreference = "Continue"
$Log = Join-Path $PSScriptRoot "boot_recover.log"
$Wsl = "$env:WINDIR\System32\wsl.exe"
function Log([string]$m) { "$(Get-Date -Format s) $m" | Add-Content -Path $Log -Encoding UTF8 }

function Fix-NetworkProfile {
    Get-NetConnectionProfile -ErrorAction SilentlyContinue |
        Where-Object { $_.NetworkCategory -eq 'Public' -and $_.InterfaceAlias -notlike 'vEthernet*' } |
        ForEach-Object {
            Set-NetConnectionProfile -InterfaceIndex $_.InterfaceIndex -NetworkCategory Private
            Log "network profile '$($_.InterfaceAlias)': Public -> Private"
        }
}

function Get-WslIp {
    $out = & $Wsl -d $Distro -u root -e hostname -I 2>$null
    if ($out) { return ($out.Trim() -split '\s+')[0] }
    return $null
}

function Fix-PortProxy([string]$ip) {
    $cur = (netsh interface portproxy show v4tov4) | Select-String "^\s*0\.0\.0\.0\s+$ListenPort\s+(\S+)\s+(\d+)"
    if ($cur -and $cur.Matches[0].Groups[1].Value -eq $ip -and $cur.Matches[0].Groups[2].Value -eq "$WslSshPort") { return }
    netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$ListenPort | Out-Null
    netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$ListenPort connectaddress=$ip connectport=$WslSshPort | Out-Null
    Log "portproxy 0.0.0.0:$ListenPort -> ${ip}:$WslSshPort"
}

function Start-KeepAlive {
    # A long-running process inside the distro keeps the WSL VM up even with no user session.
    $p = Start-Process -FilePath $Wsl -ArgumentList "-d $Distro -u root -e sleep infinity" -WindowStyle Hidden -PassThru
    Log "wsl keepalive started (pid $($p.Id))"
    return $p
}

Log "start as $env:USERNAME"
for ($i = 0; $i -lt 36; $i++) {   # wait up to 3 min for the network
    if (Test-Connection -ComputerName 1.1.1.1 -Count 1 -Quiet) { break }
    Start-Sleep -Seconds 5
}
Fix-NetworkProfile
$keep = Start-KeepAlive
while ($true) {
    $ip = Get-WslIp
    if ($ip) { Fix-PortProxy $ip } else { Log "WSL IP unavailable (WSL not up yet?)" }
    if ($keep.HasExited) { $keep = Start-KeepAlive }
    Fix-NetworkProfile
    Start-Sleep -Seconds 60
}
