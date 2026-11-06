# Server boot recovery (2026-10-22: a Windows Update reboot left WSL off, the LAN profile Public and the
# 2222 portproxy pointing at the old WSL IP -> dashboard 1033 and no SSH until a human fixed it by hand).
# Run by the scheduled task "CryptoWSLBootRecover" (install_boot_task.ps1). Idempotent; safe to run any time.
#   1) LAN profile Public -> Private   2) start WSL and keep it alive   3) portproxy 0.0.0.0:2222 -> <WSL IP>:22
#   then re-check every 60 s (WSL IP changes on every WSL start).
#   4) 2026-11-07: attach and mount the NVMe VHDX (/mnt/nvme) and the 6TB USB disk (/mnt/data2); fstab has both by UUID.
#      -MountOnly does just that once and exits (manual recovery, also how this was tested).
# ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.
param([string]$Distro = "Ubuntu", [int]$ListenPort = 2222, [int]$WslSshPort = 22, [switch]$MountOnly)
$ErrorActionPreference = "Continue"
$Log = Join-Path $PSScriptRoot "boot_recover.log"
$Wsl = "$env:WINDIR\System32\wsl.exe"
$FastVhd = "C:\wsl\fast\fast.vhdx"
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
    # wsl.exe prints errors in UTF-16: without this check the error text became the portproxy target and SSH died until the next cycle (2026-11-06, 11-07).
    $ip = if ($out) { ([string]$out).Trim() -split '\s+' | Select-Object -First 1 } else { $null }
    if ($ip -match '^\d{1,3}(\.\d{1,3}){3}$') { return $ip }
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

function Test-Mounted([string]$mp) {
    & $Wsl -d $Distro -u root -e findmnt -n $mp *> $null
    return ($LASTEXITCODE -eq 0)
}

function Mount-Target([string]$mp, [string[]]$attach, [string]$what) {
    # wsl --mount acts on the calling user's WSL VM, so this must run here (kbj20, elevated), not as SYSTEM.
    & $Wsl @attach *> $null          # already attached is an error here, and harmless
    Start-Sleep -Seconds 2
    & $Wsl -d $Distro -u root -e mount $mp *> $null
    if (Test-Mounted $mp) { Log "$what mounted -> $mp" } else { Log "$what mount FAILED -> $mp" }
}

function Fix-DataDisk {
    # 2026-11-07 /mnt/nvme = VHDX on C: holding data/hot, data/live and logs. cron waits for it at boot (see docs/data_storage_map_20261107.md).
    if (-not (Test-Mounted /mnt/nvme)) {
        if (Test-Path $FastVhd) { Mount-Target /mnt/nvme @("--mount", "--vhd", $FastVhd, "--bare") "fast vhd" }
        else { Log "fast vhd missing: $FastVhd" }
    }
    # /mnt/data2 = 6TB USB. Picked by size, not number: PHYSICALDRIVE numbers change when a disk is added.
    if (-not (Test-Mounted /mnt/data2)) {
        $d = @(Get-Disk | Where-Object { $_.BusType -eq 'USB' -and $_.Size -gt 5TB })
        if ($d.Count -ne 1) { Log "data disk: want exactly 1 USB disk over 5TB, found $($d.Count)"; return }
        Mount-Target /mnt/data2 @("--mount", "\\.\PHYSICALDRIVE$($d[0].Number)", "--bare") "6TB data disk"
    }
}

if ($MountOnly) { Fix-DataDisk; return }

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
    Fix-DataDisk
    Fix-NetworkProfile
    Start-Sleep -Seconds 60
}
