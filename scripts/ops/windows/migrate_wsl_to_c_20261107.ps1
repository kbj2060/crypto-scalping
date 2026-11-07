# Move the server's WSL disk (ext4.vhdx) from D: to C: (2026-11-07). See docs/data_storage_map_20261107.md.
# Stops WSL, compacts the vhdx in place, moves it, restarts WSL through the boot task. A failed step leaves the disk
# where it is and WSL is started again either way.
#   -Check   print sizes and what would happen, change nothing
# Run it as a scheduled task or from a Windows shell, not from inside WSL: "wsl --shutdown" ends every WSL session.
# Everything is logged to migrate_wsl.log next to this file.
# ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.
param([switch]$Check, [string]$Distro = "Ubuntu", [string]$Dest = "C:\wsl\Ubuntu", [int]$MinFreeAfterGB = 80)
$ErrorActionPreference = "Continue"
$Log = Join-Path $PSScriptRoot "migrate_wsl.log"
$Wsl = "$env:WINDIR\System32\wsl.exe"
$BootTask = "CryptoWSLBootRecover"
$TunnelTask = "CryptoScalpingPublicDashboard"
function Log([string]$m) { $l = "$(Get-Date -Format s) $m"; Write-Host $l; $l | Add-Content -Path $Log -Encoding UTF8 }
function FreeGB([string]$drive) { [math]::Round((Get-PSDrive $drive).Free / 1GB, 1) }
function SizeGB([string]$p) { [math]::Round((Get-Item $p).Length / 1GB, 1) }

$base = (Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss\*" | Where-Object { $_.DistributionName -eq $Distro }).BasePath -replace '^\\\\\?\\', ''
$vhdx = Join-Path $base "ext4.vhdx"
Log "vhdx=$vhdx size=$(SizeGB $vhdx)GB C_free=$(FreeGB C)GB D_free=$(FreeGB D)GB dest=$Dest"
if ($Check) { & $Wsl -d $Distro -e df -h / | Out-String | Write-Host; return }
if ($vhdx -like "C:*") { Log "already on C:, nothing to do"; return }
if ((Test-Path $Dest) -and (Get-ChildItem $Dest -Force | Measure-Object).Count -gt 0) { Log "destination not empty, stopping"; return }

try {
    & $Wsl -d $Distro -u root -e fstrim -v / 2>&1 | Out-String | ForEach-Object { Log "fstrim: $($_.Trim())" }
    Stop-ScheduledTask -TaskName $BootTask      # its loop would start WSL again within 60 s
    & $Wsl --shutdown
    Log "wsl shutdown rc=$LASTEXITCODE"
    Start-Sleep -Seconds 15

    $dp = Join-Path $env:TEMP "compact_vhdx.txt"
    "select vdisk file=`"$vhdx`"`r`nattach vdisk readonly`r`ncompact vdisk`r`ndetach vdisk`r`nexit" | Set-Content -Path $dp -Encoding ASCII
    Log "compact start"
    $out = diskpart /s $dp 2>&1 | Out-String
    Log "compact rc=$LASTEXITCODE size=$(SizeGB $vhdx)GB"
    Remove-Item $dp -ErrorAction SilentlyContinue

    $after = (FreeGB C) - (SizeGB $vhdx)
    if ($after -lt $MinFreeAfterGB) {
        Log "NOT moving: C: would have $after GB free, want $MinFreeAfterGB. Disk stays on D:"
    } else {
        Log "move start"
        & $Wsl --manage $Distro --move $Dest 2>&1 | Out-String | ForEach-Object { Log "move: $(($_ -replace "`0", '').Trim())" }
        Log "move rc=$LASTEXITCODE; on C: $(Test-Path (Join-Path $Dest 'ext4.vhdx')); on old path $(Test-Path $vhdx)"
    }
} finally {
    Start-ScheduledTask -TaskName $BootTask     # starts WSL, keepalive, 6TB mount, portproxy
    Start-Sleep -Seconds 30
    Start-ScheduledTask -TaskName $TunnelTask -ErrorAction SilentlyContinue
    Log "boot task restarted. C_free=$(FreeGB C)GB D_free=$(FreeGB D)GB"
    Unregister-ScheduledTask -TaskName "CryptoWSLMigrate" -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "CryptoWSLMigrateFallback" -Confirm:$false -ErrorAction SilentlyContinue
}
