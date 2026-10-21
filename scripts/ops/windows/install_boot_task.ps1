#Requires -RunAsAdministrator
# One-time setup on the server (2026-10-22 outage). Run in an *administrator* PowerShell:
#   powershell -ExecutionPolicy Bypass -File C:\Users\kbj20\crypto-ops\install_boot_task.ps1
# It changes Windows settings, so read before running. Every change is printed.
#   1) scheduled task CryptoWSLBootRecover: at startup (+30 s), runs wsl_boot_recover.ps1 whether or not
#      anyone is logged on (asks for this account's Windows password once), restarts it if it dies
#   2) Windows Update: no automatic restart while a user is logged on; active hours 08:00-02:00 (18 h max)
#   3) never sleep / hibernate on AC power
#   4) physical network adapters: turn off "allow the computer to turn off this device to save power"
# Undo: Unregister-ScheduledTask CryptoWSLBootRecover; delete the four policy values below; powercfg back.
# ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.
$ErrorActionPreference = "Stop"
$Dir = $PSScriptRoot
$Task = "CryptoWSLBootRecover"

# 1) boot task
$cred = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" -Message "Windows password for $env:USERNAME (the boot task runs without login)"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Dir\wsl_boot_recover.ps1`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT30S"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $Task -Action $action -Trigger $trigger -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Highest -Force | Out-Null
Write-Host "[1] scheduled task '$Task' registered (at startup +30s, runs without login)"

# 2) Windows Update restart limits
$wu = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate"
$au = "$wu\AU"
New-Item -Path $au -Force | Out-Null
Set-ItemProperty -Path $au -Name NoAutoRebootWithLoggedOnUsers -Value 1 -Type DWord
Set-ItemProperty -Path $wu -Name SetActiveHours -Value 1 -Type DWord
Set-ItemProperty -Path $wu -Name ActiveHoursStart -Value 8 -Type DWord
Set-ItemProperty -Path $wu -Name ActiveHoursEnd -Value 2 -Type DWord
Write-Host "[2] Windows Update: no auto restart while logged on; active hours 08:00-02:00"

# 3) no sleep / hibernate on AC
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
Write-Host "[3] sleep and hibernate on AC: never"

# 4) NIC power saving off (may blink the network for a second)
Get-NetAdapter -Physical | ForEach-Object {
    Disable-NetAdapterPowerManagement -Name $_.Name -NoRestart -ErrorAction SilentlyContinue
    Write-Host "[4] power saving off: $($_.Name)"
}

Start-ScheduledTask -TaskName $Task
Start-Sleep -Seconds 20
Write-Host "task state: $((Get-ScheduledTask -TaskName $Task).State)"
Write-Host "log: $Dir\boot_recover.log"
Get-Content "$Dir\boot_recover.log" -Tail 5 -ErrorAction SilentlyContinue
netsh interface portproxy show v4tov4
