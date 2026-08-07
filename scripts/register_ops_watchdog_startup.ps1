$ErrorActionPreference = "Stop"

$taskName = "CryptoScalpingOpsWatchdog"
$wslDistro = $env:CRYPTO_SCALPING_WSL_DISTRO
if ([string]::IsNullOrWhiteSpace($wslDistro)) { $wslDistro = "Ubuntu" }

$bashCommand = "cd /home/llewyn/crypto-scalping && exec scripts/ops/supervisor_ops_watchdog.sh"
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d $wslDistro bash -lc `"$bashCommand`""
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$startupTrigger = New-ScheduledTaskTrigger -AtStartup
$startupTrigger.Delay = "PT30S"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 365) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($logonTrigger, $startupTrigger) -Settings $settings -Principal $principal -Description "Keeps the crypto-scalping operations watchdog supervisor alive in WSL." -Force | Out-Null
Write-Output "Registered scheduled task: $taskName"
