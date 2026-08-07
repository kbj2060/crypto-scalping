$ErrorActionPreference = "Stop"

$taskName = "CryptoScalpingTradingBot"
$wslDistro = $env:CRYPTO_SCALPING_WSL_DISTRO
if ([string]::IsNullOrWhiteSpace($wslDistro)) {
  $wslDistro = "Ubuntu"
}

$bashCommand = "cd /home/llewyn/crypto-scalping && scripts/start_trading_bot.sh"
$wslArgs = "-d $wslDistro bash -lc `"$bashCommand`""

$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument $wslArgs

# AtLogOn covers normal interactive sign-in; AtStartup covers reboots where the
# user never completes an interactive logon (e.g. Windows Update auto-restarts
# left sitting at the lock screen), which previously left the bot down until
# someone manually signed in.
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$startupTrigger = New-ScheduledTaskTrigger -AtStartup
$startupTrigger.Delay = "PT30S"

$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Days 365) `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -MultipleInstances IgnoreNew

# S4U lets the task run as $env:USERNAME without requiring an interactive
# logon (no stored password needed), so the AtStartup trigger can reach the
# user's own WSL distro registration even if nobody has signed in yet.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

Register-ScheduledTask `
  -TaskName $taskName `
  -Action $action `
  -Trigger @($logonTrigger, $startupTrigger) `
  -Settings $settings `
  -Principal $principal `
  -Description "Starts the crypto-scalping trading bot supervisor in WSL." `
  -Force | Out-Null

Write-Output "Registered scheduled task: $taskName"
Write-Output "Run now:"
Write-Output "  Start-ScheduledTask -TaskName $taskName"
Write-Output "Remove:"
Write-Output "  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
