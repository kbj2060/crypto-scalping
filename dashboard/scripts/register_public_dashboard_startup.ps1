$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$taskName = "CryptoScalpingPublicDashboard"
$wslDistro = $env:CRYPTO_SCALPING_WSL_DISTRO
if ([string]::IsNullOrWhiteSpace($wslDistro)) {
  $wslDistro = "Ubuntu"
}

$bashCommand = "cd /home/llewyn/crypto-scalping && dashboard/scripts/start_public_dashboard.sh"
$wslArgs = "-d $wslDistro bash -lc `"$bashCommand`""

$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument $wslArgs
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Days 365) `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
  -TaskName $taskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Starts the crypto-scalping dashboard origin and Cloudflare tunnel in WSL." `
  -Force | Out-Null

Write-Output "Registered scheduled task: $taskName"
Write-Output "Run now:"
Write-Output "  Start-ScheduledTask -TaskName $taskName"
Write-Output "Remove:"
Write-Output "  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
