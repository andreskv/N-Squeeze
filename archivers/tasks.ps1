<#
.SYNOPSIS
  Start, stop and inspect the N-Squeeze scheduled archivers.

.EXAMPLE
  powershell -File archivers\tasks.ps1                              # list state, last run, result, next run
  powershell -File archivers\tasks.ps1 -Action enable -Source agsi
  powershell -File archivers\tasks.ps1 -Action enable -Source all
  powershell -File archivers\tasks.ps1 -Action disable -Source ais

.NOTES
  A daily archiver re-fetches a window of days, so disable and enable act as a safe
  pause and resume: the next run fills the gap. Long back-fills use `nsq jobs`.
  Output of every run: data\logs\archivers.log (AIS: data\logs\ais_stream.log).
#>
param(
  [ValidateSet("list", "enable", "disable")][string]$Action = "list",
  [string]$Source = "all"
)
$filter = if ($Source -eq "all") { "N-Squeeze *" } else { "N-Squeeze $Source" }
$tasks = Get-ScheduledTask -TaskName $filter -ErrorAction SilentlyContinue
if (-not $tasks) { Write-Host "No task matches '$filter'. Register the tasks with archivers\schedule_windows.ps1."; exit 1 }
switch ($Action) {
  "enable" { $tasks | Enable-ScheduledTask | Out-Null; Write-Host "enabled: $($tasks.TaskName -join ', ')" }
  "disable" {
    $tasks | Where-Object State -eq "Running" | Stop-ScheduledTask
    $tasks | Disable-ScheduledTask | Out-Null
    Write-Host "disabled: $($tasks.TaskName -join ', ')"
  }
}
Get-ScheduledTask -TaskName $filter | ForEach-Object {
  $i = $_ | Get-ScheduledTaskInfo
  [pscustomobject]@{
    Task = $_.TaskName -replace "^N-Squeeze ", ""; State = $_.State
    LastRun = $i.LastRunTime; Result = $i.LastTaskResult; NextRun = $i.NextRunTime
  }
} | Format-Table -AutoSize
