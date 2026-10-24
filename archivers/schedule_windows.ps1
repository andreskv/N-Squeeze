<#
.SYNOPSIS
  Registers the N-Squeeze archivers in Windows Task Scheduler (current user).

.DESCRIPTION
  Review before running: this creates scheduled tasks named "N-Squeeze <source>".
  Every task is registered DISABLED. Start and stop them with archivers\tasks.ps1.
  Remove them with:  Get-ScheduledTask -TaskName "N-Squeeze*" | Unregister-ScheduledTask -Confirm:$false
  Keys are read from .env by run_archiver.ps1 at run time; nothing secret is stored in the task.
  -Unattended (always-on machine, run as administrator): tasks run whether or not the user
  is logged on (S4U logon, no stored password), and the AIS stream starts at boot.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File archivers\schedule_windows.ps1 -WhatIf
  powershell -ExecutionPolicy Bypass -File archivers\schedule_windows.ps1 -Unattended
#>
[CmdletBinding(SupportsShouldProcess)]
param([string]$Repo, [switch]$Unattended)
# Not a param default: Windows PowerShell 5.1 can leave $PSScriptRoot empty there,
# which pointed every task at C:\archivers.
if (-not $Repo) { $Repo = Split-Path -Parent $PSScriptRoot }
if (-not (Test-Path (Join-Path $Repo "archivers\run_archiver.ps1"))) { throw "Repo not found: $Repo" }

$runner = Join-Path $Repo "archivers\run_archiver.ps1"
$logon = if ($Unattended) { "S4U" } else { "Interactive" }
# The token's own name: $env:USERDOMAIN can be wrong in an SSH session
$me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $me -LogonType $logon
$jobs = @(
  # GIE publishes once a day at 19:30
  @{ Source = "agsi";      At = "08:30" },
  @{ Source = "alsi";      At = "08:35" },
  @{ Source = "entsog";    At = "08:40" },
  @{ Source = "weather";   At = "09:00" },
  @{ Source = "entsoe";    At = "09:05" },
  @{ Source = "ecb_fx";    At = "18:45" },
  @{ Source = "eua_auctions"; At = "18:50" },
  @{ Source = "henry_hub"; At = "23:50" },
  @{ Source = "ttf_daily_candidate"; At = "23:55" },
  # Weekly (Monday): snapshots of revisable or slow-moving sources
  @{ Source = "portwatch";  At = "07:00"; Weekly = $true; Days = 21 },
  @{ Source = "comext";     At = "07:10"; Weekly = $true; Days = 400 },
  @{ Source = "pink_sheet_monthly"; At = "07:30"; Weekly = $true; Days = 400 },  # World Bank, updated early each month
  @{ Source = "eurostat_hdd"; At = "07:15"; Weekly = $true; Days = 400 },
  @{ Source = "eurostat_population"; At = "07:20"; Weekly = $true; Days = 800 },
  @{ Source = "eurostat_gas_trade"; At = "07:25"; Weekly = $true; Days = 400 },
  @{ Source = "s2_swir"; At = "07:40"; Weekly = $true; Days = 21 },
  @{ Source = "eurostat_c2015"; At = "07:35"; Weekly = $true; Days = 800 },
  @{ Source = "ets_registry"; At = "07:45"; Weekly = $true; Days = 4400 },  # one file a year (April); vintages kept
  @{ Source = "fan_chart"; At = "2027-01-04 09:30"; Weekly = $true },  # first entry; after the 08:30 AGSI+ update
  # Live AIS stream: starts at log-on (or boot) and runs until the task is disabled. Unattended, it is
  # also started every hour; Task Scheduler ignores the start while the stream already runs
  @{ Source = "ais"; AtLogOn = $true }
)

foreach ($j in $jobs) {
  $name = "N-Squeeze $($j.Source)"
  $days = if ($j.Days) { $j.Days } else { 10 }
  $action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -Source $($j.Source) -Days $days" `
    -WorkingDirectory $Repo
  $trigger = if ($j.AtLogOn -and $Unattended) {
               @((New-ScheduledTaskTrigger -AtStartup),
                 (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)))
             }
             elseif ($j.AtLogOn) { New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME }
             elseif ($j.Weekly) { New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At $j.At }
             else { New-ScheduledTaskTrigger -Daily -At $j.At }
  $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 30)
  if ($j.AtLogOn) { $settings.ExecutionTimeLimit = "PT0S" }  # the stream has no time limit
  $when = if ($j.AtLogOn -and $Unattended) { "at boot" } elseif ($j.AtLogOn) { "at log-on" } else { $j.At }
  if ($PSCmdlet.ShouldProcess($name, "Register scheduled task ($when), disabled")) {
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force -ErrorAction Stop | Out-Null
    Disable-ScheduledTask -TaskName $name -ErrorAction Stop | Out-Null  # I enable each task by hand
    Write-Host "registered $name ($when, disabled)"
  }
}
