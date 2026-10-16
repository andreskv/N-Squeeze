<#
.SYNOPSIS
  Runs one archiver with keys loaded from .env. Called by Task Scheduler.
#>
param([Parameter(Mandatory)][string]$Source, [int]$Days = 10)

$repo = (Resolve-Path "$PSScriptRoot\..").Path
$envFile = Join-Path $repo ".env"
if (Test-Path $envFile) {
  Get-Content $envFile | Where-Object { $_ -match '^\s*([A-Z_]+)\s*=\s*([^#\s]+)' } | ForEach-Object {
    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
  }
}
$log = Join-Path $repo "data\logs\archivers.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
if ($Source -eq "ais") {
  # Long-running stream, it reconnects by itself. Stop it by disabling the task (tasks.ps1).
  & (Join-Path $repo ".venv\Scripts\python.exe") (Join-Path $repo "archivers\ais_stream.py") 2>&1 |
    Out-File -Append -Encoding utf8 (Join-Path $repo "data\logs\ais_stream.log")
  exit $LASTEXITCODE
}
if ($Source -eq "fan_chart") {
  # Weekly forecast entry; nsq refuses to write outside the pre-registered window.
  $out = & (Join-Path $repo ".venv\Scripts\nsq.exe") fan-chart 2>&1
} else {
  $out = & (Join-Path $repo ".venv\Scripts\nsq.exe") archive $Source --days $Days 2>&1
}
"$stamp $Source exit=$LASTEXITCODE $out" | Out-File -Append -Encoding utf8 $log
exit $LASTEXITCODE
