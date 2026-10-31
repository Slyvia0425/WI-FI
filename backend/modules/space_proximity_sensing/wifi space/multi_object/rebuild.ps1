$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
foreach ($taskScript in @('multi_simulate.py','multi_train.py','occupancy.py','quality_audit.py','make_report.py')) {
  & $taskPython -u $taskScript
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
