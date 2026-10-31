$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
foreach ($taskScript in @('coverage_generate.py','coverage_train.py','coverage_activate.py','coverage_report.py')) {
    & $taskPython -u $taskScript
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Output 'Reload http://127.0.0.1:8766/ and run verify.py with the service running.'
