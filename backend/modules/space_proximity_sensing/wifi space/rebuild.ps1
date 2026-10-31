$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create the Python 3.12 virtual environment and install requirements.txt first.' }
if (-not (Test-Path 'data\manifest_phase1.json')) {
    & $taskPython -u simulate.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $taskPython -u train.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $taskPython -u robust_generate.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -u train.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -u make_reports.py
exit $LASTEXITCODE
