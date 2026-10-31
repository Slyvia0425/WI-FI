$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
                                                                       
& $taskPython -u opt_v2_generate.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -u opt_v2_train.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -u opt_v2_diagnostics.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -u opt_v2_report.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Results are ready. Review OPTIMIZATION_RESULTS.md before activating a new model.'
