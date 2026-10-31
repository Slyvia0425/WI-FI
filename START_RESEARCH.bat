@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\serve_integrated.py --spatial-mode research
) else (
  python scripts\serve_integrated.py --spatial-mode research
)
if errorlevel 1 pause
