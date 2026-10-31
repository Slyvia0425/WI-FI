@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\serve_integrated.py --spatial-mode lite
) else (
  python scripts\serve_integrated.py --spatial-mode lite
)
if errorlevel 1 pause
