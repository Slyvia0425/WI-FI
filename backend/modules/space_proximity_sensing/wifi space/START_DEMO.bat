@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Install the Python environment using the README first.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8766/"
".venv\Scripts\python.exe" -u app.py
pause
