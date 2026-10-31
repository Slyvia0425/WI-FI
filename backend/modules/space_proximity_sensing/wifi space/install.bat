@echo off
cd /d "%~dp0"
py -3.12 fetch_data.py
if errorlevel 1 exit /b 1
py -3.12 -m venv .venv
".venv\Scripts\python.exe" -m pip install -r multi_object\requirements-night.txt
if errorlevel 1 pause
