@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r "backend\modules\space_proximity_sensing\wifi space\multi_object\requirements-night.txt"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" scripts\prepare_spatial_lite.py
if errorlevel 1 pause
