@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "multi_object\outputs\spatial_v6\default\model.joblib" (
  call install.bat
  if errorlevel 1 exit /b 1
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py
) else if exist "D:\Users\AIS课件\wifi_spatial_sensing\static_obstacle\.venv\Scripts\python.exe" (
  "D:\Users\AIS课件\wifi_spatial_sensing\static_obstacle\.venv\Scripts\python.exe" run.py
) else (
  py -3.12 run.py
)
if errorlevel 1 pause
