@echo off
cd /d "%~dp0"
python scripts\serve_integrated.py
if errorlevel 1 pause
