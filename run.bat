@echo off
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "run_preview.py"
exit /b 0
