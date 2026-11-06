@echo off
if exist "%~dp0mumu-panel.exe" (
    start "" "%~dp0mumu-panel.exe" desktop
) else if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0desktop_entry.py" desktop
) else (
    echo Please extract the complete Windows release first.
    pause
)
exit /b
