@echo off
REM Open the OVERBOARD browser studio. Double-click this file, or run it from
REM a terminal. Press Ctrl+C in the terminal to stop the studio.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH. Install Python 3.8 or newer and try again.
  pause
  exit /b 1
)

python main.py --doctor
if errorlevel 1 (
  echo.
  echo The environment check failed. Fix the [fail] lines above, then run this file again.
  pause
  exit /b 1
)

python main.py --gui %*
if errorlevel 1 pause
endlocal
