@echo off
REM Open the OVERBOARD browser studio. Double-click this file, or run it from
REM a terminal. Press Ctrl+C in the terminal to stop the studio.
REM
REM On a machine that has never run the studio, this installs the virtual
REM environment first, so one double-click is enough to get going.
setlocal
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo.
  echo No virtual environment yet. Installing it now - this takes a few minutes.
  echo.
  call install.bat --quiet
  if errorlevel 1 (
    echo.
    echo The install failed. Fix the [fail] lines above, then run this file again.
    pause
    exit /b 1
  )
)

if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

where python >nul 2>nul
if errorlevel 1 if not exist "%~dp0.venv\Scripts\python.exe" (
  echo.
  echo Python was not found on PATH. Install Python 3.9 or newer from
  echo https://www.python.org/downloads/ , or run install.bat to set
  echo everything up, then run this file again.
  pause
  exit /b 1
)

echo Checking the environment...
"%PY%" -m overboard.main --doctor
if errorlevel 1 (
  echo.
  echo The environment check failed. Fix the [fail] lines above, then run this file again.
  echo If the packages are missing, run install.bat first.
  pause
  exit /b 1
)

echo.
echo Starting the studio...
"%PY%" -m overboard.main --gui %*
if errorlevel 1 pause
endlocal
