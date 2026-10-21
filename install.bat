@echo off
REM Install OVERBOARD into a local virtual environment (.venv) and check it.
REM
REM Double-click this file, or run it from a terminal. It is safe to run it
REM again: an existing .venv is reused and the packages are simply updated.
REM
REM Pass --quiet to skip the "press a key" at the end. start_studio.bat does
REM that when it calls this file.
setlocal
cd /d "%~dp0"

echo.
echo OVERBOARD setup
echo ==============
echo.

REM --- 1. find a Python 3.9 or newer -------------------------------------
set "PY="
call :try_python "py -3"
call :try_python "python"
call :try_python "python3"
if not defined PY goto :no_python
echo [ok]   Python found: %PY%

REM --- 2. the virtual environment -----------------------------------------
if exist "%~dp0.venv\Scripts\python.exe" (
  echo [ok]   Reusing the existing .venv
) else (
  echo [....] Creating the virtual environment .venv
  %PY% -m venv "%~dp0.venv"
  if errorlevel 1 goto :no_venv
)
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto :no_venv

REM --- 3. the packages ----------------------------------------------------
echo [....] Installing the solver and the studio. This can take a few minutes.
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :no_packages
echo [ok]   Core packages installed

echo [....] Installing the optional extras (validation, calibration, hub push)
"%VENV_PY%" -m pip install -r requirements-optional.txt
if errorlevel 1 (
  echo [warn] The optional extras did not install. The solver and the studio
  echo        still work, but --validate and the hub tools do not.
  echo        Run this file again once the network behaves.
) else (
  echo [ok]   Optional extras installed
)

REM --- 4. the check -------------------------------------------------------
echo.
"%VENV_PY%" -m overboard.main --doctor
if errorlevel 1 goto :failed

echo.
echo Ready. Start the studio by double-clicking start_studio.bat,
echo or from a terminal with:
echo     .venv\Scripts\python -m overboard.main --gui
echo.
if "%~1"=="" pause
endlocal
exit /b 0

:try_python
if defined PY goto :eof
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
goto :eof

:no_python
echo [fail] Python 3.9 or newer was not found on this computer.
echo.
echo Install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH" in the installer. Then run this file again.
goto :failed

:no_venv
echo [fail] The virtual environment .venv could not be created.
echo        Check the messages above, then run this file again.
goto :failed

:no_packages
echo [fail] The install failed. The pip messages above say why.
echo        Check your network or proxy and run this file again.
goto :failed

:failed
echo.
echo The setup did not finish. Fix the lines above and run this file again.
if "%~1"=="" pause
endlocal
exit /b 1
