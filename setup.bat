@echo off
rem Create .venv and install the dependencies listed in requirements.txt.
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

%PY% -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>nul
if errorlevel 1 (
    echo Python 3.10 or later is required: https://www.python.org/downloads/
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv || goto :error
)

echo Installing dependencies from requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

echo.
echo Setup complete. Start the dashboard with run.bat
exit /b 0

:error
echo.
echo Setup failed. See the messages above.
exit /b 1
