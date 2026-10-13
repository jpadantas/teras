@echo off
rem Start the TERAS dashboard (local access only). Extra arguments go to Streamlit.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run apps\dashboard.py --server.address localhost %*
