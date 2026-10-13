#!/usr/bin/env sh
# Create .venv and install the dependencies listed in requirements.txt.
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || {
    echo "Python 3.10 or later is required."; exit 1; }

VPY=.venv/bin/python
[ -x .venv/Scripts/python.exe ] && VPY=.venv/Scripts/python.exe   # Git Bash on Windows
if [ ! -x "$VPY" ]; then
    echo "Creating virtual environment in .venv ..."
    "$PY" -m venv .venv
    [ -x .venv/Scripts/python.exe ] && VPY=.venv/Scripts/python.exe
fi

echo "Installing dependencies from requirements.txt ..."
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt

echo
echo "Setup complete. Start the dashboard with ./run.sh"
