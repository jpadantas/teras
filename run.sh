#!/usr/bin/env sh
# Start the TERAS dashboard (local access only). Extra arguments go to Streamlit.
set -e
cd "$(dirname "$0")"

VPY=.venv/bin/python
[ -x .venv/Scripts/python.exe ] && VPY=.venv/Scripts/python.exe   # Git Bash on Windows
if [ ! -x "$VPY" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

exec "$VPY" -m streamlit run apps/dashboard.py --server.address localhost "$@"
