#!/bin/sh
# Open the OVERBOARD browser studio. Press Ctrl+C to stop it.
#
# Run it with: sh start_studio.sh
# On a machine that has never run the studio, this installs the virtual
# environment first, so one command is enough to get going.
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo
    echo "No virtual environment yet. Installing it now - this takes a few minutes."
    echo
    sh install.sh || exit 1
fi

# The virtual environment from install.sh is used when it is there.
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 was not found on PATH." >&2
        echo "Install Python 3.9 or newer, or run sh install.sh to set everything up." >&2
        exit 1
    fi
    PY=python3
fi

"$PY" main.py --doctor || exit 1

echo
echo "Starting the studio..."
exec "$PY" main.py --gui "$@"
