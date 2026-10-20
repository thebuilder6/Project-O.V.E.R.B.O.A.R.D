#!/bin/sh
# Open the OVERBOARD browser studio. Press Ctrl+C to stop it.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found on PATH. Install Python 3.8 or newer and try again."
    exit 1
fi

python3 main.py --doctor
python3 main.py --gui "$@"
