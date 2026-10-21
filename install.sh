#!/bin/sh
# Install OVERBOARD into a local virtual environment (.venv) and check it.
#
# Run it with: sh install.sh
# It is safe to run it again: an existing .venv is reused and the packages are
# simply updated.
set -e
cd "$(dirname "$0")"

echo
echo "OVERBOARD setup"
echo "=============="
echo

# --- 1. find a Python 3.9 or newer ---------------------------------------
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "[fail] Python 3.9 or newer was not found on this computer."
    echo "       Install it from https://www.python.org/downloads/ and run this again."
    exit 1
fi
echo "[ok]   Python found: $PY"

# --- 2. the virtual environment -------------------------------------------
if [ -x ".venv/bin/python" ]; then
    echo "[ok]   Reusing the existing .venv"
else
    echo "[....] Creating the virtual environment .venv"
    "$PY" -m venv .venv || {
        echo "[fail] The virtual environment .venv could not be created."
        echo "       On Debian/Ubuntu: sudo apt install python3-venv"
        exit 1
    }
fi
VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "[fail] The virtual environment is missing $VENV_PY"
    exit 1
fi

# --- 3. the packages -------------------------------------------------------
echo "[....] Installing the solver and the studio. This can take a few minutes."
"$VENV_PY" -m pip install --upgrade pip ||
    echo "[warn] pip could not be upgraded; the one that came with Python works."
"$VENV_PY" -m pip install -r requirements.txt
echo "[ok]   Core packages installed"

# The optional stack (JAX) has no wheels for 3.9, so it is skipped there
# instead of failing the whole install.
if "$VENV_PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "[....] Installing the optional extras (validation, calibration, hub push)"
    if "$VENV_PY" -m pip install -r requirements-optional.txt; then
        echo "[ok]   Optional extras installed"
    else
        echo "[warn] The optional extras did not install. The solver and the studio still"
        echo "       work, but --validate and the hub tools do not."
    fi
else
    echo "[warn] Python 3.9 or older: the optional extras (JAX) are skipped."
    echo "       Install Python 3.10 or newer and run this again for --validate."
fi

# --- 4. the check ----------------------------------------------------------
echo
"$VENV_PY" -m overboard.main --doctor

echo
echo "Ready. Start the studio with sh start_studio.sh,"
echo "or from a terminal with:"
echo "    .venv/bin/python -m overboard.main --gui"
