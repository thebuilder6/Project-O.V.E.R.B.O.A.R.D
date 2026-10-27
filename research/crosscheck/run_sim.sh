#!/usr/bin/env bash
# Run the cross-check step profile in a third-party simulator.
#
# The simulator is an Electron/TypeScript project that needs Node 22 and
# pnpm, and it only runs on Linux. This script drives it inside WSL2, copies
# the trace back into this repository, and leaves the simulator's checkout
# alone except for one copied script.
#
# Prerequisites (once):
#   git clone --depth 1 https://github.com/martijas/fll-sim ~/fll-sim
#   cd ~/fll-sim && pnpm install --filter "@fll-sim/sim-cli..."
#
# Usage:
#   bash research/crosscheck/run_sim.sh [out_csv] [command_dps] [stop_mode] [run_ms]
#
#   run_ms defaults to 600: long enough to reach steady speed at 10000 dps^2
#   ramps, short enough to stay clear of the walls and mission models (a
#   10 s drive collides with the field and corrupts the measurement).
#
# The simulator's repository has no licence file, so nothing from it is
# copied into this repository. This script runs it as a separate program and
# compares the numbers, which is all a cross-check needs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CHECKOUT="${FLLSIM_CHECKOUT:-$HOME/fll-sim}"

OUT="${1:-$REPO/research/crosscheck/trace.csv}"
COMMAND_DPS="${2:-1110}"
STOP_MODE="${3:-brake}"
RUN_MS="${4:-600}"

if [ ! -d "$CHECKOUT" ]; then
  echo "The simulator is not checked out at $CHECKOUT." >&2
  echo "Clone it there, or set FLLSIM_CHECKOUT." >&2
  exit 1
fi

export PATH="$HOME/.local/node/node-v22.14.0-linux-x64/bin:$PATH"
# pnpm keeps tsx out of the checkout's root .bin; resolve it explicitly.
TSX="$CHECKOUT/node_modules/.pnpm/node_modules/.bin/tsx"
if [ ! -x "$TSX" ]; then TSX="$CHECKOUT/tools/sim-cli/node_modules/.bin/tsx"; fi
if [ ! -x "$TSX" ]; then
  echo "tsx not found under $CHECKOUT/node_modules (pnpm install --filter '@fll-sim/sim-cli...')." >&2
  exit 1
fi

# The script imports the simulator's own modules, so it has to sit next to
# them. It is copied, not linked, and it is the only file that is touched.
cp "$HERE/sim_step.ts" "$CHECKOUT/crosscheck.ts"

cd "$CHECKOUT"
"$TSX" crosscheck.ts /tmp/crosscheck_trace.csv "$COMMAND_DPS" "$STOP_MODE" "$RUN_MS"
cp /tmp/crosscheck_trace.csv "$OUT"

echo "trace written to $OUT"
