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
#   bash research/crosscheck/run_sim.sh [out_csv] [command_dps] [stop_mode]
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

if [ ! -d "$CHECKOUT" ]; then
  echo "The simulator is not checked out at $CHECKOUT." >&2
  echo "Clone it there, or set FLLSIM_CHECKOUT." >&2
  exit 1
fi

export PATH="$HOME/.local/node/node-v22.14.0-linux-x64/bin:$PATH"

# The script imports the simulator's own modules, so it has to sit next to
# them. It is copied, not linked, and it is the only file that is touched.
cp "$HERE/sim_step.ts" "$CHECKOUT/crosscheck.ts"

cd "$CHECKOUT"
npx tsx crosscheck.ts /tmp/crosscheck_trace.csv "$COMMAND_DPS" "$STOP_MODE" 10000
cp /tmp/crosscheck_trace.csv "$OUT"

echo "trace written to $OUT"
