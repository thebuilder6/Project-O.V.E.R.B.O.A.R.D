#!/usr/bin/env bash
# Run a closed-loop trajectory in the third-party simulator (Q25 oracle).
#
# Same setup and rules as run_sim.sh: the simulator is a Linux/TypeScript
# project run inside WSL2; the driver is copied into its checkout, run, and
# the trace is copied back. The simulator's repository has no licence file,
# so nothing from it is copied into this repository.
#
# Usage:
#   bash research/crosscheck/run_traj.sh [spec] [out_csv] [mode]
#
#   spec     path to a spec JSON, or "straight-turn" (the built-in sanity
#            profile). Paths must be readable from inside WSL (/mnt/c/...).
#   out_csv  default research/crosscheck/traj_trace.csv
#   mode     closedloop (default) | openloop

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CHECKOUT="${FLLSIM_CHECKOUT:-$HOME/fll-sim}"

SPEC="${1:-straight-turn}"
OUT="${2:-$REPO/research/crosscheck/traj_trace.csv}"
MODE="${3:-closedloop}"

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
cp "$HERE/sim_traj.ts" "$CHECKOUT/crosscheck_traj.ts"

if [ "$SPEC" = "straight-turn" ]; then
  SPEC_ARG="--straight-turn"
else
  SPEC_ARG="$SPEC"
fi

cd "$CHECKOUT"
"$TSX" crosscheck_traj.ts "$SPEC_ARG" /tmp/crosscheck_traj_trace.csv "$MODE"
cp /tmp/crosscheck_traj_trace.csv "$OUT"

echo "trace written to $OUT"
