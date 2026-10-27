# Cross-check against a third-party simulator

This is measurement code. It compares OVERBOARD's drivetrain model with an
independent rigid-body simulator, so that the model is checked against
something that was not written by the same person.

## What it does

1. `sim_step.ts` drives a wheel-speed step profile in the reference
   simulator — run up to a steady speed, then stop — and writes what
   happened as CSV: the motor speeds, the motor duty, and the robot's pose.
2. `compare_step.py` reads that trace and asks OVERBOARD's own model the
   same question: how fast does the robot run, and how far does it travel
   after the command is dropped?
3. `sim_traj.ts` drives a **closed-loop trajectory** (Q25's trajectory
   oracle): the deployed chain from `pybricks_code/ramsete.py` — reference
   interpolation with the tick-midpoint lead, Ramsete on the robot's own
   encoder odometry, the KinematicLimiter — per-wheel speed commands held
   over the 20 ms heartbeat. `--straight-turn` is the built-in sanity
   profile; a spec JSON (`wheelDiameterMm`, `trackWidthMm`, `robot`, and
   `samples: [{t,x,y,heading,vl,vr,omega}]` in the planner frame) drives a
   mission's profile. `run_traj.sh` runs it the way `run_sim.sh` runs the
   step harness. Mode `openloop` commands the reference (vl, vr) directly.
4. `compare_traj.py` matches a twin prediction (`t,x,y,heading[,
   tube_radius_m]` at plan sample instants) against a trajectory trace at
   the sample instants and reports `e_pos`, `e_head` and tube containment,
   per trajectory and aggregated over a set.

The two models are not the same model. The reference simulator is a rigid
body with a motor model, a tyre model, a caster and contact friction; ours
is an analytic motor and drivetrain. The point is not to agree to the last
decimal. The point is to see whether the *structure* of ours holds up.

## The reference simulator

<https://github.com/martijas/fll-sim> — a FIRST LEGO League simulator:
SPIKE Prime Python on a to-scale field with Rapier physics. Its `sim`
package is pure TypeScript plus Rapier, so it runs headless; the Electron
app is only the user interface.

It has **no licence file**, so nothing from it is copied into this
repository. It is run as a separate program, and only its numbers are
compared. The SPIKE Prime motor constants that OVERBOARD's model uses are
cited from it as literature values (see `drivetrain.py`).

## Running it

The simulator is a Linux project that needs Node 22 and pnpm. On Windows it
runs in WSL2.

```powershell
# once, inside WSL
wsl -d Ubuntu -- bash -lc "mkdir -p ~/.local/node && cd ~/.local/node && \
  curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz \
  -o node.tar.xz && tar -xf node.tar.xz && rm node.tar.xz"
wsl -d Ubuntu -- bash -lc "export PATH=\$HOME/.local/node/node-v22.14.0-linux-x64/bin:\$PATH && \
  npm install -g pnpm@12.6.0 --force && \
  git clone --depth 1 https://github.com/martijas/fll-sim ~/fll-sim && \
  cd ~/fll-sim && pnpm install --filter '@fll-sim/sim-cli...'"
```

Then, for every run:

```powershell
wsl -d Ubuntu -- bash -lc "bash /mnt/c/Users/<you>/AppData/Local/Temp/opencode/run_sim.sh"
python research/crosscheck/compare_step.py
```

`run_sim.sh` in this directory does the same thing from inside WSL. It
copies `sim_step.ts` into the simulator's checkout, runs it with `tsx`, and
copies the trace back as `trace.csv` (and `trace_coast.csv` for the coast
run).

## What it found

The results and what they mean are in
[`../../docs/research/RQ-B_drivetrain_model.md`](../../docs/research/RQ-B_drivetrain_model.md).
The short version:

- the motor model's **structure holds**: the top speed is predicted to
  within 7 % of a rigid-body simulation,
- the model has **no rolling resistance**, so any deceleration it predicts
  is an upper bound,
- the reference simulator's **brake mode reverses the wheels** instead of
  braking them, so its braking distance is not a usable reference.

The trajectory oracle's feasibility spike, sanity gates and repeatability
floor (Q25, B1) are recorded in
[`../results/q25_b1_oracle.md`](../results/q25_b1_oracle.md) (gitignored
results directory): closed-loop trajectories work, the step regression
reproduces RQ-B's numbers bit-identically, and the oracle's run-to-run
noise floor is exactly zero (deterministic fixed-step physics).
