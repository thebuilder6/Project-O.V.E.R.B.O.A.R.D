# The two propagations, side by side — 2026-10-20

The closed-loop tick map and the shipped open-loop plant propagation, run
through the CLI on the six example missions, on the same tree and the same
robot, with the same candidates.

The question: the tick map is the *correct* model of the loop the robot
runs — the controller is inside the map, so the map's Jacobian carries the
feedback term and the tube contracts the way the robot's does. The plant
propagation evaluates the controller *outside* the propagation, at the tube's
midpoint, so what it bounds is the reachable set of an open-loop plant
driven by a pre-recorded command. Does the correct model change anything?

## How to reproduce it

```powershell
foreach ($m in @('example_straight', 'example_mission_first4', 'example_s_curve',
                 'example_with_stops', 'example_with_events', 'example_complex_mission')) {
    foreach ($p in @('plant', 'tick')) {
        python main.py -w "examples/$m.json" -o "$m.traj" --validate `
            --candidates 2 --propagation $p
    }
}
```

## The result

The selected candidate of each run. The duration and the tracking error are
the **same** for both propagations on every mission, to the last digit: the
tick map changes the *tube*, not the plan and not the tracking metric.

| mission | duration | envelope (plant) | envelope (tick) | tracking | verdict |
|---|---:|---:|---:|---:|---|
| example_straight | 2.377 s | 0.305 cm | **0.292 cm** | 0.341 cm | pass, both |
| example_mission_first4 | 1.833 s | 0.310 cm | **0.195 cm** | 0.235 cm | pass, both |
| example_s_curve | 3.667 s | 0.732 cm | **0.250 cm** | 1.468 cm | fail, both |
| example_with_stops | 6.249 s | 0.915 cm | **0.283 cm** | 0.962 cm | fail, both |
| example_with_events | 4.416 s | 0.816 cm | **0.474 cm** | 0.459 cm | pass, both |
| example_complex_mission | 6.072 s | 1.239 cm | **0.283 cm** | 0.437 cm | pass, both |

**The tick map is uniformly tighter**, and much tighter on the curves: 2.9x
on the S-curve, 3.2x on the stops, 4.4x on the complex mission, 1.7x on the
events, 1.6x on the first four waypoints, and 1.04x on the straight. The
straight is the case the feedback does not help with: on a straight the
command difference is exactly zero, so there is nothing for the feedback to
contract.

**No verdict changes.** Four missions pass and two fail under both
propagations, and the two that fail fail on the **tracking** gate (the
discretisation mismatch RQ-A describes), which the tick map does not touch:
the tracking error is the tube's midpoint against the plan, so it is
decided by the discretisation and not by the uncertainty.

**The envelope was never the failing gate on this set.** The declared
uncertainty ranges in `examples/fll_choreo.json` are test assumptions and
they are narrow, so the plant tube stays inside the 2 cm limit on every
example. The plant propagation's own numbers are the ones the
experiments of 2026-10-20 were about, and they are much worse on a
*calibrated* robot: with a +/-1 % gain band the S-curve's plant tube is cut
off at the 50 mm guard (53.6 mm) and the tick map with the mixed-Jacobian
gives 18.8 mm, which passes the gate (`docs/research/rq_a/exp20_closed_loop_tick.py`).

## What this says

1. **The tick map is the better model and it is now reachable.** It is
   tighter everywhere, it is the model of the loop the robot actually runs,
   and `--propagation tick` runs it end to end.
2. **It does not rescue a mission here, and it is not supposed to.** The
   two failing example missions fail on the tracking gate, which is a
   different defect (KI-020, RQ-A). Tier 1 targeted the envelope; the
   envelope was not what was failing.
3. **The default stays on the plant propagation.** The tick map's own gain
   is on the paths where the plant propagation *diverges* rather than merely
   widens — the calibrated case — and with the digital twin it is **wider**
   (12.1 cm against 5.1 cm on the S-curve), because the twin's plant state
   is two interval coordinates whose spread feeds the command and the
   interval arithmetic counts that growth twice
   (`docs/research/rq_a/exp23_twin_tick_belief.py`). A wider tube is the
   conservative direction, so the flag is allowed with a fit; the report
   names the model, so the number is read as what it is.
4. **It costs 7-10x the validation time** with the mixed-Jacobian
   inclusion, and 2-3x with the Jacobian. The tick propagation needs a
   Jacobian inclusion: with the natural extension the closed-loop map
   explodes (627 mm on a 1 m straight).

## The tree these numbers describe

The run was measured on the working tree, whose provenance files hash to:

```
optimizer.py          9F8CD3D4B66BF3B452723BB54C040441
main.py               8CD6567FD17B32816329FBC7CA518905
immrax_validator.py   AE390475CC51DEB238F57D8C2C4305D6
validator.py          D7374FAE647300C6A3ACD85CAAD4D38D
robot_model.py        6748938B0E26034B3641FE9386ACA9ED
multiverse_optimizer.py 5D71943AD4EF313D8F9D83B9069E95F7
tree token            57F6845E775A
```

Environment: Windows 11, Python 3.12.10, `casadi` 3.8.1, `jax`/`jaxlib`
0.12.0, `immrax` 0.3.8, `numpy` 2.5.4.

## A note on honesty

Every number here is a **model prediction**. The declared uncertainty
ranges are test assumptions, not measurements, and no hardware logs exist
in the repository. A `pass` means "the trajectory passes the gates of an
uncalibrated model", not "the robot can drive it".
