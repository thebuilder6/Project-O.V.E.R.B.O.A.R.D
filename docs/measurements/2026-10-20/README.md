# Review plots — 2026-10-20

Two figures comparing the two timing models on the current tree, produced by
[`dt_modes_plots.py`](dt_modes_plots.py):

```bash
python docs/measurements/2026-10-20/dt_modes_plots.py
```

| file | what it shows |
|---|---|
| `dt_modes.png` | the complex mission: path, reachability tube, closed-loop tracking error, time per segment, the wheel-force step at each waypoint, and the wheel-force profile |
| `uneven_dt_modes.png` | the uneven 1.0 m + 0.2 m path, the same six panels |

## What the panels say

**Time per segment** is the whole story of the two models. The equal-duration
mode shares one `dt` across all segments, and because every segment has the
same sample count that also forces every segment to take the **same time** —
so on the uneven path the 0.2 m hop inherits the 1.0 m segment's 2.23 s and
the robot crawls through it, overshooting the last waypoint and coming back
(the black path in the uneven figure). The per-piece mode gives each segment
its own duration and drives straight through.

**Wheel-force step through each waypoint** is the KI-024 property. The
per-piece model may re-time at a waypoint, but it may not step the wheel
force there: the per-piece bars are flat on the floor at every waypoint the
robot drives through, while the equal-duration bars reach a few m/s². The
hatched bars are waypoints the robot *stops* at, where the wheel force has to
reverse sign and a step is expected.

**The tracking and force panels** are measured with the validator mirroring
the controller that runs (KI-028): a synced 20 ms command hold, an unrounded
command and the gain recomputed from the interpolated speed and turn rate. The
numbers in the older 2026-10-19 evidence were measured against a mirror that
rounded the command and interpolated the gain, so they are not comparable
cell for cell.
