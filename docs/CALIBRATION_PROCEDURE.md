# Drivebase Calibration Procedure

A repeatable way to measure the robot, so the planner's model matches the
robot that is actually on the field.

The procedure has two halves:

1. **Collect** — the robot runs the test suite on the hub while you measure
   where it ended. The hub drives; you hold the tape measure.
2. **Fit** — the measurements and the log become `sysid_fit.json`, which
   `main.py --sysid-fit` then uses to plan with.

Budget about half an hour. The robot drives for about six minutes of that;
the rest is your setup between tests and the tape work.

---

## What you need

- The robot, charged, hub running the Pybricks firmware.
- A tape measure, at least 1 m long.
- A clear floor with 2 m of runway in both directions, and room to spin.
- A rigid wall the front bumper can press against.
- Masking tape, to mark the start pose and the stop marks.
- A laptop with Bluetooth (or a USB cable to the hub).

Prepare the drivebase profile first, because the fit needs the wheel radius:

```bash
cp examples/drivebase_profile.example.json my_drivebase.json
```

Then open `my_drivebase.json` and set `wheel_radius_m` and `track_width_m` for
your robot. The rest of this document assumes that file exists.

---

## The measurement convention

Read this once; it is the part that is easy to get wrong.

Every measurement is in the **field frame**: the frame the robot sat in at the
start of the test. `x` points along the runway, `y` points to the left.

| Column | Meaning |
|---|---|
| `x_mm` | How far the robot ended **forward** from the start pose. `998` means it went 998 mm forward. |
| `y_mm` | How far it ended to the **left** of the runway, from the start pose. `0` means it stayed on the line, `20` means it drifted 20 mm left. |
| `turn_right_deg` | How far the robot **turned to the right** from the way it was pointing at the start. `0` means it kept its heading, `90` means it ended pointing 90° to the right of the runway. |

Put a piece of masking tape where the robot's centre sits at the start of each
test. That is the zero you measure from. For the straight tests, also mark
the point 1 m ahead, so you can check the robot's own distance reading
against the tape.

The turn is recorded as a turn to the **right** because that is how you read
it off the robot. The tool flips the sign, because the planner counts a turn
to the left as positive.

---

## Why the tests run three times

A single run cannot say how much the robot varies. The spread of the steps
*inside* one run is not that either: the steps are sequential, so a sagging
battery or a warming motor maps straight onto the step order and the steps
stop being comparable.

So the tests that produce the numbers the planner uses are run **three
times**, and the fit reports the spread over those three runs. Three is the
smallest number that gives a spread at all, and the interval it produces is
deliberately wide, because three is a small sample. The step order is
reversed on every other repeat, so a drift over the run cannot land on the
step index.

The cost is that you measure more. Seven poses per repeat, three repeats,
**21 measurements**.

---

## Step 1 — collect the session

```bash
python sysid_collect.py --session session.csv
```

This pushes `pybricks_code/sysid_digital_twin.py` to the hub over Bluetooth
and captures everything it prints. The session is written as it arrives, so a
dropped link costs the last test and not the whole run.

**What happens on the hub:** the display lights yellow and the hub waits for
the centre button. The terminal prints the name of the test and its setup
line. Do the setup, press the **centre button**, and the test runs. The
**Bluetooth button is the emergency stop** — press it if the robot does
something unexpected.

The tests run in this order. "×3" means the test runs three times, and you
measure it each time:

| # | Test | Setup | What you measure |
|---|---|---|---|
| 1 | `static` | Robot still and level on the floor. Do not touch it. | — |
| 2 | `straight_steps_fwd` ×3 | 2+ m clear runway. Robot at rest, facing along the runway. | — |
| 3 | `straight_steps_rev` ×3 | Same, but it drives backwards. | — |
| 4 | `turn_steps_cw` ×3 | Open space. The robot spins in place; keep the hub clear of the ground. | — |
| 5 | `turn_steps_ccw` ×3 | Same, the other way. | — |
| 6 | `curvature_cw` | Large open area. The robot follows constant-curvature arcs. | — |
| 7 | `curvature_ccw` | Same, the other way. | — |
| 8 | `reversal_backlash` | 2 m clear runway. The robot oscillates forward and back. | — |
| 9 | `chirp_straight` | 2 m clear runway. The robot sweeps forward and back around the start. | — |
| 10 | `chirp_turn` | Open space. The robot spins in place with a sweep. | — |
| 11 | `odometry_straight` ×3 | 2 m clear runway. The robot drives straight at 100 mm/s for 10 s. | **the end pose** |
| 12 | `odometry_turn_cw` ×3 | Open space. The robot turns clockwise at 50 deg/s for 1.8 s. | **the end pose** |
| 13 | `multi_stop` ×3 | 2 m runway. Mark the start pose and a measuring spot at each stop. | **the pose at each of 3 stops** |
| 14 | `brake_from_300` ×3 | Mark the start and final pose. 2 m runway required. | **the end pose** |
| 15 | `brake_from_600` ×3 | Same, at 600 mm/s. | **the end pose** |
| 16 | `wall_stall` | Front bumper flat against a rigid wall. | — |
| 17 | `replay_complex_mission` | Place the robot at the trajectory start pose. Clear the full path. | — |

Only tests 11 to 15 need a tape measure. The rest are for the actuator model
and the gyro bias, and you only have to watch them.

**Measure these seven, in each of the three repeats:**

- **Test 11, `odometry_straight`** — how far it went, and whether it drifted
  sideways. It should read about 1000 mm.
- **Test 12, `odometry_turn_cw`** — where it ended and how far it turned. It
  should have turned about 90° to the right.
- **Test 13, `multi_stop`** — the pose at each of the three stops, at roughly
  500, 1000 and 1500 mm. The hub pauses at each stop for you.
- **Test 14 and 15, `brake_from_300` and `brake_from_600`** — where it ended
  after the stop. The robot was told to brake from a speed, so it may have
  skidded and ended short of where it was driving to. That is the point of
  the test: the gap is the brake slip.

Write the numbers on a sheet of paper as you go, one line per measurement,
with the repeat number on the line. A phone photo of the sheet is a fine
backup.

When the hub is finished it turns green and the command prints the list of
measurements it still needs.

---

## Step 2 — write the measurement sheet

```bash
python sysid_fit.py session.csv --write-sheet calibration_sheet.csv
```

This writes a sheet with one row per measurement the session needs, and
nothing else:

```csv
test,repeat,stop,x_mm,y_mm,turn_right_deg
odometry_straight,0,0,,,
odometry_turn_cw,0,0,,,
multi_stop,0,0,,,
multi_stop,0,1,,,
multi_stop,0,2,,,
brake_from_300,0,0,,,
brake_from_600,0,0,,,
odometry_straight,1,0,,,
...
```

The `test`, `repeat` and `stop` columns are already filled in; they say which
measurement goes where. Fill in the three number columns from your paper.

- `test` is the test name the hub printed.
- `repeat` is 0, 1 or 2, counting the runs of that test.
- `stop` is 0 for a single measurement, and 0, 1, 2 for the three stops of
  `multi_stop`.
- `x_mm`, `y_mm` and `turn_right_deg` are in millimetres and degrees, as
  described above.

Leave a row empty if you did not measure it. The tool tells you which
measurements are missing instead of guessing.

---

## Step 3 — turn the sheet into measured poses

```bash
python sysid_fit.py session.csv --from-sheet calibration_sheet.csv \
  --out-poses poses.csv
```

The sheet is keyed by test name and repeat, so it does not matter which
order the tests ran in or which ones you skipped. The tool looks each one up
in the session, converts millimetres to metres, flips the turn, and writes the
rows the fitter reads:

```csv
run_id,stop,x_m,y_m,heading_deg
11,0,0.998,0.002,0.00
...
```

It warns about a row for a test that did not run, and about a measurement
that is missing.

---

## Step 4 — fit the model

```bash
python sysid_fit.py session.csv --poses poses.csv \
  --profile my_drivebase.json --out sysid_fit.json
```

The fitter prints what it found:

```
Parsed 42 runs from 1 file(s).
Straight: gain=0.981, tau=0.081s, delay=0.021s, n=42 steps in 3 repeats
  gain: 95% interval [0.974, 0.988] over 3 repeats (std 0.0042).
Turn: gain=0.976, tau=0.079s, delay=0.018s, n=36 steps in 3 repeats
  gain: 95% interval [0.968, 0.984] over 3 repeats (std 0.0048).
Gyro z bias: 0.0041 deg/s (std 0.0039)
Odometry: distance_scale=0.9980, heading_scale=1.0000
  distance_scale: 95% interval [0.9961, 0.9999] over 3 repeats (std 0.0011).
Brake slip: median=0.011, range=[0.008, 0.013]
The intervals cover the run-to-run variation of the robot. They do not cover
the error of the tape measure, which is the same every time and never averages
out.
```

- **Straight and turn gain** — how much of the commanded speed the robot
  actually delivers. A value below 1 means the motors are slower than the
  model thinks.
- **tau and delay** — how long the robot takes to react to a command. This is
  the part the planner could not know before.
- **Gyro z bias** — how much the gyro reads when the robot is still. The
  planner subtracts it.
- **Odometry distance scale** — the ratio between how far the robot really
  went and how far its wheel encoders think it went. `0.998` means the
  encoders over-read by 2 mm per metre, which is the usual amount of tyre
  squash.
- **Odometry heading scale** — the same thing for turns.
- **Brake slip** — how much of the commanded distance the robot loses when it
  brakes. A median of `0.011` means it slides about 11 mm for every metre it
  brakes from speed.

### What the interval covers, and what it does not

The interval is the 95% t-interval on the mean of the three repeats. It
covers:

- how much the robot varies from run to run,
- and any drift over the session, such as a battery that sags as the tests go
  on. That makes it conservative in the right direction: the band covers the
  robot at the tired end of a session, not only at the fresh end.

It does **not** cover the error of your tape measure. The tape error is in
the numerator of every measurement, it is the same every time, and no number
of repeats averages it out. If your tape is off by 5 mm per metre, the
distance scale is off by 0.5% and the interval will not say so. Measure the
same distance twice at some point to get a feel for it.

---

## Step 5 — use the fit

Plan with it:

```bash
python main.py -w examples/example_straight.json -o output.traj \
  --validate --sysid-fit sysid_fit.json
```

`--sysid-fit` requires `--validate`. The validator applies the fitted
actuator response and the odometry scale, and it uses the **gain interval**
as the band it plans against, so a robot that varies a lot produces a
conservative plan.

Check the plan against the robot you measured:

```bash
python sysid_compare.py output.traj poses.csv --run-id 11 \
  --profile my_drivebase.json
```

The comparator prints the position and heading error of the planned pose
against the measured one, and says whether the measured pose is inside the
reachability envelope. It is a report, not a gate: the 5 mm and 2 cm
thresholds do not change.

---

## If something goes wrong

- **The robot will not start a test.** The hub waits for the centre button.
  Press it once the setup is right. The Bluetooth button is the emergency
  stop, not the start.
- **The robot ran into something.** Press the Bluetooth button. The session
  keeps what it captured; re-run step 1.
- **A measurement looks wrong.** Compare it with the robot's own distance
  reading. The hub prints `STOP_AT` and `FINAL_STATE` lines with the odometry
  reading for the multi-stop and brake tests. If the two disagree by more
  than a few millimetres, you measured the wrong spot.
- **`the sheet has 'x' but the session does not have that test`** — the sheet
  names a test the session does not contain. Check the spelling against the
  `META,test` line in `session.csv`, and the repeat against `META,repeat`.
- **`'x' repeat 0 stop 0 was not measured`** — that measurement is missing,
  so its scale will not be calibrated. Fill it in and re-run steps 3 and 4.
- **The fit says "one run only, no interval".** The session has a single run
  of that test, so there is no spread to report. Collect a session with the
  repeats, or accept the point estimate with no interval.
- **The link dropped mid-session.** The session file keeps whatever arrived.
  Re-run step 1 to get a clean one; the fit is only as good as the log.
- **`odometry heading_scale must be > 0`** — the fit is broken, not the robot.
  This means the sheet and the session disagree about the sign of the turn.
  Check that a turn to the right is a positive `turn_right_deg`.

---

## Limits

- The poses are measured by hand, so they are only as good as the tape
  measure and the tape marks. Aim for a few millimetres.
- The pose of the robot means the pose of the **centre between the wheels**,
  not the front bumper.
- The odometry scale is calibrated from the straight test and the multi-stop
  test only. A robot that slips on one wheel will not show up there.
- Three repeats give a wide interval on purpose. If you want a tighter one,
  collect more repeats, not a cleverer analysis.
- Braking slip is fitted but the plant does not apply it yet (KI-020).
- Do not tune `b`, `zeta`, or the validation thresholds to a single path.
