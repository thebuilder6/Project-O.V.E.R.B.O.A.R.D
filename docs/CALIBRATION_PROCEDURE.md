# Drivebase Calibration Procedure

A repeatable way to measure the robot, so the planner's model matches the
robot that is actually on the field.

The procedure has two halves:

1. **Collect** — the robot runs the test suite on the hub while you measure
   where it ended. The hub drives; you hold the tape measure.
2. **Fit** — the measurements and the log become `sysid_fit.json`, which
   `python -m overboard.main --sysid-fit` then uses to plan with.

Budget about half an hour. The robot drives for about six minutes of that;
the rest is your setup between tests and the tape work.

## The whole thing, in two commands and one form

```bash
# 0. make the drivebase profile (once per robot; asks a few questions)
python -m overboard.sysid.profile --out my_drivebase.json

# 1. collect: the robot runs the tests, you press the hub button between them
python -m overboard.sysid.collect --conn ble --profile my_drivebase.json \
  --session session.csv --sheet calibration_sheet.csv

# 2. fill in calibration_sheet.html in a browser, click Download CSV

# 3. fit: the sheet is the measurement, so there is no file to convert
python -m overboard.sysid.fit session.csv --sheet calibration_sheet.csv \
  --profile my_drivebase.json --out sysid_fit.json
```

The steps below are the same three things, in detail.

---

## What you need

- The robot, charged, hub running the Pybricks firmware.
- A tape measure, at least 1 m long.
- A kitchen scale, for the weigh-in (the robot's mass, with the battery).
- A clear floor with 2 m of runway in both directions, and room to spin.
- A rigid wall the front bumper can press against.
- Masking tape, to mark the start pose and the stop marks.
- A laptop with Bluetooth (or a USB cable to the hub).

Prepare the drivebase profile first, because the fit needs the wheel radius.
The wizard asks a few questions, with a tip on how to measure each number,
and checks the answers are plausible before it writes the file:

```bash
python -m overboard.sysid.profile --out my_drivebase.json
```

It asks for the robot's name, the wheel radius in millimetres (roll the
robot one wheel turn and divide the distance by 6.28), the track width in
millimetres (centre-to-centre of the wheel contact patches; the fit refines
it from the turn tests), the gearing — how many times the motor turns for
one turn of the wheel, 1 for direct drive — and optionally the wall distance
below. To write the file by hand instead, copy the example and set the same
fields:

```bash
cp examples/drivebase_profile.example.json my_drivebase.json
```

Then open `my_drivebase.json` and set `wheel_radius_m`, `track_width_m` and
`gearing` for your robot. The rest of this document assumes that file exists.

### The wall, if you have one

The `wall` block is optional. It says how far the start mark is from a wall
the robot can drive into, and it is what makes the wall homing test work:

```json
"wall": { "distance_m": 1.05 }
```

Measure it **to where the robot's centre rests against the wall**, not to
the wall's surface, so the bumper's offset does not have to be known. Do it
once: put the robot against the wall, mark where its centre is, then measure
from that mark to the wall and put the robot back on the mark for the
calibration.

With a wall in the profile, the robot can check its own odometry against
something that does not move and that nobody has to read, so the scale can
be re-measured without a human. Without one, leave the block out and the
homing test is skipped.

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
**21 measurements**, and the robot's mass three times on the scale.

---

## Step 1 — collect the session

```bash
python -m overboard.sysid.collect --conn ble --profile my_drivebase.json \
  --session session.csv --sheet calibration_sheet.csv
```

This pushes `pybricks_code/sysid_digital_twin.py` to the hub over Bluetooth
and captures everything it prints. The session is written as it arrives, so a
dropped link costs the last test and not the whole run.

`--profile` is only needed for the fit, so the command also works without it
(then the session is collected and the model is not fitted). `--sheet` writes
the measurement sheet for the session: the CSV the fitter reads
(`calibration_sheet.csv`) and a form in a browser next to it
(`calibration_sheet.html`) that you fill in instead of editing a spreadsheet.

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
| 16 | `wall_homing` ×3 | Robot's centre on the start mark, square to the wall. | — |
| 17 | `wall_stall` ×3 | The robot is against the wall now, from the homing run. | — |
| 18 | `top_speed` ×3 | **Lift the robot so both wheels spin free.** They must not touch the mat. | — |
| 19 | `replay_complex_mission` | Place the robot at the trajectory start pose. Clear the full path. | — |

Only tests 11 to 15 need a tape measure. The rest are for the actuator model
and the gyro bias, and you only have to watch them.

The table lists the tests in the order they first appear. The repeats are
interleaved: the forward and reverse step tests alternate, and the five
measured tests run as one group of five per repeat, so the tape work is done
in one go. The step order is reversed on every other repeat, so a battery
that sags over the session cannot land on the step index.

**Test 16, `wall_homing`, checks the distance scale against the wall.** The
robot drives slowly forward from the start mark and stops when it is blocked,
so the distance it covered is the distance to the wall. The wall does not
move and nobody has to read it, so the scale can be re-checked later
without a human. It needs the `wall` block in the drivebase profile.

**Test 17, `wall_stall`, measures the motor's stall torque** from the same
wall, which is why it runs straight after the homing run: the robot is
already against it.

**Test 18, `top_speed`, needs the robot off the ground.** With the wheels
lifted the motor pushes nothing, so it runs up to the speed at which it makes
no torque. That speed is the other end of the motor's torque-speed line, and
the robot config otherwise has to declare it. It runs last, after the
measuring tests, so the runway is not disturbed. The command is deliberately
larger than any LEGO motor can reach, so the motor is torque-limited rather
than speed-limited; if the robot somehow reached the commanded speed, the
fit rejects the number and says why.

**Measure these seven, in each of the three repeats:**

- **Test 11, `odometry_straight`** — how far it went, and whether it drifted
  sideways. It should read about 1000 mm. **The sideways offset matters as
  much as the distance**: it is how the fit sees the robot curving when it is
  told to drive straight, and the fit checks its own gyro estimate against
  it. If the two disagree by more than a degree per metre, the report says
  so, and the drift term should not be trusted.
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

## Step 2 — fill in the measurement sheet

`python -m overboard.sysid.collect --sheet` already wrote it. If you collected the session
some other way, write it here:

```bash
python -m overboard.sysid.fit session.csv --write-sheet calibration_sheet.csv
```

The sheet has one row per measurement the session needs, and three weigh-in
rows at the end:

```csv
test,repeat,stop,x_mm,y_mm,turn_right_deg,mass_kg
odometry_straight,0,0,,,,
odometry_turn_cw,0,0,,,,
multi_stop,0,0,,,,
multi_stop,0,1,,,,
multi_stop,0,2,,,,
brake_from_300,0,0,,,,
brake_from_600,0,0,,,,
odometry_straight,1,0,,,,
...
weigh_in,0,0,,,,
weigh_in,1,0,,,,
weigh_in,2,0,,,,
```

The `test`, `repeat` and `stop` columns are already filled in; they say which
measurement goes where. Fill in the three number columns from your paper, and
the `mass_kg` column on the weigh-in rows.

- `test` is the test name the hub printed.
- `repeat` is 0, 1 or 2, counting the runs of that test.
- `stop` is 0 for a single measurement, and 0, 1, 2 for the three stops of
  `multi_stop`.
- `x_mm`, `y_mm` and `turn_right_deg` are in millimetres and degrees, as
  described above.
- `mass_kg` is the robot's mass **with the battery in**, in kilograms, on a
  kitchen scale. Weigh it three times -- the same three-repeat discipline as
  the hub tests, so the fit can report a 95% t-interval over the weigh-ins
  instead of an assumed range. Pick the robot up, put it down, read the
  scale; a repeat that re-weighs the same static reading is not a repeat.

Leave a row empty if you did not measure it. The tool tells you which
measurements are missing instead of guessing.

**Fill it in the browser, not in a text editor.** The collect command wrote a
form next to the sheet:

```bash
python -m overboard.sysid.fit session.csv --write-form calibration_sheet.html
```

Open `calibration_sheet.html` in a browser. It has one row per measurement,
with the convention written at the top, and it counts what is still missing.
Type the numbers, click **Download CSV**, and save the file in the folder you
run the commands from. It is the same CSV as the sheet, so it drops straight
into the next step. The form also takes a sheet you saved earlier, so a
session you measured in two goes is not measured twice.

---

## Step 3 — fit the model

```bash
python -m overboard.sysid.fit session.csv --sheet calibration_sheet.csv \
  --profile my_drivebase.json --out sysid_fit.json
```

The sheet is keyed by test name and repeat, so it does not matter which
order the tests ran in or which ones you skipped. The tool looks each
measurement up in the session, converts millimetres to metres, flips the
turn, and fits from that. There is no intermediate pose file.

It warns about a row for a test that did not run, and about a measurement
that is missing. A missing measurement is not fatal: the scales that need it
stay uncalibrated, and the report says so.

If you want the measured poses as a file of their own (to compare the plan
against later, for instance), write them out:

```bash
python -m overboard.sysid.fit session.csv --from-sheet calibration_sheet.csv \
  --out-poses poses.csv
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
Straight drift: 1.472 deg/m (the robot curves when told to drive straight)
  tape says 1.455 deg/m, disagreement 0.017 deg/m. A big disagreement means
  the gyro bias or the start mark is off, so do not trust the drift term.
Stall torque: 0.1314 Nm
No-load speed: 21.318 rad/s at the wheel (the speed at which the motor makes
  no torque)
  no_load_speed_rad_s: 95% interval [21.105, 21.531] over 3 repeats (std 0.206).
Mass: 0.8504 kg (weighed 3x, with battery, source weigh_in)
  mass_kg: 95% interval [0.8492, 0.8509] over 3 repeats (std 0.0006).
  the straight steps say 0.8574 kg (cross-check)
Inertia: 0.003119 kg m^2 (from the turn steps, tau_mech=0.081s)
  inertia_kg_m2: 95% interval [0.0031, 0.0031] over 3 repeats (std 0.0000).
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
  squash. The wall and the tape are two estimates of the same number, so
  the report gives both and the difference between them: a big disagreement
  means the start mark or the tape is off.
- **Odometry heading scale** — the same thing for turns.
- **Brake slip** — how much of the commanded distance the robot loses when it
  brakes. A median of `0.011` means it slides about 11 mm for every metre it
  brakes from speed.
- **Straight drift** — how much the robot curves per metre when it is told
  to drive straight. The plan's straight segments are straight, so this is a
  tracking error that grows with the distance: **1 deg/m is 8.7 mm of lateral
  error per metre, and 78 mm over a 3 m path.** The fit takes it from the
  step tests, where the robot drives straight at five speeds: the heading
  rate against the speed is a straight line whose slope is the drift and
  whose intercept is the gyro's bias. It checks that against the tape's
  lateral offset.
- **Stall torque** — what the motors can actually push with, measured by
  pushing into a wall. The robot config can only declare it, so this
  replaces a guess. The planner uses it as the limit on the force a wheel may
  need.
- **No-load speed** — the speed the motor reaches with the wheels off the
  ground, which is the speed at which it makes no torque. Together with the
  stall torque it is the whole torque-speed line the planner uses, so both
  numbers are now measured instead of declared. The planner, the validator
  and the exported controller all use the measured pair.
- **Mass** — the robot's mass with the battery, from the sheet's weigh-in
  rows. It lands in the robot config, with the 95% t-interval over the three
  weigh-ins as its range. The straight steps also give a derived mass from
  the response (below); it is reported as a cross-check only, and the
  weigh-in is what lands.
- **Inertia** — the yaw inertia, derived from the turn steps plus the
  measured torque-speed line. No new hardware test: the turn steps are
  already in the session, and the line is the stall torque and no-load speed
  you measured above.

  The step tests drive `drivebase.drive()`, which is a closed-loop speed
  command: its step response hides the robot's mechanics behind the speed
  controller, so the fitted tau alone cannot give the inertia. What the
  session does carry is what the closed loop actually applied -- every
  rate-test row logs the motor torque. From that actuation the fit reads
  both factors of the line `J = tau_mech * (B + K_v)`: the steady tail of
  the step gives `(B + K_v) = K0 / y_ss`, and the transient gives the
  mechanical time constant `tau_mech`, and with the wheel radius, the track
  width and the gearing from your profile, the yaw inertia falls out. The
  same machinery on the straight steps gives the derived mass cross-check.

  Two conventions matter when reading the numbers. The value is the
  **effective body-side inertia** the force model consumes: it already
  includes the reflected motor/gear rotor inertia, so the config stores it
  as `inertia` and never adds `reflected_inertia_kg_m2` on top. And the
  range is measured, like the others: the assumed +/-10% default in the
  config's uncertainties is replaced by the interval over the repeats.

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

The drift is the one number that is measured twice, by two different
instruments, because it is the one that matters most and the one that is
hardest to get right. The **step tests** give it as a slope: the robot
drives straight at five speeds, and the heading rate against the speed is a
line whose slope is the drift and whose intercept is the gyro's bias. That
matters because a bias that leaks into the drift is multiplied by the run's
length over its distance, which is ten for the one-metre odometry run. As a
slope, the bias is only an intercept and cannot leak. The **tape** gives an
independent estimate from the lateral offset the drift leaves behind: a pure
length measurement with no sensor bias in it, but it carries the operator's
placement error, and the placement error is the same every time. The fit
reports both, the difference between them, and the gyro's bias measured two
ways (standing still and moving). If the two drift estimates disagree by
more than a degree per metre, one of the two systematic errors is wrong,
the report says so, and the drift term should not be used until you have
found out which.

---

## Step 4 — use the fit

Plan with it:

```bash
python -m overboard.main -w examples/example_straight.json -o output.traj \
  --validate --sysid-fit sysid_fit.json
```

`--sysid-fit` requires `--validate` (the studio passes the same fit to every solve the
studio runs). The validator applies the fitted actuator response, the odometry belief and the
measured braking shortfall, and it uses the **gain interval** as the band it plans against, so a
robot that varies a lot produces a conservative plan.

Check the plan against the robot you measured. The comparator reads the
measured poses, so write them out of the sheet first:

```bash
python -m overboard.sysid.fit session.csv --from-sheet calibration_sheet.csv \
  --out-poses poses.csv
python -m overboard.sysid.compare output.traj poses.csv --run-id 11 \
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
  so its scale will not be calibrated. Fill it in and re-run step 3.
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
- Braking slip is fitted and applied by the digital-twin plant, which slows the robot's
  ground truth while it brakes. Slip during acceleration is still unmodelled (KI-020).
- The mass and inertia derivation is validated against the **digital twin
  only** (the roundtrip recovers both within 2% of a known robot). Its
  real-hardware accuracy is unvalidated until the hardware window, so treat
  the numbers as measured-but-unproven until then (KI-020 stays open).
- Do not tune `b`, `zeta`, or the validation thresholds to a single path.
