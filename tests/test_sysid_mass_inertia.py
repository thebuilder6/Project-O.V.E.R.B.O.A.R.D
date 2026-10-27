"""Mass and inertia in the sysid pipeline: formulas, roundtrip, the sheet.

Q19: inertia ranks second in the one-knob robot sensitivity and mass is
minor but real, and no procedure step landed either into the robot config.
The sheet weigh-in lands the mass; the turn steps plus the measured
torque-speed line derive the inertia. The derivation is validated against
the digital twin here only -- real hardware stays unvalidated (KI-020).
"""

import csv
import json
import math
import os
import tempfile
import unittest

import numpy as np

from overboard.sysid import fit as fit_mod
from overboard.sysid.fit import parse_session, write_sheet
from overboard.sysid.sheet_form import (
    SHEET_FIELDS,
    WEIGH_IN_TEST,
    WEIGH_REPEATS,
    weigh_in_rows,
    write_form,
)

# The tolerance is pinned before anything runs, and is never relaxed after
# seeing numbers. The weighed mass and the derived inertia must both come
# back within 2% of the sim's truth. The straight-step derived mass is a
# cross-check that rides the encoder gradient instead of the weigh-in, so
# it carries its own, wider bound.
TOLERANCE = 0.02
CROSS_CHECK_TOLERANCE = 0.05

# The digital truth of the simulated session.
R_W = 0.028
TRACK = 0.0965
GEARING = 1.35
T_STALL = 0.08          # N m at the motor shaft
NO_LOAD_WHEEL = 18.0    # rad/s at the wheel
M_TRUE = 0.85           # kg
I_TRUE = 0.0031         # kg m^2
B_V = 1.2               # N s/m, the drag the logged torque fights
B_W = 0.0035            # N m s

V_NL = R_W * NO_LOAD_WHEEL
W_NL = 2.0 * R_W * NO_LOAD_WHEEL / TRACK

_RATE_HEADER = ("t_ms,loop_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg,"
                "gyro_z_rad_s,ax_m_s2,ay_m_s2,batt_mv,batt_ma,"
                "l_load_mNm,r_load_mNm")

# The hub's own formatter, so the fixture logs the same digits the robot
# does: encoder angles land on whole degrees (Pybricks motor.angle()).
_HUB_RATE_FORMAT = ("%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.5f,"
                    "%.3f,%.3f,%.0f,%.0f,%.1f,%.1f")


def _clip(value):
    return max(-1.0, min(1.0, value))


def _command_at(sequence, t_ms):
    """The command in SI units: m/s and rad/s (the sequence is hub units)."""
    elapsed = 0.0
    for duration, v_cmd, w_cmd in sequence:
        elapsed += duration
        if t_ms < elapsed:
            return v_cmd / 1000.0, w_cmd * math.pi / 180.0
    return sequence[-1][1] / 1000.0, sequence[-1][2] * math.pi / 180.0


def _simulate(sequence, axis, seed):
    """One run through the closed loop the deployed drivebase runs.

    The semantics are the ones DrivetrainModel.step_response documents: the
    reference ramps at the acceleration limit on every drive() call, the
    duty is back-EMF feed-forward plus a speed-error term, and the motor is
    on its torque-speed line. The plant is the first-order body model with
    the true mass and inertia. What is logged is what the closed loop
    actually applied -- the motor torques -- which is exactly what the
    derivation inverts; the command column alone could never carry it.
    """
    rng = np.random.default_rng(seed)
    control_period = 0.020
    sample_period = 0.010
    sub_dt = 0.002
    sub_per_control = int(round(control_period / sub_dt))
    sub_per_sample = int(round(sample_period / sub_dt))
    steps = int(round(sum(seg[0] for seg in sequence) / 1000.0 / sub_dt))

    v = 0.0
    omega = 0.0
    ref_v = 0.0
    ref_w = 0.0
    integ_v = 0.0
    integ_w = 0.0
    duty_v = 0.0
    duty_w = 0.0
    angle_l = 0.0
    angle_r = 0.0
    torque_l = 0.0
    torque_r = 0.0
    rows = []
    for i in range(steps):
        t_ms = i * sub_dt * 1000.0
        v_cmd, w_cmd = _command_at(sequence, t_ms)
        if i % sub_per_control == 0:
            # The reference never steps: it ramps at the acceleration limit.
            delta = v_cmd - ref_v
            ref_v += max(-3.0 * control_period, min(3.0 * control_period, delta))
            delta = w_cmd - ref_w
            ref_w += max(-25.0 * control_period, min(25.0 * control_period, delta))
            err_v = ref_v - v
            err_w = ref_w - omega
            duty_v = _clip(ref_v / V_NL + 2.5 * err_v + integ_v)
            duty_w = _clip(ref_w / W_NL + 0.21 * err_w + integ_w)
            integ_v += 60.0 * err_v * control_period
            integ_w += 5.2 * err_w * control_period

        if axis == "turn":
            torque = T_STALL * (
                duty_w - omega * TRACK / (2.0 * R_W * NO_LOAD_WHEEL))
            torque_l = torque
            torque_r = -torque
            moment = torque * GEARING * TRACK / R_W
            omega += (moment - B_W * omega) / I_TRUE * sub_dt
            v = 0.0
        elif axis == "wall_stall":
            torque = T_STALL * duty_v
            torque_l = torque_r = torque
            v = 0.0
            omega = 0.0
        elif axis == "top_speed":
            torque = T_STALL * (duty_v - v / V_NL)
            torque_l = torque_r = torque
            v += (2.0 * torque * GEARING / R_W) / 1.2 * sub_dt
            omega = 0.0
        else:
            torque = T_STALL * (duty_v - v / V_NL)
            torque_l = torque_r = torque
            force = 2.0 * torque * GEARING / R_W
            v += (force - B_V * v) / M_TRUE * sub_dt
            omega = 0.0

        if axis == "turn":
            n_l = omega * TRACK / (2.0 * R_W)
            n_r = -n_l
        else:
            n_l = n_r = v / R_W
        angle_l += math.degrees(n_l * sub_dt)
        angle_r += math.degrees(n_r * sub_dt)

        if i % sub_per_sample == 0:
            rows.append(_HUB_RATE_FORMAT % (
                t_ms, 10.0, v_cmd * 1000.0, w_cmd * 180.0 / math.pi,
                float(round(angle_l)), float(round(angle_r)),
                -omega + rng.normal(0.0, 0.002),
                0.0, 0.0, 8000.0, 0.0,
                torque_l * 1000.0 + rng.normal(0.0, 0.3),
                torque_r * 1000.0 + rng.normal(0.0, 0.3),
            ))
    return rows


def _session_text(runs):
    lines = []
    for run_id, run in enumerate(runs, start=1):
        name, repeat, rows = run[0], run[1], run[2]
        poses = run[3] if len(run) > 3 else 0
        header = run[4] if len(run) > 4 else _RATE_HEADER
        lines.append("TEST_START " + name)
        lines.append("META,run_id,%d" % run_id)
        lines.append("META,test," + name)
        lines.append("META,repeat,%d" % repeat)
        if poses:
            lines.append("META,poses,%d" % poses)
        lines.append("DATA")
        lines.append(header)
        lines.extend(rows)
        lines.append("TEST_END,%d" % len(rows))
    return "\n".join(lines) + "\n"


def _simulated_session(seed=20261026):
    """3 repeats of every rate test the collector runs (KI-010)."""
    runs = []
    for repeat in range(3):
        straight_levels = [50, 100, 200, 400, 600]
        turn_levels = [30, 60, 120, 240]
        if repeat % 2:
            # The step order reverses on every other repeat.
            straight_levels = straight_levels[::-1]
            turn_levels = turn_levels[::-1]
        kinds = (
            ("straight_steps_fwd", straight_levels, 1, "straight"),
            ("straight_steps_rev", straight_levels, -1, "straight"),
            ("turn_steps_cw", turn_levels, -1, "turn"),
            ("turn_steps_ccw", turn_levels, 1, "turn"),
        )
        for index, (name, levels, sign, axis) in enumerate(kinds):
            sequence = []
            for level in levels:
                if axis == "straight":
                    sequence.append((900, sign * level, 0))
                else:
                    sequence.append((900, 0, sign * level))
                sequence.append((600, 0, 0))
            runs.append((name, repeat,
                         _simulate(sequence, axis, seed + 10 * repeat + index)))
        runs.append(("wall_stall", repeat,
                     _simulate([(2500, 30, 0)], "wall_stall",
                               seed + 10 * repeat + 4)))
        runs.append(("top_speed", repeat,
                     _simulate([(3000, 1000, 0)], "top_speed",
                               seed + 10 * repeat + 5)))
    # One pose run, so the sheet has a measurement to land: an old sheet
    # without the weigh-in still has this row filled in.
    runs.append(("odometry_straight", 0,
                 ["10000.0,100.0,0.0,1000.0,0.0,0.0,0.0,8000.0"], 1,
                 "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,"
                 "db_a_deg,db_w_deg_s,batt_mv"))
    return runs


def _fill_sheet(sheet_path, weigh_in_values):
    """Fill the sheet the way the operator does: the tape rows, and the
    weigh-in rows when the robot was weighed."""
    with open(sheet_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["test"] == WEIGH_IN_TEST:
            continue
        row["x_mm"] = "998"
        row["y_mm"] = "2"
        row["turn_right_deg"] = "0"
    if weigh_in_values:
        filled = [row for row in rows if row["test"] == WEIGH_IN_TEST]
        for row, value in zip(filled, weigh_in_values):
            row["mass_kg"] = "%.4f" % value
    with open(sheet_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestMassInertiaFormulas(unittest.TestCase):
    """The pinned line J = tau_mech * (B + K_v) on synthetic steps."""

    def test_the_straight_step_recovers_the_mass(self):
        j_true = M_TRUE
        damping = 5.0
        back_emf = 15.3
        drive0 = 5.0  # the zero-speed drive the loop holds
        total = damping + back_emf
        y_ss = drive0 / total
        times = np.arange(0.0, 0.9, 0.01)
        speed = y_ss * (1.0 - np.exp(-total * times / j_true))
        # What the motor torque logs is the drive after the back-EMF.
        logged = drive0 - back_emf * speed

        estimate = fit_mod._step_j_estimate(
            times, speed, logged, back_emf)

        self.assertIsNotNone(estimate)
        j, b, tau_mech = estimate
        self.assertAlmostEqual(j, j_true, delta=j_true * 1e-3)
        self.assertAlmostEqual(b, damping, delta=damping * 1e-3)
        self.assertAlmostEqual(
            tau_mech, j_true / (damping + back_emf),
            delta=tau_mech * 1e-3)
        # The pinned line, both ways: J = tau_mech * (B + K_v) and
        # J = tau_mech * K0 / y_ss with K0 = G_ss + K_v * y_ss.
        g_ss = float(np.median(logged[-30:]))
        k0 = g_ss + back_emf * (g_ss / b)
        self.assertAlmostEqual(j, tau_mech * (b + back_emf), delta=j * 1e-9)
        self.assertAlmostEqual(
            j, tau_mech * k0 / (g_ss / b), delta=j * 1e-3)

    def test_the_reverse_step_recovers_the_same_mass(self):
        j_true = M_TRUE
        damping = 5.0
        back_emf = 15.3
        drive0 = -5.0
        total = damping + back_emf
        y_ss = drive0 / total
        times = np.arange(0.0, 0.9, 0.01)
        speed = y_ss * (1.0 - np.exp(-total * times / j_true))
        logged = drive0 - back_emf * speed

        estimate = fit_mod._step_j_estimate(
            times, speed, logged, back_emf)

        self.assertIsNotNone(estimate)
        j, b, _tau = estimate
        self.assertAlmostEqual(j, j_true, delta=j_true * 1e-3)
        self.assertAlmostEqual(b, damping, delta=damping * 1e-3)

    def test_the_turn_step_recovers_the_inertia(self):
        """The generalized force from the logged torques, signs included."""
        j_true = I_TRUE
        damping = 0.03
        back_emf = fit_mod._back_emf_slope(
            "turn", T_STALL, NO_LOAD_WHEEL, R_W, TRACK, GEARING)
        torque0 = 0.006
        drive0 = torque0 * GEARING * TRACK / R_W
        total = damping + back_emf
        y_ss = drive0 / total
        times = np.arange(0.0, 0.9, 0.01)
        omega = y_ss * (1.0 - np.exp(-total * times / j_true))
        # The logged torque is the drive after the back-EMF, left wheel
        # forward and right wheel back for a clockwise spin: +T and -T.
        torque = (drive0 - back_emf * omega) * R_W / (GEARING * TRACK)
        rows = [
            _HUB_RATE_FORMAT % (
                t * 1000.0, 10.0, 0.0, y_ss * 180.0 / math.pi,
                0.0, 0.0, -omega[i], 0.0, 0.0, 8000.0, 0.0,
                torque[i] * 1000.0, -torque[i] * 1000.0)
            for i, t in enumerate(times)
        ]
        runs = parse_session(_session_text([("turn_steps_ccw", 0, rows)]))
        fit_mod._add_measured_outputs(runs, R_W)
        step = {"repeat": 0, "_run": runs[0], "_span": (0, len(times))}

        estimates = fit_mod._axis_j_estimates(
            [step], "turn", "w_meas_rad_s", R_W, TRACK, GEARING,
            T_STALL, NO_LOAD_WHEEL)

        self.assertEqual(len(estimates), 1)
        # This one goes through the hub's row format, which logs the motor
        # torque as 0.1 mNm -- about 2% of this step's torque. The mapping
        # and the signs are what is pinned here; the noiseless formula is
        # pinned by the straight test above.
        self.assertAlmostEqual(
            estimates[0]["j"], j_true, delta=j_true * 2e-2)
        self.assertAlmostEqual(
            estimates[0]["damping"], damping, delta=damping * 2e-2)
        self.assertAlmostEqual(
            estimates[0]["tau_mech_s"], j_true / (damping + back_emf),
            delta=j_true / (damping + back_emf) * 2e-2)

    def test_the_reflected_inertia_is_not_added_twice(self):
        """The identified J is the body-side value the force model reads.

        diff_drive_wheel_forces takes (mass, inertia) and never adds
        reflected_inertia_kg_m2, so the fit must write the effective value
        to inertia alone, never on top of the reflected one.
        """
        from overboard.models.robot_model import RobotConfig
        from overboard.sysid.digital_twin import DigitalTwinConfig

        fit_data = {"inertia": {
            "calibrated": True,
            "inertia_kg_m2": I_TRUE,
            "inertia_kg_m2_ci95": [I_TRUE * 0.99, I_TRUE * 1.01],
            "n_sources": 3,
        }}
        cfg = RobotConfig({"robot": {"reflected_inertia_kg_m2": 0.0004}})
        self.assertGreater(cfg.reflected_inertia_kg_m2, 0.0)
        cfg.apply_sysid_fit(DigitalTwinConfig.from_dict(fit_data))
        self.assertAlmostEqual(cfg.inertia, I_TRUE, delta=I_TRUE * 1e-9)


class TestMassInertiaRoundtrip(unittest.TestCase):
    """The digital-twin roundtrip: simulate, fit, recover within 2%."""

    @classmethod
    def setUpClass(cls):
        cls.session_text = _session_text(_simulated_session())

    def _fit(self, tmp, weigh_in_values):
        session = os.path.join(tmp, "session.csv")
        with open(session, "w", encoding="utf-8") as handle:
            handle.write(self.session_text)
        sheet = os.path.join(tmp, "sheet.csv")
        write_sheet(parse_session(self.session_text), sheet)
        _fill_sheet(sheet, weigh_in_values)
        profile = os.path.join(tmp, "profile.json")
        with open(profile, "w", encoding="utf-8") as handle:
            json.dump({"wheel_radius_m": R_W, "track_width_m": TRACK,
                       "gearing": GEARING}, handle)
        out = os.path.join(tmp, "sysid_fit.json")
        code = fit_mod.main(
            [session, "--sheet", sheet, "--profile", profile, "--out", out])
        self.assertEqual(code, 0)
        with open(out, "r", encoding="utf-8") as handle:
            return out, json.load(handle)

    def test_the_fit_recovers_the_mass_and_the_inertia(self):
        rng = np.random.default_rng(7)
        weigh_in = list(M_TRUE * (1.0 + rng.normal(0.0, 0.0015, WEIGH_REPEATS)))
        with tempfile.TemporaryDirectory() as tmp:
            out, result = self._fit(tmp, weigh_in)

            # The pinned tolerance: 2% of truth for both.
            mass = result["mass"]
            self.assertTrue(mass["calibrated"])
            self.assertAlmostEqual(
                mass["mass_kg"], M_TRUE, delta=TOLERANCE * M_TRUE)
            inertia = result["inertia"]
            self.assertTrue(inertia["calibrated"])
            self.assertAlmostEqual(
                inertia["inertia_kg_m2"], I_TRUE, delta=TOLERANCE * I_TRUE)

            # The straight steps cross-check the mass on the same machinery.
            self.assertAlmostEqual(
                mass["derived_mass_kg"], M_TRUE,
                delta=CROSS_CHECK_TOLERANCE * M_TRUE)

            # Measured ranges, not the assumed +/-10%.
            self.assertIn("mass_kg_ci95", mass)
            self.assertIn("inertia_kg_m2_ci95", inertia)
            self.assertGreaterEqual(inertia["n_sources"], 3)

            # And they land in the robot config.
            from overboard.models.robot_model import RobotConfig
            from overboard.sysid.digital_twin import DigitalTwinConfig
            cfg = RobotConfig({})
            cfg.apply_sysid_fit(DigitalTwinConfig.load(out))
            self.assertAlmostEqual(
                cfg.mass, M_TRUE, delta=TOLERANCE * M_TRUE)
            self.assertAlmostEqual(
                cfg.inertia, I_TRUE, delta=TOLERANCE * I_TRUE)
            low, high = cfg.uncertainties["inertia_kg_m2"]
            # The measured KI-010 interval, not the assumed +/-10%: it
            # holds the measured value and is centred on it (the default
            # would sit around the unmeasured default instead).
            self.assertLessEqual(low, cfg.inertia)
            self.assertGreaterEqual(high, cfg.inertia)
            self.assertAlmostEqual(
                0.5 * (low + high), cfg.inertia, delta=TOLERANCE * I_TRUE)
            self.assertIn("mass_kg", cfg.uncertainties)
            mass_low, mass_high = cfg.uncertainties["mass_kg"]
            self.assertLessEqual(mass_low, cfg.mass)
            self.assertGreaterEqual(mass_high, cfg.mass)

    def test_an_old_sheet_without_the_weigh_in_leaves_the_mass_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            _out, result = self._fit(tmp, weigh_in_values=None)

            self.assertFalse(result["mass"]["calibrated"])
            self.assertNotIn("mass_kg", result["mass"])
            # The inertia does not need the weigh-in: it is derived.
            self.assertTrue(result["inertia"]["calibrated"])
            self.assertAlmostEqual(
                result["inertia"]["inertia_kg_m2"], I_TRUE,
                delta=TOLERANCE * I_TRUE)


class TestWeighInSheet(unittest.TestCase):
    """The weigh-in survives the CSV and the form roundtrips."""

    def _small_runs(self):
        rows = [_HUB_RATE_FORMAT % (
            i * 10.0, 10.0, 100.0, 0.0, float(i), float(i), 0.0,
            0.0, 0.0, 8000.0, 0.0, 40.0, 40.0)
            for i in range(30)]
        return parse_session(
            _session_text([("odometry_straight", 0, rows)]))

    def test_the_weigh_in_survives_the_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.csv")
            write_sheet(self._small_runs(), path)

            # A blank sheet measures nothing yet, and its weigh rows are
            # not poses.
            self.assertEqual(fit_mod.read_weigh_in(path), [])
            self.assertEqual(fit_mod.read_sheet(path), [])

            values = [0.851, 0.849, 0.850]
            _fill_sheet(path, values)
            pairs = fit_mod.read_weigh_in(path)
            self.assertEqual([repeat for repeat, _kg in pairs], [0, 1, 2])
            for (_repeat, kg), expected in zip(pairs, values):
                self.assertAlmostEqual(kg, expected)
            # The weigh rows still do not become pose rows.
            self.assertEqual(fit_mod.read_sheet(path), [])

    def test_the_form_carries_the_weigh_in_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.html")
            write_form(
                [{"test": "odometry_straight", "repeat": 0, "stop": 0}],
                path, csv_name="calibration_sheet.csv",
                fit_command="python -m overboard.sysid.fit session.csv")
            with open(path, encoding="utf-8") as handle:
                text = handle.read()

        self.assertIn("<th>mass_kg</th>", text)
        self.assertIn(
            'var HEADER = "test,repeat,stop,x_mm,y_mm,turn_right_deg,mass_kg"',
            text)
        self.assertEqual(text.count('class="m"'), WEIGH_REPEATS)
        self.assertEqual(
            text.count('data-test="%s"' % WEIGH_IN_TEST), WEIGH_REPEATS)
        self.assertIn('data-test="%s" data-repeat="2"' % WEIGH_IN_TEST, text)
        self.assertEqual(len(weigh_in_rows()), WEIGH_REPEATS)

    def test_a_sheet_from_before_the_mass_column_still_reads(self):
        csv_text = ("test,repeat,stop,x_mm,y_mm,turn_right_deg\n"
                    "odometry_straight,0,0,998,2,0\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(csv_text)
            rows = fit_mod.read_sheet(path)
            weigh_in = fit_mod.read_weigh_in(path)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["x_mm"], 998.0)
        self.assertEqual(weigh_in, [])
