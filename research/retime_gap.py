"""E1: the timing gap between the NLP and the time floor of its own path.

Fixes each solved plan's geometry (the sample path and headings) and re-times
it with a global forward-backward pass (TOPP-style, on E = v^2) under the same
bounds the NLP enforces: the wheel-speed box (0.99 x max linear speed), the
motor torque-speed line, traction against the lateral weight transfer, and the
wheel acceleration ramp. The re-time is globally time-optimal for that
geometry, so the gap

    gap = T_nlp - T_topp

is what the NLP's jerk/discretisation pricing plus its local solve put between
the plan and the time floor of its own path. Small gap => the timing layer is
already near the floor; large gap => the NLP pays visibly for smoothness (or
leaks time), and a convex timing layer is worth measuring against it.

    python research/retime_gap.py                  # gap on every saved plan
    python research/retime_gap.py --limit 5        # quick smoke run
    python research/retime_gap.py --validate       # + Immrax gate verdicts

--validate runs the production reachability gate (plant propagation) on the
original and the re-timed plan and counts verdict flips. Slow: two validator
runs per plan.

Interior samples with vl = vr = 0 are treated as required stops (the NLP pins
them with equalities; an incidental zero-velocity dip is rare and forcing a
stop there can only shrink the gap, so the reported gap is a lower bound).
The saved corpus is whatever research runs exported to research/results/traj
(103 plans), not the 32-mission suite itself.

Self-check: every plan's recorded wheel forces are recomputed from its own
(vl, vr) with the production force model; the run aborts if the config parse
or the formulas have drifted from what produced the plans.
"""

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from overboard.models.drivetrain_physics import (
    diff_drive_wheel_forces,
    wheel_normal_forces,
)
from overboard.models.robot_model import CG_HEIGHT_M, RobotConfig

DEFAULT_TRAJ_DIR = Path(__file__).resolve().parent / "results" / "traj"
DEFAULT_ACCEL_LIMIT = 7.8  # the DriveBase's measured ramp (m/s^2), q18_decompose
SUBDIV = 4
TOPP_PASSES = 3
EPS = 1e-9


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Profile:
    """The NLP's bounds as a feasible-set oracle on the (s, E) plane.

    Matches the solver's default robust arm (``apply_headroom=True``): kinematics
    on the nominal track width, the force split on max inertia and the minimum
    track width, traction against the minimum cof, and the motor line from
    ``get_motor_linear_limits(apply_headroom=True)``.
    """

    def __init__(self, cfg, accel_limit):
        self.mass = cfg.mass
        self.g = cfg.g
        unc = cfg.uncertainties
        self.epi_inertia = max(unc["inertia_kg_m2"])
        self.epi_tw = min(unc["track_width_range"])
        self.cof = min(unc["cof"])
        self.f_stall, self.v_noload = cfg.get_motor_linear_limits(apply_headroom=True)
        self.W = cfg.track_width  # kinematics, nominal, as the NLP's vl/vr rows
        self.v_bound = 0.99 * cfg.max_linear_speed(apply_headroom=True)
        self.accel_limit = accel_limit

    def feasible_Eprime(self, sigma, kappa, kappa_p, E, E_prime):
        u = math.sqrt(max(E, 0.0))
        vl = u * (sigma - 0.5 * self.W * kappa)
        vr = u * (sigma + 0.5 * self.W * kappa)
        if abs(vl) > self.v_bound or abs(vr) > self.v_bound:
            return False
        al = 0.5 * E_prime * (sigma - 0.5 * self.W * kappa) - 0.5 * self.W * E * kappa_p
        ar = 0.5 * E_prime * (sigma + 0.5 * self.W * kappa) + 0.5 * self.W * E * kappa_p
        if abs(al) > self.accel_limit or abs(ar) > self.accel_limit:
            return False
        fl, fr = diff_drive_wheel_forces(al, ar, self.mass, self.epi_inertia, self.epi_tw)
        # The epigraph rows: motor torque-speed line at both wheel speeds and
        # traction against the lateral weight transfer (v * omega on epi_tw).
        v1 = 0.5 * (vl + vr)
        omega_epi = (vr - vl) / self.epi_tw
        nl, nr = wheel_normal_forces(self.mass, self.g, CG_HEIGHT_M, self.epi_tw,
                                     v1 * omega_epi)
        fl_max = min(self.f_stall * (1.0 - abs(vl) / self.v_noload),
                     self.cof * max(nl, 0.0))
        fr_max = min(self.f_stall * (1.0 - abs(vr) / self.v_noload),
                     self.cof * max(nr, 0.0))
        return abs(fl) <= fl_max and abs(fr) <= fr_max

    def bisect_Eprime(self, sigma, kappa, kappa_p, E, sign):
        """Largest feasible E' with the given sign (0 if even a step fails)."""
        lo, hi = 0.0, 1.0
        while hi < 1e4 and self.feasible_Eprime(sigma, kappa, kappa_p, E, sign * hi):
            lo, hi = hi, hi * 2.0
        if not self.feasible_Eprime(sigma, kappa, kappa_p, E, sign * lo):
            return 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.feasible_Eprime(sigma, kappa, kappa_p, E, sign * mid):
                lo = mid
            else:
                hi = mid
        return sign * lo


def load_plan(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["trajectory"]["config"], raw["trajectory"]["samples"]


def self_check(cfg, samples, path):
    """The recorded forces must be reproducible from the recorded accelerations.

    At a stitched piece boundary (an interior stop) the exporter stores
    al = ar = 0 while the interval leaving the stop carries real acceleration,
    so the finite-difference replay is only a diagnostic there.
    """
    max_f_err = 0.0
    max_a_err = 0.0
    for k in range(len(samples) - 1):
        if "al" in samples[k] and "fl" in samples[k]:
            fl, fr = diff_drive_wheel_forces(samples[k]["al"], samples[k]["ar"],
                                             cfg.mass, cfg.inertia, cfg.track_width)
            max_f_err = max(max_f_err, abs(fl - samples[k]["fl"]),
                            abs(fr - samples[k]["fr"]))
        dt = samples[k + 1]["t"] - samples[k]["t"]
        if dt <= 0:
            continue
        al = (samples[k + 1]["vl"] - samples[k]["vl"]) / dt
        ar = (samples[k + 1]["vr"] - samples[k]["vr"]) / dt
        interior_stop = (0 < k < len(samples) - 2
                         and abs(samples[k]["vl"]) < EPS and abs(samples[k]["vr"]) < EPS)
        if "al" in samples[k] and not interior_stop:
            max_a_err = max(max_a_err, abs(al - samples[k]["al"]),
                            abs(ar - samples[k]["ar"]))
    if max_f_err > 1e-5:
        raise SystemExit(f"self-check failed on {path}: force model drift {max_f_err:.3e} N")
    return max_f_err, max_a_err


def plan_geometry(cfg, samples):
    """Nodes: arc length, kappa, travel direction. Runs: stop-to-stop pieces."""
    n = len(samples)
    ths = [q["heading"] for q in samples]
    ds, sigma, dth = [], [], []
    for k in range(n - 1):
        dx = samples[k + 1]["x"] - samples[k]["x"]
        dy = samples[k + 1]["y"] - samples[k]["y"]
        d = math.hypot(dx, dy)
        ds.append(d)
        dot = dx * math.cos(ths[k]) + dy * math.sin(ths[k])
        sigma.append(1.0 if dot >= 0 else -1.0)
        dth.append(wrap(ths[k + 1] - ths[k]))
    s_node = [0.0]
    for d in ds:
        s_node.append(s_node[-1] + d)
    kappa_n = []
    for k in range(n):
        if k == 0:
            kappa_n.append(dth[0] / ds[0] if ds[0] > EPS else 0.0)
        elif k == n - 1:
            kappa_n.append(dth[-1] / ds[-1] if ds[-1] > EPS else 0.0)
        else:
            den = ds[k - 1] + ds[k]
            kappa_n.append((dth[k - 1] + dth[k]) / den if den > EPS else 0.0)
    # Run boundaries: start, end, required stops (vl = vr = 0), direction flips.
    bounds = {0, n - 1}
    for k in range(1, n - 1):
        if abs(samples[k]["vl"]) < EPS and abs(samples[k]["vr"]) < EPS:
            bounds.add(k)
        if ds[k - 1] > EPS and ds[k] > EPS and sigma[k - 1] != sigma[k]:
            bounds.add(k)
    bounds = sorted(bounds)
    runs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]
    return s_node, kappa_n, sigma, runs


def retime(profile, s_node, kappa_n, sigma, runs):
    """Forward-backward pass on E = v^2 per run.

    Returns node energies E_node and node times t_node at the original sample
    nodes, plus the re-timed profile's own metrics.
    """
    E_node = [0.0] * len(s_node)
    t_node = [0.0] * len(s_node)
    peak_al = peak_ar = peak_jerk = 0.0
    smooth = disc = 0.0
    for (a, b) in runs:
        # Grid: SUBDIV steps per interval. Nodes j = 0..SUBDIV*n_steps.
        steps = []  # per step: (kappa_mid, kappa_p, sigma, ds_step)
        nodes_kappa = []
        n_steps = 0
        for k in range(a, b):
            d = (s_node[k + 1] - s_node[k]) / SUBDIV
            kap0, kap1 = kappa_n[k], kappa_n[k + 1]
            kp = (kap1 - kap0) / max(s_node[k + 1] - s_node[k], EPS)
            if n_steps == 0:
                nodes_kappa.append(kap0)
            for j in range(SUBDIV):
                f_mid = (j + 0.5) / SUBDIV
                steps.append((kap0 + f_mid * (kap1 - kap0), kp, sigma[k], d))
                nodes_kappa.append(kap1 if j == SUBDIV - 1 else kap0 + (j + 1) / SUBDIV * (kap1 - kap0))
                n_steps += 1
        if n_steps == 0:
            continue
        ceil = []
        for kap in nodes_kappa:
            u = profile.v_bound / (1.0 + 0.5 * profile.W * abs(kap))
            ceil.append(u * u)
        E = [0.0] * (n_steps + 1)
        for _ in range(TOPP_PASSES):
            for i in range(n_steps):
                kap, kp, sg, d = steps[i]
                emax = profile.bisect_Eprime(sg, kap, kp, E[i], +1.0)
                E[i + 1] = min(ceil[i + 1], E[i] + emax * d)
            E[n_steps] = 0.0
            for i in range(n_steps - 1, -1, -1):
                kap, kp, sg, d = steps[i]
                emin = profile.bisect_Eprime(sg, kap, kp, E[i + 1], -1.0)
                E[i] = min(E[i], E[i + 1] + abs(emin) * d, ceil[i])
            E[0] = 0.0
        # Time integration (exact for constant tangential accel) + metrics.
        acc = t_node[a]
        al_prev = ar_prev = None
        for i in range(n_steps):
            kap, kp, sg, d = steps[i]
            u0 = math.sqrt(max(E[i], 0.0))
            u1 = math.sqrt(max(E[i + 1], 0.0))
            dt = 2.0 * d / (u0 + u1) if (u0 + u1) > EPS else 0.0
            acc += dt
            e_p = (E[i + 1] - E[i]) / d if d > 0 else 0.0
            E_mid = 0.5 * (E[i] + E[i + 1])
            al = 0.5 * e_p * (sg - 0.5 * profile.W * kap) - 0.5 * profile.W * E_mid * kp
            ar = 0.5 * e_p * (sg + 0.5 * profile.W * kap) + 0.5 * profile.W * E_mid * kp
            peak_al = max(peak_al, abs(al))
            peak_ar = max(peak_ar, abs(ar))
            if al_prev is not None and dt > 0:
                peak_jerk = max(peak_jerk, abs(al - al_prev) / dt, abs(ar - ar_prev) / dt)
                smooth += (al - al_prev) ** 2 + (ar - ar_prev) ** 2
            al_prev, ar_prev = al, ar
            if dt > 0:
                v = 0.5 * (u0 + u1)
                om0, om1 = kap * u0, kap * u1
                sagitta = 0.125 * v * 0.5 * (om0 + om1) * dt * dt
                parab = 0.125 * v * (om1 - om0) * dt * dt
                hold = 0.5 * (u1 - u0) * 0.020
                disc += sagitta * sagitta + parab * parab + 0.25 * hold * hold
            if (i + 1) % SUBDIV == 0:
                t_node[a + (i + 1) // SUBDIV] = acc
        for k in range(a, b + 1):
            E_node[k] = E[(k - a) * SUBDIV]
    return E_node, t_node, {
        "peak_al": peak_al, "peak_ar": peak_ar, "peak_jerk": peak_jerk,
        "smoothness": smooth, "disc": disc}


def plan_metrics(samples, track_width, boundaries):
    """The plan's own profile metrics; jerk/smoothness never cross run
    boundaries (the NLP's pieces are independent at stops, KI-024)."""
    peak_al = peak_ar = peak_jerk = 0.0
    smooth = disc = 0.0
    al_prev = ar_prev = None
    for k in range(len(samples) - 1):
        q, r = samples[k], samples[k + 1]
        dt = r["t"] - q["t"]
        if dt <= 0:
            continue
        al, ar = (r["vl"] - q["vl"]) / dt, (r["vr"] - q["vr"]) / dt
        peak_al = max(peak_al, abs(al))
        peak_ar = max(peak_ar, abs(ar))
        if al_prev is not None and k not in boundaries:
            peak_jerk = max(peak_jerk, abs(al - al_prev) / dt, abs(ar - ar_prev) / dt)
            smooth += (al - al_prev) ** 2 + (ar - ar_prev) ** 2
        al_prev, ar_prev = al, ar
        v0 = 0.5 * (q["vl"] + q["vr"])
        v1 = 0.5 * (r["vl"] + r["vr"])
        om0 = q.get("omega", (q["vr"] - q["vl"]) / track_width)
        om1 = r.get("omega", (r["vr"] - r["vl"]) / track_width)
        v = 0.5 * (v0 + v1)
        sagitta = 0.125 * v * 0.5 * (om0 + om1) * dt * dt
        parab = 0.125 * v * (om1 - om0) * dt * dt
        hold = 0.5 * (v1 - v0) * 0.020
        disc += sagitta * sagitta + parab * parab + 0.25 * hold * hold
    return {"peak_al": peak_al, "peak_ar": peak_ar, "peak_jerk": peak_jerk,
            "smoothness": smooth, "disc": disc}


def build_samples(cfg, samples, kappa_n, sigma, E_node, t_node):
    """The re-timed plan at the original sample nodes, same sample schema."""
    out = []
    W = cfg.track_width
    for k, q in enumerate(samples):
        u = math.sqrt(max(E_node[k], 0.0))
        sg = sigma[k] if k < len(sigma) else sigma[-1]
        kap = kappa_n[k]
        out.append({"t": t_node[k], "x": q["x"], "y": q["y"], "heading": q["heading"],
                    "vl": u * (sg - 0.5 * W * kap),
                    "vr": u * (sg + 0.5 * W * kap),
                    "omega": kap * u})
    for k in range(len(out) - 1):
        dt = out[k + 1]["t"] - out[k]["t"]
        if dt > 0:
            out[k]["al"] = (out[k + 1]["vl"] - out[k]["vl"]) / dt
            out[k]["ar"] = (out[k + 1]["vr"] - out[k]["vr"]) / dt
    if out:
        out[-1]["al"] = 0.0
        out[-1]["ar"] = 0.0
    return out


def validate_pair(cfg, orig, retimed):
    from overboard.models.jax_robot_model import JAXRobotConfig
    from overboard.verification.immrax_validator import ImmraxValidator
    u = cfg.uncertainties
    unc = (u["cof"], u["torque_headroom"], u["gear_backlash_m"],
           u["track_width_range"], u["wheel_radius_range"], u["inertia_kg_m2"])
    validator = ImmraxValidator(JAXRobotConfig(cfg), propagation="plant")
    out = {}
    for label, samples in (("nlp", orig), ("topp", retimed)):
        report = validator.validate_trajectory(samples, *unc)
        env = report["reachability"]["envelope"]
        radius = max(math.hypot((e["x_max"] - e["x_min"]) / 2.0,
                                (e["y_max"] - e["y_min"]) / 2.0) for e in env)
        out[label] = {
            "tracking_cm": float(report.get("max_tracking_error_m", 0.0)) * 100.0,
            "envelope_cm": radius * 100.0,
            "motor_violation_N": float(report.get("max_motor_violation_N", 0.0)),
            "slip_l_N": float(report.get("max_slip_l_N", 0.0)),
            "slip_r_N": float(report.get("max_slip_r_N", 0.0)),
            "speed_violation_m_s": float(report.get("speed_violation_m_s", 0.0)),
            "diverged": bool(report["reachability"].get("diverged", False)),
            "passed": bool(report.get("passed")),
        }
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", default=str(DEFAULT_TRAJ_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--accel-limit", type=float, default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "retime_gap.jsonl"))
    args = ap.parse_args(argv)

    paths = sorted(Path(args.traj_dir).glob("*.traj"))
    if args.limit:
        paths = paths[: args.limit]
    accel_default = args.accel_limit or DEFAULT_ACCEL_LIMIT
    records = []
    tot_f_err = 0.0
    gaps = []
    n_stop = n_flip = n_accel_viol = 0
    for path in paths:
        cfg_dict, samples = load_plan(path)
        cfg = RobotConfig(cfg_dict["robot"] if "robot" in cfg_dict else cfg_dict)
        f_err, a_err = self_check(cfg, samples, path)
        tot_f_err = max(tot_f_err, f_err)
        accel_limit = args.accel_limit
        if accel_limit is None:
            accel_limit = getattr(cfg, "accel_limit_m_s2", None) or DEFAULT_ACCEL_LIMIT
        profile = Profile(cfg, accel_limit)
        s_node, kappa_n, sigma, runs = plan_geometry(cfg, samples)
        n_stop += sum(1 for k in range(1, len(samples) - 1)
                      if abs(samples[k]["vl"]) < EPS and abs(samples[k]["vr"]) < EPS)
        n_flip += sum(1 for k in range(1, len(sigma)) if sigma[k] != sigma[k - 1])
        E_node, t_node, m_topp = retime(profile, s_node, kappa_n, sigma, runs)
        boundaries = {a for (a, _b) in runs if a > 0}
        m_nlp = plan_metrics(samples, cfg.track_width, boundaries)
        T_nlp = samples[-1]["t"] - samples[0]["t"]
        T_topp = t_node[-1] - t_node[0]
        if m_nlp["peak_al"] > accel_limit + 1e-6:
            n_accel_viol += 1
        rec = {
            "plan": path.stem,
            "samples": len(samples),
            "runs": len(runs),
            "T_nlp_s": T_nlp,
            "T_topp_s": T_topp,
            "gap_s": T_nlp - T_topp,
            "gap_pct": 100.0 * (T_nlp - T_topp) / T_topp if T_topp > 0 else float("nan"),
            "nlp_peak_accel": max(m_nlp["peak_al"], m_nlp["peak_ar"]),
            "topp_peak_accel": max(m_topp["peak_al"], m_topp["peak_ar"]),
            "nlp_peak_jerk": m_nlp["peak_jerk"],
            "topp_peak_jerk": m_topp["peak_jerk"],
            "nlp_smoothness": m_nlp["smoothness"],
            "topp_smoothness": m_topp["smoothness"],
            "nlp_disc": m_nlp["disc"],
            "topp_disc": m_topp["disc"],
            "accel_limit": accel_limit,
            "orig_exceeds_ramp": m_nlp["peak_al"] > accel_limit + 1e-6,
        }
        if args.validate:
            retimed = build_samples(cfg, samples, kappa_n, sigma, E_node, t_node)
            rec["gates"] = validate_pair(cfg, samples, retimed)
        records.append(rec)
        gaps.append(rec["gap_pct"])

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    gaps_sorted = sorted(gaps)

    def pct(p):
        return gaps_sorted[min(len(gaps_sorted) - 1, int(p / 100.0 * len(gaps_sorted)))] if gaps_sorted else float("nan")

    print(f"plans: {len(records)}  (self-check force-model error <= {tot_f_err:.2e} N, "
          f"accel replay error <= {a_err:.2e} m/s^2)")
    print(f"interior required stops: {n_stop}   direction flips: {n_flip}   "
          f"plans over the {accel_default} m/s^2 ramp: {n_accel_viol}")
    if gaps:
        print("gap T_nlp - T_topp [% of the time floor]")
        print(f"  median {pct(50):6.2f}   p25 {pct(25):6.2f}   p75 {pct(75):6.2f}   "
              f"min {min(gaps):6.2f}   max {max(gaps):6.2f}")
        print(f"  plans within 1% of the floor: {sum(1 for g in gaps if g < 1.0)}/{len(gaps)}   "
              f"within 5%: {sum(1 for g in gaps if g < 5.0)}/{len(gaps)}   "
              f"over 10%: {sum(1 for g in gaps if g > 10.0)}/{len(gaps)}")
    if args.validate:
        flips = sum(1 for rec in records
                    if rec.get("gates") and
                    rec["gates"]["nlp"]["passed"] != rec["gates"]["topp"]["passed"])
        n_nlp = sum(1 for rec in records if rec.get("gates") and rec["gates"]["nlp"]["passed"])
        n_topp = sum(1 for rec in records if rec.get("gates") and rec["gates"]["topp"]["passed"])
        print(f"verdict flips under the plant gate: {flips}/{len(records)}   "
              f"(passing: nlp {n_nlp}, topp {n_topp})")
        for label in ("nlp", "topp"):
            motor = sum(1 for rec in records if rec.get("gates")
                        and rec["gates"][label]["motor_violation_N"] > 1e-3)
            slip = sum(1 for rec in records if rec.get("gates")
                       and max(rec["gates"][label]["slip_l_N"],
                               rec["gates"][label]["slip_r_N"]) > 1e-3)
            track = sum(1 for rec in records if rec.get("gates")
                        and rec["gates"][label]["tracking_cm"] > 0.5)
            div = sum(1 for rec in records if rec.get("gates")
                      and rec["gates"][label]["diverged"])
            print(f"  {label:<5} failures: motor {motor}  slip {slip}  "
                  f"tracking>5mm {track}  diverged {div}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
