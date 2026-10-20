"""Compare the optimization variants, on the missions, with the figures.

The optimizer grew a set of switches in one day: the collocation, the
sample layout, the path splitting, the cost metric, the shape of the force
bound, the weight on the force, and the timing model. Each one changes the
plan, and none of them has been compared with the others on the same
missions with the same gates.

This runs them against each other, one switch at a time from the shipped
default, and writes the scorecard and the per-mission figures. A full cross
product is 2**8, so the switches are turned on one at a time: a switch's
effect is then its own, and the combinations are a follow-up.

The gates are the production ones: main.py's own failure reasons, the CPU
audit from validator.py, and the same thresholds. The scoring is the same
code path the CLI uses, so a number here means what it means there.

Run it from the repository root:

    python docs/measurements/2026-10-20-optimizations/compare_optimizations.py

The tree's fingerprint is taken before and after the run. A tree that
changed while the measurement was running is detectable rather than
silently wrong - the same rule research/harness.py follows.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import main as cli
from optimizer import TrajectoryOptimizer
from validator import validate_samples

OUT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(REPO, "examples", "fll_choreo.json")
SAMPLES = 10

MISSIONS = [
    "example_straight",
    "example_s_curve",
    "example_mission_first4",
    "example_complex_mission",
    "example_with_stops",
    "example_with_events",
]

# One switch at a time from the shipped default. The default is first, so
# every row of the scorecard can be read against it.
VARIANTS = [
    ("default", {}),
    ("trapezoid", {"self_consistent": False}),
    ("uniform-samples", {"proportional_samples": False}),
    ("joint-nlp", {"split_at_boundaries": False}),
    ("discretization-cost", {"cost_metric": "discretization"}),
    ("epigraph-force", {"epigraph_constraints": True}),
    ("force-weight-0.05", {"force_weight": 0.05}),
    ("force-weight-0.5", {"force_weight": 0.5}),
]

# The two timing models. The CLI's candidate set runs both and keeps the
# better, so both are measured and the better of the two is the answer.
TIMINGS = [("equal-duration", False), ("per-segment", True)]

TREE_FILES = [
    "optimizer.py", "main.py", "immrax_validator.py", "validator.py",
    "robot_model.py",
]

ENVELOPE_GATE_M = 0.02
TRACKING_GATE_M = 0.005


# --------------------------------------------------------------------------
# The tree's fingerprint
# --------------------------------------------------------------------------
def fingerprint():
    out = {}
    for name in TREE_FILES:
        path = os.path.join(REPO, name)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                out[name] = hashlib.md5(handle.read()).hexdigest().upper()
    return out


# --------------------------------------------------------------------------
# One measurement
# --------------------------------------------------------------------------
def measure(mission, variant, variant_kwargs, timing_name, per_segment_dt,
            robot_cfg, validator_cls, robot_config_cls, thresholds,
            stop_indices, events, waypoints, propagation="plant"):
    """Solve one variant under one timing model and score it with the gates."""
    started = time.time()
    optimizer = TrajectoryOptimizer(robot_cfg)
    samples, stats = optimizer.solve(
        waypoints,
        num_samples_per_segment=SAMPLES,
        accuracy_weight=0.0,
        stop_waypoint_indices=stop_indices,
        waypoint_events=events,
        guess_style="baseline",
        per_segment_dt=per_segment_dt,
        capture_iterations=True,
        verbose=False,
        **variant_kwargs)
    wall = time.time() - started

    record = {
        "mission": mission,
        "variant": variant,
        "timing": timing_name,
        "converged": bool(stats.get("converged", False)),
        "duration_s": float(samples[-1]["t"]) if samples else float("inf"),
        "iter_count": int(stats.get("iter_count", -1)),
        "final_cost": float(stats.get("final_cost", float("inf"))),
        "segment_samples": [int(c) for c in (stats.get("segment_samples") or [])],
        "wall_s": wall,
        "reasons": [],
        "ok": False,
    }
    if not samples:
        record["reasons"] = ["no samples"]
        return record, samples, stats, None

    # The gate chain, as main.py runs it.
    jax_cfg = robot_config_cls(robot_cfg)
    immrax_val = validator_cls(jax_cfg, propagation=propagation)
    unc = robot_cfg.uncertainties
    report = immrax_val.validate_trajectory(
        samples,
        unc["cof"], unc["torque_headroom"], unc["gear_backlash_m"],
        unc["track_width_range"], unc["wheel_radius_range"],
        unc["inertia_kg_m2"])

    for i, env in enumerate(report["reachability"]["envelope"]):
        if i < len(samples):
            samples[i]["reachability_envelope"] = env

    bounds = cli._waypoint_boundaries(stats.get("segment_samples"))
    failed_threshold = False
    max_vio = 0.0
    max_err = 0.0
    for i in range(len(samples)):
        wp_idx = cli._waypoint_of(bounds, i)
        local = cli._segment_limit(thresholds, wp_idx)
        env = samples[i].get("reachability_envelope", {})
        if not env:
            continue
        err = float(np.hypot((env["x_max"] - env["x_min"]) / 2.0,
                             (env["y_max"] - env["y_min"]) / 2.0))
        max_err = max(max_err, err)
        if env.get("unknown") or err > local:
            failed_threshold = True
            max_vio = max(max_vio, err)

    record["envelope_cm"] = max_err * 100.0
    record["tracking_cm"] = float(report.get("max_tracking_error_m", 0.0)) * 100.0
    record["envelope_truncated"] = bool(
        report["reachability"].get("envelope_truncated", False))
    record["diverged"] = bool(report["reachability"].get("diverged", False))

    reasons = list(cli._immrax_failure_reasons(report, failed_threshold, max_vio))
    cpu = validate_samples(samples, robot_cfg)
    record["jerk"] = float(cpu.get("metrics", {}).get("max_jerk_m_s3", 0.0) or 0.0)
    if not cpu["passed"]:
        reasons += list(cli._cpu_failure_reasons(cpu))
    record["reasons"] = reasons
    record["ok"] = bool(record["converged"] and not failed_threshold
                        and report["passed"] and cpu["passed"])
    # The convergence history holds numpy arrays, so it is kept beside the
    # record for the figure and not inside the record that is written out.
    history = []
    for entry in optimizer.iteration_history:
        history.append({k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in entry.items()})
    return record, samples, stats, report, history


# --------------------------------------------------------------------------
# The figures
# --------------------------------------------------------------------------
def radius(env):
    return float(np.hypot((env["x_max"] - env["x_min"]) / 2.0,
                           (env["y_max"] - env["y_min"]) / 2.0))


def midpoint_error(env, sample):
    return float(np.hypot((env["x_min"] + env["x_max"]) / 2.0 - sample["x"],
                           (env["y_min"] + env["y_max"]) / 2.0 - sample["y"]))


def detail_figure(mission, runs, waypoints, path):
    """The six-panel review figure, for a handful of variants."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    ax_path, ax_env, ax_track, ax_dt, ax_force, ax_table = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for run, color in zip(runs, colors):
        samples = run["samples"]
        envs = [s.get("reachability_envelope", {}) for s in samples]
        label = "%s / %s" % (run["variant"], run["timing"])
        style = "-" if run["record"]["ok"] else "--"
        # The default is drawn last and thickest, so it is not hidden under
        # the variants that land on top of it.
        is_default = run["variant"] == "default"
        z = 10 if is_default else 2
        width = 3.0 if is_default else 1.8
        ax_path.plot([s["x"] for s in samples], [s["y"] for s in samples],
                     color=color, linewidth=width, linestyle=style, zorder=z,
                     label=label)
        ax_env.plot([s["t"] for s in samples],
                    [radius(e) * 100 for e in envs if e],
                    color=color, linewidth=width, linestyle=style, zorder=z,
                    label=label)
        ax_track.plot([s["t"] for s in samples],
                      [midpoint_error(e, s) * 100 for e, s in zip(envs, samples) if e],
                      color=color, linewidth=width, linestyle=style, zorder=z,
                      label=label)
        ax_force.plot([s["t"] for s in samples], [s.get("fl", 0.0) for s in samples],
                      color=color, linewidth=width, linestyle=style, zorder=z,
                      label=label)

    # The segment durations, from the samples themselves.
    width = 0.8 / max(1, len(runs))
    for j, (run, color) in enumerate(zip(runs, colors)):
        samples = run["samples"]
        segs = run["record"].get("segment_samples") or []
        bounds = cli._waypoint_boundaries(segs)
        durations = []
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            if b < len(samples) and a < len(samples):
                durations.append(samples[b - 1]["t"] - samples[a]["t"])
        x = np.arange(len(durations))
        ax_dt.bar(x + (j - (len(runs) - 1) / 2.0) * width, durations,
                  width=width, color=color,
                  label="%s/%s" % (run["variant"], run["timing"]))

    ax_path.scatter([w[0] for w in waypoints], [w[1] for w in waypoints],
                    color='black', zorder=5)
    for i, w in enumerate(waypoints):
        ax_path.annotate(str(i), (w[0], w[1]), textcoords='offset points',
                         xytext=(5, 5))
    ax_path.set_xlabel('X (m)')
    ax_path.set_ylabel('Y (m)')
    ax_path.set_title('Path')
    xs = [w[0] for w in waypoints]
    ys = [w[1] for w in waypoints]
    span = max(max(xs) - min(xs), 0.1)
    ax_path.set_xlim(min(xs) - 0.15 * span, max(xs) + 0.15 * span)
    ax_path.set_ylim(min(ys) - 0.15 * span, max(ys) + 0.15 * span)
    ax_path.set_aspect('equal')
    ax_path.grid(True, linestyle='--', alpha=0.4)
    ax_path.legend(fontsize=8)

    ax_env.axhline(ENVELOPE_GATE_M * 100, color='darkorange', linestyle=':',
                   linewidth=1.2, label='2 cm transit threshold')
    ax_env.axhline(5.0, color='black', linestyle='--', linewidth=1.2,
                   label='5 cm divergence guard')
    ax_env.set_xlabel('Time (s)')
    ax_env.set_ylabel('Tube radius (cm)')
    ax_env.set_title('Reachability tube')
    ax_env.grid(True, linestyle='--', alpha=0.4)
    ax_env.legend(fontsize=8)

    ax_track.axhline(TRACKING_GATE_M * 100, color='darkorange', linestyle=':',
                     linewidth=1.2, label='0.5 cm tracking gate')
    ax_track.set_xlabel('Time (s)')
    ax_track.set_ylabel('Tracking error (cm)')
    ax_track.set_title('Closed-loop tracking error')
    ax_track.grid(True, linestyle='--', alpha=0.4)
    ax_track.legend(fontsize=8)

    ax_dt.set_xlabel('Segment index')
    ax_dt.set_ylabel('Segment duration (s)')
    ax_dt.set_title('Time spent per segment\n(the sample layout decides the split)')
    ax_dt.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_dt.legend(fontsize=8)

    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Left wheel force (N)')
    ax_force.set_title('Left wheel force profile')
    ax_force.grid(True, linestyle='--', alpha=0.4)
    ax_force.legend(fontsize=8)

    # The numbers, in one place.
    ax_table.axis('off')
    header = ("variant", "timing", "dur (s)", "env (cm)", "trk (cm)",
              "jerk", "iter", "gate")
    rows = [header]
    for run in runs:
        rec = run["record"]
        rows.append((
            rec["variant"],
            "equal-dur" if rec["timing"] == "equal-duration" else "per-seg",
            "%.3f" % rec["duration_s"],
            "%.2f" % rec.get("envelope_cm", float("nan")),
            "%.2f" % rec.get("tracking_cm", float("nan")),
            "%.1f" % rec.get("jerk", float("nan")),
            "%d" % rec.get("iter_count", -1),
            "ok" if rec["ok"] else "FAIL"))
    table = ax_table.table(cellText=rows[1:], colLabels=rows[0],
                            loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax_table.set_title('The numbers', fontsize=11)

    fig.suptitle("%s - the optimization variants" % mission, fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def scorecard_figure(records, path, missions=None):
    """Every mission against every variant: the many-variants view."""
    variants = sorted({r["variant"] for r in records})
    missions = missions or sorted({r["mission"] for r in records})
    fig, axes = plt.subplots(1, 4, figsize=(26, 9),
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    ax_grid, ax_dur, ax_env, ax_track = axes

    # The grid: one cell per (mission, variant), the better timing model.
    for col, variant in enumerate(variants):
        for row, mission in enumerate(MISSIONS):
            best = None
            for rec in records:
                if rec["mission"] == mission and rec["variant"] == variant:
                    if best is None or (rec["ok"], -rec["duration_s"]) > \
                            (best["ok"], -best["duration_s"]):
                        best = rec
            if best is None:
                continue
            ax_grid.add_patch(plt.Rectangle(
                (col + 0.05, len(MISSIONS) - row - 0.95), 0.9, 0.9,
                facecolor=("#d7f0d7" if best["ok"] else "#f7d7d7"),
                edgecolor='black', linewidth=0.6))
            ax_grid.text(col + 0.5, len(MISSIONS) - row - 0.5,
                         "%.2f s" % best["duration_s"], ha='center',
                         va='center', fontsize=8)
    ax_grid.set_xlim(0, len(variants))
    ax_grid.set_ylim(0, len(MISSIONS))
    ax_grid.set_xticks([i + 0.5 for i in range(len(variants))])
    ax_grid.set_xticklabels(variants, rotation=30, ha='right', fontsize=9)
    ax_grid.set_yticks([len(MISSIONS) - i - 0.5 for i in range(len(MISSIONS))])
    ax_grid.set_yticklabels(MISSIONS, fontsize=9)
    ax_grid.set_title('The better timing model per cell\n(green passes the gates, red does not)')
    ax_grid.tick_params(length=0)

    def bars(ax, key, title, ylabel, gate=None):
        width = 0.8 / len(variants)
        x = np.arange(len(MISSIONS))
        for j, variant in enumerate(variants):
            values, colors = [], []
            for mission in MISSIONS:
                best = None
                for rec in records:
                    if rec["mission"] == mission and rec["variant"] == variant:
                        if best is None or (rec["ok"], -rec["duration_s"]) > \
                                (best["ok"], -best["duration_s"]):
                            best = rec
                values.append(best.get(key, float("nan")) if best else float("nan"))
                colors.append("#4c78a8" if (best and best["ok"]) else "#e45756")
            ax.bar(x + (j - (len(variants) - 1) / 2.0) * width, values,
                   width=width, color=colors, label=variant)
        if gate is not None:
            ax.axhline(gate, color='darkorange', linestyle=':', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(MISSIONS, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    bars(ax_dur, "duration_s", "Duration (s)", "Duration (s)")
    bars(ax_env, "envelope_cm", "Envelope radius (cm)", "Envelope (cm)",
         gate=ENVELOPE_GATE_M * 100)
    bars(ax_track, "tracking_cm", "Tracking error (cm)", "Tracking (cm)",
         gate=TRACKING_GATE_M * 100)

    fig.suptitle("The optimization variants, on the example missions",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


# --------------------------------------------------------------------------
def main():
    # A pilot run: --missions and --variants narrow the set, so the harness
    # can be exercised on one mission before the whole matrix is run.
    wanted_missions = MISSIONS
    wanted_variants = [v for v, _ in VARIANTS]
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("--missions="):
            wanted_missions = arg.split("=", 1)[1].split(",")
        if arg.startswith("--variants="):
            wanted_variants = arg.split("=", 1)[1].split(",")

    before = fingerprint()
    config_data, robot_cfg = cli._load_config(CONFIG)
    validator_cls, robot_config_cls = cli._ensure_jax_classes()

    records = []
    runs = []
    for mission in wanted_missions:
        waypoints, events, thresholds, stop_indices = cli._load_waypoints(
            os.path.join(REPO, "examples", "%s.json" % mission))
        for variant, kwargs in VARIANTS:
            if variant not in wanted_variants:
                continue
            for timing_name, per_segment_dt in TIMINGS:
                print("running %s / %s / %s" % (mission, variant, timing_name),
                      flush=True)
                rec, samples, stats, report, history = measure(
                    mission, variant, kwargs, timing_name, per_segment_dt,
                    robot_cfg, validator_cls, robot_config_cls, thresholds,
                    stop_indices, events, waypoints)
                records.append(rec)
                runs.append({
                    "mission": mission, "variant": variant, "timing": timing_name,
                    "record": rec, "samples": samples, "stats": stats,
                    "history": history,
                })
                print("    %s | %s | %s" % (
                    "PASS" if rec["ok"] else "FAIL",
                    "%.3f s" % rec["duration_s"],
                    "; ".join(rec["reasons"]) or "every gate passed"), flush=True)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"records": records, "tree": before}, handle, indent=1)

    # The detail figure: the default and the three best alternatives.
    for mission in wanted_missions:
        by_variant = {}
        for run in runs:
            if run["mission"] != mission:
                continue
            key = run["variant"]
            if key not in by_variant or (run["record"]["ok"], -run["record"]["duration_s"]) > \
                    (by_variant[key]["record"]["ok"], -by_variant[key]["record"]["duration_s"]):
                by_variant[key] = run
        chosen = []
        if "default" in by_variant:
            chosen.append(by_variant["default"])
        others = sorted(
            (r for k, r in by_variant.items() if k != "default"),
            key=lambda r: (not r["record"]["ok"], r["record"]["duration_s"]))
        chosen.extend(others[:3])
        if not chosen:
            continue
        waypoints, _e, _t, _s = cli._load_waypoints(
            os.path.join(REPO, "examples", "%s.json" % mission))
        detail_figure(mission, chosen, waypoints,
                      os.path.join(OUT, "%s.png" % mission))

    scorecard_figure(records, os.path.join(OUT, "scorecard.png"),
                     missions=wanted_missions)

    after = fingerprint()
    changed = [k for k in before if before.get(k) != after.get(k)]
    if changed:
        print("!! THE TREE CHANGED DURING THE RUN: %s" % ", ".join(changed))
        print("   The numbers above straddle two trees and cannot be quoted "
              "as one measurement. Re-run the measurement on a settled tree.")
        return
    print("the tree did not change during the run")
    print("wrote", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
