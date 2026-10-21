"""
FLL Trajectory Optimizer CLI.

Generates time-optimal trajectories for LEGO differential-drive robots.
"""

import importlib.util
import json
import math
import os
import webbrowser
from pathlib import Path

# A first-time user who runs this before installing anything should get the
# environment report with the install command, not an ImportError traceback.
# diagnostics is stdlib only, so it works before anything is installed.
from diagnostics import missing_core_packages, run_doctor

if __name__ == "__main__" and missing_core_packages():
    raise SystemExit(run_doctor())

import click
import numpy as np

from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer, GUESS_STYLES
from validator import (
    validate_trajectory,
    validate_samples,
    check_final_waypoint,
    compute_metrics,
    compute_path_fitness,
)
from export import write_controller_file, write_python_file

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "examples/fll_choreo.json"
JAX_HINT = "Install the optional dependencies with: pip install -r requirements-optional.txt"

# Where the hub side of a push lives. The runner imports the trajectory from
# hub_mission.py next to it, and pybricksdev bundles both plus the robot
# config when it pushes.
HUB_CODE_DIR = REPO_ROOT / "pybricks_code"
HUB_MISSION_MODULE = HUB_CODE_DIR / "hub_mission.py"
HUB_RUNNER = HUB_CODE_DIR / "hub_runner.py"

# A machine-readable validation summary on one line. The browser studio reads
# it to show which part of the path failed; a human never has to look at it.
VALIDATION_MARKER = "##OVERBOARD-VALIDATION## "


def _module_available(name):
    """True when a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# The JAX stack is slow to import and a simple solve does not need it, so it is
# imported on demand. The names stay at module level so tests can patch them.
HAS_JAX = _module_available("jax") and _module_available("immrax")
HAS_MULTIVERSE = HAS_JAX and _module_available("jaxopt")
ImmraxValidator = None
JAXRobotConfig = None

# Named starting points for the flags that matter. A preset only fills in
# options the user did not set explicitly.
PRESETS = {
    "fast": {
        "samples": 8,
        "accuracy_weight": 0.0,
        "candidates": 1,
        "validate": False,
        "export_format": "none",
    },
    "balanced": {
        "samples": 10,
        "accuracy_weight": 0.0,
        "candidates": 2,
        "validate": False,
        "export_format": "none",
    },
    "safe": {
        "samples": 14,
        "accuracy_weight": 0.005,
        "candidates": 3,
        "validate": True,
        "export_format": "controller",
    },
}


def _ensure_jax_classes():
    """Import the JAX validators on demand."""
    global ImmraxValidator, JAXRobotConfig
    if ImmraxValidator is None:
        from immrax_validator import ImmraxValidator as _ImmraxValidator
        ImmraxValidator = _ImmraxValidator
    if JAXRobotConfig is None:
        from jax_robot_model import JAXRobotConfig as _JAXRobotConfig
        JAXRobotConfig = _JAXRobotConfig
    return ImmraxValidator, JAXRobotConfig


def _select_attempt(attempts, prefer_duration: bool = False):
    """Choose the candidate to export: a passing converged attempt with the
    lowest 5-part time-equivalent fitness score (when fitness is scored), or
    smallest reachability envelope then shortest duration as tie-breakers.
    When prefer_duration=True (Layer 5 Gate-Certified Selection), passing
    candidates are ranked by shortest duration first, then smallest envelope.
    Otherwise the converged attempt with the smallest quantified violation.
    Non-converged attempts are never eligible (KI-001)."""
    candidates = [a for a in attempts if a.get("converged", False)]
    if not candidates:
        return None, False
    passing = [a for a in candidates if a["ok"]]
    if passing:
        if prefer_duration:
            return min(
                passing,
                key=lambda a: (
                    a.get("duration", a["cost"]),
                    a.get("tube", float("inf")),
                ),
            ), True
        if any(
            a.get("fitness") is not None and math.isfinite(float(a["fitness"]))
            for a in passing
        ):
            return min(
                passing,
                key=lambda a: (
                    float(a.get("fitness", float("inf"))),
                    a.get("tube", float("inf")),
                    a.get("duration", a["cost"]),
                ),
            ), True
        return min(passing, key=lambda a: (a.get("tube", float("inf")),
                                           a.get("duration", a["cost"]))), True
    return min(candidates, key=lambda a: (a["violation"], a["cost"])), False


CANDIDATE_WEIGHT_STEP = 2.0  # legacy ladder for the Multi-Verse path

# Accuracy weights the sweep tries. The weight trades time against the
# wheel-jerk discipline, and which value is best depends on the path: a
# mission with long straight segments wants a low weight, an uneven path
# wants a little more. Measured on the example missions: the sweep finds
# 18 % on examples/example_mission_first4.json and a passing trajectory on
# examples/example_with_events.json, which the default candidate set cannot
# solve at all.
SWEEP_WEIGHTS = (0.0, 0.001, 0.002, 0.005, 0.01)

# Thread limits handed to the candidate worker processes. Several candidates
# solve at the same time, so each one keeps a single compute thread.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def _dt_modes(dt_mode):
    """The timing models a candidate set cycles through.

    The two modes are not peers: "uniform" shares one dt across all
    segments, which because every segment has the same sample count also
    forces every segment to take the SAME time; on a path with unequal
    segment lengths that is a large artificial penalty. "per-segment" gives
    each segment its own dt and is the general model (KI-022).
    """
    if dt_mode == "uniform":
        return [False]
    if dt_mode == "per-segment":
        return [True]
    if dt_mode == "auto":
        return [False, True]
    raise ValueError(f"Unknown dt_mode: {dt_mode}")


def _rescue_specs(specs, base_weight, use_styles, dt_mode, validated):
    """The candidates to try when the run would otherwise produce nothing.

    With validation, nothing passed the gates; without it, nothing
    converged. Either way the mission ends with no trajectory, and the
    default candidate set cannot help: its two candidates are the *same*
    seed (the baseline) on the two timing models, so the other seeds are
    never tried (KI-048).

    The rescue is the seeds the set did not use. On a validated run it uses
    the per-segment model, because that is where the measured rescues came
    from: of the eighteen mission-robot pairs the default set lost on
    random paths, the full stock set rescued two, both of them per-segment,
    and the other seeds' equal-duration candidates rescued none (the stress
    run of 2026-10-20). On an unvalidated run the model is not changed: the
    operator asked for one plan without a check, so the rescue is the same
    plan from a different starting point.

    The Multi-Verse path walks an accuracy-weight ladder rather than the
    seeds, and each of its candidates costs a whole pipeline, so it is left
    alone.
    """
    if not use_styles:
        return []
    used = {spec["guess_style"] for spec in specs}
    if validated and dt_mode != "uniform":
        modes = [True]
    else:
        modes = []
        for spec in specs:
            mode = bool(spec.get("per_segment_dt", False))
            if mode not in modes:
                modes.append(mode)
    rescue = []
    for style in GUESS_STYLES:
        if style in used:
            continue
        for mode in modes:
            rescue.append({"accuracy_weight": base_weight,
                           "guess_style": style, "per_segment_dt": mode})
    return rescue


def _candidate_specs(base_accuracy_weight, count, use_styles=True, dt_mode="auto",
                     sweep=False, refined=False):
    """Definitions for the candidate set.

    With the simple solver, each candidate is a distinct initial-guess seed at
    the base accuracy weight (KI-020: accuracy_weight is a time/smoothness
    knob, not a candidate generator) combined with a timing mode. The default
    "auto" timing mode alternates the two modes for each seed, so validation can
    choose the timing model that works on this path. "uniform" and "per-segment"
    force one mode.

    The two modes are not peers. "uniform" shares one dt across all segments,
    which because every segment has the same sample count also forces every
    segment to take the SAME time; on a path with unequal segment lengths that
    is a large, artificial penalty. "per-segment" gives each segment its own
    dt and is the general model (KI-022).

    sweep=True adds the timing-mode x accuracy-weight grid to the set. It adds
    candidates, it never replaces them, so the default candidates stay in the
    comparison and the usual rule (the passing candidate with the smallest
    envelope, the shortest one breaking the tie) still decides. A sweep costs
    more solves and more validation passes.

    refined=True adds the Multi-Verse pipeline's own refined trajectory as a
    candidate (KI-004). The pipeline stops being a second solver and becomes
    a *seed provider*: its phases 1 to 4 hand the real solver a trajectory
    that passes through every waypoint and carries a per-segment interval for
    every segment. The refined trajectory is only meaningful under the
    per-segment timing model - it is the model that reads its intervals - so
    that is the mode it is used with. Like the sweep, it adds candidates and
    never replaces them, and it needs JAX.

    The Multi-Verse path has no seed or timing plumbing yet, so it keeps the
    historical accuracy-weight ladder.
    """
    count = max(1, int(count))
    if not use_styles:
        return [{"accuracy_weight": base_accuracy_weight + CANDIDATE_WEIGHT_STEP * k,
                 "guess_style": "baseline", "per_segment_dt": False}
                for k in range(count)]

    modes = _dt_modes(dt_mode)

    specs = []
    for k in range(count):
        # auto: baseline uniform, baseline per-segment, rolling uniform, ...
        style = GUESS_STYLES[(k // len(modes)) % len(GUESS_STYLES)]
        specs.append({
            "accuracy_weight": base_accuracy_weight,
            "guess_style": style,
            "per_segment_dt": modes[k % len(modes)],
        })

    if sweep:
        # The timing-mode x accuracy-weight grid. The base weight is always
        # in the set, so a sweep can only add options. A combination the
        # default set already covers is not solved twice.
        seen = {(spec["accuracy_weight"], spec["guess_style"], spec["per_segment_dt"])
                for spec in specs}
        weights = []
        for weight in (base_accuracy_weight,) + SWEEP_WEIGHTS:
            if weight not in weights:
                weights.append(weight)
        for weight in weights:
            for per_segment_dt in modes:
                key = (weight, "baseline", per_segment_dt)
                if key in seen:
                    continue
                seen.add(key)
                specs.append({
                    "accuracy_weight": weight,
                    "guess_style": "baseline",
                    "per_segment_dt": per_segment_dt,
                })

    if refined:
        # The pipeline's refined trajectory, over the accuracy-weight grid:
        # the weight is what buys the tracking error, so the seed has to be
        # judged with it (Q6). The refined guess is a *different* starting
        # point from the stock one, so it is added even where the stock set
        # already covers that (seed, timing, weight).
        weights = []
        for weight in (base_accuracy_weight,) + SWEEP_WEIGHTS:
            if weight not in weights:
                weights.append(weight)
        for weight in weights:
            specs.append({
                "accuracy_weight": weight,
                "guess_style": "baseline",
                "per_segment_dt": True,
                "guess": "refined",
            })
    return specs


def _build_refined_guess(robot_cfg, waypoints, num_samples_per_segment, stops,
                         events, accuracy_weight):
    """The Multi-Verse pipeline's refined trajectory, as a starting point.

    Phases 1 to 4 of the pipeline: the Reeds-Shepp bootstrap, the
    kinematic profile, the critic and the segment-preserving refinement.
    The result passes through every waypoint and carries a per-segment
    interval for every segment, so it is a starting point the per-segment
    timing model can read (KI-004).

    The pipeline is imported here, not at the top of the module, so a plain
    solve never pays for JAX.
    """
    from multiverse_optimizer import MasterTrajectoryOptimizer
    from optimizer import TrajectoryOptimizer

    pipeline = MasterTrajectoryOptimizer(robot_cfg, enable_parallel=False,
                                         verbose=False)
    geometry = pipeline.bootstrapper.generate_baseline(
        list(waypoints), num_samples_per_segment)
    trajectory = pipeline._profile_kinematics(
        geometry, num_samples_per_segment, list(stops), dict(events), True)
    bad_windows = pipeline.critic.evaluate(trajectory, num_samples_per_segment)
    pipeline._refine_segments(trajectory, num_samples_per_segment, bad_windows,
                              accuracy_weight, True)
    return TrajectoryOptimizer(robot_cfg).guess_from_samples(
        trajectory, len(waypoints) - 1, num_samples_per_segment)


def _solve_candidate_worker(payload):
    """Solve one candidate. Runs in a worker process when candidates are parallel.

    Returns (samples, stats, history). A candidate that raises is reported as a
    non-converged attempt, so the export gate stays closed for it (KI-001).

    The history is the solver's iteration log. It travels with the result
    because the solver runs in *this* worker, and the caller has no other
    handle on it.
    """
    robot_cfg, wps, spec, solve_kwargs, use_multiverse = payload
    try:
        if use_multiverse:
            from multiverse_optimizer import MasterTrajectoryOptimizer
            optimizer = MasterTrajectoryOptimizer(robot_cfg, enable_parallel=False,
                                                  verbose=False)
        else:
            optimizer = TrajectoryOptimizer(robot_cfg)
        if spec.get("guess") == "refined":
            # The Multi-Verse pipeline's refined trajectory as the starting
            # point (KI-004). The guess is built here, in the worker that
            # needs it, so only that candidate pays for the refinement and
            # a plain solve never imports JAX.
            guess = _build_refined_guess(
                robot_cfg, wps,
                solve_kwargs.get("num_samples_per_segment", 10),
                solve_kwargs.get("stop_waypoint_indices") or [],
                solve_kwargs.get("waypoint_events") or {},
                spec.get("accuracy_weight", 0.0))
            solve_kwargs = dict(solve_kwargs)
            solve_kwargs["initial_guess"] = guess
            solve_kwargs["per_segment_dt"] = True
        samples, stats = optimizer.solve(wps, **solve_kwargs)
        return samples, stats, list(getattr(optimizer, "iteration_history", []))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return [], {"converged": False, "final_cost": float("inf"),
                    "total_time": 0.0, "error": str(exc)}, []


def _solve_candidates(payloads, workers, quiet, parallel, local_optimizer=None):
    """Solve every candidate and return the results in candidate order.

    Candidates are independent, so more than one can be solved in parallel
    worker processes. Validation deliberately stays in the calling process:
    the JAX validator is warm there, and every worker would pay the import
    and the compile cost again.

    Returns ``(results, histories)``. ``results`` is a list of
    ``(samples, stats)`` and ``histories`` a list of the solver's iteration
    log, one per candidate. The log is empty when the candidate was not
    asked to capture it.

    The log travels with the result because the solver may run in a worker
    process: the caller holds no optimizer of its own in that case, so
    without this the convergence plot would be empty whenever the
    candidates are solved in parallel.
    """
    if not payloads:
        return [], []
    if len(payloads) == 1 or not parallel:
        if local_optimizer is not None:
            # Solve here, with the optimizer this process already holds. The
            # convergence capture stays available.
            results = []
            histories = []
            for robot_cfg, wps, spec, solve_kwargs, use_multiverse in payloads:
                try:
                    before = len(getattr(local_optimizer, "iteration_history", []))
                    if spec.get("guess") == "refined":
                        solve_kwargs = dict(solve_kwargs)
                        solve_kwargs["initial_guess"] = _build_refined_guess(
                            robot_cfg, wps,
                            solve_kwargs.get("num_samples_per_segment", 10),
                            solve_kwargs.get("stop_waypoint_indices") or [],
                            solve_kwargs.get("waypoint_events") or {},
                            spec.get("accuracy_weight", 0.0))
                        solve_kwargs["per_segment_dt"] = True
                    samples, stats = local_optimizer.solve(wps, **solve_kwargs)
                    history = list(
                        getattr(local_optimizer, "iteration_history", []))[before:]
                    results.append((samples, stats))
                    histories.append(history)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    results.append(([], {"converged": False,
                                         "final_cost": float("inf"),
                                         "total_time": 0.0, "error": str(exc)}))
                    histories.append([])
            return results, histories
        solved = [_solve_candidate_worker(p) for p in payloads]
        return [pair[:2] for pair in solved], [pair[2] for pair in solved]

    max_workers = max(1, min(len(payloads), int(workers)))
    # Every worker is its own process. One BLAS thread per worker stops the
    # candidates from oversubscribing the machine while they solve together.
    saved_env = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    for name in _THREAD_ENV_VARS:
        os.environ[name] = "1"
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            solved = list(pool.map(_solve_candidate_worker, payloads))
    except Exception as exc:
        # A pool that cannot start must not lose the run: solve here instead.
        if not quiet:
            click.echo(f"  Parallel solving unavailable ({exc}); solving sequentially.")
        solved = [_solve_candidate_worker(p) for p in payloads]
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return [pair[:2] for pair in solved], [pair[2] for pair in solved]


def _export_block_reason(stats, validate, attempt_passed):
    """Return why the result must not be written, or None if it may be."""
    if not stats.get("converged", False):
        return "solver did not converge"
    if validate and not attempt_passed:
        return "validation failed"
    return None


def _immrax_failure_reasons(report, failed_threshold=False, max_violation=0.0):
    """Explain an Immrax failure in the units a user cares about."""
    reasons = []
    motor = float(report.get("max_motor_violation_N", 0.0))
    slip = max(float(report.get("max_slip_l_N", 0.0)),
               float(report.get("max_slip_r_N", 0.0)))
    tracking = float(report.get("max_tracking_error_m", 0.0))

    if failed_threshold:
        reasons.append(f"envelope radius {max_violation * 100:.2f} cm exceeds the local threshold")
    if motor > 1e-3:
        reasons.append(f"motor force {motor:.3f} N over limit")
    if slip > 1e-3:
        reasons.append(f"wheel slip {slip:.3f} N")
    speed = float(report.get("speed_violation_m_s", 0.0))
    if speed > 1e-3:
        reasons.append(f"wheel speed {speed:.3f} m/s over limit")
    if tracking > 0.005:
        reasons.append(f"tracking error {tracking * 100:.2f} cm > 0.50 cm")
    return reasons


def _finite(value):
    """A JSON-safe float: None instead of inf or nan."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _segment_limit(thresholds, index, default=0.02):
    """The envelope limit for one waypoint segment.

    A waypoint's limit is the limit for the way *into* it: the robot has to
    be tight where it arrives, not where it leaves. So the segment that ends
    at a waypoint is judged by that waypoint's limit, and the segment that
    starts there falls back to it when the next waypoint declares none.

    The rule matters in two places. The last waypoint's limit now governs
    the last segment, so it is never dropped: with the old rule the final
    waypoint had no segment after it and its limit was never read at all.
    And a waypoint marked as a precision one is now held to its tight limit
    on the approach, which is what the operator asked for.

    The thresholds are keyed by waypoint index; index is the segment index,
    and segment ``index`` runs from waypoint ``index`` to waypoint
    ``index + 1``.
    """
    if index + 1 in thresholds:
        return float(thresholds[index + 1])
    if index in thresholds:
        return float(thresholds[index])
    return float(default)


def _waypoint_boundaries(segment_samples):
    """The sample index of every waypoint.

    The optimizer splits the sample budget across the waypoint segments in
    proportion to their length, so the boundaries are NOT k *
    samples_per_segment any more - they are the optimizer's own
    waypoint_sample_indices table, which it reports as
    stats["segment_samples"]. The gate and the segment report both have to
    use it, or a waypoint's envelope threshold is applied to the wrong
    samples and the segments are grouped wrongly.

    An empty list means "no counts were reported", and the caller falls
    back to the uniform layout.
    """
    counts = [int(c) for c in (segment_samples or [])]
    if not counts:
        return []
    bounds = [0]
    for c in counts:
        bounds.append(bounds[-1] + c)
    return bounds


def _waypoint_of(bounds, index):
    """The waypoint whose segment the sample ``index`` belongs to."""
    if not bounds:
        return 0
    for waypoint in range(len(bounds) - 1):
        if bounds[waypoint] <= index < bounds[waypoint + 1]:
            return waypoint
    return max(0, len(bounds) - 2)


def _segment_report(samples_data, samples_per_segment, thresholds, immrax_report, cpu_summary,
                    segment_samples=None):
    """Per-segment validation detail: envelope radius, local limit, violations.

    The gate checks sample i against the threshold of the waypoint whose
    segment i is in, so a segment is judged by the threshold of the waypoint
    it starts from. The report says the same thing, per segment, instead of
    one reason string. The drift is the forward-integration error inside the
    segment: it shows where the path drifts away from the plan.
    """
    segments = []
    count = len(samples_data)
    if count == 0:
        return segments
    if isinstance(samples_per_segment, (list, tuple, np.ndarray)):
        if len(samples_per_segment) == 0:
            return segments
        if segment_samples is None:
            segment_samples = list(samples_per_segment)
        nominal_sps = max(1, int(round(float(np.mean(samples_per_segment)))))
    else:
        if samples_per_segment <= 0:
            return segments
        nominal_sps = int(samples_per_segment)

    envelopes = []
    if immrax_report:
        envelopes = immrax_report.get("reachability", {}).get("envelope", []) or []
    violating = set()
    slip = set()
    drift = [0.0] * count
    if cpu_summary:
        violating = set(cpu_summary["audit"].get("violating_sample_indices", []))
        # slip_points is a list of detail dicts; the index is inside.
        slip = {point.get("index") for point in cpu_summary["audit"].get("slip_points", [])}
        integrated = cpu_summary.get("integrated") or []
        if integrated:
            times = np.array([point["t"] for point in integrated])
            xs = np.array([point["x"] for point in integrated])
            ys = np.array([point["y"] for point in integrated])
            for i, sample in enumerate(samples_data):
                j = int(np.clip(np.searchsorted(times, sample.get("t", 0.0)),
                                0, len(times) - 1))
                drift[i] = math.hypot(xs[j] - sample.get("x", xs[j]),
                                       ys[j] - sample.get("y", ys[j]))

    bounds = _waypoint_boundaries(segment_samples)
    if len(bounds) > 1:
        # The optimizer's own layout: the boundaries are the waypoints.
        n_segments = len(bounds) - 1
        starts = list(bounds[:-1])
        stops = [min(stop, count) for stop in bounds[1:]]
    else:
        n_segments = max(1, (count - 1) // nominal_sps)
        starts = [index * nominal_sps for index in range(n_segments)]
        stops = [min(start + nominal_sps, count) for start in starts]
    for index in range(n_segments):
        start = starts[index]
        stop = stops[index]
        limit = _segment_limit(thresholds, index)
        radius = 0.0
        for i in range(start, stop):
            env = envelopes[i] if i < len(envelopes) else None
            if not env:
                continue
            radius = max(radius, math.hypot((env["x_max"] - env["x_min"]) / 2.0,
                                             (env["y_max"] - env["y_min"]) / 2.0))
        violations = sum(1 for i in range(start, stop) if i in violating)
        slip_points = sum(1 for i in range(start, stop) if i in slip)
        segments.append({
            "from": index,
            "to": index + 1,
            "limit_m": limit,
            "envelope_m": radius,
            "drift_m": max(drift[start:stop]) if stop > start else 0.0,
            "violations": violations,
            "slip_points": slip_points,
            "ok": radius <= limit and violations == 0 and slip_points == 0,
        })
    return segments


def _emit_validation_report(attempts, selected_index, propagation=None):
    """Print the machine-readable validation summary.

    Every candidate carries its score vector: the duration, the reachability
    envelope, the tracking error, the peak wheel jerk, and the biggest
    excess of the envelope over the local threshold. The selection rule can
    therefore be re-run offline on the numbers, without solving anything
    again (Q3's step 1, Q10's step 1).

    Every candidate also carries the **model** that produced its numbers: the
    propagation ("plant" or "tick") and the inclusion functions it used. The
    envelope is the width of a reachability tube, and the tube is a property
    of the propagation, so a number whose model is not stated cannot be
    compared with a number from another run (KI-049).
    """
    candidates = []
    for number, attempt in enumerate(attempts, 1):
        segments = attempt.get("segments", []) or []
        excess = max((float(segment.get("envelope_m", 0.0))
                      - float(segment.get("limit_m", 0.0))
                      for segment in segments), default=0.0)
        candidates.append({
            "index": number,
            "style": attempt.get("guess_style"),
            "guess": attempt.get("guess", attempt.get("guess_style")),
            "timing": "per-segment" if attempt.get("per_segment_dt") else "equal-duration",
            "accuracy_weight": _finite(attempt.get("accuracy_weight")),
            "duration_s": _finite(attempt.get("duration")),
            "envelope_cm": _finite(attempt.get("tube", 0.0) * 100.0),
            # True when the propagation was cut off, so envelope_cm is a
            # lower bound and a property of the 50 mm guard (KI-049).
            "envelope_truncated": bool(attempt.get("envelope_truncated")),
            # The model the numbers came from, so no envelope is ever read
            # without it.
            "propagation": attempt.get("propagation", propagation),
            "inclusion": attempt.get("inclusion", {}),
            "tracking_cm": _finite(attempt.get("tracking_m", 0.0) * 100.0),
            "max_excess_cm": _finite(excess * 100.0),
            "peak_jerk_m_s3": _finite(attempt.get("peak_jerk_m_s3")),
            "heading_error_deg": _finite(
                (attempt.get("heading_rad", 0.0) or 0.0) * 180.0 / math.pi),
            "fitness_s": _finite(attempt.get("fitness")),
            "stop_spread_cm": _finite(
                (attempt.get("stop_spread_m", 0.0) or 0.0) * 100.0),
            "slip_risk": _finite(attempt.get("slip_risk", 0.0)),
            "rms_force_n": _finite(attempt.get("rms_force_n", 0.0)),
            "rms_force_util": _finite(attempt.get("rms_force_util", 0.0)),
            "ok": bool(attempt.get("ok")),
            "converged": bool(attempt.get("converged")),
            "reasons": list(attempt.get("reasons", [])),
            "segments": segments,
        })
    payload = {"selected": selected_index, "candidates": candidates,
               "propagation": propagation}
    click.echo(VALIDATION_MARKER + json.dumps(payload))


def _cpu_failure_reasons(summary, waypoint_check=None):
    """Explain a CPU-validator failure."""
    audit = summary["audit"]
    errors = summary["errors"]
    reasons = []
    if audit["num_violating_samples"]:
        reasons.append(f"{audit['num_violating_samples']} constraint violations")
    if audit["num_slip_points"]:
        reasons.append(f"{audit['num_slip_points']} wheel-slip points")
    speed = float(audit.get("speed_violation_m_s", 0.0))
    if speed > 1e-6:
        reasons.append(f"wheel speed {speed:.3f} m/s over limit")
    if errors["max_pos_error_m"] >= 0.01:
        reasons.append(f"forward-integration error {errors['max_pos_error_m'] * 100:.2f} cm")
    if errors["final_pos_error_m"] >= 0.01:
        reasons.append(f"final integration error {errors['final_pos_error_m'] * 100:.2f} cm")
    if waypoint_check is not None and not waypoint_check["passed"]:
        reasons.append(f"final waypoint error {waypoint_check['pos_error_m'] * 100:.2f} cm")
        if waypoint_check["heading_error_rad"] > 0.02:
            reasons.append(f"final heading error {waypoint_check['heading_error_rad']:.3f} rad")
    return reasons


def _apply_preset(ctx, preset, values):
    """Fill in preset values for options the user did not set explicitly."""
    for name, value in PRESETS[preset].items():
        if name not in values:
            continue
        source = ctx.get_parameter_source(name)
        if source is not None and source.name in ("DEFAULT", "DEFAULT_MAP"):
            values[name] = value
    return values


def _studio_options(samples, accuracy_weight, candidates, dt_mode, validate,
                    export_format, controller_dt, fast_polish, sysid_fit=None,
                    hub_conn=None, hub_name=None, self_consistent=True):
    """The options the studio starts with, so --live reproduces this run."""
    return {
        "samples": samples,
        "accuracy_weight": accuracy_weight,
        "candidates": candidates,
        "timing": dt_mode,
        "validate": validate,
        "export_format": export_format,
        "controller_dt": controller_dt,
        "fast_polish": fast_polish,
        "sysid_fit": sysid_fit,
        "hub_conn": hub_conn,
        "hub_name": hub_name,
        "self_consistent": self_consistent,
    }


def _load_config(path):
    """Load a robot config and report problems in plain language."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise click.ClickException(
            f"Robot config '{path}' not found. Pass -c/--config with your own robot file."
        )
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Robot config '{path}' is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}."
        )
    try:
        return data, RobotConfig(data)
    except Exception as exc:
        raise click.ClickException(f"Invalid robot config '{path}': {exc}")


def _number(value, label, index, path):
    """Read a finite number, or explain which waypoint field is wrong."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise click.ClickException(
            f"Waypoint {index} in '{path}': '{label}' must be a number, got {value!r}."
        )
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise click.ClickException(
            f"Waypoint {index} in '{path}': '{label}' must be a finite number."
        )
    return value


def _load_waypoints(path):
    """Load a waypoint file into (waypoints, events, thresholds, stop_indices)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise click.ClickException(f"Waypoint file '{path}' not found.")
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Waypoint file '{path}' is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}."
        )

    if not isinstance(data, list):
        raise click.ClickException(
            f"Waypoint file '{path}' must contain a list of waypoints."
        )
    if len(data) < 2:
        raise click.ClickException(
            f"Waypoint file '{path}' needs at least two waypoints."
        )

    wps = []
    events = {}
    thresholds = {}
    stop_indices = []
    for index, item in enumerate(data):
        if isinstance(item, dict):
            if "x" not in item or "y" not in item:
                raise click.ClickException(
                    f"Waypoint {index} in '{path}' needs both 'x' and 'y'."
                )
            x = _number(item["x"], "x", index, path)
            y = _number(item["y"], "y", index, path)
            heading = item.get("heading")
            if heading is not None:
                heading = _number(heading, "heading", index, path)
            wps.append((x, y, heading))
            if "event" in item:
                name = item["event"]
                if not isinstance(name, str) or not name.strip():
                    raise click.ClickException(
                        f"Waypoint {index} in '{path}': 'event' must be a non-empty name."
                    )
                events[index] = name.strip()
            if item.get("stop"):
                stop_indices.append(index)
            if "error_threshold" in item:
                threshold = _number(item["error_threshold"], "error_threshold", index, path)
                if threshold <= 0:
                    raise click.ClickException(
                        f"Waypoint {index} in '{path}': 'error_threshold' must be positive."
                    )
                thresholds[index] = threshold
            elif item.get("precision_mode"):
                thresholds[index] = 0.005  # 0.5 cm
        elif isinstance(item, (list, tuple)):
            if len(item) < 2:
                raise click.ClickException(
                    f"Waypoint {index} in '{path}' needs at least an x and a y value."
                )
            x = _number(item[0], "x", index, path)
            y = _number(item[1], "y", index, path)
            heading = item[2] if len(item) > 2 else None
            if heading is not None:
                heading = _number(heading, "heading", index, path)
            wps.append((x, y, heading))
        else:
            raise click.ClickException(
                f"Waypoint {index} in '{path}': expected an object or a [x, y] list, "
                f"got {type(item).__name__}."
            )
    return wps, events, thresholds, stop_indices


def _parse_stop_waypoints(raw, count):
    """Parse --stop-waypoints and reject out-of-range indices."""
    if not raw:
        return []
    try:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise click.BadParameter(
            "expected comma-separated waypoint indices, for example '2,5,7'",
            param_hint="'--stop-waypoints'",
        )
    for index in indices:
        if not 0 <= index < count:
            raise click.BadParameter(
                f"waypoint index {index} is out of range (0..{count - 1})",
                param_hint="'--stop-waypoints'",
            )
    return indices


def _parse_events(raw, count):
    """Parse --events and reject malformed pairs or out-of-range indices."""
    if not raw:
        return {}
    events = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise click.BadParameter(
                "expected 'index:event' pairs, for example '2:lower_arm,5:release'",
                param_hint="'--events'",
            )
        index_text, name = pair.split(":", 1)
        try:
            index = int(index_text.strip())
        except ValueError:
            raise click.BadParameter(
                f"'{index_text.strip()}' is not a waypoint index",
                param_hint="'--events'",
            )
        if not 0 <= index < count:
            raise click.BadParameter(
                f"waypoint index {index} is out of range (0..{count - 1})",
                param_hint="'--events'",
            )
        name = name.strip()
        if not name:
            raise click.BadParameter(
                f"the event name for waypoint {index} is empty",
                param_hint="'--events'",
            )
        events[index] = name
    return events


def _push_to_hub(traj_file, connection, name, log_path, quiet):
    """Push the written trajectory to the hub and run it there.

    The hub entry point is pybricks_code/hub_runner.py; it imports the
    trajectory from pybricks_code/hub_mission.py, so both files travel with
    the push. Returns the lines the hub printed.
    """
    from hub_deploy import HUB_RUNNER, push_trajectory, write_session

    if not quiet:
        click.echo(f"Pushing {HUB_RUNNER.name} to the hub over {connection}...")
    try:
        lines = push_trajectory(traj_file, connection=connection, name=name,
                                log_path=log_path,
                                mission_module=HUB_MISSION_MODULE,
                                runner=HUB_RUNNER)
    except RuntimeError as exc:
        # The log is streamed to disk as the run arrives, so what the hub
        # printed before the link dropped is still there. The other two
        # entry points (hub_deploy.py --out and sysid_collect.py) keep the
        # session the same way; this one used to lose all of it.
        message = str(exc)
        if log_path and os.path.exists(log_path):
            message += f" (the hub output that arrived is in {log_path})"
        raise click.ClickException(message)
    if log_path:
        # The streamed file holds the same lines in the hub's own line
        # endings. Rewrite it in the text form the other entry points
        # write, so one reader serves all of them.
        write_session(lines, log_path)
        if not quiet:
            click.echo(f"  wrote {len(lines)} lines of hub output to {log_path}")
    return lines


def _print_summary(output, samples_data, robot_cfg, validate, attempt_passed, tube, files):
    """Print what was produced, in the units the user cares about."""
    metrics = compute_metrics(samples_data, robot_cfg)
    click.echo("")
    click.secho(f"Trajectory written to {output}", fg="green", bold=True)
    click.echo(
        f"  duration {metrics['total_time_s']:.3f} s | "
        f"path {metrics['path_length_m']:.3f} m | "
        f"peak wheel speed {metrics['max_wheel_speed_m_s']:.2f} m/s | "
        f"{len(samples_data)} samples"
    )
    if validate:
        if attempt_passed:
            detail = f"envelope {tube * 100:.2f} cm" if tube == tube else "envelope n/a"
            click.secho(f"  validation: passed ({detail})", fg="green")
        else:
            click.secho("  validation: failed (nothing was written)", fg="red")
    else:
        click.secho(
            "  validation: not run - add --validate before driving this robot",
            fg="yellow",
        )
    for path in files:
        click.echo(f"  also wrote {path}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
# --- Input / Output Options ---
@click.option('-c', '--config', default=DEFAULT_CONFIG, type=click.Path(dir_okay=False),
              help='Path to the robot configuration file (.json).')
@click.option('-w', '--waypoints', default=None, type=click.Path(dir_okay=False),
              help='Path to waypoints JSON file. Required unless --gui or --doctor.')
@click.option('-o', '--output', default='output.traj', type=str,
              help='Output trajectory file path.')

# --- Modes ---
@click.option('--gui', is_flag=True,
              help='Open the browser studio: draw a mission, solve it, export it.')
@click.option('--gui-port', default=8000, type=int,
              help='Port for the browser studio.')
@click.option('--doctor', is_flag=True,
              help='Check the install, dependencies, examples and ports, then exit.')
@click.option('--multiverse', is_flag=True,
              help='(Deprecated: superseded by --unified) Use legacy Multi-Verse pipeline in legacy/.')
@click.option('--simple', is_flag=True,
              help='(Deprecated: CasADi is already standard) Standard CasADi optimizer.')
@click.option('--preset', type=click.Choice(['fast', 'balanced', 'safe'], case_sensitive=False),
              default=None,
              help='Named starting point. fast: 1 candidate, no validation. '
                   'balanced: the defaults. safe: 3 candidates, --validate and a controller export. '
                   'Options you set yourself always win.')

# --- Solver Parameters ---
@click.option('-n', '--samples', default=10, type=click.IntRange(2, 200),
              help='Samples per segment.')
@click.option('-a', '--accuracy-weight', default=0.0, type=click.FloatRange(min=0.0),
              help='Smoothness/accuracy weight (0 = pure time-optimal).')
@click.option('--force-weight', default=0.0, type=click.FloatRange(min=0.0),
              help='Soft wheel-force and traction-utilization penalty weight in the NLP '
                   'objective (simple solver only).')
@click.option('--candidates', default=2, type=click.IntRange(min=1),
              help='Candidates to solve and compare. With the simple solver these combine initial-guess '
                   'seeds (baseline, rolling, bow_left, bow_right, turn_first) with the timing '
                   'mode: by default candidate 1 is equal-duration baseline, candidate 2 is '
                   'per-segment baseline, then the remaining seeds alternate. The Multi-Verse path uses '
                   'accuracy_weight steps of 2.0 instead. Validation picks the passing candidate '
                   'with the smallest envelope, then the fastest. Without --validate only the first '
                   'candidate is solved. Use 1 for a single solve.')
@click.option('--stop-waypoints', default=None, type=str,
              help='Comma-separated waypoint indices where robot must stop (e.g., "2,5,7").')
@click.option('--events', default=None, type=str,
              help='Comma-separated waypoint:event pairs (e.g., "2:lower_arm,5:release").')

# --- Optimizer Strategy ---
@click.option('--per-segment-dt', is_flag=True,
              help='Force each waypoint segment to have its own sample interval (simple solver only). '
                   'This is the general timing model. By default candidates mix both modes.')
@click.option('--uniform-dt', is_flag=True,
              help='(Deprecated: equal-duration dt severely penalizes variable-length segments) '
                   'Force one shared sample interval across all segments.')
@click.option('--no-parallel', is_flag=True,
              help='Solve the candidates in this process instead of in worker processes, and '
                   'disable the parallel Multi-Verse refinement (no effect on the result).')
@click.option('--workers', default=8, type=int,
              help='Number of worker processes used to solve candidates in parallel.')
@click.option('--fast-polish', is_flag=True,
              help='Solve large missions faster. The warm-start solve can crawl through thousands of '
                   'IPOPT iterations; this gives it an iteration budget and finishes from that point '
                   'with a cold pass. Missions that solve inside the budget return the same '
                   'trajectory as before.')
@click.option('--sweep', is_flag=True,
              help='(Deprecated: 80 ms invariant mesh eliminates manual grid sweep) '
                   'Search the timing model and accuracy weight grid.')
@click.option('--self-consistent/--no-self-consistent', default=True,
              help='Plan a self-consistent trajectory: the samples are placed on the path the '
                   'trajectory\'s own (v, omega) profile traces, so the headings are its tangents '
                   '(simple solver only). On by default (KI-026).')
@click.option('--refine', is_flag=True,
              help='(Deprecated: relied on legacy multiverse pipeline) '
                   'Add the Multi-Verse pipeline\'s refined trajectory as a candidate.')
@click.option('--unified/--no-unified', default=True,
              help='Run the 3-Pillar Unified Trajectory Engine (default: on): '
                   'Kinetic Frontier Seeds + 80 ms Invariant Mesh + '
                   'Discretization-Consistent Collocation.')
@click.option('--surrogate-polish', is_flag=True,
              help='Run the 5-Layer Unified Surrogate-to-Truth parallel pipeline: '
                   'CasADi Discretization-Aware Metric + Linear Epigraph Force Bounds + '
                   'Single-Process ThreadPoolExecutor + Compiled float64 jax.lax.scan '
                   'Reachability + Differentiable Immrax Polish + Gate-Certified Min-Time Selection.')
@click.option('--pyramid', is_flag=True, default=False,
              help='Multi-Fidelity "Screen-then-Refine" Pyramid: vectorized JAX sieve '
                   'over 100-350 hypotheses, topological clustering into survivor seeds, '
                   'and process-isolated parallel IPOPT solves.')
@click.option('--pyramid-candidates', default=None, type=click.IntRange(min=1),
              help='Number of top distinct homotopy finalists to solve in parallel during '
                   '--pyramid mode (defaults to 10).')
@click.option('--recombine-segments', is_flag=True, default=False,
              help='Advanced Optimization: Segment-wise recombination (chimera splicing) '
                   'that extracts the best-performing segments from different candidates, '
                   'splices them at waypoint knots, and rapidly warm-polishes the composite.')
@click.option('--memetic-evolution', is_flag=True, default=False,
              help='Memetic Evolution: population-based genetic algorithm with segment-wise '
                   'chimera crossover, creep mutation, and rapid warm IPOPT local search.')
@click.option('--generations', default=3, type=click.IntRange(min=1),
              help='Number of evolution generations for --memetic-evolution (defaults to 3).')
@click.option('--population-size', default=10, type=click.IntRange(min=4),
              help='Population size for --memetic-evolution (defaults to 10).')
@click.option('--h-adaptive', is_flag=True, default=False,
              help='Adaptive mesh refinement (h-refinement): selectively allocates higher node density '
                   '(8-10 nodes) to curved apex segments, lowering forward-integration discretization '
                   'error to sub-centimeter levels (<1.0 cm).')

# --- Export & Validation ---
@click.option('--validate/--no-validate', default=False,
              help='Run validation and block export on failure (requires JAX/Immrax).')
@click.option('--sysid-fit', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Optional sysid_fit.json digital-twin fit (requires --validate or --live).')
@click.option('--export-format', type=click.Choice(['none', 'controller', 'python', 'python-compact'], case_sensitive=False),
              default='none', help='Export format for controller consumption (python-compact is the packed Pybricks format).')
@click.option('--controller-dt', default=0.02, type=click.FloatRange(min=0.001, max=1.0),
              help='Fixed timestep for controller export (seconds).')
@click.option('--control-dt', default=0.02, type=click.FloatRange(min=0.001, max=0.2),
              help='The control period the validator models, in seconds. The deployed '
                   'controllers hold their command for 20 ms, but a robot that cannot '
                   'sustain it runs slower; set this to the period the robot actually '
                   'achieves (the pushed session reports it) so the gate is measured '
                   'against the controller that will run.')
@click.option('--propagation', type=click.Choice(['plant', 'tick'], case_sensitive=False),
              default='plant',
              help='Which reachability propagation the envelope gate uses. "plant" is '
                   'the default: the plant is embedded and propagated, and the '
                   'controller is evaluated outside the propagation, so the tube bounds '
                   'an open-loop plant driven by a pre-recorded command. "tick" embeds '
                   'one control tick of the closed loop - controller, pose estimate and '
                   'delayed command inside the map - and propagates that, so the '
                   'feedback contracts the tube the way the robot\'s does. It needs a '
                   'Jacobian inclusion and costs 7-10x the validation time. The '
                   'propagation in force is reported with every candidate.')
@click.option('--benchmark', is_flag=True,
              help='Collect comprehensive benchmarking data for whitepaper.')

# --- Hub Push ---
@click.option('--push-hub', is_flag=True,
              help='Push the trajectory to a Pybricks hub and run it there '
                   '(needs pybricksdev and a hub with Pybricks firmware).')
@click.option('--hub-conn', type=click.Choice(['ble', 'usb'], case_sensitive=False),
              default='ble', help='How to reach the hub when pushing.')
@click.option('--hub-name', default=None, type=str,
              help='Bluetooth name or address of the hub to push to.')
@click.option('--hub-log', default=None, type=click.Path(dir_okay=False),
              help='Write the hub console output to this file. The runner '
                   'reports the measured pose as a POSE line in it.')

# --- Visualization ---
@click.option('--plot', is_flag=True, help='Plot the resulting trajectory.')
@click.option('--animate', is_flag=True, help='Animate the trajectory in real-time.')
@click.option('--live', is_flag=True,
              help='Open the browser studio on this mission and solve it there (blocking).')
@click.option('--show-convergence', is_flag=True, help='Show convergence visualization.')
@click.option('--iteration-history', type=click.Path(dir_okay=False), default=None,
              help='Write the solver\'s iteration history to this JSON file, for the '
                   'studio to plot how the trajectory converged.')
@click.option('--convergence-mode', type=click.Choice(['parallel', 'best', 'layered'], case_sensitive=False),
              default='best', help='Convergence visualization mode.')

# --- Miscellaneous ---
@click.option('-v', '--verbose', is_flag=True, help='Show solver internals.')
@click.option('--quiet', '-q', is_flag=True, help='Suppress verbose output.')
@click.pass_context
def main(ctx, config, waypoints, output, gui, gui_port, doctor, multiverse, simple, preset,
         samples, accuracy_weight, force_weight, candidates, stop_waypoints, events, per_segment_dt, uniform_dt,
         no_parallel, workers, fast_polish, sweep, self_consistent, refine, unified, surrogate_polish, pyramid, pyramid_candidates, recombine_segments, memetic_evolution, generations, population_size, h_adaptive, validate,
         sysid_fit, export_format, controller_dt, control_dt, propagation, benchmark,
         push_hub, hub_conn, hub_name, hub_log,
         plot, animate, live, show_convergence, convergence_mode, iteration_history,
         verbose, quiet):
    """
    FLL Trajectory Optimizer CLI.
    Generates time-optimal trajectories for Lego robots.

    \b
    Examples:
      python main.py -w examples/example_straight.json
      python main.py -w mission.json --preset safe -o mission.traj
      python main.py --gui
      python main.py --doctor
    """
    if quiet and verbose:
        raise click.ClickException("--quiet and --verbose cannot be combined.")
    if simple and multiverse:
        raise click.ClickException(
            "--simple is the default; use --multiverse for the experimental pipeline (not both)."
        )
    if simple and not quiet:
        click.secho(
            "Note: --simple is deprecated; CasADi is already standard.",
            fg="yellow",
        )
    if multiverse and not quiet:
        click.secho(
            "Warning: --multiverse is deprecated and superseded by --unified (legacy pipeline in legacy/).",
            fg="yellow",
        )
    if refine and not quiet:
        click.secho(
            "Warning: --refine is deprecated (relied on legacy multiverse pipeline).",
            fg="yellow",
        )
    if sweep and not quiet:
        click.secho(
            "Warning: --sweep is deprecated (80 ms invariant mesh eliminates manual grid sweep).",
            fg="yellow",
        )
    if uniform_dt and not quiet:
        click.secho(
            "Warning: --uniform-dt is deprecated (equal-duration dt penalizes variable-length segments; per-segment dt is standard).",
            fg="yellow",
        )
    if ctx.get_parameter_source('unified') == click.core.ParameterSource.DEFAULT:
        if simple or uniform_dt or sweep or refine or multiverse:
            unified = False
    if per_segment_dt and uniform_dt:
        raise click.ClickException(
            "--per-segment-dt and --uniform-dt are mutually exclusive."
        )
    if (per_segment_dt or uniform_dt) and multiverse:
        raise click.ClickException(
            "--per-segment-dt/--uniform-dt are only supported by the simple solver."
        )
    if sweep and multiverse:
        raise click.ClickException(
            "--sweep is only supported by the simple solver."
        )
    if multiverse and self_consistent:
        if ctx.get_parameter_source("self_consistent") == click.core.ParameterSource.COMMANDLINE:
            raise click.ClickException(
                "--self-consistent is only supported by the simple solver."
            )
        self_consistent = False
    if refine and multiverse:
        raise click.ClickException(
            "--refine is only supported by the simple solver."
        )
    if force_weight > 0.0 and multiverse:
        raise click.ClickException(
            "--force-weight is only supported by the simple solver."
        )

    if doctor:
        from diagnostics import run_doctor
        raise SystemExit(run_doctor(config_path=config))

    if waypoints is None and not (gui or live):
        raise click.UsageError(
            "Missing option '-w' / '--waypoints'. "
            "Example: python main.py -w examples/example_straight.json"
        )

    values = {"samples": samples, "accuracy_weight": accuracy_weight,
              "candidates": candidates, "validate": validate,
              "export_format": export_format}
    if preset:
        values = _apply_preset(ctx, preset, values)
        samples = values["samples"]
        accuracy_weight = values["accuracy_weight"]
        candidates = values["candidates"]
        validate = values["validate"]
        export_format = values["export_format"]

    if validate and not HAS_JAX:
        raise click.ClickException(
            f"--validate needs JAX and Immrax. {JAX_HINT}"
        )
    if multiverse and not HAS_MULTIVERSE:
        raise click.ClickException(
            f"--multiverse needs JAX, JAXOpt and Immrax. {JAX_HINT}"
        )

    dt_mode = "per-segment" if per_segment_dt else "uniform" if uniform_dt else "auto"

    if gui or live:
        if multiverse:
            raise click.ClickException(
                "--multiverse is not available in the studio yet; use the command line."
            )
        if gui and live:
            raise click.ClickException(
                "--gui and --live both open the studio; use one of them."
            )
        if live and waypoints is None:
            raise click.UsageError(
                "Missing option '-w' / '--waypoints'. "
                "Example: python main.py --live -w examples/example_straight.json"
            )
        if plot or animate:
            raise click.ClickException(
                "--gui and --live cannot be combined with --plot or --animate."
            )
        from gui_server import run_studio
        run_studio(config_path=config, waypoints_path=waypoints,
                   port=gui_port, presets=PRESETS, auto_solve=live,
                   options=_studio_options(samples, accuracy_weight, candidates,
                                           dt_mode, validate, export_format,
                                           controller_dt, fast_polish, sysid_fit,
                                           hub_conn, hub_name, self_consistent))
        return

    config_data, robot_cfg = _load_config(config)
    if not quiet and os.path.abspath(config) == os.path.abspath(DEFAULT_CONFIG):
        click.secho(
            f"Using the example robot config ({DEFAULT_CONFIG}). "
            "Pass -c/--config with your own robot to plan for your build.",
            fg="yellow",
        )

    digital_twin = None
    if sysid_fit:
        if not (validate or live):
            raise click.ClickException(
                "--sysid-fit requires --validate or --live so the fitted model is used."
            )
        if not HAS_JAX:
            raise click.ClickException(
                f"--sysid-fit needs JAX and Immrax. {JAX_HINT}"
            )
        from digital_twin import load_sysid_fit
        try:
            digital_twin = load_sysid_fit(sysid_fit)
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                f"Could not load digital-twin fit '{sysid_fit}': {exc}"
            ) from exc
        if not quiet:
            if digital_twin.enabled:
                gain_lo, gain_hi = digital_twin.actuator_gain_range()
                click.echo(
                    "Digital twin: gain {:.3f}-{:.3f}, tau {:.3f}s, "
                    "delay {:.3f}s, odometry {:.4f}/{:.4f}".format(
                        gain_lo, gain_hi,
                        digital_twin.actuator_tau_s(),
                        digital_twin.actuator_delay_s(),
                        digital_twin.odometry.distance_scale,
                        digital_twin.odometry.heading_scale,
                    )
                )
                if digital_twin.slip.calibrated:
                    click.echo(
                        "  Brake slip is loaded but not applied yet "
                        "(no braking state in the plant)."
                    )
                drivebase = digital_twin.drivebase
                if drivebase:
                    label = drivebase.get("name") or "unnamed"
                    parts = []
                    radius = drivebase.get("wheel_radius_m")
                    track = drivebase.get("track_width_m")
                    if radius is not None:
                        parts.append(f"wheel radius {float(radius) * 1000.0:.2f} mm")
                    if track is not None:
                        parts.append(f"track width {float(track) * 1000.0:.2f} mm")
                    suffix = f" ({', '.join(parts)})" if parts else ""
                    click.echo(f"  Drivebase profile: {label}{suffix}")
                    checks = (
                        ("wheel radius", radius, robot_cfg.wheel_radius),
                        ("track width", track, robot_cfg.track_width),
                    )
                    for name, fitted, configured in checks:
                        if fitted is None:
                            continue
                        fitted = float(fitted)
                        configured = float(configured)
                        if abs(fitted - configured) > max(
                                1e-9, 1e-3 * abs(fitted)):
                            click.echo(
                                f"  WARNING: fit {name} {fitted:g} differs from "
                                f"config {configured:g}; the trajectory may be "
                                "planned for a different drivebase."
                            )
            else:
                click.echo(
                    "Digital-twin fit has no calibrated parameters; "
                    "using the nominal model."
                )

        # A measured digital twin replaces the declared numbers in the config,
        # so that the optimizer, the CPU audit, the Immrax validator and the
        # exported controller all plan against the same calibrated robot.
        changes = robot_cfg.apply_sysid_fit(digital_twin)
        if changes:
            if not quiet:
                for name, change in changes.items():
                    prefix = "Measured motor" if name in ("t_max_nm", "v_max_rad_s", "t_max_range") else "Measured fit"
                    if isinstance(change, dict) and "declared" in change and "measured" in change:
                        click.echo(
                            "  {p}: {n} {d:.4f} -> {m:.4f}".format(
                                p=prefix, n=name, d=change["declared"], m=change["measured"])
                        )
                    else:
                        click.echo(f"  {prefix}: {name} {change}")
        elif not quiet:
            click.echo(
                "  The fit has no measured parameters; the config's "
                "declared values are used."
            )

    # A motor written from memory is a guess. The published SPIKE Prime
    # motors say whether the numbers the plan will use are in the
    # neighbourhood of a real LEGO motor.
    #
    # The check is made here, after the fit has replaced the declared
    # numbers, so it looks at the motor the plan is really made against.
    # A robot that was measured does not need this, and a warning about its
    # old guess would only train the operator to ignore the warning.
    if not quiet:
        from drivetrain import compare_declared_motor
        motor_check = compare_declared_motor(
            robot_cfg.t_max_nm, robot_cfg.v_max_rad_s,
            robot_cfg.wheel_radius, robot_cfg.gearing)
        if motor_check.get("suspicious"):
            click.secho(f"  Motor: {motor_check['note']}.", fg="yellow")

    # Choose optimizer
    if multiverse:
        if not quiet:
            click.echo(f"Using the Multi-Verse optimizer (parallel={not no_parallel}, workers={workers})")
        from multiverse_optimizer import MasterTrajectoryOptimizer
        try:
            optimizer = MasterTrajectoryOptimizer(
                robot_cfg, enable_parallel=not no_parallel,
                num_workers=workers, verbose=not quiet)
        except ImportError as exc:
            raise click.ClickException(f"{exc} {JAX_HINT}")
    else:
        if not quiet:
            click.echo("Using the simple CasADi optimizer")
        optimizer = TrajectoryOptimizer(robot_cfg)

    wps, waypoint_events, waypoint_thresholds, json_stop_indices = _load_waypoints(waypoints)
    stop_indices = _parse_stop_waypoints(stop_waypoints, len(wps))
    waypoint_events.update(_parse_events(events, len(wps)))

    # sorted() so the constraint order (and therefore the IPOPT iterate) is
    # reproducible; set() iteration order changes the NLP ordering.
    all_stop_indices = sorted(set(stop_indices + json_stop_indices))

    if not quiet:
        # The two modes are not peers: uniform dt forces every waypoint segment
        # to take the SAME time, so a path with unequal segment lengths pays
        # for the long one on every segment. Per-segment dt lets each segment
        # have its own duration and is the general model (KI-022).
        dt_label = {
            "auto": "mixed equal-duration/per-segment dt",
            "uniform": "equal-duration segments (legacy)",
            "per-segment": "per-segment dt",
        }[dt_mode]
        click.echo(
            f"Solving {len(wps)} waypoints (samples={samples}, "
            f"accuracy_weight={accuracy_weight}, candidates={max(1, candidates)}, "
            f"{dt_label})..."
        )

    capture_iterations = show_convergence or bool(iteration_history)

    # --- CANDIDATE LOOP (validate and select; KI-020) ---
    # Solve several candidates, score each with the same Immrax check, and let
    # _select_attempt pick the winner: passing first, then the smallest envelope,
    # then the fastest. With the simple solver the candidates are different
    # initial-guess seeds (different local optima), not different accuracy
    # weights: the old accuracy_weight ladder silently traded speed for a bound
    # that could get worse (longer trajectories accumulate more uncertainty).
    candidate_specs = _candidate_specs(accuracy_weight, candidates,
                                       use_styles=not multiverse, dt_mode=dt_mode,
                                       sweep=sweep, refined=refine)
    attempt_passed = False
    best_reasons = []
    all_attempts = []  # Each candidate: samples, stats, cost, ok, violation, tube, converged, reasons

    # --- SOLVE PHASE ---
    # The candidates are independent, so they are solved first and scored
    # afterwards. Without validation only the first candidate is solved.
    use_validation = HAS_JAX and validate
    specs_to_solve = candidate_specs if use_validation else candidate_specs[:1]

    if unified and not multiverse:
        from unified_planner import UnifiedPathEngine
        engine = UnifiedPathEngine(robot_cfg)
        mesh_samples = engine.compute_invariant_mesh_samples(wps)
        if ctx.get_parameter_source('samples') == click.core.ParameterSource.DEFAULT and preset is None:
            samples = mesh_samples
        specs_to_solve = [
            {"accuracy_weight": 0.001, "guess_style": "frontier_sprint", "per_segment_dt": True, "cost_metric": "discretization", "epigraph_constraints": True},
            {"accuracy_weight": 0.002, "guess_style": "frontier_sweeper", "per_segment_dt": True, "cost_metric": "discretization", "epigraph_constraints": True},
            {"accuracy_weight": 0.001, "guess_style": "frontier_exploratory", "per_segment_dt": True, "cost_metric": "discretization", "epigraph_constraints": True},
        ]
        if ctx.get_parameter_source('accuracy_weight') != click.core.ParameterSource.DEFAULT:
            for s in specs_to_solve:
                s['accuracy_weight'] = float(accuracy_weight)
        if candidates is not None and candidates < len(specs_to_solve):
            specs_to_solve = specs_to_solve[:max(1, int(candidates))]
        if not use_validation:
            specs_to_solve = specs_to_solve[:1]

    if (pyramid or memetic_evolution) and not multiverse:
        if not HAS_JAX:
            raise click.ClickException(f"--pyramid/--memetic-evolution requires JAX. {JAX_HINT}")
        from surrogate_polish import (
            generate_homotopy_bank,
            screen_candidates_jax,
            select_topological_representatives,
        )
        from reachability_engine import CompiledReachabilityEngine
        validator_cls, robot_config_cls = _ensure_jax_classes()
        jax_cfg = robot_config_cls(robot_cfg)
        immrax_val = validator_cls(jax_cfg, control_dt=control_dt, propagation=propagation)
        engine = CompiledReachabilityEngine(immrax_val)

        if h_adaptive:
            segment_samples = optimizer.get_adaptive_mesh_sample_counts(wps)
        else:
            segment_samples = optimizer.get_segment_sample_counts(wps, samples, proportional=True)
        segment_samples = [int(s) for s in segment_samples]
        sample_to_segment = list(np.repeat(np.arange(len(wps) - 1), segment_samples))

        if not quiet:
            click.echo("Running Pyramid Stage 1: Vectorized Homotopy Bank Generation...")
        cand_states, cand_dts, metadata = generate_homotopy_bank(
            wps,
            segment_samples,
            track_width=float(robot_cfg.track_width),
            v_max=float(robot_cfg.max_linear_speed()),
            stop_waypoint_indices=all_stop_indices,
        )
        if not quiet:
            click.echo(f"  Generated {len(cand_states)} hypotheses. Stage 2: Sieve screening via compiled JAX...")
        surrogate_costs = screen_candidates_jax(
            engine,
            cand_states,
            cand_dts,
            sample_to_segment,
        )
        if pyramid_candidates is not None:
            target_k = pyramid_candidates
        elif memetic_evolution:
            target_k = population_size
        elif ctx.get_parameter_source('candidates') != click.core.ParameterSource.DEFAULT:
            target_k = candidates
        else:
            target_k = 10

        k_clusters = max(2, min(target_k, len(cand_states)))
        selected_indices = select_topological_representatives(
            cand_states,
            surrogate_costs,
            k_clusters=k_clusters,
            min_separation_m=0.12,
        )
        if not quiet:
            click.echo(f"  Stage 3: Clustered into Top {len(selected_indices)} distinct homotopy seeds.")
        specs_to_solve = [
            {
                "guess_style": f"pyramid_{i+1}",
                "guess": f"pyramid_cluster_{idx}",
                "per_segment_dt": True,
                "accuracy_weight": accuracy_weight,
                "initial_guess": np.concatenate([cand_dts[idx], cand_states[idx].flatten()]),
            }
            for i, idx in enumerate(selected_indices)
        ]

    def _payloads_for(specs):
        """The solve payloads for a set of candidates."""
        payloads = []
        for spec in specs:
            solve_kwargs = dict(
                num_samples_per_segment=samples,
                accuracy_weight=spec["accuracy_weight"],
                stop_waypoint_indices=all_stop_indices,
                waypoint_events=waypoint_events,
                verbose=verbose and not quiet,
                capture_iterations=capture_iterations,
            )
            if not multiverse:
                solve_kwargs["guess_style"] = spec["guess_style"]
                solve_kwargs["per_segment_dt"] = spec.get("per_segment_dt", False)
                solve_kwargs["self_consistent"] = bool(self_consistent)
                if force_weight > 0.0:
                    solve_kwargs["force_weight"] = float(force_weight)
                if "initial_guess" in spec:
                    solve_kwargs["initial_guess"] = spec["initial_guess"]
                    solve_kwargs["cost_metric"] = "discretization"
                    solve_kwargs["epigraph_constraints"] = True
                if "cost_metric" in spec:
                    solve_kwargs["cost_metric"] = spec["cost_metric"]
                if "epigraph_constraints" in spec:
                    solve_kwargs["epigraph_constraints"] = spec["epigraph_constraints"]
            if fast_polish:
                solve_kwargs["fast_polish"] = True
            payloads.append((robot_cfg, wps, spec, solve_kwargs, multiverse))
        return payloads

    # The live view and the convergence plot both need the solver in this
    # process, so those runs stay sequential. Worker processes import the
    # optimizer module themselves, so an injected or patched optimizer would be
    # ignored there: only the stock solvers may run in a worker.
    stock_optimizer = type(optimizer).__module__ in ("optimizer", "multiverse_optimizer")

    def _solve_batch(specs):
        """Solve one set of candidates and return their results and logs."""
        payloads = _payloads_for(specs)
        parallel = (len(payloads) > 1 and not live and not show_convergence
                    and not no_parallel and stock_optimizer)
        if (pyramid or memetic_evolution) and not multiverse:
            from surrogate_polish import solve_pyramid_candidates
            if not quiet:
                click.echo(f"Solving {len(payloads)} survivor candidates in parallel (ProcessPoolExecutor)...")
            survivor_guesses = [spec.get("initial_guess") for spec in specs if "initial_guess" in spec]
            if len(survivor_guesses) == len(specs):
                base_kw = dict(payloads[0][3])
                base_kw.pop("initial_guess", None)
                max_w = min(len(payloads), max(workers, os.cpu_count() or 8)) if parallel else 1
                solved = solve_pyramid_candidates(
                    robot_cfg, wps, survivor_guesses, base_kw, max_workers=max_w
                )
                histories = [[] for _ in solved]
                return solved, histories
        if surrogate_polish and not multiverse:
            from surrogate_polish import solve_candidates_threaded
            if not quiet:
                click.echo(f"Solving {len(payloads)} candidates via 5-Layer Surrogate-Polish pipeline (ThreadPoolExecutor)...")
            return solve_candidates_threaded(
                payloads, workers=workers if parallel else 1
            )
        if not quiet and len(payloads) > 1:
            click.echo(f"Solving {len(payloads)} candidates "
                       f"({'in parallel' if parallel else 'sequentially'})...")
        return _solve_candidates(payloads, workers, quiet, parallel,
                                  local_optimizer=optimizer)

    # The candidates are solved in rounds. The first round is the candidate
    # set; if it produced nothing usable, the seeds the set did not use are
    # tried as a rescue (KI-048).
    batch = list(specs_to_solve)
    is_rescue = False
    while batch:
        solved, histories = _solve_batch(batch)

        if recombine_segments and len(solved) > 1 and not multiverse and not is_rescue:
            from surrogate_polish import recombine_candidate_segments
            from reachability_engine import CompiledReachabilityEngine
            validator_cls, robot_config_cls = _ensure_jax_classes()
            jax_cfg = robot_config_cls(robot_cfg)
            immrax_val = validator_cls(jax_cfg, control_dt=control_dt, propagation=propagation)
            rec_engine = CompiledReachabilityEngine(immrax_val)

            base_kw = {
                "accuracy_weight": accuracy_weight,
                "stop_waypoint_indices": all_stop_indices,
                "waypoint_events": waypoint_events,
                "force_weight": force_weight,
                "per_segment_dt": True,
            }
            if not quiet:
                click.echo("Running Segment-Wise Recombination Splicer (Chimera Crossover)...")
            chimera_samples, chimera_stats, recomb_rep = recombine_candidate_segments(
                solved, wps, robot_cfg, rec_engine,
                stop_waypoint_indices=all_stop_indices,
                accuracy_weight=accuracy_weight,
                force_weight=force_weight,
                solve_kwargs=base_kw,
                max_polish_iters=250,
            )
            if chimera_samples and chimera_stats and recomb_rep.get("recombined"):
                if not quiet:
                    click.echo(
                        f"  Synthesized Chimera trajectory (contributors: {recomb_rep['unique_contributors']}, "
                        f"polished in {recomb_rep['polish_iters']} iters, duration={recomb_rep['duration_s']:.3f} s)."
                    )
                solved.append((chimera_samples, chimera_stats))
                histories.append([])
                batch.append({
                    "guess_style": "chimera_recombined",
                    "guess": "chimera",
                    "per_segment_dt": True,
                    "accuracy_weight": accuracy_weight,
                })
            else:
                if not quiet:
                    click.echo(f"  Recombination notice: {recomb_rep.get('reason', 'unknown')}")

        if memetic_evolution and len(solved) > 1 and not multiverse and not is_rescue:
            from surrogate_polish import run_memetic_evolution
            from reachability_engine import CompiledReachabilityEngine
            validator_cls, robot_config_cls = _ensure_jax_classes()
            jax_cfg = robot_config_cls(robot_cfg)
            immrax_val = validator_cls(jax_cfg, control_dt=control_dt, propagation=propagation)
            mem_engine = CompiledReachabilityEngine(immrax_val)

            base_kw = {
                "accuracy_weight": accuracy_weight,
                "stop_waypoint_indices": all_stop_indices,
                "waypoint_events": waypoint_events,
                "force_weight": force_weight,
                "per_segment_dt": True,
            }
            if not quiet:
                click.echo(f"Running Memetic Evolution ({generations} generations, population={population_size})...")
            champ_samples, champ_stats, evo_report = run_memetic_evolution(
                solved, wps, robot_cfg, mem_engine,
                num_generations=generations,
                population_size=population_size,
                stop_waypoint_indices=all_stop_indices,
                accuracy_weight=accuracy_weight,
                force_weight=force_weight,
                solve_kwargs=base_kw,
                max_polish_iters=150,
                max_workers=workers if not no_parallel else 1,
                verbose=not quiet,
            )
            if champ_samples and champ_stats:
                if not quiet:
                    click.echo(
                        f"  Memetic Evolution completed: Champion duration={champ_stats.get('memetic_evolution', {}).get('champion_duration', 0.0):.3f} s "
                        f"(score={champ_stats.get('memetic_evolution', {}).get('champion_score', 0.0):.3f}, "
                        f"origin '{champ_stats.get('memetic_evolution', {}).get('champion_tag', '')}')."
                    )
                solved.append((champ_samples, champ_stats))
                histories.append([])
                batch.append({
                    "guess_style": "memetic_champion",
                    "guess": "memetic",
                    "per_segment_dt": True,
                    "accuracy_weight": accuracy_weight,
                })

        for cand_idx, (spec, solve_result) in enumerate(zip(batch, solved)):
            curr_accuracy_weight = spec["accuracy_weight"]
            guess_style = spec["guess_style"]
            per_segment_dt = spec.get("per_segment_dt", False)
            number = (f"Rescue {cand_idx + 1}/{len(batch)}" if is_rescue
                      else f"Candidate {cand_idx + 1}/{len(batch)}")
            if not quiet:
                if multiverse:
                    click.echo(f"{number} "
                               f"(accuracy_weight={curr_accuracy_weight})...")
                else:
                    timing = "per-segment" if per_segment_dt else "equal-duration"
                    weight = f", a={curr_accuracy_weight:g}" if sweep else ""
                    click.echo(f"{number} "
                               f"(guess_style={guess_style}, {timing} dt{weight})...")

            samples_data, stats = solve_result

            # The sample layout the optimizer used: the sample budget is
            # split across the waypoint segments in proportion to their
            # length, so the waypoints are not k * samples_per_segment apart.
            # The gate and the segment report have to use the optimizer's own
            # table, or a waypoint's threshold is applied to the wrong
            # samples.
            waypoint_bounds = _waypoint_boundaries(stats.get("segment_samples") or (samples if isinstance(samples, (list, tuple, np.ndarray)) else None))

            # Track this candidate. A non-converged candidate is never eligible for export (KI-001).
            attempt_record = {
                "samples": samples_data, "stats": stats,
                "cost": stats.get('final_cost', float('inf')),
                "ok": False, "violation": float("inf"),
                "tube": float("inf"),
                "duration": float(samples_data[-1]['t']) if samples_data else float("inf"),
                "diverged": False,
                "converged": bool(stats.get("converged", False)),
                "guess_style": guess_style,
                # Which starting point produced it: the seed, or the Multi-Verse
                # pipeline's refined trajectory. The style alone does not say:
                # a refined candidate carries the baseline style.
                "guess": spec.get("guess", guess_style),
                "per_segment_dt": per_segment_dt,
                "accuracy_weight": curr_accuracy_weight,
                "iteration_history": (histories[cand_idx]
                                      if cand_idx < len(histories) else []),
                "reasons": [],
                # A rescue candidate is insurance: it is only chosen when the
                # candidate set itself produced nothing usable.
                "rescue": is_rescue,
            }
            all_attempts.append(attempt_record)

            # Always compute reachability envelope if JAX is available and we are in live/validate mode
            if not use_validation:
                continue

            try:
                cached_immrax = stats.get("immrax_report") if isinstance(stats, dict) else None
                if (
                    surrogate_polish
                    and cached_immrax is not None
                    and digital_twin is None
                    and propagation == "plant"
                ):
                    if not quiet:
                        click.echo("  Running the Immrax robustness check (compiled lax.scan)...")
                    immrax_report = cached_immrax
                else:
                    validator_cls, robot_config_cls = _ensure_jax_classes()
                    jax_cfg = robot_config_cls(robot_cfg)
                    if digital_twin is None:
                        immrax_val = validator_cls(jax_cfg, control_dt=control_dt,
                                                   propagation=propagation)
                    else:
                        immrax_val = validator_cls(jax_cfg, digital_twin=digital_twin,
                                                   control_dt=control_dt,
                                                   propagation=propagation)

                    # Use the uncertainty ranges declared in the config (KI-009).
                    cof_range = robot_cfg.uncertainties["cof"]
                    torque_range = robot_cfg.uncertainties["torque_headroom"]
                    backlash_range = robot_cfg.uncertainties["gear_backlash_m"]
                    track_width_range = robot_cfg.uncertainties["track_width_range"]
                    wheel_radius_range = robot_cfg.uncertainties["wheel_radius_range"]
                    inertia_range = robot_cfg.uncertainties["inertia_kg_m2"]

                    if not quiet:
                        click.echo("  Running the Immrax robustness check...")
                    immrax_report = immrax_val.validate_trajectory(
                        samples_data, cof_range, torque_range, backlash_range,
                        track_width_range, wheel_radius_range, inertia_range)
            except Exception as e:
                if not quiet:
                    click.echo(f"  ERROR: Immrax validation failed: {e}")
                import traceback
                traceback.print_exc()
                # A validator crash is not a pass (KI-002).
                attempt_record["reasons"] = [f"Immrax validation error: {e}"]
                if not validate:
                    break
                continue

            # Attach envelope to samples for export/viz (do this even if it failed, so user can see it)
            for i, env in enumerate(immrax_report['reachability']['envelope']):
                if i < len(samples_data):
                    samples_data[i]['reachability_envelope'] = env

            # CONTEXT-AWARE SAFETY CHECK (score every candidate on the same scale)
            failed_threshold = False
            max_vio_val = 0.0
            max_err_all = 0.0

            for i in range(len(samples_data)):
                # Determine local threshold. The segment's limit is the one of
                # the waypoint the segment arrives at (see _segment_limit), so a
                # precision waypoint is held to its tight limit on the
                # approach. The sample's waypoint comes from the optimizer's
                # own layout: the sample budget is split across the segments
                # in proportion to their length, so the waypoints are not
                # k * samples_per_segment apart.
                if waypoint_bounds:
                    wp_idx = _waypoint_of(waypoint_bounds, i)
                else:
                    s_val = samples[0] if isinstance(samples, (list, tuple, np.ndarray)) else samples
                    wp_idx = i // max(1, int(s_val))
                local_threshold = _segment_limit(waypoint_thresholds, wp_idx)

                # Check envelope radius (max error)
                env = samples_data[i].get('reachability_envelope', {})
                if env:
                    err_x = (env['x_max'] - env['x_min']) / 2.0
                    err_y = (env['y_max'] - env['y_min']) / 2.0
                    err = np.sqrt(err_x**2 + err_y**2)
                    max_err_all = max(max_err_all, err)
                    if env.get('unknown') or err > local_threshold:
                        failed_threshold = True
                        max_vio_val = max(max_vio_val, err)

            attempt_record["tube"] = float(max_err_all)
            attempt_record["diverged"] = bool(immrax_report['reachability'].get('diverged', False))
            # A cut-off propagation reports the width the tube had when it
            # crossed the guard, not the width it would have reached, so the
            # number is a property of the guard and not of the path. Say so
            # next to it (KI-049).
            attempt_record["envelope_truncated"] = bool(
                immrax_report['reachability'].get('envelope_truncated',
                                                  attempt_record["diverged"]))
            # Which model produced the number. The envelope is the width of a
            # tube, and the tube is a property of the propagation, so a
            # number without its model cannot be read (KI-049).
            attempt_record["propagation"] = str(
                immrax_report.get("propagation", propagation))
            attempt_record["inclusion"] = dict(
                immrax_report.get("inclusion", {}) or {})
            attempt_record["tracking_m"] = float(immrax_report.get("max_tracking_error_m", 0.0))
            attempt_record["heading_rad"] = float(immrax_report.get("max_heading_error_rad", 0.0))
            attempt_record["ok"] = (attempt_record["converged"] and not failed_threshold
                                    and bool(immrax_report['passed']))
            attempt_record["reasons"] = _immrax_failure_reasons(immrax_report, failed_threshold, max_vio_val)
            if failed_threshold:
                attempt_record["violation"] = max_vio_val
            else:
                # Tracking is a quantified failure too; rank failed candidates on it
                # instead of falling back to pure cost (KI-020 scoring).
                tracking_error = float(immrax_report.get("max_tracking_error_m", 0.0))
                if tracking_error > 0.005:
                    attempt_record["violation"] = tracking_error

            cpu_summary = None
            if validate:
                # The CPU audit runs on every candidate, not only on the winner, so
                # a candidate that clears Immrax but breaks the force, slip or
                # integration limits cannot block a later candidate that passes
                # every gate.
                cpu_summary = validate_samples(samples_data, robot_cfg)
                if not cpu_summary["passed"]:
                    attempt_record["ok"] = False
                    attempt_record["reasons"] = list(attempt_record["reasons"]) + \
                        _cpu_failure_reasons(cpu_summary)
                # The peak wheel jerk, for the score vector. It is the quantity
                # the accuracy weight trades against, so it belongs beside the
                # other scores (Q10's step 1).
                attempt_record["peak_jerk_m_s3"] = float(
                    cpu_summary.get("metrics", {}).get("max_jerk_m_s3", 0.0) or 0.0)

            # 5-part time-equivalent path fitness score (Stage 3: precision > accuracy)
            if waypoint_bounds:
                stop_sample_indices = [
                    waypoint_bounds[w] for w in all_stop_indices
                    if 0 <= w < len(waypoint_bounds)
                ]
            else:
                stop_sample_indices = [
                    w * samples for w in all_stop_indices
                    if 0 <= w * samples < len(samples_data)
                ]
            fitness_info = compute_path_fitness(
                samples_data,
                robot_cfg,
                stop_sample_indices=stop_sample_indices,
                immrax_report=immrax_report,
                cpu_summary=cpu_summary,
            )
            attempt_record["fitness"] = float(fitness_info["fitness_score"])
            attempt_record["stop_spread_m"] = float(
                fitness_info["stop_precision_spread_m"])
            attempt_record["slip_risk"] = float(
                fitness_info["slip_risk_penalty"])
            attempt_record["rms_force_n"] = float(
                fitness_info["rms_wheel_force_n"])
            attempt_record["rms_force_util"] = float(
                fitness_info["rms_force_utilization"])
            attempt_record["fitness_info"] = fitness_info

            # Per-segment detail, so a failure can be pinned to a part of the path.
            attempt_record["segments"] = _segment_report(
                samples_data, samples, waypoint_thresholds, immrax_report,
                cpu_summary, segment_samples=stats.get("segment_samples"))

            if not quiet:
                score = "envelope diverged" if attempt_record["diverged"] else f"envelope {attempt_record['tube']*100:.2f} cm"
                if attempt_record["ok"]:
                    click.echo(
                        f"  Passed! {score}, fitness {attempt_record['fitness']:.3f} s "
                        f"(stop spread {attempt_record['stop_spread_m']*100:.2f} cm), "
                        f"all segments within thresholds."
                    )
                else:
                    reason_text = "; ".join(attempt_record["reasons"]) or "validation failed"
                    click.echo(f"  FAILED: {reason_text}.")

            if not validate:  # Live-only: keep the result for visualization.
                break

        # Did the run produce something usable? With validation that means a
        # candidate that passed every gate; without it, a candidate that
        # converged. If not, the seeds the candidate set did not use are
        # tried before the mission is given up on (KI-048): the set's two
        # candidates share one seed, so without the rescue a path that beats
        # the baseline seed ends with no trajectory at all.
        usable = [a for a in all_attempts
                  if (a.get("ok") if use_validation else a.get("converged"))]
        if usable or is_rescue:
            break
        batch = _rescue_specs(specs_to_solve, accuracy_weight, not multiverse,
                              dt_mode, use_validation)
        if not batch:
            break
        is_rescue = True
        if not quiet:
            if use_validation:
                click.echo("  No candidate passed the gates; trying the other "
                           "seeds.")
            else:
                click.echo("  No candidate converged; trying the other seeds.")

    if validate:
        # Never silently export a failed candidate: prefer a passing one with the
        # smallest envelope, otherwise the converged candidate closest to passing.
        # A rescue candidate is insurance: it is only chosen when the candidate
        # set itself produced nothing that passed.
        primary = [a for a in all_attempts if not a.get("rescue")]
        rescue = [a for a in all_attempts if a.get("rescue")]
        best, passed_any = _select_attempt(primary, prefer_duration=bool(surrogate_polish or pyramid or memetic_evolution or unified))
        if not passed_any and rescue:
            rescue_best, rescue_passed = _select_attempt(rescue, prefer_duration=bool(surrogate_polish or pyramid or memetic_evolution or unified))
            if rescue_passed or best is None:
                best, passed_any = rescue_best, rescue_passed
        if best is None:
            raise click.ClickException(
                f"Solver did not converge on any of the {len(all_attempts)} "
                f"attempts. No trajectory written.")
        selected_k = next(k for k, a in enumerate(all_attempts, 1) if a is best)
        if not quiet and len(all_attempts) > 1:
            for k, a in enumerate(all_attempts, 1):
                if a.get("diverged"):
                    score = "envelope diverged"
                elif np.isfinite(a.get("tube", float("inf"))):
                    score = f"envelope {a['tube']*100:.2f} cm"
                else:
                    score = "no envelope"
                state = "ok" if a["ok"] else "failed"
                if multiverse:
                    label = f"Candidate {k}"
                else:
                    # Name the mode by what it constrains, and show how the
                    # time was split, so a uniform win is self-explanatory:
                    # equal-duration segments are forced to take the same time
                    # each, which is why they are slow on an uneven path
                    # (KI-022).
                    timing = "per-seg" if a.get("per_segment_dt") else "equal-dur"
                    weight = f", a={a.get('accuracy_weight', 0.0):g}" if sweep else ""
                    label = (f"Candidate {k} ({a.get('guess_style', 'baseline')}, "
                             f"{timing}{weight})")
                click.echo(f"  {label}: {a.get('duration', float('inf')):6.3f} s, {score} ({state})")
                seg_dt = (a.get("stats") or {}).get("segment_dt")
                if seg_dt and len(seg_dt) > 1:
                    seg_samples_list = (a.get("stats") or {}).get("segment_samples") or (samples if isinstance(samples, (list, tuple)) else [samples] * len(seg_dt))
                    seg_times = [d * s for d, s in zip(seg_dt, seg_samples_list)]
                    spread = "  segment times: " + " ".join("%.2f" % t for t in seg_times) + " s"
                    click.echo(f"    {spread}")
            click.echo(f"  Selected candidate {selected_k}.")
        samples_data, stats = best["samples"], best["stats"]
        attempt_passed = passed_any
        best_reasons = best["reasons"]
        selected_tube = best.get("tube", float("inf"))
        selected_index = selected_k
    else:
        # Without validation the first candidate is the result. After a
        # rescue round it is the last candidate that converged, not merely
        # the last one tried.
        converged = [a for a in all_attempts if a.get("converged")]
        chosen = converged[-1] if converged else (
            all_attempts[-1] if all_attempts else None)
        if chosen is not None:
            samples_data, stats = chosen["samples"], chosen["stats"]
            best_reasons = chosen["reasons"]
            selected_tube = chosen.get("tube", float("inf"))
        else:
            best_reasons = []
            selected_tube = float("inf")
        selected_index = 1 if all_attempts else None

    if use_validation:
        # The studio reads this to show which part of the path failed.
        _emit_validation_report(all_attempts, selected_index, propagation=propagation)

    # --- EXPORT GATE (KI-001 / KI-002) ---
    block_reason = _export_block_reason(stats, validate, attempt_passed)
    if block_reason:
        detail = "; ".join(best_reasons)
        message = f"{block_reason}: {detail}" if detail else block_reason
        raise click.ClickException(
            f"{message[0].upper() + message[1:]}. No trajectory written to '{output}'."
        )

    # Second line of defence: a converged solve must still reach the requested
    # final pose, even when --validate is not set.
    waypoint_check = check_final_waypoint(samples_data, wps[-1])
    if not waypoint_check["passed"]:
        raise click.ClickException(
            f"Final waypoint not reached (position error {waypoint_check['pos_error_m']*100:.2f} cm, "
            f"heading error {waypoint_check['heading_error_rad']:.3f} rad). "
            f"No trajectory written to '{output}'."
        )

    if validate:
        cpu_summary = validate_samples(samples_data, robot_cfg)
        if not cpu_summary["passed"]:
            reasons = _cpu_failure_reasons(cpu_summary, waypoint_check)
            raise click.ClickException(
                f"Validator rejected the trajectory: {'; '.join(reasons) or 'unknown reason'}. "
                f"No trajectory written to '{output}'."
            )

    result = {
        "name": os.path.basename(output).split('.')[0],
        "version": 3,
        "trajectory": {
            "config": config_data.get("robot", config_data.get("config", config_data)),
            "samples": samples_data
        }
    }
    with open(output, 'w') as f:
        json.dump(result, f, indent=1)

    written = [output]

    if benchmark:
        quality_metrics = compute_metrics(samples_data, robot_cfg)
        stats_to_write = {k: v for k, v in stats.items() if k != "immrax_report"}
        stats_to_write["quality_metrics"] = quality_metrics
        stats_to_write["robot_config"] = {
            "mass": robot_cfg.mass, "inertia": robot_cfg.inertia, "track_width": robot_cfg.track_width,
            "wheel_radius": robot_cfg.wheel_radius, "v_max_rad_s": robot_cfg.v_max_rad_s,
            "t_max_nm": robot_cfg.t_max_nm, "gearing": robot_cfg.gearing, "cof": robot_cfg.cof
        }
        stats_output = os.path.splitext(output)[0] + '_stats.json'
        with open(stats_output, 'w') as f:
            json.dump(stats_to_write, f, indent=2)
        written.append(stats_output)

    if validate:
        validate_trajectory(output, config, waypoints=wps)

    if export_format == 'controller':
        ctrl_output = os.path.splitext(output)[0] + '_controller.json'
        write_controller_file(output, ctrl_output, target_dt=controller_dt, track_width=robot_cfg.track_width)
        written.append(ctrl_output)

    if export_format in ('python', 'python-compact'):
        py_output = os.path.splitext(output)[0] + '.py'
        write_python_file(output, py_output, compact=(export_format == 'python-compact'))
        written.append(py_output)

    if not quiet:
        _print_summary(output, samples_data, robot_cfg, validate, attempt_passed,
                       selected_tube, written[1:])

    if push_hub:
        try:
            robot_cfg.sync_pybricks_robot_config()
        except Exception:
            pass
        _push_to_hub(output, hub_conn, hub_name, hub_log, quiet)

    if plot:
        from plotter import plot_trajectory, plot_reachability_envelope
        if not quiet:
            click.echo("Generating plot...")
            click.echo("  Generating standard trajectory plot...")
        plot_trajectory(samples_data, waypoints=wps, title=f"Trajectory: {os.path.basename(output)}")

        # Also show reachability envelope plot if Immrax data is available
        if HAS_JAX and any('reachability_envelope' in s for s in samples_data):
            if not quiet:
                click.echo("  Generating reachability envelope plot...")
            plot_reachability_envelope(samples_data, waypoints=wps, title=f"Reachability Envelope: {os.path.basename(output)}", block=True)

    if animate:
        from animated_plotter import animate_trajectory
        animate_trajectory(samples_data, waypoints=wps, title=f"Trajectory: {os.path.basename(output)}")

    if show_convergence:
        if hasattr(optimizer, 'iteration_history') and optimizer.iteration_history:
            from convergence_plotter import plot_convergence
            plot_convergence(optimizer.iteration_history, mode=convergence_mode, waypoints=wps, title=f"Convergence: {os.path.basename(output)}")

    if iteration_history:
        # The way the trajectory converged, for a tool that plots it.
        #
        # The log belongs to the candidate that was selected: the candidates
        # are independent, so the log of a candidate that lost describes a
        # trajectory that was never written. The candidates may have been
        # solved in worker processes, so the log comes with the result.
        history = []
        if all_attempts:
            chosen = all_attempts[max(0, (selected_index or 1) - 1)]
            history = chosen.get("iteration_history") or []
        entries = []
        for entry in history:
            item = dict(entry)
            for key, value in list(item.items()):
                if hasattr(value, "tolist"):
                    item[key] = value.tolist()
            entries.append(item)
        with open(iteration_history, "w", encoding="utf-8") as handle:
            json.dump(entries, handle)
        if not quiet:
            click.echo(f"  wrote {len(entries)} iterations to {iteration_history}")


if __name__ == '__main__':
    main()
