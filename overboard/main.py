"""
FLL Trajectory Optimizer CLI.

Generates time-optimal trajectories for LEGO differential-drive robots.
"""

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from typing import Any

# A first-time user who runs this before installing anything should get the
# environment report with the install command, not an ImportError traceback.
# diagnostics is stdlib only, so it works before anything is installed.
from overboard.diagnostics import missing_core_packages, run_doctor

if __name__ == "__main__" and missing_core_packages():
    raise SystemExit(run_doctor())

import click
import numpy as np

from overboard.paths import HUB_MISSION_MODULE, HUB_RUNNER
from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer
from overboard.verification.validator import (
    validate_trajectory,
    validate_samples,
    check_final_waypoint,
    compute_metrics,
    compute_path_fitness,
)
from overboard.deploy.export import write_controller_file, write_python_file

DEFAULT_CONFIG = "examples/fll_choreo.json"
JAX_HINT = "Install the optional dependencies with: pip install -r requirements-optional.txt"

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
        # The widened config no longer fits an open-loop tube, so the safe
        # preset pays for the closed-loop model (KI-049/KI-050). An explicit
        # --propagation still wins.
        "propagation": "tick",
    },
}


def _ensure_jax_classes():
    """Import the JAX validators on demand."""
    global ImmraxValidator, JAXRobotConfig
    if ImmraxValidator is None:
        from overboard.verification.immrax_validator import ImmraxValidator as _ImmraxValidator
        ImmraxValidator = _ImmraxValidator
    if JAXRobotConfig is None:
        from overboard.models.jax_robot_model import JAXRobotConfig as _JAXRobotConfig
        JAXRobotConfig = _JAXRobotConfig
    return ImmraxValidator, JAXRobotConfig


def _select_attempt(attempts):
    """Choose the candidate to export: the passing attempt with the shortest
    duration, the reachability envelope breaking a tie.

    The 5-part fitness score is reported for every candidate, so the rule
    can be re-run offline, but it does not decide: the planner is
    time-optimal, so the shortest passing plan is the one that ships. If
    nothing passed, the converged attempt with the smallest quantified
    violation wins, and `passed_any` is False - so the export is still
    blocked. Non-converged attempts are never eligible (KI-001)."""
    candidates = [a for a in attempts if a.get("converged", False)]
    if not candidates:
        return None, False
    passing = [a for a in candidates if a["ok"]]
    if passing:
        return min(
            passing,
            key=lambda a: (
                a.get("duration", a["cost"]),
                a.get("tube", float("inf")),
            ),
        ), True
    return min(candidates, key=lambda a: (a["violation"], a["cost"])), False



# The timing model the planner uses: one interval per waypoint segment.
TIMING_MODEL = "per-segment"

# The fallback seeds, in the order they are tried when the planner's own
# candidates produce nothing usable (KI-048). The order is the priority: the
# seeds that change the geometry come first, because those are the ones that
# can find a different optimum; the plain baseline is last, because it is the
# guess the planner's candidates already beat.
FALLBACK_CANDIDATE_STYLES = ("rolling", "frontier_sprint", "frontier_sweeper", "frontier_exploratory",
                             "baseline")

# Thread limits handed to the candidate worker processes. Several candidates
# solve at the same time, so each one keeps a single compute thread.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def _fallback_specs(accuracy_weight):
    """The fallback batch: the seeds the planner's candidates cannot provide.

    The planner's own candidates are three variants of one frontier family,
    so when none of them works the other seeds have to be tried (KI-048).
    The timing model is per-segment, because that is where the measured
    rescues came from: of the eighteen mission-robot pairs the planner's
    candidates lost on random paths, the stock set rescued two, both of them
    per-segment, and the equal-duration candidates rescued none (the stress
    run of 2026-10-20).
    """
    return [{"accuracy_weight": accuracy_weight,
             "guess_style": style,
             "per_segment_dt": True}
            for style in FALLBACK_CANDIDATE_STYLES]


def _solve_candidate_worker(payload):
    """Solve one candidate. Runs in a worker process when candidates are parallel.

    Returns (samples, stats, history). A candidate that raises is reported as a
    non-converged attempt, so the export gate stays closed for it (KI-001).

    The history is the solver's iteration log. It travels with the result
    because the solver runs in *this* worker, and the caller has no other
    handle on it.
    """
    robot_cfg, wps, solve_kwargs = payload
    try:
        optimizer = TrajectoryOptimizer(robot_cfg)
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
            for robot_cfg, wps, solve_kwargs in payloads:
                try:
                    before = len(getattr(local_optimizer, "iteration_history", []))
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


@dataclass
class _SolveContext:
    """Everything a candidate solve needs, outside the candidate spec itself.

    These values are fixed for the whole run: the path, the robot, the
    sampling and the output mode. They used to be captured implicitly by two
    closures inside ``main``, which made the solve path unreadable and
    untestable; naming them is what lets the solver be lifted out of the CLI.
    """

    robot_cfg: Any
    waypoints: Any
    samples: int
    stop_indices: Any
    events: Any
    verbose: bool
    quiet: bool
    capture_iterations: bool
    force_weight: float
    fast_polish: bool
    workers: int
    no_parallel: bool
    planner: Any
    polish: bool
    optimizer: Any

    def may_run_parallel(self, payload_count):
        """True when the candidates may be solved in worker processes.

        Worker processes import the optimizer module themselves, so an
        injected or patched optimizer would be ignored there: only the stock
        solver may run in a worker.
        """
        stock_optimizer = type(self.optimizer).__module__ == "overboard.core.optimizer"
        return (payload_count > 1 and not self.no_parallel and stock_optimizer)


def _solve_payloads(specs, ctx):
    """The solve payloads for a set of candidates."""
    payloads = []
    for spec in specs:
        solve_kwargs = dict(
            num_samples_per_segment=ctx.samples,
            accuracy_weight=spec["accuracy_weight"],
            stop_waypoint_indices=ctx.stop_indices,
            waypoint_events=ctx.events,
            verbose=ctx.verbose and not ctx.quiet,
            capture_iterations=ctx.capture_iterations,
        )
        solve_kwargs["guess_style"] = spec["guess_style"]
        solve_kwargs["per_segment_dt"] = spec.get("per_segment_dt", False)
        if ctx.force_weight > 0.0:
            solve_kwargs["force_weight"] = float(ctx.force_weight)
        if "initial_guess" in spec:
            solve_kwargs["initial_guess"] = spec["initial_guess"]
            solve_kwargs["cost_metric"] = "discretization"
            solve_kwargs["epigraph_constraints"] = True
        if "cost_metric" in spec:
            solve_kwargs["cost_metric"] = spec["cost_metric"]
        if "epigraph_constraints" in spec:
            solve_kwargs["epigraph_constraints"] = spec["epigraph_constraints"]
        if ctx.fast_polish:
            solve_kwargs["fast_polish"] = True
        payloads.append((ctx.robot_cfg, ctx.waypoints, solve_kwargs))
    return payloads


def _dispatch_solve(specs, ctx):
    """Solve one set of candidates with the one engine there is.

    The candidates are solved by the CasADi optimizer, in worker processes
    when there is more than one of them. The reachability check and the
    optional gradient polish stay in this process, where the JAX stack is
    warm.
    """
    payloads = _solve_payloads(specs, ctx)
    parallel = ctx.may_run_parallel(len(payloads))
    if not ctx.quiet and len(payloads) > 1:
        click.echo(f"Solving {len(payloads)} candidates "
                   f"({'in parallel' if parallel else 'sequentially'})...")
    return _solve_candidates(payloads, ctx.workers, ctx.quiet, parallel,
                             local_optimizer=ctx.optimizer)


def _export_block_reason(stats, validate, attempt_passed, best_effort=False):
    """Return why the result must not be written, or None if it may be.

    ``best_effort`` is the operator's explicit acceptance of an accuracy
    excess: the path is written and marked, because a path that misses the
    envelope is still a plan to look at. It is not a pass - the file says
    so, the summary says so, and the studio's push stays gated on the
    verdict. It does not cover a plan that is physically infeasible: motor
    force, wheel slip, a diverged tube and a non-converged solve are refused
    before the gate (`_choose_attempt`).
    """
    if not stats.get("converged", False):
        return "solver did not converge"
    if validate and not attempt_passed and not best_effort:
        return "validation failed"
    return None


def _mark_best_effort(stats, attempts, selected_index):
    """Mark an exported trajectory as a best-effort, unvalidated plan.

    The verdict travels with the trajectory: the `.traj` root and the
    `_stats.json` both carry it, so a file read later cannot be mistaken for
    an approved one. Returns the biggest excess over the local limit, in
    metres, for the summary line.
    """
    attempt = attempts[max(0, selected_index - 1)]
    segments = attempt.get("segments", []) or []
    excess = max((float(segment.get("envelope_m", 0.0))
                  - float(segment.get("limit_m", 0.0))
                  for segment in segments), default=0.0)
    stats["validation"] = {
        "passed": False,
        "best_effort": True,
        "max_excess_m": excess,
        "reasons": list(attempt.get("reasons", [])),
        "segments": segments,
    }
    return excess


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


# How far the reachable set may spread around the plan at any sample, in
# metres, when neither the run nor a waypoint declares one. The tube is
# propagated open-loop by default, so this is effectively a duration budget:
# a long path can fail for being long (KI-054). A waypoint's
# `error_threshold` overrides it; `--envelope-limit` sets the run-wide value.
DEFAULT_ENVELOPE_LIMIT_M = 0.02
# The propagation cuts the tube off at this radius (KI-049), so a looser
# limit could never be measured, let alone failed.
MAX_ENVELOPE_LIMIT_M = 0.05


def _segment_limit(thresholds, index, default=DEFAULT_ENVELOPE_LIMIT_M):
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
                    segment_samples=None, default_limit=DEFAULT_ENVELOPE_LIMIT_M):
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
        limit = _segment_limit(thresholds, index, default_limit)
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


def _emit_validation_report(attempts, selected_index, propagation=None,
                            best_effort=False, envelope_limit=None):
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
               "propagation": propagation,
               # The run-wide default the gate applied. A verdict depends on
               # it, so it is never left implicit (KI-054).
               "envelope_limit_m": _finite(envelope_limit),
               # True when a failed candidate was exported anyway, so the
               # studio can say "failed, shown" instead of "not checked".
               "best_effort": bool(best_effort)}
    click.echo(VALIDATION_MARKER + json.dumps(payload))


def _candidate_summary(attempts, selected_index):
    """Every solved candidate, samples included, for a tool that draws them.

    The comparison is only worth showing if the paths are there: the numbers
    say one candidate is 0.04 s faster, and only the paths say why. The
    summary is diagnostics - it is written next to the run, not exported, so
    a failed validation still leaves the operator with something to look at.
    """
    candidates = []
    for number, attempt in enumerate(attempts, 1):
        candidates.append({
            "index": number,
            "style": attempt.get("guess_style"),
            "ok": bool(attempt.get("ok")),
            "converged": bool(attempt.get("converged")),
            "rescue": bool(attempt.get("rescue")),
            "duration_s": _finite(attempt.get("duration")),
            "envelope_cm": _finite(attempt.get("tube", 0.0) * 100.0),
            "tracking_cm": _finite(attempt.get("tracking_m", 0.0) * 100.0),
            "fitness_s": _finite(attempt.get("fitness")),
            "propagation": attempt.get("propagation"),
            "reasons": list(attempt.get("reasons", [])),
            "samples": attempt.get("samples") or [],
        })
    return {"selected": selected_index, "candidates": candidates}


def _write_candidates(path, attempts, selected_index, quiet):
    """Write the candidate comparison, when the caller asked for it.

    Written through a temporary file and renamed, because a reader (the
    studio) polls it while the run is still going: a half-written path list
    is not a comparison. ``selected_index`` is None for a progress write,
    before the rule has chosen.
    """
    if not path:
        return
    payload = _candidate_summary(attempts, selected_index)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)
    if not quiet:
        click.echo(f"  wrote {len(payload['candidates'])} candidates to {path}")


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
                    hub_conn=None, hub_name=None, propagation="plant",
                    auto_mesh=True, polish=False):
    """The options the studio starts with, so a run in the studio is the same run."""
    return {
        "samples": samples,
        "auto_mesh": auto_mesh,
        "accuracy_weight": accuracy_weight,
        "candidates": candidates,
        "timing": dt_mode,
        "validate": validate,
        "export_format": export_format,
        "controller_dt": controller_dt,
        "fast_polish": fast_polish,
        "propagation": propagation,
        "polish": polish,
        "sysid_fit": sysid_fit,
        "hub_conn": hub_conn,
        "hub_name": hub_name,
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
    from overboard.deploy.hub_deploy import HUB_RUNNER, push_trajectory, write_session

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


def _print_summary(output, samples_data, robot_cfg, validate, attempt_passed, tube, files,
                   best_effort=False):
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
        elif best_effort:
            detail = f"envelope {tube * 100:.2f} cm" if tube == tube else "envelope n/a"
            click.secho(
                f"  validation: FAILED - best effort exported ({detail}); "
                "not approved for the hub", fg="red")
        else:
            click.secho("  validation: failed (nothing was written)", fg="red")
    else:
        click.secho(
            "  validation: not run - add --validate before driving this robot",
            fg="yellow",
        )
    for path in files:
        click.echo(f"  also wrote {path}")


@dataclass
class _ScoreContext:
    """Fixed inputs the per-candidate scoring needs.

    These are the same for every candidate in a run, so they are passed as
    one object instead of a dozen arguments. The candidate itself lives in
    its attempt record.
    """

    robot_cfg: Any
    samples: Any
    stop_indices: Any
    waypoint_thresholds: Any
    validate: bool
    use_validation: bool
    digital_twin: Any
    propagation: str
    control_dt: float
    quiet: bool
    envelope_limit: float = DEFAULT_ENVELOPE_LIMIT_M


def _record_candidate(spec, solve_result, cand_idx, batch_len, histories, *,
                      is_rescue, quiet):
    """Turn one solver result into an attempt record and announce it."""
    accuracy_weight = spec["accuracy_weight"]
    guess_style = spec["guess_style"]
    per_segment_dt = spec.get("per_segment_dt", False)
    if not quiet:
        number = (f"Rescue {cand_idx + 1}/{batch_len}" if is_rescue
                  else f"Candidate {cand_idx + 1}/{batch_len}")
        timing = "per-segment" if per_segment_dt else "equal-duration"
        click.echo(f"{number} "
                   f"(guess_style={guess_style}, {timing} dt, aw={accuracy_weight})...")

    samples_data, stats = solve_result
    return {
        "samples": samples_data, "stats": stats,
        "cost": stats.get('final_cost', float('inf')),
        "ok": False, "violation": float("inf"),
        "accuracy_only_failure": False,
        "tube": float("inf"),
        "duration": float(samples_data[-1]['t']) if samples_data else float("inf"),
        "diverged": False,
        "converged": bool(stats.get("converged", False)),
        "guess_style": guess_style,
        "per_segment_dt": per_segment_dt,
        "accuracy_weight": accuracy_weight,
        "iteration_history": (histories[cand_idx]
                              if cand_idx < len(histories) else []),
        "reasons": [],
        # A rescue candidate is insurance: it is only chosen when the
        # candidate set itself produced nothing usable.
        "rescue": is_rescue,
    }


def _score_attempt(attempt_record, ctx):
    """Validate, audit and score one candidate.

    Returns True when the batch must stop here: an unvalidated live run
    keeps the first result for the visualisation, and a validator crash on
    such a run is final (KI-001, KI-002).
    """
    if not ctx.use_validation:
        return False

    samples_data = attempt_record["samples"]
    stats = attempt_record["stats"]
    samples = ctx.samples

    # The sample layout the optimizer used: the sample budget is split
    # across the waypoint segments in proportion to their length, so the
    # waypoints are not k * samples_per_segment apart. The gate and the
    # segment report have to use the optimizer's own table, or a waypoint's
    # threshold is applied to the wrong samples.
    waypoint_bounds = _waypoint_boundaries(
        stats.get("segment_samples")
        or (samples if isinstance(samples, (list, tuple, np.ndarray)) else None))

    try:
        validator_cls, robot_config_cls = _ensure_jax_classes()
        jax_cfg = robot_config_cls(ctx.robot_cfg)
        if ctx.digital_twin is None:
            immrax_val = validator_cls(jax_cfg, control_dt=ctx.control_dt,
                                       propagation=ctx.propagation)
        else:
            immrax_val = validator_cls(jax_cfg, digital_twin=ctx.digital_twin,
                                       control_dt=ctx.control_dt,
                                       propagation=ctx.propagation)

        # Use the uncertainty ranges declared in the config (KI-009).
        cof_range = ctx.robot_cfg.uncertainties["cof"]
        torque_range = ctx.robot_cfg.uncertainties["torque_headroom"]
        backlash_range = ctx.robot_cfg.uncertainties["gear_backlash_m"]
        track_width_range = ctx.robot_cfg.uncertainties["track_width_range"]
        wheel_radius_range = ctx.robot_cfg.uncertainties["wheel_radius_range"]
        inertia_range = ctx.robot_cfg.uncertainties["inertia_kg_m2"]

        if not ctx.quiet:
            click.echo("  Running the Immrax robustness check...")
        immrax_report = immrax_val.validate_trajectory(
            samples_data, cof_range, torque_range, backlash_range,
            track_width_range, wheel_radius_range, inertia_range)
    except Exception as e:
        if not ctx.quiet:
            click.echo(f"  ERROR: Immrax validation failed: {e}")
        import traceback
        traceback.print_exc()
        # A validator crash is not a pass (KI-002).
        attempt_record["reasons"] = [f"Immrax validation error: {e}"]
        return not ctx.validate

    # Attach the envelope to the samples for export/viz, even when it failed,
    # so the operator can see it.
    for i, env in enumerate(immrax_report['reachability']['envelope']):
        if i < len(samples_data):
            samples_data[i]['reachability_envelope'] = env

    # CONTEXT-AWARE SAFETY CHECK (score every candidate on the same scale)
    failed_threshold = False
    unknown_envelope = False
    max_vio_val = 0.0
    max_err_all = 0.0

    for i in range(len(samples_data)):
        # The segment's limit is the one of the waypoint the segment arrives
        # at (see _segment_limit), so a precision waypoint is held to its
        # tight limit on the approach.
        if waypoint_bounds:
            wp_idx = _waypoint_of(waypoint_bounds, i)
        else:
            s_val = samples[0] if isinstance(samples, (list, tuple, np.ndarray)) else samples
            wp_idx = i // max(1, int(s_val))
        local_threshold = _segment_limit(ctx.waypoint_thresholds, wp_idx,
                                        ctx.envelope_limit)

        env = samples_data[i].get('reachability_envelope', {})
        if env:
            err_x = (env['x_max'] - env['x_min']) / 2.0
            err_y = (env['y_max'] - env['y_min']) / 2.0
            err = np.sqrt(err_x**2 + err_y**2)
            max_err_all = max(max_err_all, err)
            if env.get('unknown'):
                # The tube could not be bounded, so there is no accuracy
                # number to accept; that is not an accuracy drop.
                unknown_envelope = True
                failed_threshold = True
                max_vio_val = max(max_vio_val, err)
            elif err > local_threshold:
                failed_threshold = True
                max_vio_val = max(max_vio_val, err)

    attempt_record["tube"] = float(max_err_all)
    attempt_record["diverged"] = bool(immrax_report['reachability'].get('diverged', False))
    # A cut-off propagation reports the width the tube had when it crossed
    # the guard, not the width it would have reached, so the number is a
    # property of the guard and not of the path (KI-049).
    attempt_record["envelope_truncated"] = bool(
        immrax_report['reachability'].get('envelope_truncated',
                                          attempt_record["diverged"]))
    # Which model produced the number. An envelope is the width of a tube,
    # and the tube is a property of the propagation (KI-049).
    attempt_record["propagation"] = str(
        immrax_report.get("propagation", ctx.propagation))
    attempt_record["inclusion"] = dict(
        immrax_report.get("inclusion", {}) or {})
    attempt_record["tracking_m"] = float(immrax_report.get("max_tracking_error_m", 0.0))
    attempt_record["heading_rad"] = float(immrax_report.get("max_heading_error_rad", 0.0))
    attempt_record["ok"] = (attempt_record["converged"] and not failed_threshold
                            and bool(immrax_report['passed']))
    # Which kind of failure this is, for --best-effort. An accuracy excess -
    # the envelope radius or the tracking error - is a number the operator
    # can read and decide about. A force, slip, divergence or unbounded-tube
    # failure is the plan being physically unsound, and showing it would not
    # make it a path.
    motor_violation = float(immrax_report.get("max_motor_violation_N", 0.0))
    traction_violation = float(immrax_report.get("max_traction_violation_N", 0.0))
    slip_violation = max(float(immrax_report.get("max_slip_l_N", 0.0)),
                         float(immrax_report.get("max_slip_r_N", 0.0)))
    speed_violation = float(immrax_report.get("speed_violation_m_s", 0.0))
    hard_failure = bool(
        attempt_record["diverged"]
        or unknown_envelope
        or motor_violation > 1e-3
        or traction_violation > 1e-3
        or slip_violation > 1e-3
        or speed_violation > 1e-6)
    attempt_record["accuracy_only_failure"] = bool(
        attempt_record["converged"] and not attempt_record["ok"]
        and not hard_failure)
    attempt_record["reasons"] = _immrax_failure_reasons(
        immrax_report, failed_threshold, max_vio_val)
    if failed_threshold:
        attempt_record["violation"] = max_vio_val
    else:
        # Tracking is a quantified failure too; rank failed candidates on it
        # instead of falling back to pure cost (KI-020 scoring).
        tracking_error = float(immrax_report.get("max_tracking_error_m", 0.0))
        if tracking_error > 0.005:
            attempt_record["violation"] = tracking_error

    cpu_summary = None
    if ctx.validate:
        # The CPU audit runs on every candidate, not only on the winner, so
        # a candidate that clears Immrax but breaks the force, slip or
        # integration limits cannot block a later candidate that passes
        # every gate.
        cpu_summary = validate_samples(samples_data, ctx.robot_cfg)
        if not cpu_summary["passed"]:
            attempt_record["ok"] = False
            # An integration or force failure is not an accuracy drop.
            attempt_record["accuracy_only_failure"] = False
            attempt_record["reasons"] = list(attempt_record["reasons"]) + \
                _cpu_failure_reasons(cpu_summary)
        # The peak wheel jerk, for the score vector. It is the quantity the
        # accuracy weight trades against, so it belongs beside the other
        # scores (Q10's step 1).
        attempt_record["peak_jerk_m_s3"] = float(
            cpu_summary.get("metrics", {}).get("max_jerk_m_s3", 0.0) or 0.0)

    # 5-part time-equivalent path fitness score (Stage 3: precision > accuracy)
    if waypoint_bounds:
        stop_sample_indices = [
            waypoint_bounds[w] for w in ctx.stop_indices
            if 0 <= w < len(waypoint_bounds)
        ]
    else:
        s_val = samples[0] if isinstance(samples, (list, tuple, np.ndarray)) else samples
        stop_sample_indices = [
            w * s_val for w in ctx.stop_indices
            if 0 <= w * s_val < len(samples_data)
        ]
    fitness_info = compute_path_fitness(
        samples_data,
        ctx.robot_cfg,
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
        samples_data, samples, ctx.waypoint_thresholds, immrax_report,
        cpu_summary, segment_samples=stats.get("segment_samples"),
        default_limit=ctx.envelope_limit)

    if not ctx.quiet:
        score = ("envelope diverged" if attempt_record["diverged"]
                 else f"envelope {attempt_record['tube']*100:.2f} cm")
        if attempt_record["ok"]:
            click.echo(
                f"  Passed! {score}, fitness {attempt_record['fitness']:.3f} s "
                f"(stop spread {attempt_record['stop_spread_m']*100:.2f} cm), "
                f"all segments within thresholds."
            )
        else:
            reason_text = "; ".join(attempt_record["reasons"]) or "validation failed"
            click.echo(f"  FAILED: {reason_text}.")

    # Live-only: keep the first validated result for the visualisation.
    return not ctx.validate


def _run_candidates(specs_to_solve, solve_ctx, score_ctx, *,
                     accuracy_weight, use_validation, quiet, progress=None):
    """Solve the candidate set, then the fallback seeds if nothing usable came out.

    The planner's own candidates come first. If they produced nothing usable
    - no candidate passed the gates, or (without validation) none converged -
    the fallback seeds are tried once, in priority order (KI-048). The rescue
    costs nothing when the plan works.

    ``progress`` is called with the attempts so far, once as each candidate
    leaves the solver and once when its check returns, so a caller that draws
    can show the path while the reachability check is still running.
    """
    attempts = []
    batch = list(specs_to_solve)
    is_fallback = False
    while batch:
        solved, histories = _dispatch_solve(batch, solve_ctx)
        if solve_ctx.polish:
            solved = _polish_candidates(solved, solve_ctx)
        for cand_idx, (spec, solve_result) in enumerate(zip(batch, solved)):
            record = _record_candidate(
                spec, solve_result, cand_idx, len(batch), histories,
                is_rescue=is_fallback, quiet=quiet)
            attempts.append(record)
            if progress is not None:
                progress(attempts)
            if _score_attempt(record, score_ctx):
                break
            if progress is not None:
                progress(attempts)

        # Did the run produce something usable? With validation that means a
        # candidate that passed every gate; without it, a candidate that
        # converged.
        usable = [a for a in attempts
                  if (a.get("ok") if use_validation else a.get("converged"))]
        if usable or is_fallback:
            break
        is_fallback = True
        batch = _fallback_specs(accuracy_weight)
        if not quiet:
            if use_validation:
                click.echo("  No candidate passed the gates; trying the "
                           "fallback seeds.")
            else:
                click.echo("  No candidate converged; trying the fallback seeds.")
    return attempts


def _polish_candidates(solved, ctx):
    """The optional post-solve gradient polish, in this process.

    The polish needs the JAX stack, and the stack is warm here: the
    candidates were solved in worker processes, but the reachability check
    and the polish stay in this one.
    """
    polished = []
    for samples, stats in solved:
        if not samples or not stats.get("converged", False):
            polished.append((samples, stats))
            continue
        if not ctx.quiet:
            click.echo("  Polishing the plan with the reachability gradient...")
        try:
            new_samples, new_stats = ctx.planner.gradient_polish(
                samples, stats, ctx.waypoints,
                stop_indices=ctx.stop_indices,
                waypoint_events=ctx.events,
            )
        except Exception as exc:  # a failed polish is not a failed plan
            if not ctx.quiet:
                click.echo(f"  Polish skipped: {exc}")
            polished.append((samples, stats))
            continue
        polished.append((new_samples, new_stats))
    return polished


def _choose_attempt(all_attempts, *, validate, quiet, samples, pick=None,
                    best_effort=False):
    """Pick the attempt to export and print the candidate table.

    ``pick`` is the operator's choice from a comparison: it exports that
    candidate instead of the one the rule picks. Only a candidate that
    passed every gate may be picked - the rule is not suspended, it is
    overridden by a human who has seen the same numbers.

    ``best_effort`` is the explicit "show me the path" request. When
    nothing passed, it exports the converged candidate whose only failure
    is an accuracy excess, preferring the smallest one. A candidate that
    failed on motor force, wheel slip, an unbounded tube or the integration
    check is never exported this way: that is not an accuracy drop.

    Returns ``(samples_data, stats, passed, reasons, tube, selected_index)``.
    """
    if validate:
        if pick is not None:
            if not 1 <= pick <= len(all_attempts):
                raise click.ClickException(
                    f"Candidate {pick} was not solved: this run has "
                    f"{len(all_attempts)} candidate(s).")
            chosen = all_attempts[pick - 1]
            if not chosen.get("ok"):
                raise click.ClickException(
                    f"Candidate {pick} did not pass the validator, so it cannot "
                    "be exported. Pick a candidate that passed, or solve again.")
            if not quiet:
                click.echo(f"  Exporting candidate {pick} (the operator's pick).")
            return (chosen["samples"], chosen["stats"], True, chosen["reasons"],
                    chosen.get("tube", float("inf")), pick)
        # Never silently export a failed candidate: prefer a passing one with
        # the smallest envelope, otherwise the converged candidate closest to
        # passing. A fallback candidate is insurance: it is only in the list
        # when the planner's own candidates produced nothing that passed
        # (KI-048), so it cannot win over a passing one.
        best, passed_any = _select_attempt(all_attempts)
        if best is None:
            raise click.ClickException(
                f"Solver did not converge on any of the {len(all_attempts)} "
                f"attempts. No trajectory written.")
        if not passed_any and best_effort:
            # The operator asked for the best path even with an accuracy
            # drop. Only an accuracy failure qualifies; the physically
            # unsound candidates stay out of the pool.
            if not best.get("accuracy_only_failure"):
                eligible = [a for a in all_attempts
                            if a.get("accuracy_only_failure")]
                if not eligible:
                    raise click.ClickException(
                        "No candidate failed on accuracy alone, so there is no "
                        "best-effort path to export: the solve did not converge, "
                        "or the plan breaks the force, slip, integration or "
                        "envelope-bound check. No trajectory written.")
                best = min(eligible, key=lambda a: (a["violation"], a["cost"]))
        selected_index = next(k for k, a in enumerate(all_attempts, 1) if a is best)
        if not quiet and len(all_attempts) > 1:
            for k, a in enumerate(all_attempts, 1):
                if a.get("diverged"):
                    score = "envelope diverged"
                elif np.isfinite(a.get("tube", float("inf"))):
                    score = f"envelope {a['tube']*100:.2f} cm"
                else:
                    score = "no envelope"
                state = "ok" if a["ok"] else "failed"
                # Name the mode by what it constrains, and show how the time
                # was split, so a uniform win is self-explanatory (KI-022).
                timing = "per-seg" if a.get("per_segment_dt") else "equal-dur"
                label = (f"Candidate {k} ({a.get('guess_style', 'baseline')}, "
                         f"{timing})")
                click.echo(f"  {label}: "
                           f"{a.get('duration', float('inf')):6.3f} s, "
                           f"{score} ({state})")
                seg_dt = (a.get("stats") or {}).get("segment_dt")
                if seg_dt and len(seg_dt) > 1:
                    seg_samples_list = (a.get("stats") or {}).get("segment_samples") or (
                        samples if isinstance(samples, (list, tuple))
                        else [samples] * len(seg_dt))
                    seg_times = [d * s for d, s in zip(seg_dt, seg_samples_list)]
                    spread = "  segment times: " + " ".join(
                        "%.2f" % t for t in seg_times) + " s"
                    click.echo(f"    {spread}")
            click.echo(f"  Selected candidate {selected_index}.")
        return (best["samples"], best["stats"], passed_any, best["reasons"],
                best.get("tube", float("inf")), selected_index)

    # Without validation the first candidate is the result. After a rescue
    # round it is the last candidate that converged, not merely the last one
    # tried.
    converged = [a for a in all_attempts if a.get("converged")]
    chosen = converged[-1] if converged else (
        all_attempts[-1] if all_attempts else None)
    if chosen is None:
        raise click.ClickException(
            f"Solver did not converge on any of the {len(all_attempts)} "
            f"attempts. No trajectory written.")
    return (chosen["samples"], chosen["stats"], False, chosen["reasons"],
            chosen.get("tube", float("inf")), 1)


def _write_outputs(output, samples_data, stats, config_data, robot_cfg, *,
                   benchmark, validate, config, waypoints, export_format,
                   controller_dt):
    """Write the trajectory and the files the flags asked for.

    Returns every path written, the trajectory first.
    """
    result = {
        "name": os.path.basename(output).split('.')[0],
        "version": 3,
        "trajectory": {
            "config": config_data.get("robot", config_data.get(
                "config", config_data)),
            "samples": samples_data
        }
    }
    # A best-effort export carries its own verdict, so the file cannot be
    # mistaken for an approved one when it is read later.
    if stats.get("validation"):
        result["validation"] = stats["validation"]
    with open(output, 'w') as f:
        json.dump(result, f, indent=1)

    written = [output]

    if benchmark:
        quality_metrics = compute_metrics(samples_data, robot_cfg)
        stats_to_write = {k: v for k, v in stats.items() if k != "immrax_report"}
        stats_to_write["quality_metrics"] = quality_metrics
        stats_to_write["robot_config"] = {
            "mass": robot_cfg.mass, "inertia": robot_cfg.inertia,
            "track_width": robot_cfg.track_width,
            "wheel_radius": robot_cfg.wheel_radius,
            "v_max_rad_s": robot_cfg.v_max_rad_s,
            "t_max_nm": robot_cfg.t_max_nm, "gearing": robot_cfg.gearing,
            "cof": robot_cfg.cof
        }
        stats_output = os.path.splitext(output)[0] + '_stats.json'
        with open(stats_output, 'w') as f:
            json.dump(stats_to_write, f, indent=2)
        written.append(stats_output)

    if validate:
        validate_trajectory(output, config, waypoints=waypoints)

    if export_format == 'controller':
        ctrl_output = os.path.splitext(output)[0] + '_controller.json'
        write_controller_file(output, ctrl_output, target_dt=controller_dt,
                              track_width=robot_cfg.track_width)
        written.append(ctrl_output)

    if export_format in ('python', 'python-compact'):
        py_output = os.path.splitext(output)[0] + '.py'
        write_python_file(output, py_output, compact=(export_format == 'python-compact'))
        written.append(py_output)

    return written


def _render(output, samples_data, wps, optimizer, all_attempts, selected_index, *,
            plot, animate, iteration_history, quiet):
    """Draw the requested views and write the iteration history."""
    if plot:
        from overboard.viz.plotter import plot_trajectory, plot_reachability_envelope
        if not quiet:
            click.echo("Generating plot...")
            click.echo("  Generating standard trajectory plot...")
        plot_trajectory(samples_data, waypoints=wps,
                        title=f"Trajectory: {os.path.basename(output)}")

        # Also show the reachability envelope if Immrax data is available.
        if HAS_JAX and any('reachability_envelope' in s for s in samples_data):
            if not quiet:
                click.echo("  Generating reachability envelope plot...")
            plot_reachability_envelope(
                samples_data, waypoints=wps,
                title=f"Reachability Envelope: {os.path.basename(output)}",
                block=True)

    if animate:
        from overboard.viz.animated_plotter import animate_trajectory
        animate_trajectory(samples_data, waypoints=wps,
                           title=f"Trajectory: {os.path.basename(output)}")

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
@click.option('--preset', type=click.Choice(['fast', 'balanced', 'safe'], case_sensitive=False),
              default=None,
              help='Named starting point. fast: 1 candidate, no validation. '
                   'balanced: the defaults. safe: 3 candidates, --validate with the '
                   'closed-loop tick propagation, and a controller export. '
                   'Options you set yourself always win.')

# --- Solver Parameters ---
@click.option('-n', '--samples', default=10, type=click.IntRange(2, 200),
              help='Samples per segment.')
@click.option('-a', '--accuracy-weight', default=0.0, type=click.FloatRange(min=0.0),
              help='Smoothness/accuracy weight (0 = pure time-optimal).')
@click.option('--force-weight', default=0.0, type=click.FloatRange(min=0.0),
              help='Soft wheel-force and traction-utilization penalty weight in the NLP '
                   'objective.')
@click.option('--candidates', default=2, type=click.IntRange(min=1),
              help='Candidates to solve and compare. The planner produces the three '
                   'frontier candidates (sprint, sweeper, exploratory); this says how '
                   'many of them are solved. Validation picks a passing one: the '
                   'fastest. Without --validate only the first candidate is solved. '
                   'Use 1 for a single solve.')
@click.option('--stop-waypoints', default=None, type=str,
              help='Comma-separated waypoint indices where robot must stop (e.g., "2,5,7").')
@click.option('--events', default=None, type=str,
              help='Comma-separated waypoint:event pairs (e.g., "2:lower_arm,5:release").')

# --- Optimizer Strategy ---
@click.option('--no-parallel', is_flag=True,
              help='Solve the candidates in this process instead of in worker processes.')
@click.option('--workers', default=8, type=int,
              help='Number of worker processes used to solve candidates in parallel.')
@click.option('--fast-polish', is_flag=True,
              help='Solve large missions faster. The warm-start solve can crawl through thousands of '
                   'IPOPT iterations; this gives it an iteration budget and finishes from that point '
                   'with a cold pass. Missions that solve inside the budget return the same '
                   'trajectory as before.')
@click.option('--polish', is_flag=True,
              help='Optional post-solve gradient polish: the reachability rollout is '
                   'differentiated, and the plan is re-solved with the samples the '
                   'gradient points at weighted (requires JAX; off by default).')

# --- Export & Validation ---
@click.option('--validate/--no-validate', default=False,
              help='Run validation and block export on failure (requires JAX/Immrax).')
@click.option('--best-effort', is_flag=True,
              help='With --validate: export the best converging candidate when the '
                   'only failure is an accuracy excess (envelope or tracking), and '
                   'mark the trajectory as failed. Motor force, wheel slip, a '
                   'diverged tube or a non-converged solve still block export.')
@click.option('--envelope-limit', default=DEFAULT_ENVELOPE_LIMIT_M,
              type=click.FloatRange(min=0.001, max=MAX_ENVELOPE_LIMIT_M),
              help='How far the reachable set may spread around the plan at any '
                   'sample, in metres (default 0.02 = 2 cm). A waypoint '
                   'error_threshold overrides it. The accuracy the mission '
                   'needs, not a number to make a run pass: the value in force '
                   'is written into the validation report and the trajectory. '
                   'Capped at 0.05, where the tube is cut off.')
@click.option('--sysid-fit', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Optional sysid_fit.json digital-twin fit (requires --validate).')
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
@click.option('--pick-candidate', default=None, type=click.IntRange(min=1),
              help='Export this candidate instead of the fastest one that passed. It has to '
                   'have passed every gate: the rule is not suspended, it is overridden by a '
                   'human who has seen the comparison. Needs --validate.')
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
@click.option('--iteration-history', type=click.Path(dir_okay=False), default=None,
              help='Write the solver\'s iteration history to this JSON file, for the '
                   'studio to plot how the trajectory converged.')
@click.option('--candidates-out', type=click.Path(dir_okay=False), default=None,
              help='Write every solved candidate to this JSON file - its seed, verdict, '
                   'score and samples - so a tool can draw the comparison. Diagnostics: it '
                   'is written even when the export gate blocks the run.')


# --- Miscellaneous ---
@click.option('-v', '--verbose', is_flag=True, help='Show solver internals.')
@click.option('--quiet', '-q', is_flag=True, help='Suppress verbose output.')
@click.pass_context
def main(ctx, config, waypoints, output, gui, gui_port, doctor, preset,
         samples, accuracy_weight, force_weight, candidates, pick_candidate,
         stop_waypoints, events,
         no_parallel, workers, fast_polish, polish, validate, best_effort,
         envelope_limit, sysid_fit, export_format, controller_dt, control_dt,
         propagation, benchmark,
         push_hub, hub_conn, hub_name, hub_log,
         plot, animate, iteration_history, candidates_out,
         verbose, quiet):
    """
    FLL Trajectory Optimizer CLI.
    Generates time-optimal trajectories for Lego robots.

    \b
    Examples:
      python -m overboard.main -w examples/example_straight.json
      python -m overboard.main -w mission.json --preset safe -o mission.traj
      python -m overboard.main --gui
      python -m overboard.main --doctor
    """
    if quiet and verbose:
        raise click.ClickException("--quiet and --verbose cannot be combined.")
    if polish and not HAS_JAX:
        raise click.ClickException(f"--polish needs JAX and Immrax. {JAX_HINT}")

    if doctor:
        from overboard.diagnostics import run_doctor
        raise SystemExit(run_doctor(config_path=config))

    if waypoints is None and not gui:
        raise click.UsageError(
            "Missing option '-w' / '--waypoints'. "
            "Example: python -m overboard.main -w examples/example_straight.json"
        )

    values = {"samples": samples, "accuracy_weight": accuracy_weight,
              "candidates": candidates, "validate": validate,
              "export_format": export_format, "propagation": propagation}
    if preset:
        values = _apply_preset(ctx, preset, values)
        samples = values["samples"]
        accuracy_weight = values["accuracy_weight"]
        candidates = values["candidates"]
        validate = values["validate"]
        export_format = values["export_format"]
        propagation = values["propagation"]

    if validate and not HAS_JAX:
        raise click.ClickException(
            f"--validate needs JAX and Immrax. {JAX_HINT}"
        )
    if pick_candidate is not None and not validate:
        raise click.ClickException(
            "--pick-candidate needs --validate: only a checked candidate may be "
            "exported.")
    if best_effort and not validate:
        raise click.ClickException(
            "--best-effort needs --validate: it exports a failed verdict, so "
            "there has to be one.")

    # The timing model is the planner's: one interval per waypoint segment.
    # The flags that used to choose it were removed, so the label and the
    # studio option say what the plan actually does.
    dt_mode = TIMING_MODEL

    if gui:
        if plot or animate:
            raise click.ClickException(
                "--gui cannot be combined with --plot or --animate."
            )
        # The studio's automatic mesh is its default; an explicit -n or a
        # preset's sample count has to reach the solve instead of being
        # overridden by it.
        source = ctx.get_parameter_source("samples")
        samples_set = source is not None and source.name not in ("DEFAULT", "DEFAULT_MAP")
        from overboard.gui_server import run_studio
        run_studio(config_path=config, waypoints_path=waypoints,
                   port=gui_port, presets=PRESETS, auto_solve=False,
                   options=_studio_options(samples, accuracy_weight, candidates,
                                           dt_mode, validate, export_format,
                                           controller_dt, fast_polish, sysid_fit,
                                           hub_conn, hub_name, propagation,
                                           auto_mesh=not samples_set and preset is None,
                                           polish=polish))
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
        if not validate:
            raise click.ClickException(
                "--sysid-fit requires --validate so the fitted model is used."
            )
        if not HAS_JAX:
            raise click.ClickException(
                f"--sysid-fit needs JAX and Immrax. {JAX_HINT}"
            )
        from overboard.sysid.digital_twin import load_sysid_fit
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
                    if digital_twin.actuator_tau_s() > 0.0:
                        # The five-state plant is in use, and it is the only
                        # plant that models the grip.
                        click.echo(
                            "  Brake slip is applied: the plant slows while braking."
                        )
                    else:
                        # The slip is loaded, but the plant is the kinematic
                        # one, which has no braking state to apply it to.
                        click.echo(
                            "  Brake slip is loaded but not applied: the fit has "
                            "no actuator time constant, so the kinematic plant "
                            "is in use."
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
        from overboard.models.drivetrain import compare_declared_motor
        motor_check = compare_declared_motor(
            robot_cfg.t_max_nm, robot_cfg.v_max_rad_s,
            robot_cfg.wheel_radius, robot_cfg.gearing)
        if motor_check.get("suspicious"):
            click.secho(f"  Motor: {motor_check['note']}.", fg="yellow")

    # Choose optimizer
    if not quiet:
        click.echo("Using the 3-Pillar Unified Trajectory Engine "
                   "(CasADi/IPOPT backend)")
    optimizer = TrajectoryOptimizer(robot_cfg)

    wps, waypoint_events, waypoint_thresholds, json_stop_indices = _load_waypoints(waypoints)
    stop_indices = _parse_stop_waypoints(stop_waypoints, len(wps))
    waypoint_events.update(_parse_events(events, len(wps)))

    # sorted() so the constraint order (and therefore the IPOPT iterate) is
    # reproducible; set() iteration order changes the NLP ordering.
    all_stop_indices = sorted(set(stop_indices + json_stop_indices))

    if not quiet:
        # Per-segment dt lets each segment have its own duration, and it is
        # the general model: the equal-duration mode forces every segment to
        # take the SAME time, which a path with unequal segment lengths pays
        # for on every segment (KI-022).
        dt_label = "per-segment dt"
        click.echo(
            f"Solving {len(wps)} waypoints (samples={samples}, "
            f"accuracy_weight={accuracy_weight}, candidates={max(1, candidates)}, "
            f"{dt_label})..."
        )

    capture_iterations = bool(iteration_history)

    # --- CANDIDATE LOOP (validate and select; KI-020) ---
    # The planner owns the candidate set: the homotopy triad, each with its
    # own accuracy weight. The CLI does not build candidates of its own.
    from overboard.core.unified_planner import UnifiedPathEngine
    planner = UnifiedPathEngine(robot_cfg)

    # The candidates are independent, so they are solved first and scored
    # afterwards. Without validation only the first candidate is solved.
    use_validation = HAS_JAX and validate

    mesh_samples = planner.compute_invariant_mesh_samples(wps)
    if ctx.get_parameter_source('samples') == click.core.ParameterSource.DEFAULT and preset is None:
        samples = mesh_samples

    weight = (float(accuracy_weight)
              if ctx.get_parameter_source('accuracy_weight')
              != click.core.ParameterSource.DEFAULT else None)
    specs_to_solve = planner.candidate_specs(
        accuracy_weight=weight,
        count=candidates if candidates is not None else None)
    if not use_validation:
        specs_to_solve = specs_to_solve[:1]

    solve_ctx = _SolveContext(
        robot_cfg=robot_cfg,
        waypoints=wps,
        samples=samples,
        stop_indices=all_stop_indices,
        events=waypoint_events,
        verbose=verbose,
        quiet=quiet,
        capture_iterations=capture_iterations,
        force_weight=force_weight,
        fast_polish=fast_polish,
        workers=workers,
        no_parallel=no_parallel,
        planner=planner,
        polish=polish,
        optimizer=optimizer,
    )

    score_ctx = _ScoreContext(
        robot_cfg=robot_cfg,
        samples=samples,
        stop_indices=all_stop_indices,
        waypoint_thresholds=waypoint_thresholds,
        validate=validate,
        use_validation=use_validation,
        digital_twin=digital_twin,
        propagation=propagation,
        control_dt=control_dt,
        quiet=quiet,
        envelope_limit=float(envelope_limit),
    )
    all_attempts = _run_candidates(
        specs_to_solve, solve_ctx, score_ctx,
        accuracy_weight=accuracy_weight,
        use_validation=use_validation, quiet=quiet,
        progress=(lambda made: _write_candidates(candidates_out, made, None, True))
        if candidates_out else None)

    (samples_data, stats, attempt_passed, best_reasons, selected_tube,
     selected_index) = _choose_attempt(
        all_attempts, validate=validate, quiet=quiet, samples=samples,
        pick=pick_candidate, best_effort=best_effort)

    # The operator asked to see the path even though it missed the accuracy
    # gates. The verdict is still a failure, and it travels with the file.
    best_effort_export = bool(validate and best_effort and not attempt_passed)
    if best_effort_export:
        excess = _mark_best_effort(stats, all_attempts, selected_index)
        click.secho(
            "  BEST EFFORT: validation failed ("
            + ("; ".join(best_reasons) or "see the candidate table")
            + (f"; up to {excess * 100:.2f} cm outside the local limit"
               if excess > 0 else "")
            + "). The path is written for inspection and is not approved "
            "for the hub.", fg="red", bold=True)

    if validate and attempt_passed:
        # The pass depends on the limit, so the file records the limit it
        # passed; a trajectory read later is not silent about its budget.
        stats["validation"] = {"passed": True,
                               "envelope_limit_m": float(envelope_limit)}

    if use_validation:
        # The studio reads this to show which part of the path failed.
        _emit_validation_report(all_attempts, selected_index,
                                propagation=propagation,
                                best_effort=best_effort_export,
                                envelope_limit=envelope_limit)
    # Every candidate, paths included, for a tool that draws the comparison.
    # Diagnostics, so it is written before the export gate: a failed run is
    # exactly when the operator wants to see what the candidates did.
    _write_candidates(candidates_out, all_attempts, selected_index, quiet)

    # --- EXPORT GATE (KI-001 / KI-002) ---
    block_reason = _export_block_reason(stats, validate, attempt_passed,
                                        best_effort=best_effort_export)
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

    written = _write_outputs(
        output, samples_data, stats, config_data, robot_cfg,
        benchmark=benchmark, validate=validate, config=config,
        waypoints=wps, export_format=export_format, controller_dt=controller_dt)

    if not quiet:
        _print_summary(output, samples_data, robot_cfg, validate, attempt_passed,
                       selected_tube, written[1:], best_effort=best_effort_export)

    if push_hub:
        try:
            robot_cfg.sync_pybricks_robot_config()
        except Exception:
            pass
        _push_to_hub(output, hub_conn, hub_name, hub_log, quiet)

    _render(output, samples_data, wps, optimizer, all_attempts, selected_index,
            plot=plot, animate=animate, iteration_history=iteration_history,
            quiet=quiet)


cli = main

if __name__ == '__main__':
    main()
