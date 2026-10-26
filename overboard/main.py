"""
FLL Trajectory Optimizer CLI.

Generates time-optimal trajectories for LEGO differential-drive robots.
"""

import importlib.util
import json
import math
import os
import time
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

from overboard.paths import EXAMPLES_DIR, FROZEN, HUB_MISSION_MODULE, HUB_RUNNER
from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer
from overboard.verification.validator import (
    validate_samples,
    check_final_waypoint,
    compute_metrics,
    compute_path_fitness,
    SCORE_VERSION,
    SCORING_PROFILES,
    DEFAULT_SCORING_PROFILE,
    DEFAULT_JERK_TARGET_M_S3,
    DEFAULT_JERK_MAX_M_S3,
    TRACKING_GATE_M,
    INTEGRATION_GATE_M,
    ratio_score,
    duration_score,
    jerk_score,
    spread_score,
    weighted_score,
)
from overboard.deploy.export import write_controller_file, write_python_file

DEFAULT_CONFIG = "examples/fll_choreo.json"
if FROZEN:
    # The desktop build runs from wherever the user is, so the shipped example
    # robot has to be named by its absolute path.
    DEFAULT_CONFIG = str(EXAMPLES_DIR / "fll_choreo.json")
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


def _attempt_accepted(attempt, accept_score):
    """True when a failed candidate is eligible under an explicit --accept-score.

    A structural failure (non-convergence, a crashed or incomplete validator,
    a diverged or unbounded tube, a missing final waypoint) is never accepted.
    """
    return bool(
        accept_score is not None
        and attempt.get("converged")
        and not attempt.get("structural_failure")
        and not attempt.get("ok")
        and attempt.get("score") is not None
        and float(attempt["score"]) >= float(accept_score)
    )


def _attempt_eligible(attempt, accept_score):
    """Ordinary passes are eligible; a failed candidate only when accepted."""
    if not attempt.get("converged") or attempt.get("structural_failure"):
        return False
    return bool(attempt.get("ok") or _attempt_accepted(attempt, accept_score))


def _select_attempt_scored(attempts, accept_score=None):
    """Choose the candidate the decision score picks.

    Among the eligible candidates: the highest score, then an ordinary pass
    before an accepted one, then the higher accuracy and physical scores, the
    shorter duration, and finally the lower candidate index. When nothing is
    eligible the least-bad converged candidate is returned with ``False``, so
    the diagnostics survive and the export gate stays closed.
    """
    candidates = [a for a in attempts if a.get("converged", False)]
    if not candidates:
        return None, False
    eligible = [a for a in candidates if _attempt_eligible(a, accept_score)]
    if eligible:
        best = min(eligible, key=lambda a: (
            -float(a.get("score") or 0.0),
            0 if a.get("ok") else 1,
            -float((a.get("categories") or {}).get("accuracy") or 0.0),
            -float((a.get("categories") or {}).get("physical") or 0.0),
            a.get("duration", a.get("cost", float("inf"))),
        ))
        # ``True`` means the chosen candidate passed the ordinary validator;
        # an accepted failure is exported with ``False`` and its own verdict.
        return best, bool(best.get("ok"))
    return min(candidates, key=lambda a: (a.get("violation", float("inf")),
                                          a.get("cost", float("inf")))), False



# The timing model the planner uses: one interval per waypoint segment.
TIMING_MODEL = "per-segment"

# The fallback seeds the rescue used to try when the planner's own candidates
# produced nothing usable (KI-048) are not run any more. On the 32-mission
# suite they rescued 0 of 22 failing missions, and 95 of their 97 failures were
# the envelope, which a seed cannot move. The specs are still in git history.

# Thread limits handed to the candidate worker processes. Several candidates
# solve at the same time, so each one keeps a single compute thread.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


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


def _export_block_reason(stats, validate, attempt_passed, best_effort=False,
                         accepted=False):
    """Return why the result must not be written, or None if it may be.

    ``best_effort`` is the operator's explicit acceptance of an accuracy
    excess: the path is written and marked, because a path that misses the
    envelope is still a plan to look at. It is not a pass - the file says
    so, the summary says so, and the hub push is refused. It does not cover a
    plan that is physically infeasible: motor force, wheel slip, a diverged
    tube and a non-converged solve are refused before the gate
    (`_choose_attempt`).

    ``accepted`` is the score contract's explicit acceptance (``--accept-score``):
    the candidate failed the ordinary validator but scored at or above the
    threshold and is structurally valid, so it may be exported and pushed.
    """
    if not stats.get("converged", False):
        return "solver did not converge"
    if validate and not attempt_passed and not best_effort and not accepted:
        return "validation failed"
    return None


def _build_validation_record(*, passed, accepted, best_effort, attempt,
                             scoring, envelope_limit, envelope_limit_per_metre,
                             trajectory_hash):
    """The validation record every artifact carries (score contract).

    One record is built for the `.traj` root and the `_stats.json`, so the
    two cannot disagree. ``hub_approved`` is the only thing the hub push
    reads: an ordinary pass or an explicitly accepted score, never a
    best-effort or an unvalidated file.
    """
    record = {
        "passed": bool(passed),
        "accepted": bool(accepted),
        "best_effort": bool(best_effort),
        "hub_approved": bool((passed or accepted) and not best_effort),
        "score": _finite(attempt.get("score")) if attempt else None,
        "score_version": SCORE_VERSION,
        "profile": scoring.get("profile"),
        "weights": scoring.get("weights"),
        "categories": attempt.get("categories") if attempt else None,
        "target_time_s": _finite(scoring.get("target_time_s")),
        "max_time_s": _finite(scoring.get("max_time_s")),
        "accept_score": scoring.get("accept_score"),
        "trajectory_sha256": trajectory_hash,
        "envelope_limit_m": float(envelope_limit),
        "envelope_limit_per_metre": float(envelope_limit_per_metre),
        # The field-derived per-sample limit and the field's placement
        # uncertainty, alongside the run-wide limits (--envelope-from-field).
        "envelope_limit_field_m": (attempt or {}).get("envelope_limit_field_m"),
        "placement_uncertainty_m": (attempt or {}).get("placement_uncertainty_m"),
        "reasons": list(attempt.get("reasons", [])) if attempt else [],
        "segments": list(attempt.get("segments", [])) if attempt else [],
    }
    segments = record["segments"]
    record["max_excess_m"] = max(
        (float(segment.get("excess_m", 0.0)) for segment in segments),
        default=0.0)
    return record


def _immrax_failure_reasons(report, failed_threshold=False, max_violation=0.0):
    """Explain an Immrax failure in the units a user cares about."""
    reasons = []
    if report.get("error"):
        # A candidate that never produced a trajectory says so; the gate
        # numbers below would describe that trajectory (KI-056).
        return [str(report["error"])]
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
# Extra allowance per metre travelled. 0.0 keeps the gate absolute (the tube
# widens with distance, so that is a duration budget in disguise); a positive
# slope prices distance instead (KI-054).
DEFAULT_ENVELOPE_LIMIT_PER_METRE = 0.0
# The field's per-sample limit (--envelope-from-field) is the clearance
# clamped to this band: 5 mm matters, and 50 mm is where the propagation
# cuts the tube off (MAX_ENVELOPE_LIMIT_M), so no looser limit can be
# measured. A NaN clearance is "no field limit" and is never clamped.
MIN_FIELD_ENVELOPE_LIMIT_M = 0.005


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
    ``index + 1``. A ``None`` default is passed through, so a caller can ask
    whether a waypoint declared a limit at all.
    """
    if index + 1 in thresholds:
        return float(thresholds[index + 1])
    if index in thresholds:
        return float(thresholds[index])
    return None if default is None else float(default)


def _travelled_m(samples_data):
    """Cumulative distance along the plan, one entry per sample, in metres."""
    travelled = [0.0]
    for a, b in zip(samples_data, samples_data[1:]):
        travelled.append(travelled[-1] + math.hypot(
            float(b.get("x", 0.0)) - float(a.get("x", 0.0)),
            float(b.get("y", 0.0)) - float(a.get("y", 0.0))))
    return travelled


def _envelope_limit_at(base, per_metre, travelled):
    """The limit in force ``travelled`` metres into the path (KI-054)."""
    return float(base) + float(per_metre) * float(travelled)


def _local_limit(thresholds, index, travelled_at, default, per_metre,
                 field_limits=None, sample=None):
    """The envelope limit in force at one sample.

    A waypoint's ``error_threshold`` / ``precision_mode`` wins locally (the
    rule of ``_segment_limit``). Otherwise the field's per-sample limit
    applies where the field has one (``--envelope-from-field``); a sample
    without one falls back to the run-wide limit and its per-metre allowance.
    """
    declared = _segment_limit(thresholds, index, default=None)
    if declared is not None:
        return _envelope_limit_at(declared, per_metre, travelled_at)
    if (field_limits is not None and sample is not None
            and sample < len(field_limits) and field_limits[sample] is not None):
        return float(field_limits[sample])
    return _envelope_limit_at(default, per_metre, travelled_at)


def _field_envelope_limits(field, robot_cfg, samples_data):
    """The field's per-sample envelope limits (``--envelope-from-field``).

    One limit per sample: the field's clearance (the robot disc resolved as
    ``chassis_radius_m`` -> track/2 -> numeric) clamped to 5-50 mm. A NaN
    clearance - a sample without usable geometry - means "no field limit
    here" and falls back to the flat run-wide limit; NaN is never clamped.
    """
    if field is None:
        return None
    limits = []
    for clearance in field.sample_clearances(samples_data, robot_cfg):
        try:
            value = float(clearance)
        except (TypeError, ValueError):
            limits.append(None)
            continue
        if math.isnan(value):
            limits.append(None)
        else:
            limits.append(min(max(value, MIN_FIELD_ENVELOPE_LIMIT_M),
                              MAX_ENVELOPE_LIMIT_M))
    return limits


def _field_limit_summary(field, field_limits):
    """What the field's per-sample limits were over one candidate."""
    if field is None:
        return None
    finite = [value for value in (field_limits or []) if value is not None]
    return {
        "season": str(field.season),
        "min_m": min(finite) if finite else None,
        "max_m": max(finite) if finite else None,
    }


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
                    segment_samples=None, default_limit=DEFAULT_ENVELOPE_LIMIT_M,
                    default_limit_per_metre=DEFAULT_ENVELOPE_LIMIT_PER_METRE,
                    field_limits=None):
    """Per-segment validation detail: envelope radius, local limit, violations.

    The gate checks sample i against the threshold of the waypoint whose
    segment i is in, so a segment is judged by the threshold of the waypoint
    it starts from. The report says the same thing, per segment, instead of
    one reason string. The drift is the forward-integration error inside the
    segment: it shows where the path drifts away from the plan. The local
    limit is resolved by ``_local_limit``, so the report cannot disagree
    with the gate about a waypoint override or a field limit.
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
    travelled = _travelled_m(samples_data)
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
        radius = 0.0
        # The limit grows along the segment, so the verdict belongs to the
        # sample with the biggest excess, not to the biggest radius.
        limit = _local_limit(thresholds, index, travelled[start], default_limit,
                             default_limit_per_metre, field_limits, start)
        worst_excess = None
        for i in range(start, stop):
            env = envelopes[i] if i < len(envelopes) else None
            if not env:
                continue
            err = math.hypot((env["x_max"] - env["x_min"]) / 2.0,
                             (env["y_max"] - env["y_min"]) / 2.0)
            local = _local_limit(thresholds, index, travelled[i], default_limit,
                                 default_limit_per_metre, field_limits, i)
            radius = max(radius, err)
            if worst_excess is None or err - local > worst_excess:
                worst_excess = err - local
                limit = local
        violations = sum(1 for i in range(start, stop) if i in violating)
        slip_points = sum(1 for i in range(start, stop) if i in slip)
        segments.append({
            "from": index,
            "to": index + 1,
            "limit_m": limit,
            "envelope_m": radius,
            "excess_m": max(0.0, worst_excess or 0.0),
            "drift_m": max(drift[start:stop]) if stop > start else 0.0,
            "violations": violations,
            "slip_points": slip_points,
            "ok": (worst_excess or 0.0) <= 0.0 and violations == 0 and slip_points == 0,
        })
    return segments


def _emit_validation_report(attempts, selected_index, propagation=None,
                            best_effort=False, envelope_limit=None,
                            envelope_limit_per_metre=None, accept_score=None,
                            profile=None, accepted=False, hub_approved=False,
                            envelope_limit_field=None, placement_uncertainty_m=None):
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
        excess = max((float(segment.get("excess_m", 0.0))
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
            # Peak utilisation of the motor limit and of the traction limit,
            # 1.0 = at the edge. Together with the envelope and tracking
            # numbers these are the margins the plan carries (Q14).
            "peak_force_util": _finite(attempt.get("peak_force_util", 0.0)),
            "peak_traction_util": _finite(attempt.get("peak_traction_util", 0.0)),
            # The decision score and what it is made of (score contract).
            "score": _finite(attempt.get("score")),
            "score_version": attempt.get("score_version"),
            "categories": attempt.get("categories"),
            "ratios": attempt.get("ratios"),
            "eligibility": bool(_attempt_eligible(attempt, accept_score)),
            "accepted": bool(_attempt_accepted(attempt, accept_score)),
            "structural_failure": bool(attempt.get("structural_failure")),
            "ok": bool(attempt.get("ok")),
            "converged": bool(attempt.get("converged")),
            "validation_skipped": bool(attempt.get("validation_skipped", False)),
            "reasons": list(attempt.get("reasons", [])),
            "segments": segments,
        })
    payload = {"selected": selected_index, "candidates": candidates,
               "propagation": propagation,
               # The run-wide default the gate applied. A verdict depends on
               # it, so it is never left implicit (KI-054).
               "envelope_limit_m": _finite(envelope_limit),
               "envelope_limit_per_metre": _finite(envelope_limit_per_metre),
               # The field-derived per-sample limit of the selected candidate
               # and the field's placement uncertainty (--envelope-from-field).
               "envelope_limit_field_m": envelope_limit_field,
               "placement_uncertainty_m": _finite(placement_uncertainty_m),
               # True when a failed candidate was exported anyway, so the
               # studio can say "failed, shown" instead of "not checked".
               "best_effort": bool(best_effort),
               # The run's decision-score contract (score_version 1).
               "score_version": SCORE_VERSION,
               "profile": profile,
               "accept_score": accept_score,
               "accepted": bool(accepted),
               "hub_approved": bool(hub_approved)}
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
            "validation_skipped": bool(attempt.get("validation_skipped", False)),
            # The decision score and its parts, so the comparison can be
            # re-run offline (score contract).
            "score": _finite(attempt.get("score")),
            "categories": attempt.get("categories"),
            "duration_s": _finite(attempt.get("duration")),
            "envelope_cm": _finite(attempt.get("tube", 0.0) * 100.0),
            "tracking_cm": _finite(attempt.get("tracking_m", 0.0) * 100.0),
            "fitness_s": _finite(attempt.get("fitness")),
            "propagation": attempt.get("propagation"),
            "reasons": list(attempt.get("reasons", [])),
            "samples": attempt.get("samples") or [],
        })
    rescues = [entry for entry in candidates if entry["rescue"]]
    return {"selected": selected_index, "candidates": candidates,
            # How the insurance did. A rescue that never pays is a cost with
            # no number attached until someone counts it.
            "rescue": {"tried": len(rescues),
                       "passed": sum(1 for entry in rescues if entry["ok"])}}


def _swap_in(temporary, target, attempts=20, delay=0.05):
    """Rename a finished temporary file over its target.

    Windows refuses the rename while another process holds the target open,
    and the studio polls these files - so one refused rename is not a failed
    write, it is a write that arrived while someone was reading.
    """
    for attempt in range(attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


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
    _swap_in(temporary, path)
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


def _config_digital_twin(config_data):
    """The digital twin a robot config declares, or None.

    A config may carry ``"digital_twin": {...}`` with documented limits
    (the DriveBase acceleration ceiling) that need no calibration.
    """
    payload = (config_data.get("digital_twin")
               if isinstance(config_data, dict) else None)
    if not isinstance(payload, dict):
        return None
    from overboard.sysid.digital_twin import DigitalTwinConfig
    try:
        return DigitalTwinConfig.from_dict(payload)
    except ValueError as exc:
        raise click.ClickException(
            f"Invalid digital_twin section in the robot config: {exc}"
        ) from exc


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


def _scoring_number(value, label, path):
    """Read one finite mission scoring number, or explain what is wrong."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise click.ClickException(
            f"Mission '{path}': scoring '{label}' must be a number, got {value!r}."
        )
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise click.ClickException(
            f"Mission '{path}': scoring '{label}' must be a finite number."
        )
    return value


def _parse_mission_scoring(raw, path):
    """Validate the mission's optional scoring section.

    ``target_time_s`` and ``max_time_s`` set the duration budget; when both
    are declared they must satisfy 0 < target < max. The jerk pair defaults
    to 10/20 m/s^3 and must satisfy target < max.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise click.ClickException(
            f"Mission '{path}': 'scoring' must be an object, got {type(raw).__name__}."
        )
    allowed = {"target_time_s", "max_time_s", "jerk_target_m_s3", "jerk_max_m_s3"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise click.ClickException(
            f"Mission '{path}': unknown scoring field(s) {', '.join(unknown)}. "
            f"Expected one of {', '.join(sorted(allowed))}."
        )
    scoring = {}
    for name in ("target_time_s", "max_time_s"):
        if name in raw and raw[name] is not None:
            scoring[name] = _scoring_number(raw[name], name, path)
    if (scoring.get("target_time_s") is not None
            and scoring.get("max_time_s") is not None
            and not 0.0 < scoring["target_time_s"] < scoring["max_time_s"]):
        raise click.ClickException(
            f"Mission '{path}': scoring needs 0 < target_time_s < max_time_s, "
            f"got target {scoring['target_time_s']} and max {scoring['max_time_s']}."
        )
    jerk_target = DEFAULT_JERK_TARGET_M_S3
    jerk_max = DEFAULT_JERK_MAX_M_S3
    if "jerk_target_m_s3" in raw and raw["jerk_target_m_s3"] is not None:
        jerk_target = _scoring_number(raw["jerk_target_m_s3"], "jerk_target_m_s3", path)
    if "jerk_max_m_s3" in raw and raw["jerk_max_m_s3"] is not None:
        jerk_max = _scoring_number(raw["jerk_max_m_s3"], "jerk_max_m_s3", path)
    if not jerk_target < jerk_max:
        raise click.ClickException(
            f"Mission '{path}': scoring needs jerk_target_m_s3 < jerk_max_m_s3, "
            f"got {jerk_target} and {jerk_max}."
        )
    scoring["jerk_target_m_s3"] = jerk_target
    scoring["jerk_max_m_s3"] = jerk_max
    return scoring


def _load_waypoints(path, with_scoring=False):
    """Load a waypoint file.

    Returns (waypoints, events, thresholds, stop_indices). A plain JSON list
    is still a valid mission; the object form
    ``{"waypoints": [...], "scoring": {...}}`` also carries the timing budget
    the decision score needs, and ``with_scoring=True`` returns it as a fifth
    element (a dict, possibly empty).
    """
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

    mission_scoring = {}
    if isinstance(data, dict):
        if "waypoints" not in data:
            raise click.ClickException(
                f"Waypoint file '{path}' must contain a list of waypoints, or an "
                "object with a 'waypoints' list (and optional 'scoring')."
            )
        mission_scoring = _parse_mission_scoring(data.get("scoring"), path)
        data = data["waypoints"]

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
    if with_scoring:
        return wps, events, thresholds, stop_indices, mission_scoring
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
                   best_effort=False, accepted=False, score=None):
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
    score_text = f", score {score:.1f}" if score is not None else ""
    if validate:
        if attempt_passed:
            detail = f"envelope {tube * 100:.2f} cm" if tube == tube else "envelope n/a"
            click.secho(f"  validation: passed ({detail}{score_text})", fg="green")
        elif accepted:
            click.secho(
                f"  validation: failed but accepted under --accept-score "
                f"({score_text.lstrip(', ')}); approved for the hub", fg="yellow")
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
    envelope_limit_per_metre: float = DEFAULT_ENVELOPE_LIMIT_PER_METRE
    # The field model behind --envelope-from-field, or None. Its per-sample
    # clearances are the envelope limits where the field has one.
    field: Any = None
    # One validator per run: its compiled scan is reused by every candidate,
    # and a fresh one per candidate pays the XLA compile again (KI-049).
    validator: Any = None
    # The decision score's run-wide timing budget and profile. A validated
    # run always has target and max; they are what the duration score and the
    # quality score's jerk term are measured against.
    target_time_s: Any = None
    max_time_s: Any = None
    jerk_target_m_s3: float = DEFAULT_JERK_TARGET_M_S3
    jerk_max_m_s3: float = DEFAULT_JERK_MAX_M_S3
    scoring_profile: str = DEFAULT_SCORING_PROFILE
    accept_score: Any = None


def _record_candidate(spec, solve_result, cand_idx, batch_len, histories, *,
                      quiet):
    """Turn one solver result into an attempt record and announce it."""
    accuracy_weight = spec["accuracy_weight"]
    guess_style = spec["guess_style"]
    per_segment_dt = spec.get("per_segment_dt", False)
    if not quiet:
        number = f"Candidate {cand_idx + 1}/{batch_len}"
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
        # The decision score (score contract): filled in by _score_attempt
        # when the run scores candidates, None when it does not.
        "score": None,
        "score_version": None,
        "categories": None,
        "ratios": None,
        "structural_failure": False,
        # True when a faster candidate already passed, so this plan was not
        # checked; the verdict is "not needed", never "failed".
        "validation_skipped": False,
        # Nothing is a rescue any more; the field stays because the studio
        # and the report readers still show it (KI-048 removed).
        "rescue": False,
    }


def _immrax_validator(ctx):
    """The run's one validator, built on first use.

    The compiled rollout lives inside the validator and is keyed by shape, so
    building a fresh validator per candidate recompiles the same graph. The
    validator holds no per-trajectory state beyond that cache.
    """
    if ctx.validator is not None:
        return ctx.validator
    validator_cls, robot_config_cls = _ensure_jax_classes()
    jax_cfg = robot_config_cls(ctx.robot_cfg)
    kwargs = {"control_dt": ctx.control_dt, "propagation": ctx.propagation}
    if ctx.digital_twin is not None:
        kwargs["digital_twin"] = ctx.digital_twin
    ctx.validator = validator_cls(jax_cfg, **kwargs)
    return ctx.validator


def _score_attempt(attempt_record, ctx):
    """Validate, audit and score one candidate.

    Returns True when the batch must stop here: an unvalidated live run
    keeps the first result for the visualisation, and a validator crash on
    such a run is final (KI-001, KI-002).
    """
    if not ctx.use_validation:
        return False
    if not attempt_record["converged"]:
        # No plan, nothing to check. The reachability rollout of a solve that
        # stopped at its budget is a minute spent on a number no reader can
        # use, and on a rescue that is five of them.
        attempt_record["reasons"] = ["solver did not converge"]
        attempt_record["structural_failure"] = True
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
    travelled = _travelled_m(samples_data)
    # The field's per-sample limits, when the run asked for them. The gate
    # and the segment report read the same table, so the two cannot disagree.
    field_limits = _field_envelope_limits(ctx.field, ctx.robot_cfg, samples_data)

    try:
        immrax_val = _immrax_validator(ctx)

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
        # A validator crash is not a pass (KI-002), and it is a structural
        # failure: there is no complete result to score or accept.
        attempt_record["reasons"] = [f"Immrax validation error: {e}"]
        attempt_record["structural_failure"] = True
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
    # The worst envelope-radius / local-limit ratio, for the accuracy score.
    max_env_ratio = 0.0

    for i in range(len(samples_data)):
        # The segment's limit is the one of the waypoint the segment arrives
        # at (see _segment_limit), so a precision waypoint is held to its
        # tight limit on the approach.
        if waypoint_bounds:
            wp_idx = _waypoint_of(waypoint_bounds, i)
        else:
            s_val = samples[0] if isinstance(samples, (list, tuple, np.ndarray)) else samples
            wp_idx = i // max(1, int(s_val))
        local_threshold = _local_limit(
            ctx.waypoint_thresholds, wp_idx, travelled[i], ctx.envelope_limit,
            ctx.envelope_limit_per_metre, field_limits, i)

        env = samples_data[i].get('reachability_envelope', {})
        if env:
            err_x = (env['x_max'] - env['x_min']) / 2.0
            err_y = (env['y_max'] - env['y_min']) / 2.0
            err = np.sqrt(err_x**2 + err_y**2)
            max_err_all = max(max_err_all, err)
            if local_threshold > 0:
                max_env_ratio = max(max_env_ratio, float(err) / local_threshold)
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
    attempt_record["envelope_limit_field_m"] = _field_limit_summary(
        ctx.field, field_limits)
    attempt_record["placement_uncertainty_m"] = (
        _finite(ctx.field.placement_uncertainty_m)
        if ctx.field is not None else None)
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
    # Structural failures (score contract): a diverged or unbounded tube and
    # an incomplete validator result can never be scored, accepted, exported
    # or pushed - only a *quantified* excess can.
    attempt_record["structural_failure"] = bool(
        attempt_record["diverged"]
        or unknown_envelope
        or immrax_report.get("error"))
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
        # How close the plan runs to the motor and to the traction limit,
        # peak (not RMS). The gate only says pass or fail, so the margin to
        # an edge is reported for the risk lever (Q14).
        attempt_record["peak_force_util"] = float(
            cpu_summary.get("metrics", {}).get("max_force_utilization", 0.0) or 0.0)
        attempt_record["peak_traction_util"] = float(
            cpu_summary.get("metrics", {}).get("peak_traction_utilization", 0.0) or 0.0)

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

    # --- Decision score (score_version 1) ---------------------------------
    # One capped, deterministic 0-100 number. It selects candidates and can
    # authorize export and a hub push. It is NOT a probability of hardware
    # success: it prices the measured excesses against the documented limits
    # and orders the candidates with the result.
    if ctx.target_time_s is not None and ctx.max_time_s is not None:
        weights = dict(SCORING_PROFILES.get(
            ctx.scoring_profile, SCORING_PROFILES[DEFAULT_SCORING_PROFILE]))
        cpu_metrics = (cpu_summary or {}).get("metrics") or {}
        cpu_audit = (cpu_summary or {}).get("audit") or {}
        cpu_errors = (cpu_summary or {}).get("errors") or {}
        immrax_util = immrax_report.get("utilization") or {}

        def _util_ratio(value, limit):
            limit = float(limit or 0.0)
            return float(value or 0.0) / limit if limit > 0 else 0.0

        # r_phys: worst motor, traction/slip and wheel-speed utilisation,
        # across the CPU audit and the Immrax interval check.
        r_phys = max(
            float(cpu_metrics.get("max_force_utilization", 0.0) or 0.0),
            float(cpu_metrics.get("peak_traction_utilization", 0.0) or 0.0),
            _util_ratio(cpu_audit.get("max_wheel_speed_m_s", 0.0),
                        cpu_audit.get("wheel_speed_limit_m_s", 0.0)),
            max((float(value) for value in immrax_util.values()
                 if value is not None), default=0.0),
        )
        # r_acc: the envelope against its active local limit (waypoint
        # overrides and per-metre allowance included), the tracking gate and
        # the RK4 integration gate.
        r_acc = max(
            max_env_ratio,
            _util_ratio(attempt_record.get("tracking_m", 0.0), TRACKING_GATE_M),
            _util_ratio(cpu_errors.get("max_pos_error_m", 0.0), INTEGRATION_GATE_M),
            _util_ratio(cpu_errors.get("final_pos_error_m", 0.0), INTEGRATION_GATE_M),
        )

        # Q: the worst stop/final spread against its own local limit, the RMS
        # force utilisation, and the peak wheel jerk.
        worst_spread_score = 100.0
        for idx, spread in zip(fitness_info.get("checkpoint_indices", []),
                               fitness_info.get("checkpoint_spreads_m", [])):
            if waypoint_bounds:
                wp_idx = _waypoint_of(waypoint_bounds, idx)
            else:
                s_val = (samples[0]
                         if isinstance(samples, (list, tuple, np.ndarray))
                         else samples)
                wp_idx = int(idx) // max(1, int(s_val))
            travelled_at = (travelled[idx] if idx < len(travelled)
                            else travelled[-1] if travelled else 0.0)
            limit = _local_limit(
                ctx.waypoint_thresholds, wp_idx, travelled_at, ctx.envelope_limit,
                ctx.envelope_limit_per_metre, field_limits, idx)
            worst_spread_score = min(
                worst_spread_score, spread_score(float(spread), limit))
        rms_score = 100.0 * max(0.0, min(
            1.0, 1.0 - float(attempt_record.get("rms_force_util", 0.0) or 0.0)))
        peak_jerk = float(attempt_record.get("peak_jerk_m_s3", 0.0) or 0.0)
        quality = (worst_spread_score + rms_score
                   + jerk_score(peak_jerk, ctx.jerk_target_m_s3,
                                ctx.jerk_max_m_s3)) / 3.0

        # D: the duration against the mission's measured budget.
        duration_cat = duration_score(
            float(attempt_record.get("duration", float("inf"))),
            float(ctx.target_time_s), float(ctx.max_time_s))

        categories = {
            "physical": ratio_score(r_phys),
            "accuracy": ratio_score(r_acc),
            "quality": float(quality),
            "duration": float(duration_cat),
        }
        base = weighted_score(categories, weights)

        cpu_physical_failed = bool(
            cpu_audit.get("num_violating_samples", 0)
            or cpu_audit.get("num_slip_points", 0)
            or float(cpu_audit.get("speed_violation_m_s", 0.0) or 0.0) > 1e-6)
        immrax_physical_failed = bool(
            motor_violation > 1e-3 or traction_violation > 1e-3
            or slip_violation > 1e-3 or speed_violation > 1e-6)
        phys_failed = bool(r_phys > 1.0 or cpu_physical_failed
                           or immrax_physical_failed)
        acc_check_failed = bool(
            failed_threshold
            or float(attempt_record.get("tracking_m", 0.0) or 0.0) > TRACKING_GATE_M
            or float(cpu_errors.get("max_pos_error_m", 0.0) or 0.0) >= INTEGRATION_GATE_M
            or float(cpu_errors.get("final_pos_error_m", 0.0) or 0.0) >= INTEGRATION_GATE_M)
        acc_failed = bool(r_acc > 1.0 or acc_check_failed)

        # The caps: a candidate never scores above a category it failed.
        score = min(base,
                    categories["physical"] if phys_failed else 100.0,
                    categories["accuracy"] if acc_failed else 100.0)
        attempt_record["score"] = float(score)
        attempt_record["score_version"] = SCORE_VERSION
        attempt_record["categories"] = {
            name: float(value) for name, value in categories.items()}
        attempt_record["ratios"] = {"r_phys": float(r_phys), "r_acc": float(r_acc)}
        attempt_record["phys_failed"] = phys_failed
        attempt_record["acc_failed"] = acc_failed

    # Per-segment detail, so a failure can be pinned to a part of the path.
    attempt_record["segments"] = _segment_report(
        samples_data, samples, ctx.waypoint_thresholds, immrax_report,
        cpu_summary, segment_samples=stats.get("segment_samples"),
        default_limit=ctx.envelope_limit,
        default_limit_per_metre=ctx.envelope_limit_per_metre,
        field_limits=field_limits)

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
                     accuracy_weight, use_validation, quiet, progress=None,
                     early_stop=True):
    """Solve the planner's candidate set.

    The fallback seeds that used to follow a failed set (KI-048) are not
    tried: on the 32-mission suite they rescued 0 of the 22 failing missions,
    and 95 of their 97 failures were the envelope, which a seed cannot move.
    If a rescue is ever wanted again it is in the git history of this function.

    ``progress`` is called with the attempts so far, once as each candidate
    leaves the solver and once when its check returns, so a caller that draws
    can show the path while the reachability check is still running.
    """
    attempts = []
    batch = list(specs_to_solve)
    solved, histories = _dispatch_solve(batch, solve_ctx)
    if solve_ctx.polish:
        solved = _polish_candidates(solved, solve_ctx)

    for cand_idx, (spec, solve_result) in enumerate(zip(batch, solved)):
        attempts.append(_record_candidate(
            spec, solve_result, cand_idx, len(batch), histories, quiet=quiet))
        if progress is not None:
            progress(attempts)

    # Check the fastest solved candidate first. Once one passes, a slower
    # candidate cannot win the duration rule, so its check is skipped; an
    # equal duration is still checked because the envelope breaks that tie.
    # A compare run (--candidates-out) or an explicit --pick-candidate wants
    # every candidate's verdict, so it turns the skip off.
    order = (sorted(range(len(attempts)), key=lambda i: attempts[i]["duration"])
             if early_stop else range(len(attempts)))
    best_passing = None
    for index in order:
        record = attempts[index]
        if (early_stop and best_passing is not None
                and record["duration"] > best_passing):
            record["validation_skipped"] = True
            record["reasons"] = [
                "validation skipped: a faster candidate already passed"]
            continue
        if _score_attempt(record, score_ctx):
            break
        if record["ok"] and (best_passing is None
                             or record["duration"] < best_passing):
            best_passing = record["duration"]
        if progress is not None:
            progress(attempts)
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
                    best_effort=False, accept_score=None, final_waypoint=None):
    """Pick the attempt to export and print the candidate table.

    ``pick`` is the operator's choice from a comparison: it exports that
    candidate instead of the one the rule picks. Only a candidate that
    passed every gate - or one explicitly accepted by an active
    ``--accept-score`` - may be picked: the rule is not suspended, it is
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
            if not _attempt_eligible(chosen, accept_score):
                if chosen.get("structural_failure"):
                    detail = ("it did not converge or the check was incomplete, "
                              "so nothing about it can be accepted")
                elif accept_score is not None and chosen.get("score") is not None:
                    detail = (f"its score {chosen['score']:.1f} is below "
                              f"--accept-score {accept_score}")
                else:
                    detail = ("it did not pass the validator and no "
                              "--accept-score accepts it")
                raise click.ClickException(
                    f"Candidate {pick} cannot be exported: {detail}. "
                    "Pick a candidate that passed, or solve again.")
            if final_waypoint is not None:
                check = check_final_waypoint(chosen["samples"], final_waypoint)
                if not check["passed"]:
                    raise click.ClickException(
                        f"Candidate {pick} does not reach the final waypoint "
                        f"(position error {check['pos_error_m']*100:.2f} cm). "
                        "No trajectory written.")
            if not quiet:
                label = ("the operator's pick" if chosen.get("ok")
                         else "accepted by --accept-score")
                click.echo(f"  Exporting candidate {pick} ({label}).")
            return (chosen["samples"], chosen["stats"], bool(chosen.get("ok")),
                    chosen["reasons"], chosen.get("tube", float("inf")), pick)
        # Never silently export a failed candidate: the decision score picks
        # among the eligible ones, and a failed candidate is only eligible
        # under an explicit --accept-score.
        best, passed_any = _select_attempt_scored(all_attempts, accept_score)
        if best is None:
            raise click.ClickException(
                f"Solver did not converge on any of the {len(all_attempts)} "
                f"attempts. No trajectory written.")
        if not passed_any and best_effort and not _attempt_accepted(best, accept_score):
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
                if a.get("score") is not None:
                    state += f", score {a['score']:.1f}"
                if _attempt_accepted(a, accept_score):
                    state += ", accepted"
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

    # Without validation the first candidate is the result; it is the last
    # candidate that converged, not merely the last one tried.
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
                   benchmark, export_format, controller_dt):
    """Write the trajectory and the files the flags asked for.

    Returns every path written, the trajectory first.
    """
    parent = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(parent):
        raise click.ClickException(
            f"Output directory '{parent}' does not exist. Create it, or pass "
            f"-o with a path in an existing directory.")
    traj_config = config_data.get("robot", config_data.get(
        "config", config_data))
    result = {
        "name": os.path.basename(output).split('.')[0],
        "version": 3,
        "trajectory": {
            "config": traj_config,
            "samples": samples_data
        }
    }
    # The validation record travels with the trajectory: the `.traj` root and
    # the `_stats.json` carry the same one, so a file read later cannot be
    # mistaken for an approved one. The hash is the one the hub push checks.
    if stats.get("validation"):
        from overboard.deploy.hub_deploy import trajectory_sha256
        record = dict(stats["validation"])
        record["trajectory_sha256"] = trajectory_sha256(traj_config, samples_data)
        stats["validation"] = record
        result["validation"] = record
    # Write beside the target and swap it in, so a failed write (disk full,
    # serialization error) keeps the previous trajectory instead of
    # truncating it. Same pattern as _write_candidates.
    temporary = f"{output}.tmp"
    with open(temporary, 'w') as f:
        json.dump(result, f, indent=1)
    _swap_in(temporary, output)

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
@click.option('--preset', type=click.Choice(['fast', 'safe'], case_sensitive=False),
              default=None,
              help='Named starting point. fast: 8 samples per segment, 1 candidate, '
                   'no validation. safe: 3 candidates, --validate with the '
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
@click.option('--envelope-limit-per-metre',
              default=DEFAULT_ENVELOPE_LIMIT_PER_METRE,
              type=click.FloatRange(min=0.0, max=0.05),
              help='Extra envelope allowance per metre travelled, in m/m, added '
                   'to the run-wide or waypoint limit (default 0 = absolute). '
                   'The tube widens with distance, so this prices the drift '
                   'instead of the elapsed time (KI-054). Recorded with the '
                   'validation report.')
@click.option('--envelope-from-field', default=None, type=str,
              help='Per-sample envelope limits from a field season model: the '
                   'robot disc clearance at each sample, clamped to 0.005-0.05 '
                   'm, replaces the run-wide limit where the field has one. A '
                   'waypoint error_threshold still wins locally; a sample the '
                   'field cannot measure falls back to --envelope-limit. The '
                   'field footprints carry about 100 mm of placement '
                   'uncertainty, so the limit is nominal geometry, not a '
                   'survey. Recorded with the validation report.')
@click.option('--target-time', default=None, type=click.FloatRange(min=0.0),
              help='Scoring budget: the duration worth 100 on the duration '
                   'score. Required with --validate when the mission does not '
                   'declare one. Overrides the mission scoring section.')
@click.option('--max-time', default=None, type=click.FloatRange(min=0.0),
              help='Scoring budget: the duration worth 0 on the duration score. '
                   'Required with --validate when the mission does not declare '
                   'one. Overrides the mission scoring section.')
@click.option('--jerk-target', default=None, type=click.FloatRange(min=0.0),
              help='Scoring: peak wheel jerk worth 100 on the motion-quality '
                   'score, in m/s^3 (default 10).')
@click.option('--jerk-max', default=None, type=click.FloatRange(min=0.0),
              help='Scoring: peak wheel jerk worth 0 on the motion-quality '
                   'score, in m/s^3 (default 20).')
@click.option('--scoring-profile',
              type=click.Choice(['precision', 'balanced', 'speed'],
                                case_sensitive=False),
              default=DEFAULT_SCORING_PROFILE,
              help='Category weights of the decision score: precision '
                   '20/35/35/10, balanced 25/25/25/25 (default), speed '
                   '20/20/20/40, in physical/accuracy/quality/duration order.')
@click.option('--accept-score', default=None, type=click.IntRange(min=1, max=100),
              help='Accept a failed candidate whose decision score is at least '
                   'this (1-100), so it can be exported and pushed. Without it '
                   'only an ordinary validator pass is eligible. Needs '
                   '--validate. A structural failure is never accepted.')
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
         envelope_limit, envelope_limit_per_metre, envelope_from_field,
         target_time, max_time, jerk_target, jerk_max,
         scoring_profile, accept_score,
         sysid_fit, export_format, controller_dt, control_dt,
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
    if accept_score is not None and not validate:
        raise click.ClickException(
            "--accept-score needs --validate: it accepts a validated verdict, "
            "so there has to be one.")
    if push_hub and not validate:
        raise click.ClickException(
            "--push-hub needs --validate: only a trajectory that passed the "
            "validator - or one accepted by --accept-score - may drive the hub.")
    scoring_profile = (scoring_profile or DEFAULT_SCORING_PROFILE).lower()

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
    if not sysid_fit and validate:
        # The config may carry a documented twin (the DriveBase acceleration
        # limit needs no calibration). --sysid-fit wins when both are given.
        digital_twin = _config_digital_twin(config_data)
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

    (wps, waypoint_events, waypoint_thresholds, json_stop_indices,
     mission_scoring) = _load_waypoints(waypoints, with_scoring=True)
    stop_indices = _parse_stop_waypoints(stop_waypoints, len(wps))
    waypoint_events.update(_parse_events(events, len(wps)))

    # The scoring timing budget: the mission's own, with the CLI flags on
    # top. A validated run needs it - the decision score's duration term and
    # its jerk term are measured against it.
    scoring = {
        "target_time_s": (float(target_time) if target_time is not None
                          else mission_scoring.get("target_time_s")),
        "max_time_s": (float(max_time) if max_time is not None
                       else mission_scoring.get("max_time_s")),
        "jerk_target_m_s3": (
            float(jerk_target) if jerk_target is not None
            else mission_scoring.get("jerk_target_m_s3",
                                     DEFAULT_JERK_TARGET_M_S3)),
        "jerk_max_m_s3": (
            float(jerk_max) if jerk_max is not None
            else mission_scoring.get("jerk_max_m_s3",
                                     DEFAULT_JERK_MAX_M_S3)),
        "profile": scoring_profile,
        "weights": dict(SCORING_PROFILES.get(
            scoring_profile, SCORING_PROFILES[DEFAULT_SCORING_PROFILE])),
        "accept_score": accept_score,
    }
    if not scoring["jerk_target_m_s3"] < scoring["jerk_max_m_s3"]:
        raise click.ClickException(
            f"Scoring needs jerk_target_m_s3 < jerk_max_m_s3, got "
            f"{scoring['jerk_target_m_s3']} and {scoring['jerk_max_m_s3']}. "
            "Set --jerk-target/--jerk-max or the mission scoring fields.")
    if validate:
        missing = []
        if scoring["target_time_s"] is None:
            missing.append(
                "target_time_s (mission 'scoring' section, or --target-time)")
        if scoring["max_time_s"] is None:
            missing.append(
                "max_time_s (mission 'scoring' section, or --max-time)")
        if missing:
            raise click.ClickException(
                "A validated run needs the scoring timing: "
                + "; ".join(missing) + ".")
        if not 0.0 < scoring["target_time_s"] < scoring["max_time_s"]:
            raise click.ClickException(
                f"Scoring needs 0 < target time < max time, got target "
                f"{scoring['target_time_s']} and max {scoring['max_time_s']}. "
                "Set --target-time/--max-time or the mission scoring fields.")

    # The field behind --envelope-from-field, loaded once and up front: a
    # typo in the season must not surface after a solve.
    field = None
    if envelope_from_field:
        from overboard.viz.fields import load_field
        field = load_field(envelope_from_field)
        if field is None:
            raise click.ClickException(
                f"There is no field for the season '{envelope_from_field}'.")

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
        envelope_limit_per_metre=float(envelope_limit_per_metre),
        field=field,
        target_time_s=scoring["target_time_s"],
        max_time_s=scoring["max_time_s"],
        jerk_target_m_s3=scoring["jerk_target_m_s3"],
        jerk_max_m_s3=scoring["jerk_max_m_s3"],
        scoring_profile=scoring_profile,
        accept_score=accept_score,
    )
    all_attempts = _run_candidates(
        specs_to_solve, solve_ctx, score_ctx,
        accuracy_weight=accuracy_weight,
        use_validation=use_validation, quiet=quiet,
        progress=(lambda made: _write_candidates(candidates_out, made, None, True))
        if candidates_out else None,
        # The decision score picks the highest score, so every candidate is
        # scored: a slower plan can outscore a faster one, and a skipped
        # candidate has no score to win with.
        early_stop=False)

    try:
        (samples_data, stats, attempt_passed, best_reasons, selected_tube,
         selected_index) = _choose_attempt(
            all_attempts, validate=validate, quiet=quiet, samples=samples,
            pick=pick_candidate, best_effort=best_effort,
            accept_score=accept_score, final_waypoint=wps[-1])
    except click.ClickException:
        # A refusal is exactly when the numbers matter, and the message can
        # only name the categories. Leave the report and the comparison
        # behind before the error ends the run.
        if use_validation:
            _emit_validation_report(all_attempts, None,
                                    propagation=propagation,
                                    best_effort=False,
                                    envelope_limit=envelope_limit,
                                    envelope_limit_per_metre=envelope_limit_per_metre,
                                    accept_score=accept_score,
                                    profile=scoring["profile"],
                                    placement_uncertainty_m=(
                                        field.placement_uncertainty_m
                                        if field is not None else None))
        _write_candidates(candidates_out, all_attempts, None, quiet)
        raise

    selected_attempt = all_attempts[max(0, selected_index - 1)]
    # Four verdicts, and the record says which one this is (score contract):
    # an ordinary pass; a failed plan accepted by --accept-score; a best-effort
    # inspection export; or an unvalidated run. Only the first two are
    # approved for the hub.
    best_effort_export = bool(validate and best_effort and not attempt_passed
                              and not _attempt_accepted(selected_attempt,
                                                        accept_score))
    accepted_export = bool(
        validate and not attempt_passed and not best_effort_export
        and _attempt_accepted(selected_attempt, accept_score))
    if validate:
        stats["validation"] = _build_validation_record(
            passed=attempt_passed, accepted=accepted_export,
            best_effort=best_effort_export,
            attempt=selected_attempt, scoring=scoring,
            envelope_limit=envelope_limit,
            envelope_limit_per_metre=envelope_limit_per_metre,
            trajectory_hash=None)  # stamped by _write_outputs over the file
    else:
        # No reachability gate ran, so the file says so. A `.traj` without
        # this could be read later as an approved plan.
        stats["validation"] = {
            "status": "unvalidated", "passed": False, "accepted": False,
            "best_effort": False, "hub_approved": False,
            "score": None, "score_version": SCORE_VERSION,
            "profile": scoring["profile"], "weights": scoring["weights"],
            "categories": None, "target_time_s": None, "max_time_s": None,
            "accept_score": None,
            "envelope_limit_m": float(envelope_limit),
            "envelope_limit_per_metre": float(envelope_limit_per_metre),
            "envelope_limit_field_m": None,
            "placement_uncertainty_m": (
                _finite(field.placement_uncertainty_m)
                if field is not None else None),
            "reasons": [], "segments": [], "max_excess_m": 0.0,
        }

    if best_effort_export:
        excess = stats["validation"]["max_excess_m"]
        click.secho(
            "  BEST EFFORT: validation failed ("
            + ("; ".join(best_reasons) or "see the candidate table")
            + (f"; up to {excess * 100:.2f} cm outside the local limit"
               if excess > 0 else "")
            + "). The path is written for inspection and is not approved "
            "for the hub.", fg="red", bold=True)
    elif accepted_export:
        click.secho(
            f"  ACCEPTED: validation failed ("
            + ("; ".join(best_reasons) or "see the candidate table")
            + f"), but the score {selected_attempt.get('score', 0.0):.1f} "
            f"meets --accept-score {accept_score}. Approved for the hub.",
            fg="yellow", bold=True)

    if use_validation:
        # The studio reads this to show which part of the path failed.
        _emit_validation_report(all_attempts, selected_index,
                                propagation=propagation,
                                best_effort=best_effort_export,
                                envelope_limit=envelope_limit,
                                envelope_limit_per_metre=envelope_limit_per_metre,
                                accept_score=accept_score,
                                profile=scoring["profile"],
                                accepted=accepted_export,
                                hub_approved=stats["validation"]["hub_approved"],
                                envelope_limit_field=(
                                    selected_attempt.get("envelope_limit_field_m")),
                                placement_uncertainty_m=(
                                    field.placement_uncertainty_m
                                    if field is not None else None))
    # Every candidate, paths included, for a tool that draws the comparison.
    # Diagnostics, so it is written before the export gate: a failed run is
    # exactly when the operator wants to see what the candidates did.
    _write_candidates(candidates_out, all_attempts, selected_index, quiet)

    # --- EXPORT GATE (KI-001 / KI-002, score contract) ---
    block_reason = _export_block_reason(stats, validate, attempt_passed,
                                        best_effort=best_effort_export,
                                        accepted=accepted_export)
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

    if validate and attempt_passed:
        cpu_summary = validate_samples(samples_data, robot_cfg)
        if not cpu_summary["passed"]:
            reasons = _cpu_failure_reasons(cpu_summary, waypoint_check)
            raise click.ClickException(
                f"Validator rejected the trajectory: {'; '.join(reasons) or 'unknown reason'}. "
                f"No trajectory written to '{output}'."
            )

    written = _write_outputs(
        output, samples_data, stats, config_data, robot_cfg,
        benchmark=benchmark, export_format=export_format, controller_dt=controller_dt)

    if not quiet:
        _print_summary(output, samples_data, robot_cfg, validate, attempt_passed,
                       selected_tube, written[1:], best_effort=best_effort_export,
                       accepted=accepted_export,
                       score=_finite(selected_attempt.get("score")))

    if push_hub:
        # The trajectory pushes enforce ordinary-pass-or-explicitly-accepted
        # approval, and push_trajectory re-checks the file against its hash.
        if not (stats.get("validation") or {}).get("hub_approved"):
            raise click.ClickException(
                "--push-hub refused: the trajectory is not approved for the "
                "hub (it needs an ordinary validator pass, or a failed plan "
                "accepted by --accept-score).")
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
