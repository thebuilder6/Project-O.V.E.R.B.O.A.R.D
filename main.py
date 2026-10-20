import click
import json
import os
import numpy as np
from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer, GUESS_STYLES
from multiverse_optimizer import MasterTrajectoryOptimizer
from plotter import plot_trajectory, plot_reachability_envelope
from animated_plotter import animate_trajectory
from validator import validate_trajectory, validate_samples, check_final_waypoint
from export import write_controller_file, write_python_file
from convergence_plotter import plot_convergence
from live_visualizer import get_visualizer
from digital_twin import load_sysid_fit

# JAX/Immrax Imports
try:
    from immrax_validator import ImmraxValidator
    from jax_robot_model import JAXRobotConfig
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _select_attempt(attempts):
    """Choose the candidate to export: a passing converged attempt with the
    smallest reachability envelope, then the shortest one (accuracy first, then
    speed). Otherwise the converged attempt with the smallest quantified
    violation. Non-converged attempts are never eligible (KI-001)."""
    candidates = [a for a in attempts if a.get("converged", False)]
    if not candidates:
        return None, False
    passing = [a for a in candidates if a["ok"]]
    if passing:
        return min(passing, key=lambda a: (a.get("tube", float("inf")),
                                           a.get("duration", a["cost"]))), True
    return min(candidates, key=lambda a: (a["violation"], a["cost"])), False


CANDIDATE_WEIGHT_STEP = 2.0  # legacy ladder for the Multi-Verse path


def _candidate_specs(base_accuracy_weight, count, use_styles=True):
    """Definitions for the candidate set.

    With --simple, each candidate is a distinct initial-guess seed at the base
    accuracy weight (KI-020: accuracy_weight is a time/smoothness knob, not a
    candidate generator). The Multi-Verse path has no seed plumbing yet, so it
    keeps the historical accuracy-weight ladder.
    """
    count = max(1, int(count))
    if use_styles:
        return [{"accuracy_weight": base_accuracy_weight, "guess_style": style}
                for style in GUESS_STYLES[:count]]
    return [{"accuracy_weight": base_accuracy_weight + CANDIDATE_WEIGHT_STEP * k,
             "guess_style": "baseline"} for k in range(count)]


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


@click.command()
# --- Input / Output Options ---
@click.option('-c', '--config', required=True, type=click.Path(exists=True), 
              help='Path to the robot configuration file (.json).')
@click.option('-w', '--waypoints', required=True, type=click.Path(exists=True), 
              help='Path to waypoints JSON file.')
@click.option('-o', '--output', default='output.traj', type=str, 
              help='Output trajectory file path.')

# --- Solver Parameters ---
@click.option('-n', '--samples', default=10, type=int, 
              help='Samples per segment.')
@click.option('-a', '--accuracy-weight', default=0.0, type=float, 
              help='Smoothness/accuracy weight (0 = pure time-optimal).')
@click.option('--candidates', default=2, type=click.IntRange(1, len(GUESS_STYLES)),
              help='Candidates to solve and compare. With --simple these are distinct '
                   'initial-guess seeds (baseline, rolling, bow_left, bow_right, turn_first); '
                   'the Multi-Verse path uses accuracy_weight steps of 2.0 instead. '
                   'Validation picks the passing candidate with the smallest envelope, then the fastest. '
                   'Use 1 for a single solve.')
@click.option('--stop-waypoints', default=None, type=str, 
              help='Comma-separated waypoint indices where robot must stop (e.g., "2,5,7").')
@click.option('--events', default=None, type=str, 
              help='Comma-separated waypoint:event pairs (e.g., "2:lower_arm,5:release").')

# --- Optimizer Strategy ---
@click.option('--simple', is_flag=True, 
              help='Use simple optimizer instead of Multi-Verse refinement.')
@click.option('--per-segment-dt', is_flag=True,
              help='Allow each waypoint segment its own sample interval (--simple only). '
                   'Default keeps one uniform dt for the validated timing model.')
@click.option('--no-parallel', is_flag=True, 
              help='Disable parallel processing for Multi-Verse refinement.')
@click.option('--workers', default=8, type=int, 
              help='Number of parallel workers for Multi-Verse refinement.')

# --- Export & Validation ---
@click.option('--validate', is_flag=True, 
              help='Run validation and block export on failure (requires JAX/Immrax).')
@click.option('--sysid-fit', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Optional sysid_fit.json digital-twin fit (requires --validate or --live).')
@click.option('--export-format', type=click.Choice(['none', 'controller', 'python', 'python-compact'], case_sensitive=False), 
              default='none', help='Export format for controller consumption (python-compact is the packed Pybricks format).')
@click.option('--controller-dt', default=0.02, type=float, 
              help='Fixed timestep for controller export (seconds).')
@click.option('--benchmark', is_flag=True, 
              help='Collect comprehensive benchmarking data for whitepaper.')

# --- Visualization ---
@click.option('--plot', is_flag=True, help='Plot the resulting trajectory.')
@click.option('--animate', is_flag=True, help='Animate the trajectory in real-time.')
@click.option('--live', is_flag=True, help='Enable live interactive visualization in browser.')
@click.option('--show-convergence', is_flag=True, help='Show convergence visualization.')
@click.option('--convergence-mode', type=click.Choice(['parallel', 'best', 'layered'], case_sensitive=False), 
              default='best', help='Convergence visualization mode.')
@click.option('--convergence-animate', is_flag=True, help='Animate convergence.')
@click.option('--convergence-output', type=str, default=None, 
              help='Save convergence plot/animation to file.')

# --- Miscellaneous ---
@click.option('--quiet', '-q', is_flag=True, help='Suppress verbose output.')
def main(config, waypoints, output, samples, accuracy_weight, candidates, stop_waypoints, events, 
         simple, per_segment_dt, no_parallel, workers, validate, sysid_fit, export_format, controller_dt, benchmark, 
         plot, animate, live, show_convergence, convergence_mode, convergence_animate, convergence_output, quiet):
    """
    FLL Trajectory Optimizer CLI.
    Generates time-optimal trajectories for Lego robots.
    """
    if not quiet:
        click.echo(f"Loading config from {config}...")
    with open(config, 'r') as f:
        config_data = json.load(f)
    
    robot_cfg = RobotConfig(config_data)

    if validate and not HAS_JAX:
        raise click.ClickException(
            "--validate requires JAX and Immrax. Install them or run without --validate."
        )
    if per_segment_dt and not simple:
        raise click.ClickException(
            "--per-segment-dt is only supported with --simple."
        )

    digital_twin = None
    if sysid_fit:
        if not (validate or live):
            raise click.ClickException(
                "--sysid-fit requires --validate or --live so the fitted model is used."
            )
        if not HAS_JAX:
            raise click.ClickException(
                "--sysid-fit requires JAX and Immrax so the fitted model can run."
            )
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

    # Choose optimizer based on --simple flag
    if simple:
        if not quiet:
            click.echo("Using simple optimizer (legacy mode)")
        optimizer = TrajectoryOptimizer(robot_cfg)
    else:
        parallel = not no_parallel
        if not quiet:
            click.echo(f"Using JAX-Enhanced Multi-Verse optimizer (parallel={parallel}, workers={workers})")
        optimizer = MasterTrajectoryOptimizer(robot_cfg, enable_parallel=parallel, num_workers=workers, verbose=not quiet)
    
    if not quiet:
        click.echo(f"Loading waypoints from {waypoints}...")
    with open(waypoints, 'r') as f:
        wp_data = json.load(f)
    
    wps = []
    waypoint_events = {}  # index -> event name
    waypoint_thresholds = {} # index -> float
    json_stop_indices = []
    for i, item in enumerate(wp_data):
        if isinstance(item, dict):
            wps.append((item['x'], item['y'], item.get('heading')))
            if 'event' in item:
                waypoint_events[i] = item['event']
            if item.get('stop'):
                json_stop_indices.append(i)
            if 'error_threshold' in item:
                waypoint_thresholds[i] = float(item['error_threshold'])
            elif item.get('precision_mode'):
                waypoint_thresholds[i] = 0.005 # 0.5 cm
        else:
            wps.append((item[0], item[1], item[2] if len(item) > 2 else None))

    if not quiet:
        if simple:
            dt_mode = "per-segment dt" if per_segment_dt else "uniform dt"
            click.echo(f"Optimizing trajectory through {len(wps)} waypoints "
                       f"(candidates={max(1, candidates)} initial-guess seeds, "
                       f"accuracy_weight={accuracy_weight}, {dt_mode})...")
        else:
            click.echo(f"Optimizing trajectory through {len(wps)} waypoints "
                       f"(base accuracy_weight={accuracy_weight}, candidates={max(1, candidates)})...")

    capture_iterations = show_convergence

    # Parse stop waypoints
    stop_indices = []
    if stop_waypoints:
        try:
            stop_indices = [int(x.strip()) for x in stop_waypoints.split(',')]
        except ValueError:
            click.echo("Invalid stop waypoints format.")

    if events:
        try:
            for pair in events.split(','):
                idx_str, event_name = pair.strip().split(':')
                waypoint_events[int(idx_str.strip())] = event_name.strip()
        except ValueError:
            click.echo("Invalid events format.")

    # sorted() so the constraint order (and therefore the IPOPT iterate) is
    # reproducible; set() iteration order changes the NLP ordering.
    all_stop_indices = sorted(set(stop_indices + json_stop_indices))

    # Start live visualizer if requested
    if live:
        viz = get_visualizer()
        wp_dicts = [{"x": w[0], "y": w[1], "heading": w[2]} for w in wps]
        viz.send_config(config_data, wp_dicts)
        if not quiet:
            click.echo("Live visualizer started. Open viz/index.html in your browser.")
            
    # --- CANDIDATE LOOP (validate and select; KI-020) ---
    # Solve several candidates, score each with the same Immrax check, and let
    # _select_attempt pick the winner: passing first, then the smallest envelope,
    # then the fastest. With --simple the candidates are different initial-guess
    # seeds (different local optima), not different accuracy weights: the old
    # accuracy_weight ladder silently traded speed for a bound that could get
    # worse (longer trajectories accumulate more uncertainty).
    candidate_specs = _candidate_specs(accuracy_weight, candidates, use_styles=simple)
    attempt_passed = False
    best_reasons = []
    all_attempts = []  # Each candidate: samples, stats, cost, ok, violation, tube, converged, reasons

    for cand_idx, spec in enumerate(candidate_specs):
        curr_accuracy_weight = spec["accuracy_weight"]
        guess_style = spec["guess_style"]
        if not quiet:
            if simple:
                click.echo(f"Candidate {cand_idx + 1}/{len(candidate_specs)} "
                           f"(guess_style={guess_style})...")
            else:
                click.echo(f"Candidate {cand_idx + 1}/{len(candidate_specs)} "
                           f"(accuracy_weight={curr_accuracy_weight})...")

        solve_kwargs = dict(
            num_samples_per_segment=samples,
            accuracy_weight=curr_accuracy_weight,
            stop_waypoint_indices=all_stop_indices,
            waypoint_events=waypoint_events,
            verbose=not quiet,
            capture_iterations=capture_iterations,
            live_viz=live,
        )
        if simple:
            solve_kwargs["guess_style"] = guess_style
            solve_kwargs["per_segment_dt"] = per_segment_dt
        samples_data, stats = optimizer.solve(wps, **solve_kwargs)

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
            "reasons": [],
        }
        all_attempts.append(attempt_record)

        # Always compute reachability envelope if JAX is available and we are in live/validate mode
        if not quiet: click.echo(f"DEBUG: HAS_JAX={HAS_JAX}, validate={validate}, live={live}")
        if not (HAS_JAX and (validate or live)):
            break

        try:
            jax_cfg = JAXRobotConfig(robot_cfg)
            if digital_twin is None:
                immrax_val = ImmraxValidator(jax_cfg)
            else:
                immrax_val = ImmraxValidator(
                    jax_cfg, digital_twin=digital_twin)

            # Use the uncertainty ranges declared in the config (KI-009).
            cof_range = robot_cfg.uncertainties["cof"]
            torque_range = robot_cfg.uncertainties["torque_headroom"]
            backlash_range = robot_cfg.uncertainties["gear_backlash_m"]
            track_width_range = robot_cfg.uncertainties["track_width_range"]
            wheel_radius_range = robot_cfg.uncertainties["wheel_radius_range"]
            inertia_range = robot_cfg.uncertainties["inertia_kg_m2"]

            if not quiet: click.echo(f"  Running Immrax Robustness Check...")
            immrax_report = immrax_val.validate_trajectory(
                samples_data, cof_range, torque_range, backlash_range,
                track_width_range, wheel_radius_range, inertia_range)
        except Exception as e:
            if not quiet: click.echo(f"  ERROR: Immrax validation failed: {e}")
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
            # Determine local threshold
            wp_idx = i // samples
            local_threshold = waypoint_thresholds.get(wp_idx, 0.02) # Default 2cm for transit

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

        if not quiet:
            score = "envelope diverged" if attempt_record["diverged"] else f"envelope {attempt_record['tube']*100:.2f} cm"
            if attempt_record["ok"]:
                click.echo(f"  Passed! {score}, all segments within thresholds.")
            else:
                reason_text = "; ".join(attempt_record["reasons"]) or "validation failed"
                click.echo(f"  FAILED: {reason_text}.")

        if not validate:  # Live-only: keep the result for visualization.
            break

    if validate:
        # Never silently export a failed candidate: prefer a passing one with the
        # smallest envelope, otherwise the converged candidate closest to passing.
        best, passed_any = _select_attempt(all_attempts)
        if best is None:
            raise click.ClickException("Solver did not converge on any attempt. No trajectory written.")
        if not quiet and len(all_attempts) > 1:
            for k, a in enumerate(all_attempts, 1):
                if a.get("diverged"):
                    score = "envelope diverged"
                elif np.isfinite(a.get("tube", float("inf"))):
                    score = f"envelope {a['tube']*100:.2f} cm"
                else:
                    score = "no envelope"
                state = "ok" if a["ok"] else "failed"
                if simple:
                    label = f"Candidate {k} ({a.get('guess_style', 'baseline')})"
                else:
                    label = f"Candidate {k}"
                click.echo(f"  {label}: {a.get('duration', float('inf')):6.3f} s, {score} ({state})")
            selected_k = next(k for k, a in enumerate(all_attempts, 1) if a is best)
            click.echo(f"  Selected candidate {selected_k}.")
        samples_data, stats = best["samples"], best["stats"]
        attempt_passed = passed_any
        best_reasons = best["reasons"]
    else:
        best_reasons = all_attempts[-1]["reasons"] if all_attempts else []

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
    
    if not quiet:
        click.echo(f"Successfully saved trajectory to {output}")

    if benchmark:
        from validator import compute_metrics
        quality_metrics = compute_metrics(samples_data, robot_cfg)
        stats["quality_metrics"] = quality_metrics
        stats["robot_config"] = {
            "mass": robot_cfg.mass, "inertia": robot_cfg.inertia, "track_width": robot_cfg.track_width,
            "wheel_radius": robot_cfg.wheel_radius, "v_max_rad_s": robot_cfg.v_max_rad_s,
            "t_max_nm": robot_cfg.t_max_nm, "gearing": robot_cfg.gearing, "cof": robot_cfg.cof
        }
        stats_output = os.path.splitext(output)[0] + '_stats.json'
        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)

    if validate:
        validate_trajectory(output, config, waypoints=wps)

    if export_format == 'controller':
        ctrl_output = os.path.splitext(output)[0] + '_controller.json'
        write_controller_file(output, ctrl_output, target_dt=controller_dt, track_width=robot_cfg.track_width)

    if export_format in ('python', 'python-compact'):
        py_output = os.path.splitext(output)[0] + '.py'
        write_python_file(output, py_output, compact=(export_format == 'python-compact'))

    if plot:
        if not quiet: click.echo("Generating plot...")
        # Always show standard trajectory plot
        if not quiet: click.echo("  Generating standard trajectory plot...")
        plot_trajectory(samples_data, waypoints=wps, title=f"Trajectory: {os.path.basename(output)}")
        
        # Also show reachability envelope plot if Immrax data is available
        if HAS_JAX and any('reachability_envelope' in s for s in samples_data):
            if not quiet: click.echo("  Generating reachability envelope plot...")
            plot_reachability_envelope(samples_data, waypoints=wps, title=f"Reachability Envelope: {os.path.basename(output)}", block=True)

    if animate:
        animate_trajectory(samples_data, waypoints=wps, title=f"Trajectory: {os.path.basename(output)}")
    
    if show_convergence:
        if hasattr(optimizer, 'iteration_history') and optimizer.iteration_history:
            plot_convergence(optimizer.iteration_history, mode=convergence_mode, waypoints=wps, title=f"Convergence: {os.path.basename(output)}")

    if live:
        viz = get_visualizer()
        # Broadcast final trajectory with envelopes
        viz.send_trajectory(samples_data, phase="final")
        click.echo("\nLive visualizer is active. Press Enter to stop and exit...")
        input()
        viz.stop()

if __name__ == '__main__':
    main()
