import click
import json
import os
import numpy as np
from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
from multiverse_optimizer import MasterTrajectoryOptimizer
from plotter import plot_trajectory
from animated_plotter import animate_trajectory
from validator import validate_trajectory
from export import write_controller_file, write_python_file
from convergence_plotter import plot_convergence, animate_convergence, ConvergencePlotter
from live_visualizer import get_visualizer

# JAX/Immrax Imports
try:
    from immrax_validator import ImmraxValidator
    from jax_robot_model import JAXRobotConfig
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

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
@click.option('--stop-waypoints', default=None, type=str, 
              help='Comma-separated waypoint indices where robot must stop (e.g., "2,5,7").')
@click.option('--events', default=None, type=str, 
              help='Comma-separated waypoint:event pairs (e.g., "2:lower_arm,5:release").')

# --- Optimizer Strategy ---
@click.option('--simple', is_flag=True, 
              help='Use simple optimizer instead of Multi-Verse refinement.')
@click.option('--no-parallel', is_flag=True, 
              help='Disable parallel processing for Multi-Verse refinement.')
@click.option('--workers', default=8, type=int, 
              help='Number of parallel workers for Multi-Verse refinement.')

# --- Export & Validation ---
@click.option('--validate', is_flag=True, 
              help='Run validation report on the generated trajectory.')
@click.option('--export-format', type=click.Choice(['none', 'controller', 'python'], case_sensitive=False), 
              default='none', help='Export format for controller consumption.')
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
def main(config, waypoints, output, samples, accuracy_weight, stop_waypoints, events, 
         simple, no_parallel, workers, validate, export_format, controller_dt, benchmark, 
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
        click.echo(f"Optimizing trajectory through {len(wps)} waypoints (accuracy_weight={accuracy_weight})...")

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

    all_stop_indices = list(set(stop_indices + json_stop_indices))

    # Start live visualizer if requested
    if live:
        viz = get_visualizer()
        wp_dicts = [{"x": w[0], "y": w[1], "heading": w[2]} for w in wps]
        viz.send_config(config_data, wp_dicts)
        if not quiet:
            click.echo("Live visualizer started. Open viz/index.html in your browser.")
            
    # --- AUTO-POLISH LOOP ---
    curr_accuracy_weight = accuracy_weight
    max_retries = 3
    passed_validation = False
    
    for attempt in range(max_retries):
        samples_data, stats = optimizer.solve(wps, num_samples_per_segment=samples, accuracy_weight=curr_accuracy_weight, stop_waypoint_indices=all_stop_indices, waypoint_events=waypoint_events, verbose=not quiet, capture_iterations=capture_iterations, live_viz=live)

        # Always compute reachability envelope if JAX is available and we are in live/validate mode
        if HAS_JAX and (validate or live):
            jax_cfg = JAXRobotConfig(config_data)
            immrax_val = ImmraxValidator(jax_cfg)

            # Default uncertainty ranges
            cof_range = (robot_cfg.cof * 0.8, robot_cfg.cof * 1.2)
            torque_range = (0.8, 1.0)
            backlash_range = (0.0001, 0.0006)

            if not quiet: click.echo(f"Attempt {attempt+1}: Running Immrax Robustness Check...")
            immrax_report = immrax_val.validate_trajectory(samples_data, cof_range, torque_range, backlash_range)

            # Attach envelope to samples for export/viz (do this even if it failed, so user can see it)
            for i, env in enumerate(immrax_report['reachability']['envelope']):
                if i < len(samples_data):
                    samples_data[i]['reachability_envelope'] = env

            # CONTEXT-AWARE SAFETY CHECK
            failed_threshold = False
            max_vio_val = 0.0

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
                    if err > local_threshold:
                        failed_threshold = True
                        max_vio_val = max(max_vio_val, err)

            if not failed_threshold and immrax_report['passed']:
                if not quiet: click.echo(f"  Passed! All segments within context-aware thresholds.")
                passed_validation = True
                break
            else:
                if not quiet:
                    if failed_threshold:
                        click.echo(f"  FAILED: Max Tracking Error {max_vio_val*100:.2f} cm exceeds local threshold.")
                    else:
                        click.echo(f"  FAILED: Physical constraints (traction/motor) violated under uncertainty.")

                if not validate: # If we aren't validating, don't retry, just show it in live viz
                    passed_validation = True
                    break
                if attempt < max_retries - 1:
                    curr_accuracy_weight += 2.0 # More aggressive escalation for tighter bounds
                    if not quiet: click.echo(f"  Increasing accuracy_weight to {curr_accuracy_weight} and re-solving...")
                else:
                    if not quiet: click.echo("  Maximum retries reached. Using best-effort trajectory.")
        else:
            passed_validation = True
            break

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

    result = {
        "name": os.path.basename(output).split('.')[0],
        "version": 3,
        "trajectory": {
            "config": config_data.get("robot", config_data.get("config", {})),
            "samples": samples_data
        }
    }
    with open(output, 'w') as f:
        json.dump(result, f, indent=1)
    
    if not quiet:
        click.echo(f"Successfully saved trajectory to {output}")

    if validate:
        validate_trajectory(output, config)

    if export_format == 'controller':
        ctrl_output = os.path.splitext(output)[0] + '_controller.json'
        write_controller_file(output, ctrl_output, target_dt=controller_dt, track_width=robot_cfg.track_width)

    if export_format == 'python':
        py_output = os.path.splitext(output)[0] + '.py'
        write_python_file(output, py_output)

    if plot:
        plot_trajectory(samples_data, waypoints=wps, title=f"Trajectory: {os.path.basename(output)}")

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
