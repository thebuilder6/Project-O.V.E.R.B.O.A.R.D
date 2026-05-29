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
            click.echo(f"Using Multi-Verse optimizer (parallel={parallel}, workers={workers})")
        optimizer = MasterTrajectoryOptimizer(robot_cfg, enable_parallel=parallel, num_workers=workers, verbose=not quiet)
    
    if not quiet:
        click.echo(f"Loading waypoints from {waypoints}...")
    with open(waypoints, 'r') as f:
        wp_data = json.load(f)
    
    # Expected wp_data: list of objects with x, y, and optionally heading and event
    # or list of lists [x, y, heading]
    wps = []
    waypoint_events = {}  # index -> event name
    json_stop_indices = []
    for i, item in enumerate(wp_data):
        if isinstance(item, dict):
            wps.append((item['x'], item['y'], item.get('heading')))
            if 'event' in item:
                waypoint_events[i] = item['event']
            if item.get('stop'):
                json_stop_indices.append(i)
        else:
            # Assume [x, y, heading]
            wps.append((item[0], item[1], item[2] if len(item) > 2 else None))

    if not quiet:
        click.echo(f"Optimizing trajectory through {len(wps)} waypoints (accuracy_weight={accuracy_weight})...")

    # Enable iteration capture if convergence visualization is requested
    capture_iterations = show_convergence

    # Parse stop waypoints
    stop_indices = []
    if stop_waypoints:
        try:
            stop_indices = [int(x.strip()) for x in stop_waypoints.split(',')]
            if not quiet:
                click.echo(f"Stop waypoints at indices: {stop_indices}")
        except ValueError:
            click.echo("Invalid stop waypoints format. Use comma-separated indices (e.g., '2,5,7').")

    # Parse events from CLI (overrides JSON)
    if events:
        try:
            for pair in events.split(','):
                idx_str, event_name = pair.strip().split(':')
                waypoint_events[int(idx_str.strip())] = event_name.strip()
            if not quiet:
                click.echo(f"Events at waypoints: {waypoint_events}")
        except ValueError:
            click.echo("Invalid events format. Use 'index:event' pairs separated by commas (e.g., '2:lower_arm,5:release').")

    # Combine JSON and CLI stop indices
    all_stop_indices = list(set(stop_indices + json_stop_indices))
    if not quiet and all_stop_indices:
        click.echo(f"Final stop waypoints: {all_stop_indices}")

    # Start live visualizer if requested
    if live:
        viz = get_visualizer()
        # Convert waypoints to dictionaries for JSON serialization
        wp_dicts = [{"x": w[0], "y": w[1], "heading": w[2]} for w in wps]
        viz.send_config(config_data, wp_dicts)
        if not quiet:
            click.echo("Live visualizer started. Open viz/index.html in your browser.")
            
    samples_data, stats = optimizer.solve(wps, num_samples_per_segment=samples, accuracy_weight=accuracy_weight, stop_waypoint_indices=all_stop_indices, waypoint_events=waypoint_events, verbose=not quiet, capture_iterations=capture_iterations, live_viz=live)
    
    if benchmark:
        # Enrich stats with quality metrics and config
        from validator import compute_metrics
        quality_metrics = compute_metrics(samples_data, robot_cfg)
        stats["quality_metrics"] = quality_metrics
        stats["robot_config"] = {
            "mass": robot_cfg.mass,
            "inertia": robot_cfg.inertia,
            "track_width": robot_cfg.track_width,
            "wheel_radius": robot_cfg.wheel_radius,
            "v_max_rad_s": robot_cfg.v_max_rad_s,
            "t_max_nm": robot_cfg.t_max_nm,
            "gearing": robot_cfg.gearing,
            "cof": robot_cfg.cof
        }

        stats_output = os.path.splitext(output)[0] + '_stats.json'
        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        if not quiet:
            click.echo(f"Benchmarking data saved to {stats_output}")

    # Construct Choreo-like output
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

    v_results = (None, None, None)
    if validate:
        v_results = validate_trajectory(output, config)

        if benchmark:
            # Save validation summary in benchmark mode
            metrics, audit, errors = v_results
            summary = {
                "metrics": metrics,
                "audit": audit,
                "errors": errors
            }
            summary_file = os.path.splitext(output)[0] + '_validation.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

    if export_format == 'controller':
        ctrl_output = os.path.splitext(output)[0] + '_controller.json'
        write_controller_file(output, ctrl_output, target_dt=controller_dt,
                              track_width=robot_cfg.track_width)

    if export_format == 'python':
        py_output = os.path.splitext(output)[0] + '.py'
        write_python_file(output, py_output)

    if plot:
        v_metrics, v_audit, v_errors = v_results
        plot_trajectory(samples_data, waypoints=wps,
                        title=f"Trajectory: {os.path.basename(output)}",
                        metrics=v_metrics, audit=v_audit, errors=v_errors)

    if animate:
        animate_trajectory(samples_data, waypoints=wps,
                          title=f"Trajectory: {os.path.basename(output)}")
    
    if show_convergence:
        if hasattr(optimizer, 'iteration_history') and optimizer.iteration_history:
            if not quiet:
                click.echo(f"Showing convergence visualization ({convergence_mode} mode)...")
            if convergence_animate:
                if convergence_output:
                    animate_convergence(optimizer.iteration_history, waypoints=wps,
                                     title=f"Convergence: {os.path.basename(output)}",
                                     save_path=convergence_output)
                    if not quiet:
                        click.echo(f"Convergence animation saved to {convergence_output}")
                else:
                    animate_convergence(optimizer.iteration_history, waypoints=wps,
                                     title=f"Convergence: {os.path.basename(output)}")
            else:
                if convergence_output:
                    # Save to HTML file
                    plot_convergence(optimizer.iteration_history, mode=convergence_mode, 
                                   waypoints=wps, title=f"Convergence: {os.path.basename(output)}",
                                   output_file=convergence_output)
                    if not quiet:
                        click.echo(f"Convergence plot saved to {convergence_output}")
                else:
                    # Open in browser
                    plot_convergence(optimizer.iteration_history, mode=convergence_mode, 
                                   waypoints=wps, title=f"Convergence: {os.path.basename(output)}")
        else:
            if not quiet:
                click.echo("No iteration history available for convergence visualization.")

    if live:
        viz = get_visualizer()

        def on_regenerate(window_label):
            click.echo(f"Regenerating window: {window_label}")
            try:
                # Parse window label "W0-W2"
                import re
                match = re.match(r"W(\d+)-W(\d+)", window_label)
                if match:
                    w_start, w_end = int(match.group(1)), int(match.group(2))

                    # Window data extraction
                    start_idx = w_start * samples
                    end_idx = min(w_end * samples, len(samples_data) - 1)

                    start_state = (
                        samples_data[start_idx]['x'], samples_data[start_idx]['y'],
                        samples_data[start_idx]['heading'], samples_data[start_idx]['vl'],
                        samples_data[start_idx]['vr']
                    )
                    end_state = (
                        samples_data[end_idx]['x'], samples_data[end_idx]['y'],
                        samples_data[end_idx]['heading'], samples_data[end_idx]['vl'],
                        samples_data[end_idx]['vr']
                    )

                    num_window_samples = end_idx - start_idx + 1

                    # Original dt for comparison
                    original_dt = (samples_data[end_idx]['t'] - samples_data[start_idx]['t']) / max(num_window_samples - 1, 1)

                    base_guess = np.zeros(1 + num_window_samples * 5)
                    base_guess[0] = original_dt if original_dt > 0.001 else 0.1
                    for j in range(num_window_samples):
                        sample = samples_data[start_idx + j]
                        base_guess[1 + j * 5] = sample['x']
                        base_guess[1 + j * 5 + 1] = sample['y']
                        base_guess[1 + j * 5 + 2] = sample['heading']
                        base_guess[1 + j * 5 + 3] = sample['vl']
                        base_guess[1 + j * 5 + 4] = sample['vr']

                    # Trigger refinement
                    refined, best_name = optimizer.refiner.refine_segment(
                        start_state, end_state, num_window_samples, base_guess,
                        live_viz=True, window_label=window_label)

                    if best_name != "None":
                        click.echo(f"Regeneration successful with {best_name}")
                        # Update samples_data
                        dt_val = refined[0]
                        states = refined[1:].reshape((num_window_samples, 5))
                        for j in range(num_window_samples):
                            idx = start_idx + j
                            samples_data[idx]['x'] = float(states[j, 0])
                            samples_data[idx]['y'] = float(states[j, 1])
                            samples_data[idx]['heading'] = float(states[j, 2])
                            samples_data[idx]['vl'] = float(states[j, 3])
                            samples_data[idx]['vr'] = float(states[j, 4])

                        # Re-broadcast updated trajectory
                        viz.send_trajectory(samples_data, phase="regenerated")
                    else:
                        click.echo("Regeneration failed to find a better path.")
            except Exception as e:
                click.echo(f"Error during regeneration: {e}")

        viz.on_regenerate = on_regenerate

        click.echo("\nLive visualizer is active. You can interact via the web interface.")
        click.echo("Press Enter to stop and exit...")
        input()
        viz.stop()

if __name__ == '__main__':
    main()
