"""Review plots for equal-duration vs per-piece timing, on the current model.

Solves the complex mission and an uneven 3-waypoint path in both timing modes,
validates each trajectory with Immrax, and writes the figures.

The two modes are not peers. The equal-duration mode shares one dt across all
segments, which because every segment has the same sample count also forces
every segment to take the SAME time. The per-piece mode lets each segment
have its own duration and is the general model.

The wheel-force-step panel is the KI-024 property: the per-piece model may
re-time at a waypoint, but it may not step the wheel force there. The hatched
bars are waypoints the robot stops at, where the force has to reverse.

Run from the repository root:

    python docs/measurements/2026-10-20/dt_modes_plots.py
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
from immrax_validator import ImmraxValidator
from jax_robot_model import JAXRobotConfig
from validator import validate_samples

OUT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = 10


def radius(env):
    return float(np.hypot((env['x_max'] - env['x_min']) / 2.0,
                           (env['y_max'] - env['y_min']) / 2.0))


def midpoint_error(env, sample):
    mx = (env['x_min'] + env['x_max']) / 2.0
    my = (env['y_min'] + env['y_max']) / 2.0
    return float(np.hypot(mx - sample['x'], my - sample['y']))


def physical_jerk(samples):
    """Peak wheel jerk over the real intervals (the solver's term set)."""
    n = len(samples)
    t = [s['t'] for s in samples]
    worst = 0.0
    for k in range(n - 2):
        span = 0.5 * ((t[k + 1] - t[k]) + (t[k + 2] - t[k + 1]))
        if span <= 1e-9:
            continue
        worst = max(worst,
                    abs(samples[k + 1]['al'] - samples[k]['al']) / span,
                    abs(samples[k + 1]['ar'] - samples[k]['ar']) / span)
    return worst


def accel_step(samples, waypoint_index, samples_per_segment=SAMPLES):
    """Acceleration step across a waypoint, in the plan's own terms.

    A non-zero step means the wheel force profile has a step exactly at the
    point the path is supposed to run through smoothly (KI-024).
    """
    k = waypoint_index * samples_per_segment
    if k <= 0 or k >= len(samples):
        return 0.0
    return max(abs(samples[k]['al'] - samples[k - 1]['al']),
               abs(samples[k]['ar'] - samples[k - 1]['ar']))


def run(cfg, raw_cfg, ur, waypoints, stops, events, per_segment, weight, label):
    opt = TrajectoryOptimizer(cfg)
    samples, stats = opt.solve(
        waypoints, SAMPLES, weight, stops, events, verbose=False,
        guess_style='baseline', per_segment_dt=per_segment)
    val = ImmraxValidator(JAXRobotConfig(raw_cfg))
    rep = val.validate_trajectory(
        samples, ur['cof'], ur['torque_headroom'], ur['gear_backlash_m'],
        ur['track_width_range'], ur['wheel_radius_range'], ur['inertia_kg_m2'])
    envs = rep['reachability']['envelope']
    for i, env in enumerate(envs):
        if i < len(samples):
            samples[i]['reachability_envelope'] = env
    cpu = validate_samples(samples, cfg)
    jerk = physical_jerk(samples)
    print('%-24s t=%6.3f track=%6.3fcm env=%6.3fcm jerk=%7.2f immrax=%-5s cpu=%-5s dts=%s' % (
        label, samples[-1]['t'], rep['max_tracking_error_m'] * 100,
        rep['max_envelope_radius_m'] * 100, jerk, rep['passed'], cpu['passed'],
        ' '.join('%.3f' % d for d in stats['segment_dt'])))
    return {
        'label': label, 'samples': samples, 'envs': envs, 'rep': rep,
        'dts': stats['segment_dt'], 'per_segment': per_segment,
        'cpu_ok': cpu['passed'], 'jerk': jerk, 'stops': stops,
    }


def plot(runs, waypoints, title, path):
    colors = ['black', 'tab:red', 'tab:blue', 'tab:green']
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    ax_path, ax_env, ax_track, ax_dt, ax_step, ax_force = axes.ravel()

    for run_data, color in zip(runs, colors):
        samples = run_data['samples']
        envs = run_data['envs']
        rep = run_data['rep']
        gates = 'immrax %s, CPU %s' % (
            'pass' if rep['passed'] else 'fail',
            'pass' if run_data['cpu_ok'] else 'fail')
        label = '%s (%.2f s, %.2f cm, %.1f m/s3; %s)' % (
            run_data['label'], samples[-1]['t'],
            rep['max_envelope_radius_m'] * 100, run_data['jerk'], gates)
        style = '-' if run_data['cpu_ok'] else '--'
        ax_path.plot([s['x'] for s in samples], [s['y'] for s in samples],
                     color=color, linewidth=2, linestyle=style, label=label)
        ax_env.plot([s['t'] for s in samples],
                    [radius(e) * 100 for e in envs],
                    color=color, linewidth=2, linestyle=style, label=label)
        ax_track.plot([s['t'] for s in samples],
                      [midpoint_error(e, s) * 100 for e, s in zip(envs, samples)],
                      color=color, linewidth=2, linestyle=style, label=label)
        ax_force.plot([s['t'] for s in samples], [s['fl'] for s in samples],
                      color=color, linewidth=1.6, linestyle=style, label=label)

    seg_count = len(runs[0]['dts'])
    width = 0.8 / len(runs)
    x = np.arange(seg_count)
    for j, (run_data, color) in enumerate(zip(runs, colors)):
        durations = [d * SAMPLES for d in run_data['dts']]
        ax_dt.bar(x + (j - (len(runs) - 1) / 2.0) * width,
                  durations, width=width, color=color, label=run_data['label'])

    wp_indices = list(range(1, len(waypoints) - 1))
    x = np.arange(len(wp_indices))
    for j, (run_data, color) in enumerate(zip(runs, colors)):
        steps = [accel_step(run_data['samples'], i) for i in wp_indices]
        bars = ax_step.bar(x + (j - (len(runs) - 1) / 2.0) * width,
                           steps, width=width, color=color,
                           label=run_data['label'])
        for bar, i in zip(bars, wp_indices):
            if run_data['stops'] and i in run_data['stops']:
                bar.set_hatch('//')
                bar.set_edgecolor('black')

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
    if (max(xs) - min(xs)) > 0 and (max(ys) - min(ys)) > 0.05 * (max(xs) - min(xs)):
        ax_path.set_aspect('equal')
    ax_path.grid(True, linestyle='--', alpha=0.4)
    ax_path.legend(fontsize=8)

    ax_env.axhline(2.0, color='darkorange', linestyle=':', linewidth=1.2,
                   label='2 cm transit threshold')
    ax_env.axhline(5.0, color='black', linestyle='--', linewidth=1.2,
                   label='5 cm divergence guard')
    ax_env.set_xlabel('Time (s)')
    ax_env.set_ylabel('Tube radius (cm)')
    ax_env.set_title('Reachability tube')
    ax_env.grid(True, linestyle='--', alpha=0.4)
    ax_env.legend(fontsize=8)

    ax_track.axhline(0.5, color='darkorange', linestyle=':', linewidth=1.2,
                     label='0.5 cm tracking gate')
    ax_track.set_xlabel('Time (s)')
    ax_track.set_ylabel('Tracking error (cm)')
    ax_track.set_title('Closed-loop tracking error')
    ax_track.grid(True, linestyle='--', alpha=0.4)
    ax_track.legend(fontsize=8)

    ax_dt.set_xlabel('Segment index')
    ax_dt.set_ylabel('Segment duration (s)')
    ax_dt.set_title('Time spent per segment\n(equal-duration mode: every bar the same)')
    ax_dt.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_dt.legend(fontsize=8)

    ax_step.set_xticks(x)
    ax_step.set_xticklabels(['%d%s' % (i, ' (stop)' if runs[0]['stops'] and i in runs[0]['stops'] else '')
                             for i in wp_indices])
    ax_step.set_xlabel('Waypoint index')
    ax_step.set_ylabel('Acceleration step at the waypoint (m/s2)')
    ax_step.set_title('Wheel-force step through each waypoint\n'
                      '(hatched = a stop, where the force must reverse;\n'
                      'per-piece keeps the force continuous elsewhere, KI-024)')
    ax_step.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_step.legend(fontsize=8)

    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Left wheel force (N)')
    ax_force.set_title('Left wheel force profile')
    ax_force.grid(True, linestyle='--', alpha=0.4)
    ax_force.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def main():
    with open('examples/fll_choreo.json') as f:
        raw_cfg = json.load(f)
    cfg = RobotConfig(raw_cfg)
    ur = cfg.uncertainties
    with open('examples/example_complex_mission.json') as f:
        raw = json.load(f)
    mission = [(w['x'], w['y'], w.get('heading')) for w in raw]
    mission_stops = sorted(i for i, w in enumerate(raw) if w.get('stop'))
    mission_events = {i: w['event'] for i, w in enumerate(raw) if 'event' in w}
    uneven = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)]

    mission_runs = [
        run(cfg, raw_cfg, ur, mission, mission_stops, mission_events, False, 0.0, 'equal-duration w=0'),
        run(cfg, raw_cfg, ur, mission, mission_stops, mission_events, True, 0.0, 'per-piece w=0'),
        run(cfg, raw_cfg, ur, mission, mission_stops, mission_events, True, 0.02, 'per-piece w=0.02'),
    ]
    plot(mission_runs, mission,
         'Complex mission: equal-duration vs per-piece timing (validated)',
         os.path.join(OUT, 'dt_modes.png'))

    uneven_runs = [
        run(cfg, raw_cfg, ur, uneven, [], {}, False, 0.0, 'equal-duration w=0'),
        run(cfg, raw_cfg, ur, uneven, [], {}, True, 0.0, 'per-piece w=0'),
        run(cfg, raw_cfg, ur, uneven, [], {}, True, 0.005, 'per-piece w=0.005'),
    ]
    plot(uneven_runs, uneven,
         'Uneven path (1.0 m + 0.2 m): equal-duration vs per-piece timing\n'
         'the equal-duration mode spends 2.23 s on BOTH segments, so the 0.2 m hop\n'
         'crawls (dashed = CPU audit rejects)',
         os.path.join(OUT, 'uneven_dt_modes.png'))


if __name__ == '__main__':
    main()
