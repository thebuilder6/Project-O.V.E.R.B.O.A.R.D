import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from matplotlib.gridspec import GridSpec


def animate_trajectory(samples, waypoints=None, title="Robot Trajectory Animation", interval=50):
    """
    Animate robot trajectory with real-time data display.
    
    Args:
        samples: list of trajectory sample dicts
        waypoints: optional list of (x, y, heading_or_None) tuples
        title: plot title
        interval: animation interval in milliseconds
    """
    t = [s['t'] for s in samples]
    x = [s['x'] for s in samples]
    y = [s['y'] for s in samples]
    h = [s['heading'] for s in samples]
    vl = [s['vl'] for s in samples]
    vr = [s['vr'] for s in samples]
    al = [s['al'] for s in samples]
    ar = [s['ar'] for s in samples]
    fl = [s['fl'] for s in samples]
    fr = [s['fr'] for s in samples]
    omega = [s['omega'] for s in samples]
    
    # Compute additional metrics
    v_lin = [(vl[i] + vr[i]) / 2 for i in range(len(samples))]
    curvature = []
    for i in range(len(samples)):
        if i < len(samples) - 1:
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            ds = np.sqrt(dx**2 + dy**2)
            if ds > 1e-6:
                dtheta = (h[i+1] - h[i] + np.pi) % (2*np.pi) - np.pi
                curvature.append(abs(dtheta / ds))
            else:
                curvature.append(0)
        else:
            curvature.append(curvature[-1] if curvature else 0)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Top-down path (main plot, spans 2x2)
    ax_path = fig.add_subplot(gs[0:2, 0:2])
    
    # 2. Velocity profiles
    ax_vel = fig.add_subplot(gs[0, 2])
    
    # 3. Acceleration profiles
    ax_acc = fig.add_subplot(gs[1, 2])
    
    # 4. Wheel forces
    ax_force = fig.add_subplot(gs[2, 0])
    
    # 5. Heading and angular velocity
    ax_heading = fig.add_subplot(gs[2, 1])
    
    # 6. Curvature
    ax_curv = fig.add_subplot(gs[2, 2])
    
    # Initialize plots
    # Path plot
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=min(v_lin), vmax=max(v_lin))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=2.5, alpha=0.6)
    lc.set_array(v_lin[:-1])
    ax_path.add_collection(lc)
    plt.colorbar(ScalarMappable(norm=norm, cmap='plasma'),
                 ax=ax_path, label='Speed (m/s)', shrink=0.8)
    
    # Waypoint overlays
    if waypoints:
        arrow_len = 0.12
        for idx, wp in enumerate(waypoints):
            wx, wy = wp[0], wp[1]
            wh = wp[2] if len(wp) > 2 else None
            ax_path.plot(wx, wy, marker='*', markersize=18,
                       color='gold', markeredgecolor='darkorange',
                       markeredgewidth=1.2, zorder=5)
            if wh is not None:
                ax_path.annotate('', xy=(wx + arrow_len*np.cos(wh),
                                       wy + arrow_len*np.sin(wh)),
                               xytext=(wx, wy),
                               arrowprops=dict(arrowstyle='->', color='darkorange',
                                             lw=2.5), zorder=6)
            ax_path.text(wx, wy + 0.05, f'W{idx}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='darkorange', zorder=7)
    
    # Robot marker (will be animated)
    robot_marker, = ax_path.plot([], [], marker='o', markersize=12, 
                                  color='steelblue', markeredgecolor='white',
                                  markeredgewidth=2, zorder=10)
    robot_heading_line, = ax_path.plot([], [], color='red', linewidth=2, zorder=10)
    trail_line, = ax_path.plot([], [], color='steelblue', linewidth=1.5, alpha=0.5)
    
    ax_path.autoscale()
    ax_path.set_xlabel('X (m)')
    ax_path.set_ylabel('Y (m)')
    ax_path.set_title('Robot Path (animated)')
    ax_path.set_aspect('equal')
    ax_path.grid(True, linestyle='--', alpha=0.5)
    
    # Velocity plot
    line_vl, = ax_vel.plot([], [], color='tomato', label='Left wheel')
    line_vr, = ax_vel.plot([], [], color='mediumseagreen', label='Right wheel')
    line_vlin, = ax_vel.plot([], [], 'k--', linewidth=1.5, label='Linear')
    current_vl = ax_vel.axhline(y=0, color='tomato', linestyle=':', alpha=0.7)
    current_vr = ax_vel.axhline(y=0, color='mediumseagreen', linestyle=':', alpha=0.7)
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('Velocity Profiles')
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, linestyle='--', alpha=0.5)
    
    # Acceleration plot
    line_al, = ax_acc.plot([], [], color='tomato', label='Left accel')
    line_ar, = ax_acc.plot([], [], color='mediumseagreen', label='Right accel')
    current_al = ax_acc.axhline(y=0, color='tomato', linestyle=':', alpha=0.7)
    current_ar = ax_acc.axhline(y=0, color='mediumseagreen', linestyle=':', alpha=0.7)
    ax_acc.set_xlabel('Time (s)')
    ax_acc.set_ylabel('Acceleration (m/s²)')
    ax_acc.set_title('Acceleration Profiles')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, linestyle='--', alpha=0.5)
    
    # Force plot
    line_fl, = ax_force.plot([], [], color='tomato', label='Left force')
    line_fr, = ax_force.plot([], [], color='mediumseagreen', label='Right force')
    current_fl = ax_force.axhline(y=0, color='tomato', linestyle=':', alpha=0.7)
    current_fr = ax_force.axhline(y=0, color='mediumseagreen', linestyle=':', alpha=0.7)
    ax_force.axhline(0, color='black', linewidth=0.8)
    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Force (N)')
    ax_force.set_title('Wheel Forces')
    ax_force.legend(fontsize=8)
    ax_force.grid(True, linestyle='--', alpha=0.5)
    
    # Heading plot
    line_heading, = ax_heading.plot([], [], color='steelblue', label='Heading')
    line_omega, = ax_heading.plot([], [], color='purple', linestyle='--', label='Angular vel')
    current_heading = ax_heading.axhline(y=0, color='steelblue', linestyle=':', alpha=0.7)
    ax_heading.set_xlabel('Time (s)')
    ax_heading.set_ylabel('Heading (rad) / ω (rad/s)')
    ax_heading.set_title('Heading & Angular Velocity')
    ax_heading.legend(fontsize=8)
    ax_heading.grid(True, linestyle='--', alpha=0.5)
    
    # Curvature plot
    line_curv, = ax_curv.plot([], [], color='orange', label='Curvature')
    current_curv = ax_curv.axhline(y=0, color='orange', linestyle=':', alpha=0.7)
    ax_curv.set_xlabel('Time (s)')
    ax_curv.set_ylabel('Curvature (rad/m)')
    ax_curv.set_title('Path Curvature')
    ax_curv.legend(fontsize=8)
    ax_curv.grid(True, linestyle='--', alpha=0.5)
    
    # Set axis limits
    ax_vel.set_xlim(min(t), max(t))
    ax_vel.set_ylim(min(min(vl), min(vr)) - 0.1, max(max(vl), max(vr)) + 0.1)
    
    ax_acc.set_xlim(min(t), max(t))
    ax_acc.set_ylim(min(min(al), min(ar)) - 0.5, max(max(al), max(ar)) + 0.5)
    
    ax_force.set_xlim(min(t), max(t))
    ax_force.set_ylim(min(min(fl), min(fr)) - 1, max(max(fl), max(fr)) + 1)
    
    ax_heading.set_xlim(min(t), max(t))
    ax_heading.set_ylim(min(min(h), min(omega)) - 0.5, max(max(h), max(omega)) + 0.5)
    
    ax_curv.set_xlim(min(t), max(t))
    ax_curv.set_ylim(0, max(curvature) + 0.1)
    
    # Animation update function
    def update(frame):
        # Update robot position
        robot_marker.set_data([x[frame]], [y[frame]])
        
        # Update heading line
        heading_len = 0.15
        hx = [x[frame], x[frame] + heading_len * np.cos(h[frame])]
        hy = [y[frame], y[frame] + heading_len * np.sin(h[frame])]
        robot_heading_line.set_data(hx, hy)
        
        # Update trail
        trail_len = max(1, frame - 20)
        trail_line.set_data(x[max(0, trail_len):frame], y[max(0, trail_len):frame])
        
        # Update velocity lines
        line_vl.set_data(t[:frame], vl[:frame])
        line_vr.set_data(t[:frame], vr[:frame])
        line_vlin.set_data(t[:frame], v_lin[:frame])
        current_vl.set_ydata([vl[frame]])
        current_vr.set_ydata([vr[frame]])
        
        # Update acceleration lines
        line_al.set_data(t[:frame], al[:frame])
        line_ar.set_data(t[:frame], ar[:frame])
        current_al.set_ydata([al[frame]])
        current_ar.set_ydata([ar[frame]])
        
        # Update force lines
        line_fl.set_data(t[:frame], fl[:frame])
        line_fr.set_data(t[:frame], fr[:frame])
        current_fl.set_ydata([fl[frame]])
        current_fr.set_ydata([fr[frame]])
        
        # Update heading lines
        line_heading.set_data(t[:frame], h[:frame])
        line_omega.set_data(t[:frame], omega[:frame])
        current_heading.set_ydata([h[frame]])
        
        # Update curvature
        line_curv.set_data(t[:frame], curvature[:frame])
        current_curv.set_ydata([curvature[frame]])
        
        return (robot_marker, robot_heading_line, trail_line, line_vl, line_vr, 
                line_vlin, current_vl, current_vr, line_al, line_ar, current_al, 
                current_ar, line_fl, line_fr, current_fl, current_fr, line_heading,
                line_omega, current_heading, line_curv, current_curv)
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(samples), 
                                   interval=interval, blit=True, repeat=True)
    
    plt.show()
    return anim


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        traj_file = sys.argv[1]
        wp_file = sys.argv[2] if len(sys.argv) > 2 else None
        with open(traj_file, 'r') as f:
            data = json.load(f)
        wps = None
        if wp_file:
            with open(wp_file, 'r') as f:
                raw = json.load(f)
            wps = [(w['x'], w['y'], w.get('heading')) for w in raw]
        animate_trajectory(data['trajectory']['samples'], waypoints=wps,
                          title=traj_file)
