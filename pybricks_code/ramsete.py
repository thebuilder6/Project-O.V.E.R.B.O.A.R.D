import umath as math
import gc
from pybricks.tools import StopWatch, wait

# The control period, in milliseconds.
#
# The drive command is held until the next iteration, and that hold is what
# the wheel-force error is proportional to, so the validator models it
# (immrax_validator.py, CONTROL_DT) and the two have to agree. It is the same
# period as the two compact controllers and as ramsete_fast.py's 50 Hz
# heartbeat.
CONTROL_MS = 20

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def _fire_event(sample, event_map, debug=False, when=0.0):
    """Fire the event attached to one sample, if it has one.

    The sample is marked as fired, so the same event cannot fire twice.
    Returns True when an action was actually run.
    """
    name = sample.get('event')
    if not name or sample.get('event_fired', False):
        return False

    sample['event_fired'] = True

    if debug:
        print(f"[{when:.2f}s] Event triggered: {name}")

    # If a dictionary was passed and the event exists in it
    if event_map and name in event_map:
        event_map[name]()
        return True

    return False


class Odometry:
    """
    Tracks the global position of the robot using Pybricks DriveBase distance and angle.
    Assumes initial pose is (x, y, heading) where heading is in radians.
    """
    def __init__(self, drivebase, initial_x=0.0, initial_y=0.0, initial_heading=0.0):
        self.db = drivebase
        self.x = initial_x
        self.y = initial_y
        self.heading = initial_heading
        
        # DriveBase metrics
        self.last_dist = self.db.distance() / 1000.0  # Convert mm to meters
        # Pybricks angle is positive-clockwise. We invert it so positive is Counter-Clockwise (standard math)
        self.last_angle = math.radians(-self.db.angle())
        
    def update(self):
        """Updates the internal (x, y, heading) based on changes since last update."""
        dist = self.db.distance() / 1000.0
        angle = math.radians(-self.db.angle())
        
        d_dist = dist - self.last_dist
        d_angle = angle - self.last_angle
        
        # Use average heading during the move for slightly better integration
        avg_angle = self.heading + d_angle / 2.0
        
        self.x += d_dist * math.cos(avg_angle)
        self.y += d_dist * math.sin(avg_angle)
        self.heading += d_angle
        self.heading = normalize_angle(self.heading)
        
        self.last_dist = dist
        self.last_angle = angle

class RamseteController:
    """
    Ramsete Controller for differential drive robots.
    b: tuning parameter for convergence rate (> 0), typically 2.0
    zeta: damping ratio (0 to 1), typically 0.7
    """
    def __init__(self, b=2.0, zeta=0.7):
        self.b = b
        self.zeta = zeta

    def calculate(self, current_x, current_y, current_theta, ref_x, ref_y, ref_theta, ref_v, ref_omega):
        """
        Calculate command velocities.
        current_pose: (x, y, theta) - current robot pose
        ref_pose: (xr, yr, thetar) - reference pose on the path
        ref_v: reference linear velocity (m/s)
        ref_omega: reference angular velocity (rad/s)
        
        Returns: (v_cmd, omega_cmd) in (m/s, rad/s)
        """
        x, y, theta = current_x, current_y, current_theta
        xr, yr, thetar = ref_x, ref_y, ref_theta

        # Error in global frame
        ex_global = xr - x
        ey_global = yr - y

        # Rotate error to local robot frame
        ex = math.cos(theta) * ex_global + math.sin(theta) * ey_global
        ey = -math.sin(theta) * ex_global + math.cos(theta) * ey_global
        etheta = normalize_angle(thetar - theta)

        # Compute gains
        k1 = 2 * self.zeta * math.sqrt(ref_omega**2 + self.b * ref_v**2)
        
        # Command linear velocity
        v_cmd = ref_v * math.cos(etheta) + k1 * ex
        
        # Command angular velocity
        if abs(etheta) < 1e-6:
            sin_etheta_over_etheta = 1.0
        else:
            sin_etheta_over_etheta = math.sin(etheta) / etheta
            
        omega_cmd = ref_omega + self.b * ref_v * sin_etheta_over_etheta * ey + k1 * etheta

        return v_cmd, omega_cmd

class KinematicLimiter:
    """
    Limits the commands based on robot's physical capabilities.
    """
    def __init__(self, config, backlash_m=0.0004):
        self.mass = config.get("mass", 0.8)
        self.inertia = config.get("inertia", 0.001)
        self.track_width = config.get("track_width", 0.0965)
        self.wheel_radius = config.get("wheel_radius", 0.028)
        self.v_max_rad_s = config.get("v_max_rad_s", 15.7)
        self.t_max_nm = config.get("t_max_nm", 0.04)
        self.gearing = config.get("gearing", 1.0)
        self.cof = config.get("cof", 1.5)
        self.g = 9.81
        self.backlash_m = backlash_m
        
        self.last_vl = 0.0
        self.last_vr = 0.0
        
    def max_force(self, v_wheel):
        omega = (abs(v_wheel) / self.wheel_radius) * self.gearing
        torque = self.t_max_nm * max(0.0, 1.0 - omega / self.v_max_rad_s)
        return (torque / self.wheel_radius) * self.gearing

    def limit(self, v_cmd, omega_cmd, current_v, current_omega, dt):
        if dt <= 0:
            return v_cmd, omega_cmd
            
        L = self.track_width
        vl_cmd = v_cmd - omega_cmd * L / 2.0
        vr_cmd = v_cmd + omega_cmd * L / 2.0
        
        vl_curr = current_v - current_omega * L / 2.0
        vr_curr = current_v + current_omega * L / 2.0
        
        al_req = (vl_cmd - vl_curr) / dt
        ar_req = (vr_cmd - vr_curr) / dt
        
        a_req = (al_req + ar_req) / 2.0
        alpha_req = (ar_req - al_req) / L
        
        F_total = self.mass * a_req
        M_total = self.inertia * alpha_req
        
        Fr_req = (F_total + 2.0 * M_total / L) / 2.0
        Fl_req = F_total - Fr_req
        
        F_max_l = self.max_force(vl_curr)
        F_max_r = self.max_force(vr_curr)
        
        Fl_ach, Fr_ach = Fl_req, Fr_req
        
        scale_l = F_max_l / abs(Fl_ach) if abs(Fl_ach) > F_max_l else 1.0
        scale_r = F_max_r / abs(Fr_ach) if abs(Fr_ach) > F_max_r else 1.0
        scale = min(1.0, scale_l, scale_r)
        
        Fl_ach *= scale
        Fr_ach *= scale
        
        traction_max = self.cof * self.mass * self.g
        if abs(Fl_ach) + abs(Fr_ach) > traction_max:
            traction_scale = traction_max / (abs(Fl_ach) + abs(Fr_ach))
            Fl_ach *= traction_scale
            Fr_ach *= traction_scale
            
        F_total_ach = Fl_ach + Fr_ach
        M_total_ach = (Fr_ach - Fl_ach) * L / 2.0
        
        a_ach = F_total_ach / self.mass
        alpha_ach = M_total_ach / self.inertia
        
        al_ach = a_ach - alpha_ach * L / 2.0
        ar_ach = a_ach + alpha_ach * L / 2.0
        
        vl_ach = vl_curr + al_ach * dt
        vr_ach = vr_curr + ar_ach * dt
        
        # Backlash is accounted for as a position uncertainty in the reachability
        # envelope (KI-025), not as an artificial velocity boost on direction
        # reversal (KI-028). The limiter purely enforces physical torque and
        # traction limits on the commanded motion.
        self.last_vl = vl_ach
        self.last_vr = vr_ach
        
        v_final = (vl_ach + vr_ach) / 2.0
        omega_final = (vr_ach - vl_ach) / L
        
        return v_final, omega_final

async def follow_trajectory(drivebase, samples, config=None, b=2.0, zeta=0.7, debug=False, event_map=None):
    """
    Follows a trajectory using a Ramsete Controller.
    This runs asynchronously and yields to other multitask operations.
    
    samples: A list of dictionaries containing trajectory points.
    Expected dict format: {'t': float, 'x': float, 'y': float, 'heading': float, 'vl': float, 'vr': float, 'omega': float}
    config: Optional robot configuration dictionary for the KinematicLimiter.
    """
    if not samples:
        print("No samples provided.")
        return
    
    print(f"Following {len(samples)} samples.")
    
    num_samples = len(samples)
    sample_idx = 0


    # Initialize Odometry with the first sample
    odom = Odometry(drivebase, samples[0]['x'], samples[0]['y'], samples[0]['heading'])
    
    controller = RamseteController(b, zeta)
    limiter = KinematicLimiter(config if config else {}, backlash_m=0.0004)
    
    watch = StopWatch()
    total_time = samples[-1]['t']
    
    last_t = 0.0
    current_v = 0.0
    current_omega = 0.0
    next_loop = watch.time()

    # Profiling variables
    loop_count = 0
    total_calc_time = 0.0
    max_calc_time = 0.0
    max_dt = 0.0
    
    # OPTIMIZATION: Force a clean garbage collection BEFORE we start driving
    gc.collect()
    gc.threshold(4096)

    # Run the control loop
    while True:
        t = watch.time() / 1000.0  # seconds
        
        if t >= total_time:
            break
            
        calc_start = watch.time()
            
        # --- EVENT HANDLING ---
        #
        # Fire the event of every sample the robot has reached, not only of
        # the one it is on. A tick that arrives late jumps the index over
        # more than one sample, and the event of a sample that was jumped
        # over would then never be looked at: the action would be silently
        # lost (KI-029). The compact controller, the one that runs on the
        # hub, fires its events the same way.
        while sample_idx < num_samples - 2 and samples[sample_idx + 1]['t'] <= t:
            sample_idx += 1
            _fire_event(samples[sample_idx], event_map, debug, t)
        _fire_event(samples[sample_idx], event_map, debug, t)

        s1 = samples[sample_idx]
        s2 = samples[sample_idx + 1]

        # Interpolate between s1 and s2
        if s2['t'] == s1['t']:
            alpha = 0.0
        else:
            alpha = (t - s1['t']) / (s2['t'] - s1['t'])
            
        xr = s1['x'] + alpha * (s2['x'] - s1['x'])
        yr = s1['y'] + alpha * (s2['y'] - s1['y'])
        
        # Handle heading interpolation around -pi/pi boundaries
        dh = normalize_angle(s2['heading'] - s1['heading'])
        thetar = normalize_angle(s1['heading'] + alpha * dh)
        
        # Calculate reference velocities
        # v = (vl + vr) / 2
        v1 = (s1['vl'] + s1['vr']) / 2.0
        v2 = (s2['vl'] + s2['vr']) / 2.0
        ref_v = v1 + alpha * (v2 - v1)

        # The turn rate is the derivative of the heading, not an independent
        # line: interpolating both separately makes them describe different
        # motions (KI-027).
        if s2['t'] == s1['t']:
            ref_omega = 0.0
        else:
            ref_omega = dh / (s2['t'] - s1['t'])

        # Update current robot pose
        odom.update()
        
        # Calculate control commands
        v_cmd, omega_cmd = controller.calculate(odom.x, odom.y, odom.heading, xr, yr, thetar, ref_v, ref_omega)
        
        # Limit commands based on physical capabilities
        dt = t - last_t
        if dt > 0:
            v_cmd, omega_cmd = limiter.limit(v_cmd, omega_cmd, current_v, current_omega, dt)
            max_dt = max(max_dt, dt)
        
        last_t = t
        current_v = v_cmd
        current_omega = omega_cmd
        
        calc_end = watch.time()
        calc_time = calc_end - calc_start
        total_calc_time += calc_time
        max_calc_time = max(max_calc_time, calc_time)
        loop_count += 1
        
        # Convert m/s to mm/s, and rad/s to deg/s
        v_mm_s = v_cmd * 1000.0
        # Pybricks expects positive turn_rate to be clockwise, so we invert our CCW-positive omega
        omega_deg_s = math.degrees(-omega_cmd)
        
        # Send commands to the DriveBase
        drivebase.drive(v_mm_s, omega_deg_s)
        
        if debug:
            # Print telemetry for external plotting: LOG,time,x,y,ref_x,ref_y
            print(f"LOG,{t:.3f},{odom.x:.4f},{odom.y:.4f},{xr:.4f},{yr:.4f}")
        
        # The 20 ms heartbeat, synced so the period is exact.
        #
        # The drive command is held until the next iteration, and that hold
        # is what the wheel-force error is proportional to, so the validator
        # models it (immrax_validator.py, CONTROL_DT) and the two have to
        # agree. A free-running loop would make the model's hold a lower
        # bound on the robot's; a synced one is exact.
        #
        # Await wait() also hands control back to the Pybricks multitask
        # loop, so other tasks (attachments) keep running.
        next_loop += CONTROL_MS
        slack = next_loop - watch.time()
        if slack > 0:
            await wait(slack)
        else:
            next_loop = watch.time()
            await wait(1)
    # --- LOOP END ---
        
    print("Trajectory complete.")
    if loop_count > 0:
        avg_calc = total_calc_time / loop_count
        avg_dt = (total_time * 1000.0) / loop_count
        print("--- Performance Metrics ---")
        print(f"Total Loops: {loop_count}")
        print(f"Avg Loop Time (dt): {avg_dt:.1f} ms ({1000.0/avg_dt:.1f} Hz)")
        print(f"Max Loop Time (dt): {max_dt * 1000.0:.1f} ms")
        print(f"Avg Math Calc Time: {avg_calc:.1f} ms")
        print(f"Max Math Calc Time: {max_calc_time:.1f} ms")
        print("---------------------------")
    drivebase.stop()

    # The loop stops before the last sample, so an event attached to the final
    # waypoint never fired (KI-012). Fire it now: the robot has arrived.
    _fire_event(samples[-1], event_map, debug, total_time)

    gc.threshold(-1)
