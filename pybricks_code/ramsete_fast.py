import umath as math
import gc
from micropython import const
from pybricks.tools import StopWatch, wait

# --- COMPILED CONSTANTS ---
_PI = const(3141593) / 1000000.0
_2PI = const(6283185) / 1000000.0
_R2D = const(5729578) / 100000.0
_D2R = const(17453) / 1000000.0
_I1000 = const(1) / 1000.0

async def follow_trajectory(drivebase, samples, config=None, b=2.0, zeta=0.7):
    if not samples: return
    
    # 1. RAM-SAFE PRE-PROCESS (Flat List)
    data = []
    while samples:
        s = samples.pop(0)
        v = (s['vl'] + s['vr']) * 0.5
        # Pre-calculating the Ramsete Gain and extending the list
        data.extend([
            float(s['t']), float(s['x']), float(s['y']), float(s['heading']), 
            float(v), float(s['omega']), math.sqrt(s['omega']**2 + b * v**2)
        ])

    num_pts = len(data) // 7
    total_time = data[(num_pts-1)*7]
    
    # Cache everything locally to bypass object attribute lookups
    db_state = drivebase.state
    db_drive = drivebase.drive
    m_cos, m_sin = math.cos, math.sin
    
    # Get Initial Hardware State
    # state() returns (dist_mm, speed_mms, angle_deg, turn_deg)
    d_mm, _, a_deg, _ = db_state()
    
    # Initial Pose
    x, y, h = data[1], data[2], data[3]
    ld, la = d_mm * _I1000, -a_deg * _D2R
    
    # Metric Trackers
    l_cnt, t_math, m_math, m_dt = 0, 0.0, 0.0, 0.0
    s_idx, l_t, c_v, c_w = 0, 0.0, 0.0, 0.0
    
    # --- THE 50HZ HEARTBEAT ---
    # 20ms is the "Goldilocks" zone for Build HAT serial communication.
    LOOP_MS = const(20) 
    watch = StopWatch()
    w_time = watch.time
    next_loop = w_time()

    print("Mission Start: Perfect 50Hz Sync...")

    try:
        while True:
            # 1. TIME SYNC
            now_ms = w_time()
            t = now_ms * _I1000
            if t >= total_time: break
            
            calc_start = now_ms

            # 2. FAST SEEK
            while s_idx < num_pts - 2 and data[(s_idx+1)*7] <= t:
                s_idx += 1
            p1 = s_idx * 7
            p2 = p1 + 7
            
            # 3. INTERPOLATION
            dt_traj = data[p2] - data[p1]
            alpha = (t - data[p1]) / dt_traj if dt_traj > 0 else 0.0
            
            xr = data[p1+1] + alpha * (data[p2+1] - data[p1+1])
            yr = data[p1+2] + alpha * (data[p2+2] - data[p1+2])
            dh = (data[p2+3] - data[p1+3] + _PI) % _2PI - _PI
            tr = (data[p1+3] + alpha * dh + _PI) % _2PI - _PI
            vr = data[p1+4] + alpha * (data[p2+4] - data[p1+4])
            wr = data[p1+5] + alpha * (data[p2+5] - data[p1+5])
            kr = data[p1+6] + alpha * (data[p2+6] - data[p1+6])

            # 4. SYNCHRONOUS ODOMETRY
            # db_state() blocks until the hardware bus is ready (~16-17ms).
            d_mm, _, a_deg, _ = db_state()
            d, a = d_mm * _I1000, -a_deg * _D2R
            
            dd, da = d - ld, a - la
            ah = h + da * 0.5
            x += dd * m_cos(ah)
            y += dd * m_sin(ah)
            h = (h + da + _PI) % _2PI - _PI
            ld, la = d, a

            # 5. RAMSETE LOGIC
            dx, dy = xr - x, yr - y
            ct, st = m_cos(h), m_sin(h)
            ex, ey = ct * dx + st * dy, -st * dx + ct * dy
            et = (tr - h + _PI) % _2PI - _PI
            
            sinc = 1.0 if abs(et) < 1e-6 else m_sin(et) / et
            kg = 2 *zeta * kr
            
            c_v = vr * m_cos(et) + kg * ex
            c_w = wr + b * vr * sinc * ey + kg * et

            # 6. HARDWARE DRIVE
            db_drive(c_v * 1000.0, -c_w * _R2D)
            
            # 7. PERFORMANCE METRICS
            dt_loop = t - l_t
            if dt_loop > m_dt: m_dt = dt_loop
            l_t = t
            
            dur = w_time() - calc_start
            t_math += dur
            if dur > m_math: m_math = dur
            l_cnt += 1

            # 8. PRECISION WAIT & MICRO-GC
            # We target a 20ms window. If the loop finished in 18ms, 
            # we have 2ms to clean memory.
            next_loop += LOOP_MS
            slack = next_loop - w_time()
            
            if slack > 1:
                gc.collect() # Quick clean
                slack = next_loop - w_time()
            
            if slack > 0:
                await wait(slack)
            else:
                # If we missed the window, don't accumulate delay
                next_loop = w_time()
                await wait(1)

    finally:
        db_drive(0, 0)
        
    if l_cnt > 0:
        print("Final Hz:", (l_cnt / total_time))
        print("Avg/Max Math:", (t_math / l_cnt), "/", m_math, "ms")
        print("Max Jitter (dt):", (m_dt * 1000.0), "ms")
