import time
import numpy as np
from robot_model import RobotConfig
from multiverse_optimizer import MasterTrajectoryOptimizer
from optimizer import TrajectoryOptimizer

def benchmark():
    config_dict = {
        "mass": 0.723,
        "inertia": 0.0024,
        "track_width": 0.0965,
        "wheel_radius": 0.028,
        "v_max_rad_s": 15.7,
        "t_max_nm": 0.04,
        "gearing": 1.0,
        "cof": 0.40,
        "gravity": 9.81,
        "torque_headroom": 0.85,
        "speed_headroom": 0.90
    }
    cfg = RobotConfig(config_dict)

    wps = [(0,0,0), (1,0,0), (2,1,np.pi/2)]
    print(f"--- Benchmarking Pipeline for {len(wps)} waypoints ---")

    # Simple
    opt_simple = TrajectoryOptimizer(cfg)
    start = time.time()
    samples_s, _ = opt_simple.solve(wps, verbose=False)
    t_simple = time.time() - start
    cost_s = samples_s[-1]['t']
    print(f"Simple CasADi Solve Time: {t_simple:.4f}s | Cost: {cost_s:.4f}s")

    # JAX
    opt_jax = MasterTrajectoryOptimizer(cfg, enable_parallel=True)
    # Warmup
    opt_jax.solve(wps, verbose=False)

    start = time.time()
    samples_j, stats = opt_jax.solve(wps, verbose=False)
    t_jax = time.time() - start
    cost_j = samples_j[-1]['t']
    print(f"JAX-Enhanced Solve Time: {t_jax:.4f}s | Cost: {cost_j:.4f}s")

    improvement = (cost_s - cost_j) / cost_s * 100
    print(f"Improvement in Cost: {improvement:.2f}%")

    # Phase 4 specific
    p4_time = stats['phase_times']['refinement']
    print(f"JAX Phase 4 (Refine) Time: {p4_time:.4f}s for {len(wps)-1} segments")

if __name__ == "__main__":
    benchmark()
