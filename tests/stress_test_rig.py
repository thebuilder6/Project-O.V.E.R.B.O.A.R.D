import json
import time
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
from multiverse_optimizer import MasterTrajectoryOptimizer
from validator import compute_metrics

class StressTestRig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config_data = json.load(f)
        self.config = RobotConfig(self.config_data)
        self.simple_opt = TrajectoryOptimizer(self.config)
        self.mv_opt = MasterTrajectoryOptimizer(self.config)
        self.results = []

    def generate_scenarios(self) -> Dict[str, List[Tuple[float, float, Optional[float]]]]:
        scenarios = {
            "Tight_U_Turn": [
                (0.0, 0.0, 0.0),
                (1.0, 0.05, None),
                (0.0, 0.1, 3.14159)
            ],
            "Zig_Zag": [
                (0.0, 0.0, 0.0),
                (0.5, 0.5, None),
                (1.0, 0.0, None),
                (1.5, 0.5, None),
                (2.0, 0.0, 0.0)
            ],
            "The_Loop_Trap": [
                (0.0, 0.0, 0.0),
                (0.5, 0.0, None), # Middle point unconstrained
                (1.0, 0.0, 0.0)
            ],
            "Sharp_Reversal": [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 3.14159), # Force 180 at same X, but it has to move
                (2.0, 0.0, 0.0)
            ],
            "Unconstrained_Sequence": [
                (0.0, 0.0, 0.0),
                (0.5, 0.2, None),
                (1.0, -0.2, None),
                (1.5, 0.2, None),
                (2.0, 0.0, 0.0)
            ],
            "False_Loop": [
                (0.0, 0.0, 0.0),
                (0.5, 0.0, None), # Straight path but unconstrained
                (1.0, 0.0, 0.0)
            ],
            "Devious_Loop": [
                (0.0, 0.0, 0.0),
                (0.5, 0.1, None),
                (0.6, 0.1, None),
                (1.1, 0.0, 0.0)
            ],
            "Spiral_Trap": [
                (0.0, 0.0, 0.0),
                (0.5, 0.5, None),
                (0.0, 1.0, None),
                (-0.5, 0.5, None),
                (0.0, 0.0, 0.0)
            ],
            "Heading_Flip_Win": [
                (0.0, 0.0, 0.0),
                (0.5, 0.05, None),
                (0.5, -0.05, None),
                (1.0, 0.0, 0.0)
            ]
        }
        return scenarios

    def run_test(self, name: str, waypoints: List[Tuple[float, float, Optional[float]]], accuracy_weight: float = 1.0):
        print(f"\n>>> Running Scenario: {name} (accuracy_weight={accuracy_weight})")

        # Simple Optimizer
        print("  Testing Simple Optimizer...")
        start = time.time()
        simple_traj, simple_stats = self.simple_opt.solve(waypoints, accuracy_weight=accuracy_weight, verbose=False)
        simple_time = time.time() - start
        simple_metrics = compute_metrics(simple_traj, self.config)

        # Multi-Verse Optimizer
        print("  Testing Multi-Verse Optimizer...")
        # Force 1 worker and no parallel to avoid Segfault in sandbox
        self.mv_opt.num_workers = 1
        self.mv_opt.enable_parallel = False
        self.mv_opt.refiner.num_workers = 1
        self.mv_opt.refiner.enable_parallel = False

        start = time.time()
        mv_traj, mv_stats = self.mv_opt.solve(waypoints, accuracy_weight=accuracy_weight, verbose=False)
        mv_time = time.time() - start
        mv_metrics = compute_metrics(mv_traj, self.config)

        result = {
            "scenario": name,
            "accuracy_weight": accuracy_weight,
            "simple": {
                "solve_time": simple_time,
                "cost": simple_stats.get("final_cost", 0.0),
                "metrics": simple_metrics
            },
            "multiverse": {
                "solve_time": mv_time,
                "cost": mv_stats.get("final_cost", 0.0),
                "metrics": mv_metrics,
                "improvement_pct": mv_stats.get("improvement_pct", 0.0),
                "refinements_solved": mv_stats.get("refinements_solved", 0),
                "heuristic_wins": mv_stats.get("heuristic_wins", [])
            }
        }
        self.results.append(result)
        self._print_comparison(result)

    def _print_comparison(self, r: Dict[str, Any]):
        s = r["simple"]
        m = r["multiverse"]
        print(f"  Comparison for {r['scenario']}: (Refinements: {m['refinements_solved']})")
        if m["heuristic_wins"]:
            print(f"    Heuristic Wins:")
            for win in m["heuristic_wins"]:
                print(f"      {win['window']}: {win['heuristic']} ({win['improvement_pct']:.2f}% improvement)")
        print(f"    {'Metric':<20} | {'Simple':<10} | {'Multi-Verse':<10} | {'Diff':<10}")
        print(f"    {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        def print_row(label, val1, val2, inverse=False):
            diff = val2 - val1
            pct = (diff / val1 * 100) if abs(val1) > 1e-6 else 0
            # For most metrics, lower is better
            color = "\033[92m" if (diff < 0 if not inverse else diff > 0) else "\033[91m"
            reset = "\033[0m"
            print(f"    {label:<20} | {val1:<10.4f} | {val2:<10.4f} | {color}{pct:>+7.1f}%{reset}")

        print_row("Solve Time (s)", s["solve_time"], m["solve_time"])
        print_row("Total Cost", s["cost"], m["cost"])
        print_row("Path Length (m)", s["metrics"]["path_length_m"], m["metrics"]["path_length_m"])
        print_row("Tortuosity", s["metrics"]["tortuosity"], m["metrics"]["tortuosity"])
        print_row("Max Jerk (m/s3)", s["metrics"]["max_jerk_m_s3"], m["metrics"]["max_jerk_m_s3"])
        print_row("Chattering", s["metrics"]["velocity_chattering"], m["metrics"]["velocity_chattering"])

    def run_all(self, subset: Optional[List[str]] = None):
        scenarios = self.generate_scenarios()
        for name, waypoints in scenarios.items():
            if subset and name not in subset:
                continue
            self.run_test(name, waypoints, accuracy_weight=0.0) # Pure time
            self.run_test(name, waypoints, accuracy_weight=2.0) # More Smoothness

        with open("stress_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to stress_test_results.json")

if __name__ == "__main__":
    rig = StressTestRig("examples/fll_choreo.json")
    # Run a subset that includes unconstrained headings
    rig.run_all(subset=["Heading_Flip_Win"])
