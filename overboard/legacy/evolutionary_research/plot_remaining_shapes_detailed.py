"""
Generate individual high-resolution 2-panel diagnostic plots for:
1. 90° Box Loop with Stops
2. Alternating Slalom
3. 180° Hairpin U-Turn
"""

import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from optimizer import TrajectoryOptimizer
from immrax_validator import ImmraxValidator
from surrogate_polish import CompiledReachabilityEngine
from scratch.compare_geometric_shapes import solve_shape
from scratch.plot_individual_shapes_and_run_s_curve import plot_detailed_shape

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"

def main():
    with open("examples/fll_choreo.json") as f:
        cfg_dict = json.load(f)
    config = RobotConfig(cfg_dict)
    jax_cfg = JAXRobotConfig(config)
    val = ImmraxValidator(jax_cfg)
    engine = CompiledReachabilityEngine(val)
    optimizer = TrajectoryOptimizer(config)

    # 1. 90° Box Loop with Pinpoint Stops
    with open("examples/example_with_stops.json") as f:
        data2 = json.load(f)
    wps2 = [(w["x"], w["y"], w.get("heading")) for w in data2]
    stops2 = [i for i, w in enumerate(data2) if w.get("stop", False)]
    res2 = solve_shape("90° Box Loop with Stops", wps2, stops2, config, optimizer, engine, cfg_dict)
    plot_detailed_shape(res2, "plot_box_detailed.png")

    # 2. Alternating Slalom
    with open("examples/example_with_events.json") as f:
        data3 = json.load(f)
    wps3 = [(w["x"], w["y"], w.get("heading")) for w in data3]
    stops3 = [i for i, w in enumerate(data3) if w.get("stop", False)]
    res3 = solve_shape("Alternating Slalom", wps3, stops3, config, optimizer, engine, cfg_dict)
    plot_detailed_shape(res3, "plot_slalom_detailed.png")

    # 3. 180° Hairpin U-Turn
    wps4 = [
        (0.0, 0.0, 0.0),
        (0.6, 0.15, None),
        (0.7, 0.45, math.pi / 2),
        (0.6, 0.75, None),
        (0.0, 0.90, math.pi),
    ]
    stops4 = [0, 4]
    res4 = solve_shape("180° Hairpin U-Turn", wps4, stops4, config, optimizer, engine, cfg_dict)
    plot_detailed_shape(res4, "plot_hairpin_detailed.png")

    print("\nAll individual shape plots rendered successfully!")

if __name__ == "__main__":
    main()
