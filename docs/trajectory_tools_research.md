# Trajectory Optimization: Tools & Research Reference

*Context: FLL differential-drive path optimizer. The current implementation uses CasADi and IPOPT with trapezoidal direct collocation; it no longer uses SLSQP or scipy.minimize.*

> **Current status (2026-10-20):** time-optimal solving, the jerk/accuracy weight, CPU validation, Immrax reachability, stops/events, and controller export are implemented. The Multi-Verse refinement is repaired and used as a **seed provider** — `main.py --refine` puts its refined trajectory into the candidate set (KI-004); it is a cheaper way to reach the stock set's speed, not a better one, so it stays opt-in. An optional fitted actuator/odometry plant (`sysid_digital_twin.py` + `sysid_fit.py` + `--sysid-fit`) exists, but the validator is not yet a full digital twin (KI-020). See `KNOWN_ISSUES.md` and `docs/WHITEPAPER.md` for the measured state.

---

## The Core Problem: Speed vs. Accuracy

The optimizer minimizes trajectory duration while enforcing wheel-speed, motor-force, and per-wheel traction limits. A time-optimal plan pushes the robot to its actuator limits, but real robots have:

- odometry drift from wheel slip and encoder error
- motor and DriveBase response lag
- frame compliance and backlash
- surface-dependent friction

A plan that is fastest on paper can still overshoot a waypoint or arrive at the wrong heading. The open question is how to encode tracking accuracy in the optimization objective without giving up too much time.

The current tree uses an optional jerk penalty (`--accuracy-weight`) as the plan-time proxy, then checks the result with a CPU audit and Immrax reachability. With `--validate`, `--candidates` solves and scores several trajectories and selects a passing one by envelope (KI-020); reachability is not yet an objective term.

---

## Comparable Tools

### Tier 1 - Directly comparable

| Tool | Platform | Method | Notes |
|---|---|---|---|
| **[Choreo](https://choreo.autos)** | FRC desktop GUI | Direct collocation via TrajoptLib + IPOPT | Optimizes time subject to drivetrain physics. Open source (Rust + TypeScript). |
| **[TrajoptLib](https://github.com/SleipnirGroup/TrajoptLib)** | C++ library | Swerve/diff-drive collocation, obstacle avoidance | The math engine behind Choreo; usable headlessly. |
| **[PathPlanner](https://pathplanner.dev)** | FRC desktop GUI | Bezier spline + velocity profile | Easier to use; does not globally minimize time. Adds event markers. |
| **[WPILib Trajectory API](https://docs.wpilib.org/en/stable/docs/software/advanced-controls/trajectories/index.html)** | Java/C++ | Trapezoidal velocity profiling on splines | Built-in, limited physics awareness. Basis for PathPlanner. |

### Tier 2 - Academic / research-grade

| Tool | Language | Method | Notes |
|---|---|---|---|
| **[CasADi](https://casadi.org)** | Python/C++/MATLAB | Symbolic autodiff + NLP (IPOPT) | Flexible; used in much robotics OCP research. This repository uses it. |
| **[ROCKIT](https://gitlab.kuleuven.be/robotgenskill/rockit)** | Python on CasADi | Higher-level OCP syntax, Radau collocation | Designed for robot motion planning. |
| **[Crocoddyl](https://github.com/loco-3d/crocoddyl)** | Python/C++ | Differential Dynamic Programming (DDP) | Good for legged/contact-rich robots. |
| **[FATROP](https://github.com/meco-group/fatrop)** | C++ | Structure-exploiting NLP solver | Can be faster than IPOPT on shooting problems. |
| **[Control Toolbox](https://github.com/ethz-adrl/control-toolbox)** | C++ | DDP, iLQR, SLQ, MPC | ETH Zurich; embedded/real-time control. |
| **[Trajopt](http://rll.berkeley.edu/trajopt/)** | Python/C++ | Sequential Convex Approximation | Manipulator planning and collision avoidance. |

### Tier 3 - Related planners

| Tool | Notes |
|---|---|
| **[OMPL](https://ompl.kavrakilab.org/)** | Sampling-based planning (RRT*, PRM*) as a front end. |
| **[MoveIt 2](https://moveit.ros.org/)** | ROS2-native; OMPL front end plus trajectory smoothing. |
| **[Drake](https://drake.mit.edu/)** | Rigorous dynamics and trajectory optimization. |

---

## Key Research Directions

### 1. Time-optimal trajectory planning

- Bobrow, Dubowsky, and Gibson, “Time-Optimal Control of Robotic Manipulators Along Specified Paths,” *IJRR*, 1985. The phase-plane/bang-bang formulation behind many minimum-time planners.
- Pfeiffer and Johanni, “Minimum-Time Trajectory Planning for Industrial Robots,” *IEEE Journal on Robotics and Automation*, 1987. Extends the phase-plane method to general arms.
- Modern differential-drive papers add obstacle constraints and actuator/thermal limits to the same minimum-time structure.

### 2. Direct collocation and NLP

- Hargraves and Paris, “Direct Collocation Methods for Trajectory Optimization in Constrained Robotic Systems,” 1987. The collocation foundation used here.
- Posa, Cantu, and Tedrake, “A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact,” *IJRR*, 2014. Contact dynamics and direct collocation.
- Fahroo and Ross, “DIDO: A MATLAB application package for solving optimal control problems.” Pseudospectral alternative to trapezoidal collocation.

### 3. Speed vs. accuracy

- Jerk-minimization papers add the derivative of acceleration to the cost. Smoother trajectories usually track better, at a time cost that depends on the path.
- Robust trajectory optimization under uncertainty adds model bounds to the optimizer. This repository currently applies those bounds to force limits but not as a reachability constraint.
- Pareto-front work treats time, energy, and tracking error as competing objectives. The `--accuracy-weight` flag is a simple scalar version of that idea.

### 4. Tracking controllers

- Campion, Bastin, and d'Andrea-Novel, “Control of Wheeled Mobile Robots: An Experimental Overview,” *IEEE Transactions on Robotics*, 1996.
- Bayle, Fourquet, and Renaud, “Ramsete: A new trajectory tracking controller for mobile robots,” 2003. The controller family used by `pybricks_code/ramsete_fast.py` and the JAX validator.
- Learning-based residual compensation (Gaussian processes or small neural networks) is a research direction for systematic tracking error, not a current feature.

---

## How Accuracy Is Implemented Here

| Mechanism | Current implementation |
|---|---|
| Jerk penalty | `--accuracy-weight` adds smoothed wheel-acceleration differences to the objective. |
| Motor/traction margin | Declared `torque_headroom`, `speed_headroom`, and uncertainty ranges. |
| Per-wheel slip check | `validator.audit_constraints` and the Immrax interval check. |
| Tracking proxy | Immrax closed-loop Ramsete reachability; post-solve gate and candidate score. |
| Final-pose check | `check_final_waypoint` at 2 cm position / 2 cm heading tolerance. |
| Controller | Fixed 20 ms sample-and-hold export; Pybricks Ramsete runtime. |

The validation thresholds (5 mm tracking, 2 cm envelope) are model checks, not hardware guarantees. Do not widen them to make a specific path pass; calibrate the model first (KI-020).

---

## Recommended Next Steps

1. **Calibrate the digital twin (KI-020).** Record hardware logs with `pybricks_code/sysid_digital_twin.py`, measure the stop/final poses, fit with `sysid_fit.py`, and compare the fitted validator with those poses. Then add integrated odometry, DriveBase response, and slip models. Keep the current gate strict until then.
2. **Make the objective answer the gate.** The Multi-Verse seed provider is repaired (KI-004) but buys speed with tracking accuracy, because its cost is duration plus a jerk term. The same lever as reachability in the objective.
3. **Fix the install path (KI-006).** Done: the optional packages live in `requirements-optional.txt`, `install.bat` / `install.sh` set up both, and `main.py --doctor` reports what is missing.
4. **Finish export/runtime edge cases (KI-011, KI-012).** Done: the headings unwrap across ±π and the final event fires.
5. **Make benchmarks trustworthy (KI-015).** Done: the pipeline mode checks convergence, the config is explicit, and the random suite generator solves every mission.

---

## References

1. Reeds and Shepp, “Optimal paths for a car that goes both forwards and backwards,” *Pacific Journal of Mathematics*, 1990.
2. Bayle, Fourquet, and Renaud, “Ramsete: A new trajectory tracking controller for mobile robots,” 2003.
3. Andersson et al., “CasADi: a software framework for nonlinear optimization and optimal control,” *Mathematical Programming Computation*, 2019.
4. Wächter and Biegler, “On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming,” *Mathematical Programming*, 2006.
5. Immrax documentation: interval analysis and mixed-monotone reachability in JAX.
6. Choreo documentation: https://choreo.autos
