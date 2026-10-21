This document outlines the system identification (SysId) tool, the telemetry recording tool, and the architectural integration into the path planner. 

This design shifts complex mathematical computation to the PC (host side), leaving the robot with lightweight execution steps that are structurally robust against real-world slip and physical interaction [1, 2].

> **Implementation status (2026-10-21)**
>
> - **Implemented:** robot-side digital-twin collector (`pybricks_code/sysid_digital_twin.py`) and host fitter (`overboard/sysid/fit.py`); the optional fitted plant (`overboard/sysid/digital_twin.py`) behind `python -m overboard.main --sysid-fit`, which applies the actuator gain, time constant, command delay, acceleration limit, odometry belief and braking shortfall; canonical/legacy/flat config parsing with declared uncertainty ranges; per-wheel force and traction limits; CPU + Immrax validation gate; controller JSON and Python export.
> - **Partial:** the robot's own odometry is modelled as a *belief* inside the validator rather than as a plant state; motor current transients, DriveBase PID, slip under load and full slip dynamics are still missing.
> - **Archived:** the JAX refinement and the Multi-Verse pipeline have been removed (git tag `archive/legacy-2026-10`) and have no CLI entry point; measured as a seed provider the refined seed rescued nothing (KI-004).
> - **Not implemented:** interaction force injection (`interactions` is parsed only as data), full motor/DriveBase/odometry/slip dynamics, and reachability-driven optimization or a feedback loop into the planner. The validator is a post-solve gate and candidate score, not a full digital twin (KI-020).
> - **Do not read this document as a claim that the robot currently guarantees zero slip.** The zero-slip target remains a design goal.

---

# Part 1: System Identification & Telemetry Tools

These tools operate in two steps: a robot-side script (running on Pybricks) to collect high-frequency sensor data [1, 2], and a PC-side script (running in Python/JAX) to analyze the data and generate nominal and interval parameters [1].

```
[Robot-Side Collector: sysid_digital_twin.py] -> [CSV Data Log + poses.csv] -> [PC-Side Fitter: overboard.sysid.fit] -> [sysid_fit.json -> overboard.main --sysid-fit] -> [overboard.sysid.compare report]
```

## 1.1 Automated SysId Test Protocol

The robot is placed on the competition mat and executes a pre-programmed sequence.

### Test A: Gear Backlash Calibration ($b$)
*   **Physical Setup:** Place the robot on a high-traction surface.
*   **Robot Action:** Apply a tiny, constant current limit to both motors. Slowly oscillate the motors back and forth ($10$ cycles) [1].
*   **Data Logged:** Wheel encoder angles, gyroscope angular velocity ($\omega_{\text{gyro}}$) [1].
*   **Mathematical Processing (PC):**
    *   Track the total encoder angular travel ($\Delta\theta_{\text{encoder}}$) before the gyroscope registers a non-zero angular velocity ($\omega_{\text{gyro}} > \epsilon$, where $\epsilon$ is the sensor noise floor) [1].
    *   Determine the backlash distance [1]: 
        $$b = \Delta\theta_{\text{encoder}} \cdot r$$
    *   **Output:** Nominal backlash and interval boundaries $[b_{\min}, b_{\max}]$ [1].

### Test B: Friction Coefficient Calibration ($\mu$)
*   **Physical Setup:** Place the robot's bumper flat against a rigid, immovable obstacle (e.g., a wall).
*   **Robot Action:** Slowly and linearly ramp up the motor duty cycle (current/torque) from $0$ to maximum.
*   **Data Logged:** Motor torque/load ($\tau$), wheel encoder speeds, accelerometer $x/y$ axes [1].
*   **Mathematical Processing (PC):**
    *   Identify the exact breakaway timestamp ($t_{\text{slip}}$) where wheel encoder speeds suddenly jump and high-frequency slip vibrations appear in the accelerometer data [1].
    *   Extract the motor torque at $t_{\text{slip}}$ ($\tau_{\text{slip}}$).
    *   Calculate maximum tractive force and friction coefficient [1]:
        $$F_{\text{traction\_max}} = \frac{\tau_{\text{slip}} \cdot G}{r}$$
        $$\mu = \frac{F_{\text{traction\_max}}}{m \cdot g}$$
    *   **Output:** Static coefficient of friction interval $[\underline{\mu}, \bar{\mu}]$ [1].

### Test C: Moment of Inertia Calibration ($I$)
*   **Physical Setup:** Place the robot in an open space on the mat.
*   **Robot Action:** Apply a sequence of step-torque commands to spin the robot in place (left motor forward, right motor backward) [1].
*   **Data Logged:** Gyroscope angular velocity ($\omega_{\text{gyro}}$), motor torques ($\tau_l, \tau_r$) [1].
*   **Mathematical Processing (PC):**
    *   Compute the angular acceleration ($\alpha = \dot{\omega}_{\text{gyro}}$) by applying a low-pass differentiator to the gyro data [1].
    *   Fit the dynamic torque balance to solve for $I$ [1]:
        $$I \cdot \alpha(t) = \frac{L}{2r} \cdot G \cdot (\tau_l(t) + \tau_r(t)) - \tau_{\text{friction}}$$
    *   **Output:** Nominal moment of inertia ($I$) and standard deviation bounds $[I_{\min}, I_{\max}]$ [1, 2].

### Test D: Motor Performance Mapping
*   **Physical Setup:** Run the robot in a straight line on the mat.
*   **Robot Action:** Apply $100\%$ duty cycle until terminal velocity is reached. Then, command a hard stop to stall the motors.
*   **Data Logged:** Terminal wheel velocity, stall motor current, battery voltage [1].
*   **Mathematical Processing (PC):**
    *   Record absolute no-load velocity ($\omega_{\max}$) at terminal speed [1].
    *   Record peak stall torque ($\tau_{\max}$) at the point of deceleration [1].
    *   **Output:** Map of the actual, battery-state-dependent motor torque curve slope [1, 2].

---

## 1.2 Mission Telemetry Recorder (Action Calibration)

To plan for push/pull interactions (which disrupt standard path following), the user executes the mission once at a controlled speed to record the environment's resistance [1].

*   **Physical Setup:** Align the robot with the targeted mission model (e.g., a lever or cart) [1].
*   **Robot Action:** Execute the push or pull action at a constant, slow velocity (e.g., $50\text{ mm/s}$).
*   **Data Logged:** Time, wheel encoder positions, motor torques, accelerometer forward axis ($a_x$) [1].
*   **Mathematical Processing (PC):**
    *   Isolate the exact interaction window $[t_{\text{start}}, t_{\text{end}}]$ using the structural transient shock in the accelerometer data [1].
    *   Because velocity was constant ($a \approx 0$), calculate the external resistive force profile spatially over distance $s$ [1]:
        $$F_{\text{push}}(s) \approx \frac{(\tau_l(s) + \tau_r(s)) \cdot G}{r}$$
    *   **Output:** A spatially indexed push force profile $F_{\text{push}}(s)$ for the specific interaction segment [1].

---

# Part 2: Data Output Schema

The fitter (`overboard/sysid/fit.py`) writes a separate `sysid_fit.json`. The CLI loads it and merges the measured numbers into the robot config for that run (`RobotConfig.apply_sysid_fit`), so the optimizer, the CPU audit, the validator and the exported controller all use the same measured motor (KI-010's rule); the config file on disk is not rewritten. The fitter accepts a minimal drivebase profile (`wheel_radius_m`, optional `track_width_m`) so a different differential drive only needs its own geometry and profile logs; `overboard/sysid/compare.py` accepts the same profile and a plain predicted-pose CSV. The longer-term design target is to merge calibrated physical ranges into the robot config schema [1, 2]:

```json
{
  "robot": {
    "mass": 0.723,
    "inertia": 0.0024,
    "track_width": 0.0965,
    "wheel_radius": 0.028,
    "v_max_rad_s": 15.7,
    "t_max_nm": 0.04,
    "gearing": 1.0,
    "uncertainties": {
      "cof": [0.35, 0.55],
      "torque_headroom": [0.75, 0.90],
      "gear_backlash_m": [0.0002, 0.0006],
      "inertia_kg_m2": [0.0022, 0.0026]
    }
  },
  "interactions": {
    "lever_push_01": {
      "start_trigger_waypoint_idx": 4,
      "end_trigger_waypoint_idx": 6,
      "force_interval_n": [0.8, 1.3]
    }
  }
}
```

---

# Part 3: Path Planner & Optimizer Architecture

This architecture integrates the calibrated system parameters and interaction force profiles into the path planning pipeline, guaranteeing zero wheel slip and robust safety under real-world tracking errors [1].

```
                     [Path Planner Pipeline]
                                │
                                ▼
         Phase A: Multi-Verse Evaluator (archived)  <── Ingests RobotConfig JSON
                                │                            (Solves unconstrained headings)
                                ▼
         Phase B: CasADi / IPOPT High-Precision Solver   <── Enforces strict physics
                                │                            (Zero wheel slip, force profiles)
                                ▼
         Phase C: Immrax Closed-Loop Robust Validator    <── Runs Monte Carlo / Intervals
                                │                            (Checks CoF, Backlash, Torque)
                                ▼
                        [Export to Pybricks]
```

## Phase A: Deterministic Homotopy Dispatch & Kinematic Velocity Profiling (Seed Generation)
Before solving the non-linear program (NLP), candidate hypotheses are generated deterministically and screened in pure JAX via `vmap` [1, 2], superseding multi-generation evolutionary/genetic search:
1.  **3-Way Geometric Classifier:** For each segment between waypoints, seeds are generated based on physical geometry:
    *   *Flowing unpinned turns:* Late-apex racing lines (shifting the curve midpoint outward).
    *   *Sharp corners ($> 90^\circ$):* Pivot-in-place turns (zero speed at waypoint, pure rotational velocity).
    *   *Stop / Interaction waypoints:* Direct chord with threshold deceleration.
2.  **Kinematic Velocity Profiling ($\Delta t_i$ Initialization):** Rather than equal time distribution, segment durations are initialized using the physical curvature speed limit:
    $$v_{\max, i} = \min\left(v_{\text{motor}}, \sqrt{\frac{\underline{\mu} g}{\kappa_i}}, \sqrt{\frac{a_{\max}}{\kappa_i}}\right), \qquad \Delta t_i = \frac{L_i}{0.75 \cdot v_{\max, i}}$$
3.  **Vectorized JAX Screening (`vmap`):** All generated candidate seeds are evaluated simultaneously in $< 25\text{ ms}$ for kinematic feasibility and estimated tracking error.
4.  **Topological Clustering:** Deduplicates candidates into the top 2--3 structurally distinct homotopies to warm-start IPOPT in parallel.

## Phase B: Discretization-Consistent CasADi / IPOPT Solver
The selected seeds are passed to IPOPT, which enforces exact hard constraints and accounts for the robot's discrete execution [1, 2, 3.2]:
1.  **Discretization-Aware Collocation & Objective:** Rather than continuous-time jerk penalties, the NLP uses 4-node Gauss-Legendre quadrature (`self_consistent=True`) and directly prices the discrete 20 ms ZOH tracking error (`cost_metric="discretization"`):
    *   *Sagitta chord-vs-arc error:* $\frac{1}{8} v_{\text{mid}} \omega_{\text{mid}} \Delta t_k^2$
    *   *Parabolic heading mismatch:* $\frac{1}{8} v_{\text{mid}} (\omega_2 - \omega_1) \Delta t_k^2$
    *   *20 ms hold lag:* $\frac{1}{2} (v_2 - v_1) \cdot 0.02$
2.  **Linear Epigraph Force Bounds:** Replaces $\sqrt{F^2 + \epsilon}$ (which produces $10^6$ Hessian spikes on saturation boundaries) with 4 smooth linear inequalities (`epigraph_constraints=True`), stabilizing interior-point convergence.
3.  **Per-Wheel Traction Enforcement:** At each timestep $k$, the solver limits each wheel separately, using the smaller normal force from lateral weight transfer and the lower calibrated CoF ($\underline{\mu}$) [1, 2]:
    $$|f_{l,k}| \le \underline{\mu} \cdot n_{l,k}, \qquad |f_{r,k}| \le \underline{\mu} \cdot n_{r,k}$$
    This replaces the older total-force friction-circle constraint (KI-010).
4.  **Motor Limits Enforcement:** The velocity-dependent maximum force is constrained using the worst-case torque headroom, wheel radius, and speed term [1, 2]:
    $$|f_{w,k}| \le \frac{1}{r_{\text{force}}} \cdot \left(\tau_{\max} \cdot \underline{h}_t \cdot \max\left(0, 1 - \frac{|\omega_k|}{\omega_{\max}}\right)\right)$$
5.  **Interaction Force Injection (design only):** The current optimizer does not read `interactions` or inject $F_{\text{push}}$. When implemented, timesteps inside an interaction window would add the recorded resistance to the dynamics:
    $$f_{\text{total\_k}} = m \cdot a_k + \max(F_{\text{push\_interval}})$$
    *This would force the optimizer to scale down entry speeds and accelerations through the mission model. It is not implemented yet (KI-020).*

## Phase C: Immrax Closed-Loop Robust Validator
Once the optimal nominal trajectory is computed, the validator checks the path against bounded real-world variations [1, 2].
1.  **Ramsete Closed-Loop Simulation:** Immrax simulates the deployed Pybricks Ramsete controller with a 20 ms sample-and-hold loop and 5 ms plant substeps [1].
2.  **Uncertainty Propagation:** 
    *   The robot mass ($m$), moment of inertia ($I$), track width ($L$), wheel radius, and CoF are modeled as interval values [1, 2].
    *   Motor backlash ($b$) is introduced as an interval step displacement at every velocity zero-crossing [1].
    *   The motor torque headroom is modeled as the interval $[\underline{h}_t, \bar{h}_t]$ [1].
3.  **Safety Verification:**
    *   Immrax calculates the spatial reachable tube (the bounding box of where the robot frame could exist) [1].
    *   It checks the worst-case per-wheel motor force, wheel slip, wheel speed, tracking error, and envelope radius [1].
4.  **Current feedback status:** the validator is a post-solve gate and candidate score. With `--validate`, `overboard/main.py` solves the planner's candidate set - the three frontier candidates (`frontier_sprint`, `frontier_sweeper`, `frontier_exploratory`) - and selects the passing one with the shortest duration, the envelope breaking a tie. The 5-part path fitness score (stop spread, slip risk, RMS wheel force, tracking error, run time) is reported for every candidate, so the rule can be re-run offline, but it does not decide. `--candidates` sets how many are solved (default 2; the flag itself has no upper bound). The optional `--polish` pass differentiates the reachability rollout and re-solves the plan with the samples the gradient points at weighted; a polished plan is kept only when it is not slower and not worse. It prices the wheel forces in the objective (`--force-weight`, off by default); it does not feed the violated segment back into the optimizer as a penalty. Reachability-driven optimization remains future work (KI-020).
    *   If validation passes, the trajectory is exported; if it fails, no output is written and the command exits non-zero.

---

# Part 4: Far Stretch Goal — 3D Simulator Mode & Interactive Mission Elements

See [`docs/STUDIO_3D_SIMULATOR_PLAN.md`](docs/STUDIO_3D_SIMULATOR_PLAN.md) for the complete design specification.

This stretch goal introduces a client-side 3D WebGL physics environment inside the browser studio (`overboard/viz/studio.html`), using Three.js for rendering and Rapier3D (WebAssembly) for 60 Hz rigid-body simulation.


[Trajectory (.traj)] ┐> [Rapier3D Physics Sandbox] <┐ [Clustered CAD (.ldr / .io)]
                                    │
                                     ▼
                     [Three.js 3D Field Viewport]
                     (Interactive Levers, Clearances, Scoring)


## 4.1 CAD Model Sources & Topological Clustering
* **Official Season Mission Models:* Available as BrickLink Studio (`.io`) models from [Komurobo](https://komurobo.com/fll/3d-models/) or scripted LDraw (`.ldr`) files from [martijas/fll-sim](https://github.com/martijas/fll-sim).
* **Topological Pin Clustering:* Solves physics stability by clustering static LEGO beams connected by friction pins (`2780`, `6558`) into compound rigid bodies, exposing only true kinematic degrees of freedom (axle rotations, slide rails).

## 4.2 Interactable Mission Mechanisms
Mission models implement an articulated `MissionElement` interface with revolute/prismatic joints and friction damping (e.g., M02 toggle riser, M03 Rock lever, loose game pieces). As the virtual robot drives over the mat and actuates mechanisms:
1. **Tool & Arm Clearances:** 3D collision checks verify physical clearances around vertical obstacles that 2D polygonal footprints cannot capture.
2. **Contact & Reaction Forces:** Simulates resistive force profiles ($F_{\text{push}}$) during mechanism actuation, bridging directly into interaction force profiles and digital twin verification.
