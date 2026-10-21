# Deriving Optimal Control from Genetic Algorithms: The Kinetic Frontier

This document details how empirical discoveries from high-generation stochastic evolutionary searches (the **5-Pillar Multi-Island Genetic Algorithm**) led directly to the formulation of analytical optimal control laws for differential-drive mobile robots, formalized in [`frontier_seed.py`](../frontier_seed.py).

---

## 1. The State-Space $(v, \omega)$ Feasibility Boundary

In a differential-drive robot, the left and right wheel velocities $(v_l, v_r)$ are constrained by the physical motor saturation limits:
$$|v_l| \le v_{\max}, \quad |v_r| \le v_{\max}$$

Transforming to unicycle coordinates $(v, \omega)$ via the differential drive kinematic relationship $v = \frac{v_r + v_l}{2}$ and $\omega = \frac{v_r - v_l}{W}$ (where $W$ is the wheel track width), the feasible control space forms a **kinematic diamond** in state space:

$$|\omega| \le \frac{2}{W} (v_{\max} - |v|)$$

![State-Space Phase Portrait](images/optimal_control_phase_portrait.png)

### What the Phase Portrait Reveals:
1. **Stock Heuristic Underutilization (Red Dashed Line)**:
   - Traditional rolling heuristics cluster passively in the conservative interior of the diamond ($v \approx 0.3 - 0.5\text{ m/s}$, $|\omega| \le 2\text{ rad/s}$).
   - The robot decelerates excessively before every turn, wasting up to 40% of its available control bandwidth.
2. **Evolutionary Discovery (Color Scatter Points)**:
   - When the 5-Pillar Genetic Algorithm optimizes for minimum time subject to physical reachability gates, it systematically pushes the robot's trajectory directly onto the **active diagonal boundary of the diamond**:
   $$|v| + \frac{W}{2}|\omega| = v_{\max}$$
   - This proves Pontryagin's Maximum Principle in practice: along time-optimal arcs, at least one actuator operates in full saturation at all times.

---

## 2. The 5 Analytical Principles of Differential-Drive Optimal Control

By mining the genomes of the top-performing attractors across 50 generations of evolutionary optimization (over 1,200 IPOPT evaluations), five invariant physical laws were extracted:

### 1. Principle of the Active Kinematic Hypotenuse
Time-optimal cornering requires tracing the linear boundary of the $(v, \omega)$ diamond. Rather than turning and translating in separate phases, the robot operates at:
$$v_{\text{corner}} = v_{\max} - \frac{W}{2} |\omega|$$
This maintains 100% actuator utilization throughout curved arcs.

### 2. The Late-Apex Dynamic Law
In contrast to geometric splines that place the peak curvature at the geometric midpoint ($u = 0.50$), the genetic algorithm consistently converges to an **asymmetric late apex**:
$$u_{\text{apex}} \in [0.65, 0.70]$$
- **Entry ($u < 0.65$)**: Trail-braking along a wide radius into the corner preserves forward speed.
- **Exit ($u > 0.70$)**: Sharp heading rotation right at the exit threshold allows the robot to straighten early and achieve 100% longitudinal acceleration out of the turn.

### 3. Waypoint Momentum Preservation
Intermediate waypoints with unpinned headings do not require zero angular error at passage. The genetic algorithm preserves momentum by passing through intermediate waypoints at:
$$v(W_i) \approx 0.68 \cdot v_{\max} \text{ to } 0.75 \cdot v_{\max}$$
The solver aligns the heading along the chord bisector tangent rather than pointing directly at subsequent coordinates.

### 4. KKT Slack Barrier Equilibrium
Physical validation imposes a hard gate: tracking error $e_{\text{track}} \le 5.0\text{ mm}$. The evolutionary search rides the active constraint boundary at **$4.6\text{ mm} - 4.8\text{ mm}$**, trading tracking margin for maximum forward velocity.

### 5. Multi-Modal Homotopy Triad
The solution landscape possesses three distinct local attractors:
- **Sprint Line**: Aggressive late apex ($u_{\text{apex}} = 0.70$), maximum lateral acceleration, shortest trajectory duration.
- **Smooth Sweeper**: Continuous curvature transition ($u_{\text{apex}} = 0.65$), minimizing wheel jerk and tracking error.
- **Exploratory Homotopy**: Wider clearance margin ($1.25\times$ apex offset) offering maximum tolerance against physical obstacle drift.

---

## 3. Kinetic Frontier Seeds vs. GA Benchmark

By distilling these five principles into closed-form equations in [`frontier_seed.py`](../frontier_seed.py), analytical seeds are generated in **$< 1.5\text{ ms}$** without running multi-minute stochastic searches.

### Quantitative Comparison on 10-Waypoint Complex Mission (`examples/example_complex_mission.json`):

| Strategy | Seed Gen Time | IPOPT Iters | Solve Time | Trajectory Duration | Tracking Error | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock Baseline (`rolling`)** | $0.87\text{ ms}$ | 150 | $3.01\text{ s}$ | $6.743\text{ s}$ | $1.10\text{ cm}$ | ❌ Fail ($> 5.0\text{ mm}$) |
| **50-Gen GA Centurion Champion** | $281,000\text{ ms}$ | 1,200 evals | $281.0\text{ s}$ | $5.572\text{ s}$ | **$0.48\text{ cm}$** | ✅ Pass |
| **Frontier Sprint Seed** | **$1.29\text{ ms}$** | **86** | **$1.89\text{ s}$** | **$5.452\text{ s}$** | **$0.49\text{ cm}$** | ✅ **Pass** |

![Frontier Seeds vs GA vs Baseline](images/plot_frontier_vs_ga_comparison.png)

### Conclusion
The analytical `frontier_sprint` seed generates a gate-certified, time-optimal trajectory (**$5.452\text{ s}$**) that is **$0.12\text{ s}$ faster than the 50-generation GA champion** and **$1.29\text{ s}$ (+19.2%) faster than stock**, while cutting solver iterations from 150 down to 86 and eliminating the need for expensive stochastic search.
