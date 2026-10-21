import os

with open('Future_Plans.md', 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = "## Phase A: JAX Pre-Solver"
end_marker = "## Phase C: Immrax Closed-Loop Robust Validator"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

assert start_idx != -1 and end_idx != -1, f"Markers not found: {start_idx}, {end_idx}"

replacement = r"""## Phase A: Deterministic Homotopy Dispatch & Kinematic Velocity Profiling (Seed Generation)
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

"""

new_text = text[:start_idx] + replacement + text[end_idx:]
with open('Future_Plans.md', 'w', encoding='utf-8') as f:
    f.write(new_text)
print("Updated Future_Plans.md successfully!")
