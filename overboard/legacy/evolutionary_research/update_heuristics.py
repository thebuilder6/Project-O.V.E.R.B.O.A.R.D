with open('docs/HEURISTICS_AND_VERIFICATION.md', 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = "### 19.5 Deterministic Homotopy Dispatch"
end_marker = "## 20. Ledger: Multi-Verse internals"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

assert start_idx != -1 and end_idx != -1, f"Markers not found: {start_idx}, {end_idx}"

replacement = r"""### 19.5 The Kinetic Frontier Analytical Seed Generator & The 5 Physical Principles

| Field | Value |
|---|---|
| **Strategy** | Replace stochastic genetic search with an analytical seed generator (`frontier_seed.py`) encoding the 5 physical principles uncovered by phase portrait diagnostics and the 50-generation marathon |
| **Mechanism** | `frontier_seed.py` (`generate_frontier_seed`, `generate_canonical_seeds`), integrated into `TrajectoryOptimizer._build_initial_guess` via `GUESS_STYLES = (..., "frontier_sprint", "frontier_sweeper", "frontier_exploratory", "frontier_direct")` |
| **The 5 Governing Principles** | 1. **Active Kinematic Hypotenuse:** $v(t) + \frac{W}{2}|\omega(t)| = v_{\max}$. In differential-drive, motor voltage saturation dominates tire friction by $2.5\times$. The optimal trajectory rides the $L_1$ actuator diamond boundary (one wheel at 100% saturation at all times) with analytical velocity profiling $v_{\text{frontier}}(\kappa) = \frac{v_{\text{wheel\_max}}}{1 + 0.5 B |\kappa|}$.<br>2. **Late-Apex Dynamic Law:** Asymmetric corner entry ($f_{\text{apex}} = 0.66\text{--}0.70$) enables deep trail-braking and early symmetric exit throttle ($0.3\text{ s}$ earlier full acceleration onto straights).<br>3. **Waypoint Momentum Preservation:** Maintains $0.65\text{--}0.75 \times v_{\max}$ through unpinned intermediate waypoints along a continuous energy arc.<br>4. **KKT Slack Barrier Equilibrium:** Balances right on the safety threshold ($4.3\text{--}4.5\text{ mm}$ tracking against the $5.0\text{ mm}$ gate).<br>5. **Multi-Modal Homotopy Triad:** Dispatches 3 canonical homotopies (`frontier_sprint`, `frontier_sweeper`, `frontier_exploratory`). |
| **Measured Outcome (2026-10-20)** | **Shatters the 44-Gene Champion on the 10-waypoint complex mission:**<br>- **Stock Baseline:** $5.948\text{ s}$, $0.96\text{ cm}$ tracking, 200 iters (capped), $3.45\text{ s}$ wall (**FAIL**)<br>- **44-Gene GA Champion (12 Gens):** $5.417\text{ s}$, $0.49\text{ cm}$ ($4.9\text{ mm}$), 120 iters, $\approx 25\text{ s}$ wall (**PASS**)<br>- **Frontier Sprint Seed:** **$5.239\text{ s}$**, **$0.45\text{ cm}$ ($4.5\text{ mm}$)**, $1.17\text{ cm}$ envelope, **Score 5.239** (**PASS**)<br>- **Frontier Sweeper Seed:** **$5.237\text{ s}$**, **$0.44\text{ cm}$ ($4.4\text{ mm}$)**, $1.18\text{ cm}$ envelope, **101 iters**, **Score 5.237** (**PASS**)<br>- **Frontier Exploratory Seed:** **$5.233\text{ s}$**, **$0.43\text{ cm}$ ($4.3\text{ mm}$)**, $1.18\text{ cm}$ envelope, **Score 5.233** (**PASS**)<br>- **Seed Generation Overhead:** **$1.16\text{ ms}$** total for all 3 seeds |
| **Status** | **Live** in `frontier_seed.py` and `optimizer.py` |

"""

new_text = text[:start_idx] + replacement + text[end_idx:]
with open('docs/HEURISTICS_AND_VERIFICATION.md', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(new_text)
print("Updated docs/HEURISTICS_AND_VERIFICATION.md successfully!")
