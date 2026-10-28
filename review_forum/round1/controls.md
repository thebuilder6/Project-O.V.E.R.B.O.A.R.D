# Round 1 v3 — controls-and-math (Gould)

F1 KKT acceptable not optimal. `optimizer.py::IPOPT-opts` tol=1e-2, acceptable_tol=1e-1 x5 iters, L-BFGS + adaptive mu; Solved_To_Acceptable counts converged. 1cm/0.01rad residual accepted — half 20mm gate consumed pre-validation.
F2 Transcription validator does not execute. Optimizer GL4 continuous linear-v/w exact for polynomial model; robot/validator 20ms ZOH + quantize + half-tick lead. ZOH errors priced only under cost_metric discretization; default jerk + accuracy_weight=0 sets both to zero — default time-optimal blind to 5mm gate; turn-boost patches downstream.
F3 Feasible-set mismatch traction; two force models. Motor both ends (correct); traction start-only, end fmin-priced with shortcut (112→3145 iters MUMPS crash); validator checks both ends. Optimizer exploits gap, validator rejects. Epigraph branch different smaller set, same gate.
F4 Ramsete singularity + epsilon drift; dead merge. k1→0 at pinned stops (loss of authority where gate tightest); floors +1e-12 vs +1e-16 vs sinc/tanh/fmax diverge inside tick Jacobian; merge agree_* computed never consumed; continuous_force only when per_segment_dt, default uniform zero enforcement with jerk weight 0.

**[VETO: FORMULATION_FLAW]** — single-dt transcription + 1/dt singularity + loose barrier = ill-conditioned NLP feasible set exceeds gate. dt 3 decades scales Hessian ~1/dt² 1e6; L-BFGS blind to sqrt kink; missing end-traction escape.

Minimal fix in budget: acceptable_tol=tol; enforce trac_end two-half-space (or where |v·ω|>10% static); dt lower bound 0.02 (control tick). Re-measure benchmark solver before/after; expect fewer converged-then-rejected.

Skipped tick Jacobian audit + live benchmarks. Risk: tightening may convert passes to max_iter — honest failure over false pass.
