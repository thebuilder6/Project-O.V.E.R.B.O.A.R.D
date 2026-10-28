# Round 1 v3 — dreamer (Jet)

Baseline (quoted): seed 1.16ms triad <2.5ms; solve 40-48 iters ~1.7s/cand; Q17 96 solves 476s, 192 solves 906s (+90% +0 missions); tick/mjac 7-10x validation, tick 2.9-4.4x tighter but 5/32→6/32; rescue 0/22 ~2x wall; collapse 3.03cm→0.013cm; anchor w=1000 still 0 flips, 2.32→3.38s iters x4.

1. Native hot loop (codegen + compiled lax.scan tick, not C++ rewrite). Win all in validation, zero in seeds. Cost medium — wire CompiledReachabilityEngine + codegen NLP once. Gate risk high if natural extension explodes (627mm vs mjac 18.8mm); parity KI-027/028 + propagation/inclusion tagging must survive bit-identical.
2. Studio auto-solve on drop (local queue, 127.0.0.1). Debounce + job reuse; failed solve leaves field empty + push off; safe costs-minutes stays labeled. No cloud/remote.
3. Basin cache (quantized-waypoint memo, validation-tagged). Key = waypoints+mesh+dt+weights+gate vector; hit skips IPOPT, always re-validates; plant-pass never reads as tick. Worthless today per Q17 (same pass/fail faster) but tiny cost.

Ledger concession: tombs stay buried (44-gene, refine, ladder, tied-dt, anchor, polish 33/33, turn_first/bow/risk-margin). Re-open only on shape-holding term at payable price or twin evidence flipping binding gate.

Skipped fresh solves. Risk: wall numbers are 10-19–26 trees, treat deltas directional.
