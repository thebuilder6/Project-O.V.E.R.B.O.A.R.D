# REJECTED — Rescue round (KI-048)

Claim: retry failures with another seed / higher weight.
Ledger proof: STRATEGY_LEDGER §18.4 + KI-048; 32-mission suite `--candidates 3 --validate --propagation plant`: 22 failed, rescue 0 rescued; 95/97 rescue candidates failed envelope.
Re-open only if: failure population not envelope-bound (e.g. controller-jerk failures where weight buys tracking without growing tube). Envelope failures never qualify.
Round rejected: R1. Veto: `[VETO: RETIRED_STRATEGY by ledger-guardian]`.
