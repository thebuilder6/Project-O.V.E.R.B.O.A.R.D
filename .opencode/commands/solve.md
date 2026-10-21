---
description: Solve a mission through the CLI and report the validation gate
---

Solve this mission: $ARGUMENTS.

Use the CLI through the compact wrapper, for example:

`python agent_report.py solve -w examples/example_straight.json -o output.traj`

If no mission is given, use `examples/example_straight.json` and `output.traj`.

Rules:
- Do not enable `--fast-polish`, `--force-weight`, `--polish`, or `--best-effort` unless the user asked for it.
- If validation is requested, pass `--validate` and report the `##OVERBOARD-VALIDATION##` fields.
- Stop at the validated file. Never push to the hub from this command.
- If validation fails, say which gate failed and write nothing extra.
