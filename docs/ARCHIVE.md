# Archived research material

The raw measurement logs, plots, JSONL records and one-off experiment scripts
live in the Git tag **`archive/research-2026-10`**. They are not part of the
product tree.

Read a file from the archive without checking the tag out:

```bash
git show archive/research-2026-10:docs/measurements/2026-10-20/matrix.jsonl
git show archive/research-2026-10:docs/research/rq_a/exp16_firmware_parity.py
```

List what was archived:

```bash
git ls-tree -r --name-only archive/research-2026-10 -- docs/measurements docs/research
```

## What stays in the product tree

- `docs/research/RQ-A_tracking_error.md` — why the tracking gate binds.
- `docs/research/RQ-B_drivetrain_model.md` — the drivetrain model check.
- `docs/research/pq_b/README.md` and `docs/research/pq_b/FINDINGS.md` — the measured
  research-agenda answers.
- `research/` — the measurement harness, random suite, robot randomizer and
  the independent cross-check, kept as developer tools.
- `overboard/legacy/` — the archived Multi-Verse pipeline, Reeds-Shepp
  bootstrap, JAX refiner and the older research prototypes. No shipped module
  imports it.
