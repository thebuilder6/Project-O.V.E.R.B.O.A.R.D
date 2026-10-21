---
description: Run the fast check loop and report only failures
---

Run the repository's compact test report.

1. If `$ARGUMENTS` names a test module, class, or file, run that first with `python -m unittest ... -v`.
2. Then run `python agent_report.py tests`.
3. Report only: exit code, tests run, successes, failures, errors, skipped, time, and the failing test IDs.

Do not paste full tracebacks. Do not edit files in this command. If the cache is cold, say so.
