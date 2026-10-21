---
description: Dispatch the read-only reviewer on the current changes
agent: reviewer
subagent: true
---

Review the current changes. Start with `git diff` and `git diff --cached`, then read the changed files and the repo rules.

Report findings in severity order with `file:line`. Do not edit, run solvers, or push.
