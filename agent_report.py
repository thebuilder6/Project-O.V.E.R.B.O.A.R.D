"""Run a noisy Overboard command and print only the part an agent needs.

Usage:
    python agent_report.py tests [run_tests.py flags]
    python agent_report.py bench [benchmark.py args]
    python agent_report.py solve [overboard.main args]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:]).strip()


def after(text: str, marker: str, fallback: int = 40) -> str:
    index = text.rfind(marker)
    if index >= 0:
        return text[index:].strip()
    return tail(text, fallback)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2 or sys.argv[1] not in {"tests", "bench", "solve"}:
        print(__doc__)
        return 2

    kind, args = sys.argv[1], sys.argv[2:]
    if kind == "tests":
        code, stdout, stderr = run([sys.executable, "run_tests.py", *args])
        if "--cache-status" in args:
            lines = stdout.splitlines()
            start = next((i for i, line in enumerate(lines) if line.startswith("TEST CACHE")), 0)
            env = next((line for line in lines[start:] if line.startswith("  environment:")), "")
            body = "\n".join(lines[start:start + 5] + ([env] if env else []))
        elif "TEST SUMMARY" in stdout:
            body = after(stdout, "TEST SUMMARY")
        else:
            body = tail(stdout + "\n" + stderr)
    elif kind == "bench":
        code, stdout, stderr = run([sys.executable, "benchmark.py", *args])
        output = stdout + "\n" + stderr
        lines = output.splitlines()
        warnings = [
            line
            for line in lines[:-35]
            if any(word in line.lower() for word in ("warn", "error", "failed", "traceback"))
        ]
        body = "\n".join(warnings[-10:] + (["..."] if warnings else []) + lines[-35:]).strip()
    else:
        code, stdout, stderr = run([sys.executable, "-m", "overboard.main", *args])
        output = stdout + "\n" + stderr
        lines = output.splitlines()
        report = [line for line in lines if line.startswith("##OVERBOARD-VALIDATION##")]
        body = "\n".join(report + lines[-12:]).strip()

    print(body)
    print(f"\n[agent_report] exit={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
