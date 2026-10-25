"""
Unit tests for the runtime footprint of a run.

A validating run opens a BLAS thread pool, an Eigen thread pool and
MUMPS's workspace. Those pools *reserve* far more than they use, so
parallel runs exhaust the machine's commit limit and the next allocation
fails inside MUMPS, which aborts the process instead of reporting a
failed solve (KI-057).
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


class TestTheRuntimeFootprint(unittest.TestCase):
    """The pools are one thread each, and MUMPS is told its limit."""

    def _run(self, code, env=None):
        environment = dict(os.environ)
        for name in THREAD_VARS:
            environment.pop(name, None)
        if env:
            environment.update(env)
        environment["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                              env=environment, capture_output=True, text=True,
                              timeout=300)

    def test_importing_the_package_sets_one_thread_per_pool(self):
        """The variables are set at import, before NumPy or JAX is loaded."""
        code = ("import os, overboard\n"
                "print(','.join(os.environ.get(n, 'unset') for n in %r))" % (THREAD_VARS,))
        result = self._run(code)
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        self.assertEqual(result.stdout.strip(), "1,1,1,1")

    def test_a_user_setting_wins(self):
        """`setdefault`, so the library default is not forced on a caller."""
        code = ("import os, overboard\n"
                "print(os.environ.get('OMP_NUM_THREADS'))")
        result = self._run(code, env={"OMP_NUM_THREADS": "4"})
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        self.assertEqual(result.stdout.strip(), "4")

    def test_mumps_is_told_how_much_memory_it_may_claim(self):
        from overboard.core.optimizer import MUMPS_MEMORY_PERCENT
        self.assertGreater(MUMPS_MEMORY_PERCENT, 0)
        # IPOPT's default is 1000, i.e. ten times the physical memory.
        self.assertLessEqual(MUMPS_MEMORY_PERCENT, 100)


if __name__ == "__main__":
    unittest.main()
