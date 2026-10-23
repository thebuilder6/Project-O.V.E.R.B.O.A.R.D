# Tests for FLL Trajectory Optimizer

import os

# The test modules import immrax directly, and on Windows its cvxopt dependency
# aborts the process before the package above can set this (overboard/__init__).
if os.name == "nt":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
