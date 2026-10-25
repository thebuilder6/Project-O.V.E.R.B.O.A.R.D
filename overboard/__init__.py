"""
Project-O.V.E.R.B.O.A.R.D
Time-optimal trajectory generator and validator for LEGO differential-drive robots.
"""

import os

# One thread per BLAS/OpenMP pool, set before NumPy, JAX or Immrax is imported.
# The pools reserve memory they never touch (4.5 GB of commit for a 0.2 GB
# import on the reference machine), so parallel validating runs exhaust the
# machine's commit limit and MUMPS aborts the process with 0xC0000005 instead
# of reporting a failed solve (KI-057). On Windows, cvxopt's own OpenBLAS copy
# (immrax -> pypoman) also aborts after printing nothing on a many-core
# machine. One thread is enough for the small matrices a solve multiplies; the
# parallelism that matters is the process pool, and `setdefault` keeps a
# caller's own setting.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
del _name

__version__ = "0.1.0a2"
