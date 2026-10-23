"""
Project-O.V.E.R.B.O.A.R.D
Time-optimal trajectory generator and validator for LEGO differential-drive robots.
"""

import os

# On Windows, cvxopt's own OpenBLAS copy (immrax -> pypoman) aborts the whole
# process on a machine with this many cores, after printing nothing. One thread
# is enough for the small matrices the validator multiplies.
if os.name == "nt":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0a1"
