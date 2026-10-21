"""
Project-O.V.E.R.B.O.A.R.D
Time-optimal trajectory generator and validator for LEGO differential-drive robots.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "TrajectoryOptimizer",
    "RobotConfig",
    "validate_trajectory",
    "run_doctor",
]


def __getattr__(name: str):
    """Lazy export loader to avoid importing heavy dependencies (CasADi, JAX) at package root."""
    if name == "TrajectoryOptimizer":
        from overboard.core.optimizer import TrajectoryOptimizer
        return TrajectoryOptimizer
    if name == "RobotConfig":
        from overboard.models.robot_model import RobotConfig
        return RobotConfig
    if name == "validate_trajectory":
        from overboard.verification.validator import validate_trajectory
        return validate_trajectory
    if name == "run_doctor":
        from overboard.diagnostics import run_doctor
        return run_doctor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
