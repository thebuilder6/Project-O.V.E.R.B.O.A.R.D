from typing import List, Tuple, Optional, Dict, Any

class Optimizer:
    """Unified interface for trajectory optimization."""

    def __init__(self, config_data: Dict[str, Any], method: str = "multiverse", **kwargs):
        from .robot_model import RobotConfig
        self.config = RobotConfig(config_data)
        self.config_data = config_data
        self.method = method
        self.kwargs = kwargs
        self._optimizer = None

    def solve(self, waypoints: List[Tuple[float, float, Optional[float]]], **kwargs):
        if self.method == "simple":
            from .optimizer import TrajectoryOptimizer
            if self._optimizer is None:
                self._optimizer = TrajectoryOptimizer(self.config)
        else:
            from .multiverse_optimizer import MasterTrajectoryOptimizer
            if self._optimizer is None:
                enable_parallel = self.kwargs.get("enable_parallel", True)
                num_workers = self.kwargs.get("num_workers", 8)
                self._optimizer = MasterTrajectoryOptimizer(
                    self.config,
                    enable_parallel=enable_parallel,
                    num_workers=num_workers
                )

        return self._optimizer.solve(waypoints, **kwargs)
