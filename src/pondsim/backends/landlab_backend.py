from __future__ import annotations
import numpy as np
from landlab.components.overland_flow import OverlandFlow
from .base import SolverBackend, BackendRegistry

class LandlabSolver(SolverBackend):
    TIER = 1
    NAME = "Landlab Native (de Almeida)"

    def __init__(self, grid_data, config, grid=None):
        if grid is None:
            raise ValueError("Landlab backend requires the full RasterModelGrid object.")
        
        self.grid = grid
        
        # Ensure mannings_n field exists (Pondsim handles this via config.manning_n)
        if "mannings_n" not in self.grid.at_node:
            if isinstance(config.manning_n, np.ndarray):
                self.grid.add_field("mannings_n", config.manning_n.ravel(), at="node")
            else:
                self.grid.add_field("mannings_n", np.full(self.grid.number_of_nodes, config.manning_n), at="node")
        
        self.of = OverlandFlow(self.grid, steep_slopes=config.steep_slopes)

    @classmethod
    def check_availability(cls):
        try:
            from landlab.components.overland_flow import OverlandFlow
            return True, "Ok"
        except ImportError:
            return False, "Landlab not installed"

    @classmethod
    def check_vram(cls, grid):
        return True

    @property
    def depth(self) -> np.ndarray:
        return self.grid.at_node["surface_water__depth"]

    def sync_to_grid(self) -> None:
        """Already synced as it works directly on grid fields."""
        pass

    def calc_time_step(self) -> float:
        return float(self.of.calc_time_step())

    def run_one_step(self, dt: float) -> None:
        self.of.run_one_step(dt)

BackendRegistry.register("landlab", LandlabSolver)
