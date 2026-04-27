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
        
        # OverlandFlow takes a scalar mannings_n; for a spatially variable roughness
        # array we use the mean (Landlab OverlandFlow doesn't support per-node n).
        if isinstance(config.manning_n, np.ndarray):
            n_scalar = float(np.mean(config.manning_n))
        else:
            n_scalar = float(config.manning_n)

        self.of = OverlandFlow(self.grid, steep_slopes=config.steep_slopes,
                               mannings_n=n_scalar)

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
        self.apply_bc()

    def apply_bc(self):
        depth = self.grid.at_node["surface_water__depth"]
        status = self.grid.status_at_node
        fixed_val = np.where(status == 1)[0]
        if len(fixed_val) > 0:
            depth[fixed_val] = 0.0

    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None:
        # Landlab grid field is linked to self.grid.at_node["surface_water__depth"]
        # Use np.add.at for safety if there are duplicate indices
        np.add.at(self.grid.at_node["surface_water__depth"], node_indices, values)

BackendRegistry.register("landlab", LandlabSolver)
