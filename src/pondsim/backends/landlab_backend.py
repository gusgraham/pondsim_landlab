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

    def _compute_neighbors(self, indices):
        neighbors = np.zeros_like(indices)
        nrows, ncols = self.grid.number_of_node_rows, self.grid.number_of_node_columns
        for i, idx in enumerate(indices):
            r, c = idx // ncols, idx % ncols
            nr, nc = r, c
            if r == 0: nr = 1
            elif r == nrows - 1: nr = nrows - 2
            if c == 0: nc = 1
            elif c == ncols - 1: nc = ncols - 2
            neighbors[i] = nr * ncols + nc
        return neighbors

    def calc_time_step(self) -> float:
        return float(self.of.calc_time_step())

    def run_one_step(self, dt: float) -> None:
        self.of.run_one_step(dt)
        self.apply_bc()

    def apply_bc(self):
        depth = self.grid.at_node["surface_water__depth"]
        status = self.grid.status_at_node
        
        # Fixed Value (Outlets)
        fixed_val = np.where(status == 1)[0]
        if len(fixed_val) > 0:
            depth[fixed_val] = 0.0
            
        # Fixed Gradient (Normal Depth)
        fixed_grad = np.where(status == 2)[0]
        if len(fixed_grad) > 0:
            # Re-calculate neighbors only if needed (cached in a real app, but this is simple)
            if not hasattr(self, "_grad_neighbors"):
                self._grad_neighbors = self._compute_neighbors(fixed_grad)
            depth[fixed_grad] = depth[self._grad_neighbors]

    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None:
        # Landlab grid field is linked to self.grid.at_node["surface_water__depth"]
        # Use np.add.at for safety if there are duplicate indices
        np.add.at(self.grid.at_node["surface_water__depth"], node_indices, values)

BackendRegistry.register("landlab", LandlabSolver)
