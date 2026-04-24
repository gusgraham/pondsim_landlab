from __future__ import annotations
import numpy as np
from .base import SolverBackend, BackendRegistry

class NumpySolver(SolverBackend):
    TIER = 0
    NAME = "NumPy Baseline"

    def __init__(self, grid_data, config, grid=None):
        self.nrows = grid_data["nrows"]
        self.ncols = grid_data["ncols"]
        self.dx = grid_data["dx"]
        self.dy = grid_data["dy"]
        self.elev = grid_data["elev"].ravel()
        self._depth = grid_data["depth"].ravel()
        self.nodes_at_link = grid_data["nodes_at_link"]
        self.links_at_node = grid_data["links_at_node"]
        self.link_dirs = grid_data["link_dirs"]
        self.node_status = grid_data["node_status"]

        self.n_nodes_num = len(self._depth)
        self.n_links_num = len(self.nodes_at_link)
        self.area = self.dx * self.dy
        self.q = np.zeros(self.n_links_num, dtype=np.float32)

        if isinstance(config.manning_n, np.ndarray):
            self.n_nodes_arr = config.manning_n.ravel()
        else:
            self.n_nodes_arr = np.full(self.n_nodes_num, config.manning_n, dtype=np.float32)

        self.n1 = self.nodes_at_link[:, 0]
        self.n2 = self.nodes_at_link[:, 1]
        self.dist = np.where(np.abs(self.n2 - self.n1) == 1, self.dx, self.dy)
        self.n_link = 0.5 * (self.n_nodes_arr[self.n1] + self.n_nodes_arr[self.n2])

        # Boundary logic
        self.boundary_indices = np.where(self.node_status != 0)[0]
        self.neighbor_indices = self._compute_neighbors()

    def _compute_neighbors(self):
        neighbors = np.zeros_like(self.boundary_indices)
        for i, idx in enumerate(self.boundary_indices):
            r, c = idx // self.ncols, idx % self.ncols
            nr, nc = r, c
            if r == 0: nr = 1
            elif r == self.nrows - 1: nr = self.nrows - 2
            if c == 0: nc = 1
            elif c == self.ncols - 1: nc = self.ncols - 2
            neighbors[i] = nr * self.ncols + nc
        return neighbors

    @classmethod
    def check_availability(cls):
        return True, "Ok"

    @classmethod
    def check_vram(cls, grid):
        return True

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    def sync_to_grid(self) -> None:
        pass

    def calc_time_step(self) -> float:
        g = 9.80665
        h_at_link = np.maximum(self._depth[self.n1], self._depth[self.n2])
        v = self.q / (h_at_link + 1e-4)
        c = np.sqrt(g * np.maximum(self._depth, 1e-4))
        dt_links = self.dist / (np.abs(v) + c[self.n1] + 1e-4)
        return float(np.min(dt_links)) * 0.2

    def run_one_step(self, dt: float) -> None:
        g = 9.80665
        
        # 1. Zero-gradient BC
        self._depth[self.boundary_indices] = self._depth[self.neighbor_indices]
        
        # 2. Local inertia update
        w = self._depth + self.elev
        S = (w[self.n2] - w[self.n1]) / self.dist
        h_flow = np.maximum(w[self.n1], w[self.n2]) - np.maximum(self.elev[self.n1], self.elev[self.n2])
        h_flow = np.maximum(h_flow, 0.0)
        
        active = (h_flow > 1e-4) | (np.abs(self.q) > 1e-6)
        
        num = self.q[active] - g * h_flow[active] * dt * S[active]
        den = 1.0 + g * dt * (self.n_link[active]**2) * np.abs(self.q[active]) / (h_flow[active]**(7/3) + 1e-6)
        self.q[active] = num / den
        
        # 3. Continuity update
        Q_links = self.q * np.where(np.abs(self.n2 - self.n1) == 1, self.dy, self.dx)
        dq = np.zeros(self.n_nodes_num, dtype=np.float32)
        np.add.at(dq, self.n1, Q_links)
        np.add.at(dq, self.n2, -Q_links)
        
        self._depth[:] -= (dt / self.area) * dq
        self._depth[self._depth < 0] = 0.0

BackendRegistry.register("numpy", NumpySolver)
