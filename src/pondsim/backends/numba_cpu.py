from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from .base import SolverBackend, BackendRegistry

@njit(parallel=True)
def update_links_numba(q, depth, elev, dist, n_link, n1, n2, active_links, dt, g, alpha):
    for i in prange(len(active_links)):
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]

        z1 = elev[idx1]
        z2 = elev[idx2]
        w1 = depth[idx1] + z1
        w2 = depth[idx2] + z2
        h_flow = max(w1, w2) - max(z1, z2)
        if h_flow < 1e-6:
            q[link_idx] = 0.0
            continue

        S = (w2 - w1) / dist[link_idx]
        num = q[link_idx] - g * h_flow * dt * S

        h_term = h_flow ** (7.0 / 3.0)
        if h_term < 1e-6:
            h_term = 1e-6
        den = 1.0 + g * dt * (n_link[link_idx] ** 2) * abs(q[link_idx]) / h_term

        new_q = num / den

        # Froude limiter — physically-based cap (Fr <= 1)
        q_froude = h_flow * math.sqrt(g * h_flow)
        if abs(new_q) > q_froude:
            new_q = (new_q / abs(new_q)) * q_froude

        # Volume-Courant limiter — prevents a single explicit step from draining a
        # cell faster than the wave speed allows.  Coefficient 0.2 was validated
        # against Landlab OverlandFlow; do not raise to alpha (0.7) — that allows
        # too much discharge at moderate depths and causes the depression to drain
        # instead of fill (injection CFL cap in engine.py handles the dry-grid
        # startup step that originally motivated the looser limiter).
        q_stability = 0.2 * h_flow * dist[link_idx] / dt
        if abs(new_q) > q_stability:
            new_q = (new_q / abs(new_q)) * q_stability

        if not math.isfinite(new_q):
            new_q = 0.0

        q[link_idx] = new_q


@njit(parallel=True)
def update_nodes_gather_numba(depth, elev, q, links_at_node, link_dirs,
                               node_status, dx, dy, dt, area, new_depths):
    for i in prange(len(depth)):
        if node_status[i] != 0:
            new_depths[i] = depth[i]
            continue

        dq = 0.0
        for j in range(4):
            link = links_at_node[i, j]
            if link != -1:
                ldir = link_dirs[i, j]
                Lperp = dy if (j == 0 or j == 2) else dx
                dq += q[link] * Lperp * ldir

        val = depth[i] + (dt / area) * dq

        if val < 0.0:
            val = 0.0
        elif val > 1000.0:
            val = 1000.0
        elif not math.isfinite(val):
            val = 0.0

        new_depths[i] = val


@njit
def calc_dt_parallel(q, depth, elev, n1, n2, active_links, dx, dy, g, alpha):
    # Bates et al. (2010) Eq. 14: dt = alpha * dx / sqrt(g * h)
    # Serial loop intentional: prange boolean reductions (found_active = True)
    # are not supported by Numba's parallel reducer and silently stay False,
    # which caused this function to always return the 1.0s fallback regardless
    # of actual water depth — killing the adaptive timestep entirely.
    # h_min matches OverlandFlow's h_init for consistent dry-grid behaviour.
    h_min = 1e-5
    mindt = 1000.0
    for i in range(len(active_links)):
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]
        w1 = depth[idx1] + elev[idx1]
        w2 = depth[idx2] + elev[idx2]
        h = max(w1, w2) - max(elev[idx1], elev[idx2])
        if h < h_min:
            h = h_min
        link_dist = dx if abs(idx2 - idx1) == 1 else dy
        dt_link = alpha * link_dist / math.sqrt(g * h)
        if dt_link < mindt:
            mindt = dt_link

    if mindt > 999.0:
        link_dist = min(dx, dy)
        return alpha * link_dist / math.sqrt(g * h_min)
    return mindt


class NumbaCpuSolver(SolverBackend):
    TIER = 2
    NAME = "Numba CPU Parallel"

    def __init__(self, grid_data, config, grid=None):
        self.nrows = grid_data["nrows"]
        self.ncols = grid_data["ncols"]
        self.dx = float(grid_data["dx"])
        self.dy = float(grid_data["dy"])
        self.elev = grid_data["elev"].ravel().astype(np.float32)
        self._depth = grid_data["depth"].ravel().astype(np.float32)
        self._grid_depth_ref = grid_data["depth"].ravel()

        self.nodes_at_link = grid_data["nodes_at_link"]
        self.links_at_node = grid_data["links_at_node"]
        self.link_dirs = grid_data["link_dirs"]
        self.node_status = grid_data["node_status"]
        self.active_links = grid_data["active_links"]

        self.n_nodes = len(self._depth)
        self.n_links = len(self.nodes_at_link)
        self.area = self.dx * self.dy

        self.q = np.zeros(self.n_links, dtype=np.float32)

        if isinstance(config.manning_n, np.ndarray):
            self.n_nodes_arr = config.manning_n.ravel().astype(np.float32)
        else:
            self.n_nodes_arr = np.full(self.n_nodes, config.manning_n, dtype=np.float32)

        self.n1 = self.nodes_at_link[:, 0]
        self.n2 = self.nodes_at_link[:, 1]
        self.dist = np.where(np.abs(self.n2 - self.n1) == 1,
                             self.dx, self.dy).astype(np.float32)
        self._new_depth = self._depth.copy()
        self.alpha = 0.7
        self._g = 9.80665
        self.n_link = 0.5 * (self.n_nodes_arr[self.n1] + self.n_nodes_arr[self.n2])

        self.fixed_val_indices = np.where(self.node_status == 1)[0]
        self.fixed_grad_indices = np.where(self.node_status == 2)[0]
        self.grad_neighbors = self._compute_neighbors(self.fixed_grad_indices).astype(np.int32)

    def _compute_neighbors(self, indices):
        neighbors = np.zeros_like(indices)
        for i, idx in enumerate(indices):
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
        try:
            import numba  # noqa: F401
            return True, "Ok"
        except ImportError:
            return False, "Numba not installed"

    @classmethod
    def check_vram(cls, grid):
        return True

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    def sync_to_grid(self) -> None:
        self._grid_depth_ref[:] = self._depth

    def add_to_depth(self, node_idx: int, value: float) -> None:
        self._depth[node_idx] += value

    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None:
        self._depth[node_indices] += values.astype(np.float32)

    def calc_time_step(self):
        return calc_dt_parallel(
            self.q, self._depth, self.elev, self.n1, self.n2,
            self.active_links, self.dx, self.dy, self._g, self.alpha
        )

    def run_one_step(self, dt: float) -> None:
        update_links_numba(
            self.q, self._depth, self.elev, self.dist,
            self.n_link, self.n1, self.n2, self.active_links,
            dt, self._g, self.alpha
        )

        update_nodes_gather_numba(
            self._depth, self.elev, self.q,
            self.links_at_node, self.link_dirs,
            self.node_status, self.dx, self.dy, dt, self.area,
            self._new_depth
        )

        self._depth, self._new_depth = self._new_depth, self._depth

        if len(self.fixed_val_indices) > 0:
            self._depth[self.fixed_val_indices] = 0.0
        if len(self.fixed_grad_indices) > 0:
            self._depth[self.fixed_grad_indices] = self._depth[self.grad_neighbors]


BackendRegistry.register("numba_cpu", NumbaCpuSolver)
