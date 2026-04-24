from __future__ import annotations
import numpy as np
from numba import njit, prange
from .base import SolverBackend, BackendRegistry

@njit(parallel=True)
def update_links_numba(q, depth, elev, dist, n_link, n1, n2, active_links, dt, g, alpha=0.7):
    for i in prange(len(active_links)):
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]
        
        d1 = depth[idx1]
        d2 = depth[idx2]
        w1 = d1 + elev[idx1]
        w2 = d2 + elev[idx2]
        # Bates height: max(w1, w2) - max(z1, z2)
        h_flow = max(w1, w2) - max(elev[idx1], elev[idx2])
        if h_flow < 1e-7: h_flow = 0.0
        
        # S = (w_head - w_tail) / L. Landlab links are tail -> head (n1 -> n2)
        S = (w2 - w1) / dist[link_idx]
        
        # Momentum update: q_new = (q - g*h*dt*S) / (1 + friction)
        num = q[link_idx] - g * h_flow * dt * S
        
        # Friction term (semi-implicit)
        h_term = h_flow**(7/3)
        if h_term < 1e-7: h_term = 1e-7
        den = 1.0 + g * dt * (n_link[link_idx]**2) * abs(q[link_idx]) / h_term
        
        new_q = num / den
        
        # --- Stability Adjustments (from Landlab deAlmeida/Bates) ---
        # 1. Supercritical flow limit (Froude number <= 1.0)
        # q_max = froude * h * sqrt(g * h)
        q_froude = h_flow * (g * h_flow)**0.5
        if abs(new_q) > q_froude:
            new_q = (new_q / abs(new_q)) * q_froude
            
        # 2. Stability limit (don't drain more than available water)
        # Landlab uses q_max = 0.2 * h * dx / dt (roughly)
        # Here we use 0.2 * h * min(dx, dy) / dt
        # Note: 'dx' and 'dy' are needed here, or similar resolution metrics
        # Given the original code signature, this assumes a rectangular grid
        # or simplified dimension logic.
        q_stability = 0.2 * h_flow * dist[link_idx] / dt
        if abs(new_q) > q_stability:
            new_q = (new_q / abs(new_q)) * q_stability
        
        if not np.isfinite(new_q):
            new_q = 0.0
            
        q[link_idx] = new_q

@njit(parallel=True)
def update_nodes_gather_numba(depth, elev, q, links_at_node, nodes_at_link, link_dirs, node_status, dx, dy, dt, area, new_depths, alpha=0.7):
    for i in prange(len(depth)):
        # Only update Core nodes (status 0)
        if node_status[i] != 0:
            new_depths[i] = depth[i]
            continue
            
        dq = 0.0
        for j in range(4):
            link = links_at_node[i, j]
            if link != -1:
                ldir = link_dirs[i, j]
                # East/West width is dy, North/South width is dx
                Lperp = dy if (j == 0 or j == 2) else dx
                dq += q[link] * Lperp * ldir
        
        val = depth[i] + (dt / area) * dq
        
        # Stability: no negative depths, cap extreme
        if val < 0: val = 0.0
        elif val > 1000.0: val = 1000.0
        elif not np.isfinite(val): val = 0.0
            
        new_depths[i] = val

@njit(parallel=True)
def calc_dt_parallel(q, depth, elev, n1, n2, active_links, dx, dy, g):
    mindt = 1000.0
    found_active = False
    for i in prange(len(active_links)):
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]
        w1 = depth[idx1] + elev[idx1]
        w2 = depth[idx2] + elev[idx2]
        zmax = elev[idx1] if elev[idx1] > elev[idx2] else elev[idx2]
        wmax = w1 if w1 > w2 else w2
        h = wmax - zmax
        if h < 1e-4: continue
        
        found_active = True
        v = abs(q[link_idx]) / h
        c = np.sqrt(g * h)
        dist = dx if abs(idx2 - idx1) == 1 else dy
        dt = dist / (v + c + 1e-4)
        if dt < mindt:
            mindt = dt
            
    if not found_active:
        return 1.0
    return mindt * 0.2

class NumbaCpuSolver(SolverBackend):
    TIER = 2
    NAME = "Numba CPU Parallel"
    
    def __init__(self, grid_data, config, grid=None):
        self.nrows = grid_data["nrows"]
        self.ncols = grid_data["ncols"]
        self.dx = float(grid_data["dx"])
        self.dy = float(grid_data["dy"])
        self.elev = grid_data["elev"].ravel().astype(np.float32)
        self._depth_f32 = grid_data["depth"].ravel().astype(np.float32)
        self._grid_depth_ref = grid_data["depth"].ravel() 
        
        self.nodes_at_link = grid_data["nodes_at_link"]
        self.links_at_node = grid_data["links_at_node"]
        self.link_dirs = grid_data["link_dirs"]
        self.node_status = grid_data["node_status"]
        self.active_links = grid_data["active_links"]
        
        self.n_nodes = len(self._depth_f32)
        self.n_links = len(self.nodes_at_link)
        self.area = self.dx * self.dy
        
        self.q = np.zeros(self.n_links, dtype=np.float32)
        
        if isinstance(config.manning_n, np.ndarray):
            self.n_nodes_arr = config.manning_n.ravel().astype(np.float32)
        else:
            self.n_nodes_arr = np.full(self.n_nodes, config.manning_n, dtype=np.float32)
            
        self.n1 = self.nodes_at_link[:, 0]
        self.n2 = self.nodes_at_link[:, 1]
        self.dist = np.where(np.abs(self.n2 - self.n1) == 1, self.dx, self.dy).astype(np.float32)
        self._new_depth_f32 = self._depth_f32.copy()
        self.alpha = 0.7
        self._g = 9.80665
        self.n_link = 0.5 * (self.n_nodes_arr[self.n1] + self.n_nodes_arr[self.n2])
        
        # BC handling
        # Status 1: Fixed Value (Outlet, depth=0)
        # Status 2: Fixed Gradient (Zero-gradient, depth=neighbor)
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
            import numba
            return True, "Ok"
        except ImportError:
            return False, "Numba not installed"

    @classmethod
    def check_vram(cls, grid):
        return True

    @property
    def depth(self) -> np.ndarray:
        return self._depth_f32

    def sync_to_grid(self) -> None:
        self._grid_depth_ref[:] = self._depth_f32.astype(np.float64)

    def add_to_depth(self, node_idx: int, value: float) -> None:
        self._depth_f32[node_idx] += value

    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None:
        self._depth_f32[node_indices] += values

    def calc_time_step(self):
        g = 9.80665
        return calc_dt_parallel(self.q, self._depth_f32, self.elev, self.n1, self.n2, self.active_links, self.dx, self.dy, g)

    def run_one_step(self, dt: float) -> None:
        update_links_numba(
            self.q, self._depth_f32, self.elev, self.dist, 
            self.n_link, self.n1, self.n2, self.active_links, 
            dt, self._g, self.alpha
        )
        
        update_nodes_gather_numba(
            self._depth_f32, self.elev, self.q, 
            self.links_at_node, self.nodes_at_link, self.link_dirs, 
            self.node_status, self.dx, self.dy, dt, self.area, 
            self._new_depth_f32, self.alpha
        )
        
        # Swap buffers
        self._depth_f32, self._new_depth_f32 = self._new_depth_f32, self._depth_f32
        
        # 3. Boundary Conditions
        if len(self.fixed_val_indices) > 0:
            self._depth_f32[self.fixed_val_indices] = 0.0
        if len(self.fixed_grad_indices) > 0:
            self._depth_f32[self.fixed_grad_indices] = self._depth_f32[self.grad_neighbors]

BackendRegistry.register("numba_cpu", NumbaCpuSolver)
