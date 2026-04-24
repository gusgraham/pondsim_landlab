from __future__ import annotations
import numpy as np
from numba import njit, prange
from .base import SolverBackend, BackendRegistry

@njit(parallel=True)
def update_links_numba(q, depth, elev, dist, n_link, n1, n2, active_links, dt, g):
    for i in prange(len(active_links)):
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]
        
        d1 = depth[idx1]
        d2 = depth[idx2]
        w1 = d1 + elev[idx1]
        w2 = d2 + elev[idx2]
        
        z1 = elev[idx1]
        z2 = elev[idx2]
        zmax = z1 if z1 > z2 else z2
        wmax = w1 if w1 > w2 else w2
        h_flow = wmax - zmax
        if h_flow < 1e-6: h_flow = 0.0
        
        # S = (w_head - w_tail) / L. Landlab links are tail -> head (n1 -> n2)
        S = (w2 - w1) / dist[link_idx]
        
        # Momentum update: q_new = (q - g*h*dt*S) / (1 + friction)
        num = q[link_idx] - g * h_flow * dt * S
        
        # Friction term (semi-implicit)
        h_term = h_flow**(7/3)
        if h_term < 1e-6: h_term = 1e-6
        den = 1.0 + g * dt * (n_link[link_idx]**2) * abs(q[link_idx]) / h_term
        
        new_q = num / den
        
        # Numerical stability: limit discharge to prevent explosions
        # Max velocity ~ 50 m/s (extreme flash flood)
        q_limit = 50.0 * (h_flow + 1e-3)
        if new_q > q_limit: new_q = q_limit
        elif new_q < -q_limit: new_q = -q_limit
        
        if not np.isfinite(new_q):
            new_q = 0.0
            
        q[link_idx] = new_q

@njit(parallel=True)
def update_nodes_gather_numba(depth, q, links_at_node, link_dirs, node_status, dx, dy, dt, area):
    for i in prange(len(depth)):
        # Only update Core nodes (status 0). 
        # Boundaries (Fixed Value/Gradient) are handled in apply_bc
        if node_status[i] != 0:
            continue
            
        dq = 0.0
        for j in range(4):
            link = links_at_node[i, j]
            if link != -1:
                ldir = link_dirs[i, j]
                Lperp = dy if (j == 0 or j == 2) else dx
                dq += q[link] * Lperp * ldir
        
        new_depth = depth[i] + (dt / area) * dq
        
        # Stability: no negative depths, and cap extreme depths
        if new_depth < 0:
            new_depth = 0.0
        elif new_depth > 1000.0:
            new_depth = 1000.0
        elif not np.isfinite(new_depth):
            new_depth = 0.0
            
        depth[i] = new_depth

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

    def calc_time_step(self):
        g = 9.80665
        return calc_dt_parallel(self.q, self._depth_f32, self.elev, self.n1, self.n2, self.active_links, self.dx, self.dy, g)

    def run_one_step(self, dt: float) -> None:
        g = 9.80665
        # 1. Update discharges
        update_links_numba(self.q, self._depth_f32, self.elev, self.dist, self.n_link, 
                           self.n1, self.n2, self.active_links, dt, g)
        
        # 2. Update depths
        update_nodes_gather_numba(self._depth_f32, self.q, self.links_at_node, self.link_dirs, 
                                  self.node_status, self.dx, self.dy, dt, self.area)
        
        # 3. Boundary Conditions
        if len(self.fixed_val_indices) > 0:
            self._depth_f32[self.fixed_val_indices] = 0.0
        if len(self.fixed_grad_indices) > 0:
            self._depth_f32[self.fixed_grad_indices] = self._depth_f32[self.grad_neighbors]

BackendRegistry.register("numba_cpu", NumbaCpuSolver)
