from __future__ import annotations
import logging
import numpy as np
from numba import cuda
import math
from .base import SolverBackend, BackendRegistry, check_cuda_vram

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CUDA Kernels
# ------------------------------------------------------------------

@cuda.reduce
def reduce_min(a, b):
    return min(a, b)

@cuda.jit
def k_add_to_depths(depth, node_indices, values):
    i = cuda.grid(1)
    if i < node_indices.size:
        idx = node_indices[i]
        val = values[i]
        cuda.atomic.add(depth, idx, val)

@cuda.jit
def k_update_links(q, depth, elev, dist, n_link, n1, n2, active_links, dt, g, alpha):
    i = cuda.grid(1)
    if i < active_links.size:
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]
        
        d1 = depth[idx1]
        d2 = depth[idx2]
        w1 = d1 + elev[idx1]
        w2 = d2 + elev[idx2]
        
        # Bates height: max(w1, w2) - max(z1, z2)
        z1 = elev[idx1]
        z2 = elev[idx2]
        h_flow = max(w1, w2) - max(z1, z2)
        
        if h_flow < 1e-7:
            q[link_idx] = 0.0
            return
            
        S = (w2 - w1) / dist[link_idx]
        num = q[link_idx] - g * h_flow * dt * S
        
        h_term = h_flow**(7/3)
        if h_term < 1e-7: h_term = 1e-7
        den = 1.0 + g * dt * (n_link[link_idx]**2) * abs(q[link_idx]) / h_term
        
        new_q = num / den
        
        # --- Stability Adjustments (from Landlab deAlmeida/Bates) ---
        # 1. Supercritical flow limit (Froude number <= 1.0)
        q_froude = h_flow * math.sqrt(g * h_flow)
        if abs(new_q) > q_froude:
            new_q = (new_q / abs(new_q)) * q_froude
            
        # 2. Stability limit (don't drain more than available water)
        # Landlab uses q_max = 0.2 * h * dx / dt (roughly)
        q_stability = 0.2 * h_flow * dist[link_idx] / dt
        if abs(new_q) > q_stability:
            new_q = (new_q / abs(new_q)) * q_stability
            
        if not math.isfinite(new_q):
            new_q = 0.0
            
        q[link_idx] = new_q

@cuda.jit
def k_update_nodes(depth, elev, q, links_at_node, nodes_at_link, link_dirs, node_status, dx, dy, dt, area, new_depths):
    i = cuda.grid(1)
    if i < depth.size:
        if node_status[i] != 0:
            new_depths[i] = depth[i]
            return
            
        dq = 0.0
        for j in range(4):
            link = links_at_node[i, j]
            if link != -1:
                ldir = link_dirs[i, j]
                Lperp = dy if (j == 0 or j == 2) else dx
                dq += q[link] * Lperp * ldir
        
        val = depth[i] + (dt / area) * dq
        if val < 0:
            val = 0.0
        elif val > 1000.0:
            val = 1000.0
        elif not math.isfinite(val):
            val = 0.0
            
        new_depths[i] = val

@cuda.jit
def k_apply_bc(depth, fixed_val_indices, fixed_grad_indices, grad_neighbors):
    # This kernel is simple enough to handle both types if we split by index range
    # But for clarity, let's just use two separate calls or a combined index check
    i = cuda.grid(1)
    
    # Handle fixed values
    if i < fixed_val_indices.size:
        idx = fixed_val_indices[i]
        depth[idx] = 0.0
        
    # Handle fixed gradients
    if i < fixed_grad_indices.size:
        idx = fixed_grad_indices[i]
        neigh = grad_neighbors[i]
        depth[idx] = depth[neigh]

@cuda.jit
def k_calc_dt(q, depth, elev, n1, n2, active_links, dist, g, dt_out):
    i = cuda.grid(1)
    if i < active_links.size:
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]
        w1 = depth[idx1] + elev[idx1]
        w2 = depth[idx2] + elev[idx2]
        zmax = max(elev[idx1], elev[idx2])
        wmax = max(w1, w2)
        h = wmax - zmax
        
        if h < 1e-7:
            dt_out[i] = 1000.0
        else:
            v = abs(q[link_idx]) / (h + 1e-4)
            c = math.sqrt(g * h)
            dt_out[i] = (dist[link_idx] / (v + c + 1e-4)) * 0.2

@cuda.reduce
def reduce_sum(a, b):
    return a + b

@cuda.jit
def k_update_max_depth(depth, max_depth):
    i = cuda.grid(1)
    if i < depth.size:
        if depth[i] > max_depth[i]:
            max_depth[i] = depth[i]

@cuda.jit
def k_update_max_level(depth, elev, max_level):
    i = cuda.grid(1)
    if i < depth.size:
        lvl = depth[i] + elev[i]
        if lvl > max_level[i]:
            max_level[i] = lvl

@cuda.jit
def k_update_max_q(q, links_at_node, max_q):
    i = cuda.grid(1)
    if i < max_q.size:
        mq = max_q[i]
        for j in range(4):
            link = links_at_node[i, j]
            if link != -1:
                val = abs(q[link])
                if val > mq:
                    mq = val
        max_q[i] = mq

class NumbaCudaSolver(SolverBackend):
    TIER = 3
    NAME = "Numba CUDA GPU"
    TRACKS_MAX_INTERNALLY = True

    def __init__(self, grid_data, config, grid=None):
        self.config = config
        self.nrows = grid_data["nrows"]
        self.ncols = grid_data["ncols"]
        self.dx = float(grid_data["dx"])
        self.dy = float(grid_data["dy"])
        self.area = self.dx * self.dy
        
        # 1. Host data setup
        elev_h = grid_data["elev"].ravel().astype(np.float32)
        depth_h = grid_data["depth"].ravel().astype(np.float32)
        self._grid_depth_ref = grid_data["depth"].ravel() 
        
        links_at_node_h = np.ascontiguousarray(grid_data["links_at_node"].astype(np.int32))
        link_dirs_h = np.ascontiguousarray(grid_data["link_dirs"].astype(np.int8))
        node_status_h = np.ascontiguousarray(grid_data["node_status"].astype(np.int8))
        active_links_h = np.ascontiguousarray(grid_data["active_links"].astype(np.int32))
        nodes_at_link_h = np.ascontiguousarray(grid_data["nodes_at_link"].astype(np.int32))
        
        n_nodes = len(depth_h)
        n_links = len(nodes_at_link_h)
        
        if isinstance(config.manning_n, np.ndarray):
            n_nodes_arr_h = config.manning_n.ravel().astype(np.float32)
        else:
            n_nodes_arr_h = np.full(n_nodes, config.manning_n, dtype=np.float32)
            
        n1_h = np.ascontiguousarray(nodes_at_link_h[:, 0])
        n2_h = np.ascontiguousarray(nodes_at_link_h[:, 1])
        dist_h = np.where(np.abs(n2_h - n1_h) == 1, self.dx, self.dy).astype(np.float32)
        n_link_h = 0.5 * (n_nodes_arr_h[n1_h] + n_nodes_arr_h[n2_h])
        
        fixed_val_h = np.ascontiguousarray(np.where(node_status_h == 1)[0].astype(np.int32))
        fixed_grad_h = np.ascontiguousarray(np.where(node_status_h == 2)[0].astype(np.int32))
        grad_neigh_h = np.ascontiguousarray(self._compute_neighbors_host(fixed_grad_h).astype(np.int32))

        # 2. Device Allocation
        self.d_elev = cuda.to_device(elev_h)
        self.d_depth = cuda.to_device(depth_h)
        self.d_depth_new = cuda.to_device(depth_h) # buffer for double-buffering
        self.d_q = cuda.to_device(np.zeros(n_links, dtype=np.float32))
        self.d_n_link = cuda.to_device(n_link_h)
        self.d_dist = cuda.to_device(dist_h)
        self.d_n1 = cuda.to_device(n1_h)
        self.d_n2 = cuda.to_device(n2_h)
        self.d_active_links = cuda.to_device(active_links_h)
        self.d_links_at_node = cuda.to_device(links_at_node_h)
        self.d_link_dirs = cuda.to_device(link_dirs_h)
        self.d_node_status = cuda.to_device(node_status_h)
        self.d_nodes_at_link = cuda.to_device(nodes_at_link_h)
        
        self.d_fixed_val = cuda.to_device(fixed_val_h)
        self.d_fixed_grad = cuda.to_device(fixed_grad_h)
        self.d_grad_neigh = cuda.to_device(grad_neigh_h)
        
        self.d_max_depth = cuda.to_device(np.zeros(n_nodes, dtype=np.float32))
        self.d_max_level = cuda.to_device(elev_h.copy()) # starts at elevation
        self.d_max_q = cuda.to_device(np.zeros(n_nodes, dtype=np.float32))
        
        self.d_dt_buffer = cuda.device_array(len(active_links_h), dtype=np.float32)
        
        # Cache for source indices to avoid redundant PCIe transfers
        self._last_source_indices_id = None
        self._d_source_indices = None
        self._d_source_values = None
        
        # Launch config
        self.tpb = 256
        self.bpg_nodes = (n_nodes + self.tpb - 1) // self.tpb
        self.bpg_links = (len(active_links_h) + self.tpb - 1) // self.tpb
        self.bpg_bc = (max(len(fixed_val_h), len(fixed_grad_h)) + self.tpb - 1) // self.tpb
        if self.bpg_bc == 0: self.bpg_bc = 1

        self.g = 9.80665
        self.alpha = 0.7

    def _compute_neighbors_host(self, indices):
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
    def check_availability(cls) -> tuple[bool, str]:
        try:
            from numba import cuda
            if cuda.is_available():
                return True, "Ok"
            else:
                return False, "CUDA hardware not detected"
        except ImportError:
            return False, "Numba/CUDA not installed"

    @classmethod
    def check_vram(cls, grid) -> bool:
        available, msg = check_cuda_vram(grid.number_of_node_rows, grid.number_of_node_columns)
        if not available:
            logger.info(f"CUDA Tier 3 bypassed: {msg}")
        return available

    @property
    def depth(self) -> np.ndarray:
        # Copy back to host only on request
        return self.d_depth.copy_to_host()

    @property
    def q(self) -> np.ndarray:
        return self.d_q.copy_to_host()

    @property
    def max_depth(self) -> np.ndarray:
        return self.d_max_depth.copy_to_host()

    @property
    def max_level(self) -> np.ndarray:
        return self.d_max_level.copy_to_host()

    @property
    def max_q(self) -> np.ndarray:
        return self.d_max_q.copy_to_host()

    def get_total_volume(self) -> float:
        return float(reduce_sum(self.d_depth)) * self.area

    def sync_to_grid(self) -> None:
        self._grid_depth_ref[:] = self.depth.astype(np.float64)

    def add_to_depth(self, node_idx: int, value: float) -> None:
        # Compatibility with protocol
        self.add_to_depths(np.array([node_idx], dtype=np.int32), np.array([value], dtype=np.float32))

    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None:
        if len(node_indices) == 0:
            return
        
        # Cache indices to avoid re-transferring constant source mappings
        indices_id = id(node_indices)
        if indices_id != self._last_source_indices_id:
            self._d_source_indices = cuda.to_device(node_indices)
            self._d_source_values = cuda.device_array(len(node_indices), dtype=np.float32)
            self._last_source_indices_id = indices_id
            
        # Copy values (always change) and launch
        self._d_source_values.copy_to_device(values)
        bpg = (len(node_indices) + self.tpb - 1) // self.tpb
        k_add_to_depths[bpg, self.tpb](self.d_depth, self._d_source_indices, self._d_source_values)

    def calc_time_step(self) -> float:
        g = 9.80665
        k_calc_dt[self.bpg_links, self.tpb](
            self.d_q, self.d_depth, self.d_elev, self.d_n1, self.d_n2, 
            self.d_active_links, self.d_dist, g, self.d_dt_buffer
        )
        # Reduction on GPU to avoid large PCIe transfers
        return float(reduce_min(self.d_dt_buffer))

    def run_one_step(self, dt: float) -> None:
        # 1. Update discharges
        k_update_links[self.bpg_links, self.tpb](
            self.d_q, self.d_depth, self.d_elev, self.d_dist, self.d_n_link, 
            self.d_n1, self.d_n2, self.d_active_links, dt, self.g, self.alpha
        )
        
        # 2. Update depths (to buffer)
        k_update_nodes[self.bpg_nodes, self.tpb](
            self.d_depth, self.d_elev, self.d_q, self.d_links_at_node, 
            self.d_nodes_at_link, self.d_link_dirs, 
            self.d_node_status, self.dx, self.dy, dt, self.area, self.d_depth_new
        )
        
        # Swap buffers
        self.d_depth, self.d_depth_new = self.d_depth_new, self.d_depth
        
        # 3. Apply BCs
        k_apply_bc[self.bpg_bc, self.tpb](
            self.d_depth, self.d_fixed_val, self.d_fixed_grad, self.d_grad_neigh
        )
        
        # 4. Update Max Tracking (on GPU)
        k_update_max_depth[self.bpg_nodes, self.tpb](self.d_depth, self.d_max_depth)
        k_update_max_level[self.bpg_nodes, self.tpb](self.d_depth, self.d_elev, self.d_max_level)
        k_update_max_q[self.bpg_nodes, self.tpb](self.d_q, self.d_links_at_node, self.d_max_q)

# Register
BackendRegistry.register("numba_cuda", NumbaCudaSolver)
