from __future__ import annotations
import logging
import math
import numpy as np
from numba import cuda
from .base import SolverBackend, BackendRegistry, check_cuda_vram

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CUDA Kernels
# ------------------------------------------------------------------


@cuda.jit
def k_add_to_depths(depth, node_indices, values):
    i = cuda.grid(1)
    if i < node_indices.size:
        cuda.atomic.add(depth, node_indices[i], values[i])

@cuda.jit
def k_update_links(q, depth, elev, dist, n_link, n1, n2, active_links, dt, g):
    i = cuda.grid(1)
    if i < active_links.size:
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
            return

        S = (w2 - w1) / dist[link_idx]
        num = q[link_idx] - g * h_flow * dt * S

        h_term = h_flow ** (7.0 / 3.0)
        if h_term < 1e-6:
            h_term = 1e-6
        den = 1.0 + g * dt * (n_link[link_idx] ** 2) * abs(q[link_idx]) / h_term

        new_q = num / den

        q_froude = h_flow * math.sqrt(g * h_flow)
        if abs(new_q) > q_froude:
            new_q = (new_q / abs(new_q)) * q_froude

        q_stability = 0.2 * h_flow * dist[link_idx] / dt
        if abs(new_q) > q_stability:
            new_q = (new_q / abs(new_q)) * q_stability

        if not math.isfinite(new_q):
            new_q = 0.0

        q[link_idx] = new_q


@cuda.jit
def k_update_nodes(depth, elev, q, links_at_node, link_dirs, node_status,
                   grad_neighbor_at_node, dx, dy, dt, area, new_depths):
    # BC is applied inline: fixed-value → 0, fixed-gradient → copy neighbour,
    # closed/looped → copy self. Core nodes receive the full hydrodynamic update.
    i = cuda.grid(1)
    if i < depth.size:
        status = node_status[i]
        if status != 0:
            if status == 1:
                new_depths[i] = 0.0
            elif status == 2:
                new_depths[i] = depth[grad_neighbor_at_node[i]]
            else:
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
        if val < 0.0:
            val = 0.0
        elif val > 1000.0:
            val = 1000.0
        elif not math.isfinite(val):
            val = 0.0

        new_depths[i] = val


@cuda.jit
def k_calc_dt(depth, elev, n1, n2, active_links, dist, g, alpha, dt_out):
    # Bates et al. (2010) Eq. 14: dt = alpha * dx / sqrt(g * h)
    # h_min matches OverlandFlow's h_init and numba_cpu for consistent dry-grid behaviour.
    i = cuda.grid(1)
    if i < active_links.size:
        link_idx = active_links[i]
        idx1 = n1[link_idx]
        idx2 = n2[link_idx]
        h = max(depth[idx1] + elev[idx1], depth[idx2] + elev[idx2]) \
            - max(elev[idx1], elev[idx2])

        if h < 1e-5:
            h = 1e-5
        dt_out[i] = alpha * dist[link_idx] / math.sqrt(g * h)


@cuda.jit
def k_update_max_all(depth, elev, q, links_at_node, max_depth, max_level, max_q):
    # Single node-pass kernel replacing the previous three separate max kernels.
    i = cuda.grid(1)
    if i < depth.size:
        d = depth[i]
        if d > max_depth[i]:
            max_depth[i] = d

        lvl = d + elev[i]
        if lvl > max_level[i]:
            max_level[i] = lvl

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

        # Host arrays — float32 for GPU; float64 throughput on consumer NVIDIA
        # hardware is 1/32–1/64 of float32, making f64 impractical here.
        elev_h = grid_data["elev"].ravel().astype(np.float32)
        depth_h = grid_data["depth"].ravel().astype(np.float32)
        self._grid_depth_ref = grid_data["depth"].ravel()

        links_at_node_h = np.ascontiguousarray(grid_data["links_at_node"].astype(np.int32))
        link_dirs_h     = np.ascontiguousarray(grid_data["link_dirs"].astype(np.int8))
        node_status_h   = np.ascontiguousarray(grid_data["node_status"].astype(np.int8))
        active_links_h  = np.ascontiguousarray(grid_data["active_links"].astype(np.int32))
        nodes_at_link_h = np.ascontiguousarray(grid_data["nodes_at_link"].astype(np.int32))

        n_nodes = len(depth_h)
        n_links = len(nodes_at_link_h)

        if isinstance(config.manning_n, np.ndarray):
            n_nodes_arr_h = config.manning_n.ravel().astype(np.float32)
        else:
            n_nodes_arr_h = np.full(n_nodes, config.manning_n, dtype=np.float32)

        n1_h = np.ascontiguousarray(nodes_at_link_h[:, 0])
        n2_h = np.ascontiguousarray(nodes_at_link_h[:, 1])
        dist_h   = np.where(np.abs(n2_h - n1_h) == 1, self.dx, self.dy).astype(np.float32)
        n_link_h = (0.5 * (n_nodes_arr_h[n1_h] + n_nodes_arr_h[n2_h])).astype(np.float32)

        # Build flat grad-neighbour lookup (n_nodes long) so BC can be applied
        # inside k_update_nodes without a separate kernel launch.
        fixed_grad_h  = np.where(node_status_h == 2)[0].astype(np.int32)
        grad_neigh_h  = self._compute_neighbors_host(fixed_grad_h).astype(np.int32)
        grad_neighbor_at_node_h = np.zeros(n_nodes, dtype=np.int32)
        for k, node_idx in enumerate(fixed_grad_h):
            grad_neighbor_at_node_h[node_idx] = grad_neigh_h[k]

        # Device allocation
        self.d_elev              = cuda.to_device(elev_h)
        self.d_depth             = cuda.to_device(depth_h)
        self.d_depth_new         = cuda.to_device(depth_h.copy())
        self.d_q                 = cuda.to_device(np.zeros(n_links, dtype=np.float32))
        self.d_n_link            = cuda.to_device(n_link_h)
        self.d_dist              = cuda.to_device(dist_h)
        self.d_n1                = cuda.to_device(n1_h)
        self.d_n2                = cuda.to_device(n2_h)
        self.d_active_links      = cuda.to_device(active_links_h)
        self.d_links_at_node     = cuda.to_device(links_at_node_h)
        self.d_link_dirs         = cuda.to_device(link_dirs_h)
        self.d_node_status       = cuda.to_device(node_status_h)
        self.d_grad_neighbor_at_node = cuda.to_device(grad_neighbor_at_node_h)

        self.d_max_depth  = cuda.to_device(np.zeros(n_nodes, dtype=np.float32))
        self.d_max_level  = cuda.to_device(elev_h.copy())   # starts at bare elevation
        self.d_max_q      = cuda.to_device(np.zeros(n_nodes, dtype=np.float32))
        self.d_dt_buffer  = cuda.device_array(len(active_links_h), dtype=np.float32)

        # Source-index cache — avoids re-uploading constant index arrays each step
        self._last_source_indices_id = None
        self._d_source_indices = None
        self._d_source_values  = None

        self.tpb       = 256
        self.bpg_nodes = (n_nodes + self.tpb - 1) // self.tpb
        self.bpg_links = max(1, (len(active_links_h) + self.tpb - 1) // self.tpb)

        self.g     = 9.80665
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
            from numba import cuda as _cuda
            if _cuda.is_available():
                return True, "Ok"
            return False, "CUDA hardware not detected"
        except ImportError:
            return False, "Numba/CUDA not installed"

    @classmethod
    def check_vram(cls, grid) -> bool:
        available, msg = check_cuda_vram(grid.number_of_node_rows,
                                          grid.number_of_node_columns)
        if not available:
            logger.info("CUDA Tier 3 bypassed: %s", msg)
        return available

    @property
    def depth(self) -> np.ndarray:
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
        return float(np.sum(self.d_depth.copy_to_host())) * self.area

    def sync_to_grid(self) -> None:
        self._grid_depth_ref[:] = self.depth.astype(np.float64)

    def add_to_depth(self, node_idx: int, value: float) -> None:
        self.add_to_depths(np.array([node_idx], dtype=np.int32),
                           np.array([value], dtype=np.float32))

    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None:
        if len(node_indices) == 0:
            return
        indices_id = id(node_indices)
        if indices_id != self._last_source_indices_id:
            self._d_source_indices = cuda.to_device(node_indices)
            self._d_source_values  = cuda.device_array(len(node_indices), dtype=np.float32)
            self._last_source_indices_id = indices_id
        self._d_source_values.copy_to_device(values.astype(np.float32))
        bpg = (len(node_indices) + self.tpb - 1) // self.tpb
        k_add_to_depths[bpg, self.tpb](self.d_depth, self._d_source_indices,
                                        self._d_source_values)

    def calc_time_step(self) -> float:
        k_calc_dt[self.bpg_links, self.tpb](
            self.d_depth, self.d_elev, self.d_n1, self.d_n2,
            self.d_active_links, self.d_dist, self.g, self.alpha,
            self.d_dt_buffer
        )
        return float(np.min(self.d_dt_buffer.copy_to_host()))

    def run_one_step(self, dt: float) -> None:
        # 1. Momentum update
        k_update_links[self.bpg_links, self.tpb](
            self.d_q, self.d_depth, self.d_elev, self.d_dist, self.d_n_link,
            self.d_n1, self.d_n2, self.d_active_links, dt, self.g
        )

        # 2. Continuity update + BC applied inline
        k_update_nodes[self.bpg_nodes, self.tpb](
            self.d_depth, self.d_elev, self.d_q, self.d_links_at_node,
            self.d_link_dirs, self.d_node_status, self.d_grad_neighbor_at_node,
            self.dx, self.dy, dt, self.area, self.d_depth_new
        )
        self.d_depth, self.d_depth_new = self.d_depth_new, self.d_depth

        # 3. Max tracking — single pass over nodes
        k_update_max_all[self.bpg_nodes, self.tpb](
            self.d_depth, self.d_elev, self.d_q, self.d_links_at_node,
            self.d_max_depth, self.d_max_level, self.d_max_q
        )


BackendRegistry.register("numba_cuda", NumbaCudaSolver)
