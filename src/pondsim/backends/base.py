from __future__ import annotations
import logging
import numpy as np
from typing import Protocol, Optional, Any, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class SolverBackend(Protocol):
    def run_one_step(self, dt: float) -> None: ...
    def calc_time_step(self) -> float: ...
    @property
    def depth(self) -> np.ndarray: ...
    def add_to_depth(self, node_idx: int, value: float) -> None: ...
    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None: ...
    def sync_to_grid(self) -> None: ...

@dataclass
class BackendInfo:
    id: str
    name: str
    tier: int
    available: bool
    status: str = "Ok"

class BackendRegistry:
    _backends: dict[str, Type] = {}

    @classmethod
    def register(cls, backend_id: str, backend_class: Type):
        cls._backends[backend_id] = backend_class
        logger.debug(f"Registered backend: {backend_id}")

    @classmethod
    def get_backend_info(cls) -> list[BackendInfo]:
        infos = []
        for bid, bcls in cls._backends.items():
            tier = getattr(bcls, "TIER", 0)
            name = getattr(bcls, "NAME", bid)
            try:
                available, status = bcls.check_availability()
                infos.append(BackendInfo(bid, name, tier, available, status))
            except Exception as e:
                infos.append(BackendInfo(bid, name, tier, False, str(e)))
        return sorted(infos, key=lambda x: x.tier, reverse=True)

    @classmethod
    def get_best_backend(cls, grid, config) -> Type:
        # Priority: 1. Manual override, 2. Best tier
        target = getattr(config, "backend", "auto")
        
        infos = cls.get_backend_info()
        
        if target != "auto":
            for info in infos:
                if info.id == target:
                    if info.available:
                        logger.info(f"Using requested backend: {info.name}")
                        return cls._backends[info.id]
                    else:
                        logger.warning(f"Requested backend {target} unavailable ({info.status}). Falling back.")

        for info in infos:
            if info.available:
                # Extra check for CUDA VRAM if it's Tier 3
                if info.tier == 3:
                    if not cls._backends[info.id].check_vram(grid):
                        logger.warning(f"CUDA Tier 3 detected but VRAM insufficient. Falling back.")
                        continue
                
                logger.info(f"Auto-selected backend: {info.name} (Tier {info.tier})")
                return cls._backends[info.id]

        raise RuntimeError("No available backends found (even NumPy baseline!)")

def check_cuda_vram(nrows: int, ncols: int) -> tuple[bool, str]:
    """Estimate VRAM requirements and check against hardware."""
    try:
        from numba import cuda
        if not cuda.is_available():
            return False, "CUDA not available to Numba"
        
        # Estimate footprint
        # Nodes: elev, depth, max_depth, max_level, h_src, etc ~ 6-8 arrays
        # Links: q, q_old, slope, friction ~ 4 arrays
        n_nodes = nrows * ncols
        n_links = (nrows - 1) * ncols + (ncols - 1) * nrows
        
        # Assume float32 (4 bytes)
        node_bytes = n_nodes * 8 * 4
        link_bytes = n_links * 6 * 4
        total_required = node_bytes + link_bytes
        
        # Buffer for overhead
        safe_required = total_required * 1.5
        
        device = cuda.get_current_device()
        free, total = cuda.current_context().get_memory_info()
        
        if safe_required > free:
            return False, f"Insufficient VRAM: Need {safe_required/1e6:.1f}MB, Free {free/1e6:.1f}MB"
        
        return True, "Ok"
    except Exception as e:
        return False, str(e)

def extract_grid_arrays(grid):
    """
    Extract raw NumPy arrays from a Landlab RasterModelGrid.
    Returns a dict of arrays and topology info.
    """
    return {
        "nrows": grid.number_of_node_rows,
        "ncols": grid.number_of_node_columns,
        "dx": grid.dx,
        "dy": grid.dy,
        "elev": grid.at_node["topographic__elevation"],
        "depth": grid.at_node["surface_water__depth"],
        "links_at_node": grid.links_at_node,
        "nodes_at_link": grid.nodes_at_link,
        "link_dirs": grid.link_dirs_at_node,
        "active_links": grid.active_links,
        # Boundary info
        "node_status": grid.status_at_node,
    }
