from __future__ import annotations
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from .engine import SimulationConfig
from .backends.base import BackendRegistry, extract_grid_arrays
from landlab import RasterModelGrid

logger = logging.getLogger(__name__)

def run_benchmarks(nrows: int = 200, ncols: int = 200, n_steps: int = 100):
    """
    Run performance and validation benchmarks across all available solver backends.
    """
    print(f"\n{'='*20} Swesim Solver Benchmarks {'='*20}")
    print(f"Grid size: {nrows} x {ncols} ({nrows*ncols:,} nodes)")
    print(f"Steps:     {n_steps}")
    print(f"{'='*67}\n")

    # 1. Setup synthetic problem
    grid = RasterModelGrid((nrows, ncols), xy_spacing=1.0)
    # Random topography with a slight slope
    y_coords = grid.node_y
    elev = (y_coords * 0.01) + np.random.rand(nrows * ncols).astype(np.float32) * 0.1
    grid.add_field("topographic__elevation", elev, at="node")
    grid.add_field("surface_water__depth", np.zeros(nrows * ncols), at="node")
    
    # Boundary status: all edges closed except bottom
    grid.status_at_node[grid.node_y == 0] = grid.BC_NODE_IS_FIXED_VALUE
    
    config = SimulationConfig(
        output_dir=".", 
        simulation_duration_s=100.0, 
        manning_n=0.03
    )
    
    # Ensure backends are loaded
    from . import backends # noqa: F401
    
    available_infos = BackendRegistry.get_backend_info()
    grid_data = extract_grid_arrays(grid)
    
    results = []
    baseline_depth = None
    numpy_time = 1.0

    for info in available_infos:
        if not info.available:
            print(f"[-] {info.name:<22}: Unavailable ({info.status})")
            continue
            
        BackendClass = BackendRegistry._backends[info.id]
        
        # Reset problem state
        grid.at_node["surface_water__depth"][:] = 0.0
        # Initialize with a "dam break" in the middle
        grid.at_node["surface_water__depth"][grid.node_y > (nrows//2)] = 1.0
        
        try:
            solver = BackendClass(grid_data, config)
        except NotImplementedError as e:
            print(f"[-] {info.name:<22}: {e}")
            continue
        except Exception as e:
            print(f"[!] {info.name:<22}: Failed to init ({e})")
            continue

        # Warmup
        solver.run_one_step(0.01)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(n_steps):
            solver.run_one_step(0.1)
        end = time.perf_counter()
        
        duration = end - start
        if info.id == "numpy":
            numpy_time = duration
            baseline_depth = grid.at_node["surface_water__depth"].copy()
            error = 0.0
        else:
            if baseline_depth is not None:
                error = np.max(np.abs(grid.at_node["surface_water__depth"] - baseline_depth))
            else:
                error = float('nan')
        
        results.append({
            "id": info.id,
            "name": info.name,
            "time": duration,
            "error": error
        })

    # Display results
    print(f"\n{'Backend':<24} | {'Steps/sec':<10} | {'Total Time':<10} | {'Speedup':<8} | {'Max Error'}")
    print("-" * 75)
    
    for res in results:
        speedup = numpy_time / res["time"]
        steps_sec = n_steps / res["time"]
        err_str = f"{res['error']:.2e}" if not np.isnan(res['error']) else "N/A"
        print(f"{res['name']:<24} | {steps_sec:<10.1f} | {res['time']:<10.3f}s | {speedup:<8.2f}x | {err_str}")

    print(f"\n{'='*67}")

if __name__ == "__main__":
    run_benchmarks()
