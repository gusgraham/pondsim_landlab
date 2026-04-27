import sys
from pathlib import Path
import logging
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swesim import (
    read_dem,
    run_simulation,
    SimulationConfig,
    HydrographSet,
    PointSources,
)
from swesim.hydrographs import make_synthetic_hydrograph
from swesim.sources import sources_from_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

def run_test(backend_id):
    logging.info(f"--- Testing Backend: {backend_id} ---")
    dem_path = "_test_data/Extract_5m_Float32.asc"
    dem = read_dem(dem_path)
    nrows, ncols = dem.shape
    
    from landlab import RasterModelGrid
    _snap_grid = RasterModelGrid((nrows, ncols), xy_spacing=(dem.dx, dem.dy), xy_of_lower_left=dem.xy_of_lower_left)
    
    x0, x1, y0, y1 = dem.xy_extent()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sources = sources_from_xy(["SYN_001"], [cx], [cy], _snap_grid)
    
    duration_s = 60 # 1 min
    hydrographs = make_synthetic_hydrograph(
        node_ids=["SYN_001"],
        volumes_m3=1000.0,
        duration_s=duration_s,
    )
    
    config = SimulationConfig(
        output_dir=f"verify_{backend_id}",
        simulation_duration_s=duration_s,
        backend=backend_id,
        fixed_timestep_s=0.1,
        fill_sinks=True,
    )
    
    import time
    start = time.time()
    result = run_simulation(dem, sources, hydrographs, config)
    duration = time.time() - start
    print(f"DEBUG: Simulation Time ({backend_id}): {duration:.2f}s")
    logging.info(f"Simulation Time ({backend_id}): {duration:.2f}s")
    
    peak_depth = float(result.max_depth.max())
    logging.info(f"Backend {backend_id} Peak Max Depth: {peak_depth:.6f} m")
    return peak_depth

if __name__ == "__main__":
    cpu_depth = run_test("numba_cpu")
    cuda_depth = run_test("numba_cuda")
    
    logging.info("--- Comparison ---")
    logging.info(f"CPU Peak Depth: {cpu_depth:.6f}")
    logging.info(f"CUDA Peak Depth: {cuda_depth:.6f}")
    
    if cuda_depth > 0:
        logging.info("SUCCESS: CUDA depth is non-zero.")
    else:
        logging.error("FAILURE: CUDA depth is still zero!")
        
    if abs(cpu_depth - cuda_depth) < 0.1: # Allow some tolerance
        logging.info("SUCCESS: CPU and CUDA results are comparable.")
    else:
        logging.warning("WARNING: Significant difference between CPU and CUDA results.")
