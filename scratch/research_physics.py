import sys
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pondsim import (
    read_dem,
    run_simulation,
    SimulationConfig,
    HydrographSet,
    PointSources,
)
from pondsim.hydrographs import make_synthetic_hydrograph
from pondsim.sources import sources_from_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

def run_test(backend_id, dem_path):
    logging.info(f"--- Running Backend: {backend_id} ---")
    # 1. Create a simple test DEM (flat with one slope)
    # dx = dy = 1.0 to isolate scaling issues
    nrows, ncols = 100, 100
    dem_arr = np.zeros((nrows, ncols))
    # Slight slope to the east
    for c in range(ncols):
        dem_arr[:, c] = 10.0 - (c * 0.01)
    
    transform = from_origin(0, 100, 1.0, 1.0)
    dem = DEM(dem_arr, transform, None, None)
    dem_path = "physics_test_dem.tif"
    write_raster(dem_path, dem.elevation, dem.transform, None)
    
    # 2. Add a source
    sources = PointSources()
    # middle node (50, 50) -> index 50*100 + 50 = 5050
    # But wait, we need world coords for add()
    sources.add(0, 50.5, 49.5, 5050)
    
    hydrographs = HydrographSet()
    duration_s = 100.0
    hydrographs.add_constant(0, 1.0) # 1.0 m3/s constant
    
    config = SimulationConfig(
        output_dir=f"cmp_{backend_id}",
        simulation_duration_s=duration_s,
        backend=backend_id,
        fill_sinks=True,
        steep_slopes=False # Use False to match the Numba implementation (no advection)
    )
    
    result = run_simulation(dem, sources, hydrographs, config)
    return result

if __name__ == "__main__":
    dem_path = "_test_data/Extract_5m_Float32.asc"
    
    # 1. Run Landlab (Baseline)
    ll_result = run_test("landlab", dem_path)
    
    # 2. Run Numba CPU
    nb_result = run_test("numba_cpu", dem_path)
    
    # 3. Compare
    ll_peak = float(ll_result.max_depth.max())
    nb_peak = float(nb_result.max_depth.max())
    
    ll_q = float(ll_result.max_q.max())
    nb_q = float(nb_result.max_q.max())
    
    logging.info("--- Comparison Results ---")
    logging.info(f"Landlab Peak Max Depth: {ll_peak:.6f} m")
    logging.info(f"Numba CPU Peak Max Depth: {nb_peak:.6f} m")
    logging.info(f"Ratio (Depth): {nb_peak/ll_peak:.4f}")
    
    logging.info(f"Landlab Peak Max Q: {ll_q:.6f} m^2/s")
    logging.info(f"Numba CPU Peak Max Q: {nb_q:.6f} m^2/s")
    logging.info(f"Ratio (Q): {nb_q/ll_q:.4f}")
    
    # Plot profile
    mid_row = ll_result.max_depth.shape[0] // 2
    plt.figure(figsize=(10, 5))
    plt.plot(ll_result.max_depth[mid_row, :], 'r-', label='Landlab')
    plt.plot(nb_result.max_depth[mid_row, :], 'b--', label='Numba CPU')
    plt.title(f"Cross-section comparison (Row {mid_row})")
    plt.legend()
    plt.savefig("physics_comparison.png")
    logging.info("Saved physics_comparison.png")
