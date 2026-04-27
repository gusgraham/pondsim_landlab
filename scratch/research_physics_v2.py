import logging
import numpy as np
from pathlib import Path
from swesim.raster import DEM, write_raster
from swesim.engine import run_simulation, SimulationConfig
from swesim.sources import PointSources
from swesim.hydrographs import HydrographSet
from rasterio.transform import from_origin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

def run_compare(dx):
    logging.info(f"=== Testing with dx={dx} ===")
    nrows, ncols = 40, 40
    dem_arr = np.zeros((nrows, ncols))
    for c in range(ncols):
        dem_arr[:, c] = 10.0 - (c * 0.01) # slope 0.01 to East
    
    transform = from_origin(0, nrows*dx, dx, dx)
    dem = DEM(dem_arr, transform, None, None)
    
    # middle-ish node
    gx = (ncols // 2 + 0.5) * dx
    gy = (nrows // 2 - 0.5) * dx
    gn = (nrows // 2) * ncols + (ncols // 2)
    sources = PointSources(
        node_ids=["S1"],
        xy=np.array([[gx, gy]]),
        grid_nodes=np.array([gn], dtype=np.int32)
    )
    
    duration_s = 10.0
    hydrographs = HydrographSet(
        times_s=np.array([0.0, 1000.0]),
        flows={"S1": np.array([2.0, 2.0])}
    )
    
    config_ll = SimulationConfig(output_dir=f"cmp_dx{dx}_ll", simulation_duration_s=duration_s, backend="landlab", fixed_timestep_s=0.1, manning_n=0.03)
    res_ll = run_simulation(dem, sources, hydrographs, config_ll)
    
    config_nb = SimulationConfig(output_dir=f"cmp_dx{dx}_nb", simulation_duration_s=duration_s, backend="numba_cuda", fixed_timestep_s=0.1, manning_n=0.03)
    res_nb = run_simulation(dem, sources, hydrographs, config_nb)
    
    ll_peak = float(res_ll.max_depth.max())
    nb_peak = float(res_nb.max_depth.max())
    
    ll_q = float(res_ll.max_q.max())
    nb_q = float(res_nb.max_q.max())
    
    ll_vol = float(np.sum(res_ll.max_depth) * dx * dx)
    nb_vol = float(np.sum(res_nb.max_depth) * dx * dx)
    
    logging.info(f"Landlab Peak Depth: {ll_peak:.6f}, Numba Peak Depth: {nb_peak:.6f}, Ratio: {nb_peak/ll_peak:.4f}")
    logging.info(f"Landlab Peak Q: {ll_q:.6f}, Numba Peak Q: {nb_q:.6f}, Ratio: {nb_q/ll_q:.4f}")
    logging.info(f"Landlab Volume: {ll_vol:.2f} m^3, Numba Volume: {nb_vol:.2f} m^3")

if __name__ == "__main__":
    run_compare(5.0)
