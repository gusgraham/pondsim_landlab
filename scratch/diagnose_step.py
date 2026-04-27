import logging
import numpy as np
from swesim.raster import DEM
from swesim.engine import run_simulation, SimulationConfig, extract_grid_arrays
from swesim.backends.numba_cpu import NumbaCpuSolver
from swesim.backends.landlab_backend import LandlabSolver
from landlab import RasterModelGrid
from rasterio.transform import from_origin

logging.basicConfig(level=logging.INFO, format="%(message)s")

def diagnose():
    dx = 5.0
    nrows, ncols = 10, 10
    dem_arr = np.zeros((nrows, ncols))
    for c in range(ncols):
        dem_arr[:, c] = 10.0 - (c * 0.1) # steep slope 0.1
    
    config = SimulationConfig(output_dir="tmp", simulation_duration_s=1.0, manning_n=0.0)
    dt = 0.01
    
    # 1. Landlab Step
    grid_ll = RasterModelGrid((nrows, ncols), xy_spacing=(dx, dx))
    grid_ll.add_field("topographic__elevation", np.flipud(dem_arr).ravel(), at="node")
    grid_ll.add_field("surface_water__depth", np.full(grid_ll.number_of_nodes, 0.1), at="node")
    grid_ll.add_field("mannings_n", np.full(grid_ll.number_of_links, 0.03), at="link")
    
    grid_data = extract_grid_arrays(grid_ll)
    ll = LandlabSolver(grid_data, config, grid=grid_ll)
    ll.of.run_one_step(dt)
    q_ll = grid_ll.at_link["surface_water__discharge"]
    h_ll = grid_ll.at_node["surface_water__depth"]
    
    # 2. Numba Step
    grid_nb = RasterModelGrid((nrows, ncols), xy_spacing=(dx, dx))
    grid_nb.add_field("topographic__elevation", np.flipud(dem_arr).ravel(), at="node")
    grid_nb.add_field("surface_water__depth", np.full(grid_nb.number_of_nodes, 0.1), at="node")
    # NumbaCpuSolver expects n_link in grid_data
    grid_nb.add_field("mannings_n", np.full(grid_nb.number_of_links, 0.03), at="link")
    grid_nb.status_at_node[:] = 0
    
    nb_grid_data = extract_grid_arrays(grid_nb)
    nb = NumbaCpuSolver(nb_grid_data, config, grid=grid_nb)
    nb.run_one_step(dt)
    q_nb = nb.q
    h_nb = nb.depth
    
    link_idx = 45 # middle of horizontal links
    node_idx = 55 # middle node
    logging.info(f"--- Step Results (dt={dt}, dx={dx}) ---")
    logging.info(f"Link {link_idx} Length: {nb_grid_data['dist'][link_idx]}")
    logging.info(f"Landlab Q[{link_idx}]: {q_ll[link_idx]:.8f} m^3/s")
    logging.info(f"Numba Q[{link_idx}]:   {q_nb[link_idx]:.8f} m^2/s")
    logging.info(f"Ratio Q:      {q_nb[link_idx]/q_ll[link_idx]:.4f}")
    
    # Compare Depth change at node 1
    dh_ll = h_ll[node_idx] - 0.1
    dh_nb = h_nb[node_idx] - 0.1
    logging.info(f"Landlab dH[{node_idx}]: {dh_ll:.8f} m")
    logging.info(f"Numba dH[{node_idx}]:   {dh_nb:.8f} m")
    logging.info(f"Ratio dH:      {dh_nb/dh_ll:.4f}")

if __name__ == "__main__":
    diagnose()
