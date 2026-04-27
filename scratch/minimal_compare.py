import logging
import numpy as np
from swesim.engine import SimulationConfig, extract_grid_arrays
from swesim.backends.numba_cpu import NumbaCpuSolver
from swesim.backends.landlab_backend import LandlabSolver
from landlab import RasterModelGrid

logging.basicConfig(level=logging.INFO, format="%(message)s")

def minimal():
    dx = 5.0
    config = SimulationConfig(output_dir="tmp", simulation_duration_s=1.0, manning_n=0.0)
    dt = 0.01
    
    # Nodes layout:
    # 6 7 8
    # 3 4 5
    # 0 1 2
    # Link 7 connects Node 4 to Node 5
    
    # 1. Landlab
    ll_grid = RasterModelGrid((3, 3), xy_spacing=(dx, dx))
    elev = np.full(9, 10.0)
    elev[5] = 9.0
    ll_grid.add_field("topographic__elevation", elev, at="node")
    ll_grid.add_field("surface_water__depth", np.full(9, 0.1), at="node")
    ll_grid.add_field("mannings_n", np.zeros(ll_grid.number_of_links), at="link")
    ll_grid.status_at_node[:] = 0 # All core to be safe
    
    ll = LandlabSolver(None, config, grid=ll_grid)
    ll.of._theta = 1.0 # Disable smoothing to match Numba
    ll.of.run_one_step(dt)
    
    # 2. Numba
    nb_grid = RasterModelGrid((3, 3), xy_spacing=(dx, dx))
    nb_grid.add_field("topographic__elevation", elev, at="node")
    nb_grid.add_field("surface_water__depth", np.full(9, 0.1), at="node")
    nb_grid.status_at_node[:] = 0
    nb_grid_data = extract_grid_arrays(nb_grid)
    nb = NumbaCpuSolver(nb_grid_data, config, grid=nb_grid)
    nb.run_one_step(dt)
    
    for i in range(ll_grid.number_of_links):
        logging.info(f"Link {i}: {ll_grid.nodes_at_link[i]}")
    
    link_idx = 6 # node 4 to 5
    q_ll = ll_grid.at_link['surface_water__discharge'][link_idx]
    q_nb = nb.q[link_idx]
    
    logging.info(f"Link {link_idx} Nodes: {ll_grid.nodes_at_link[link_idx]}")
    logging.info(f"Landlab Q: {q_ll:.8f}")
    logging.info(f"Numba Q:   {q_nb:.8f}")
    logging.info(f"Ratio Q:   {q_nb/q_ll:.4f}")

if __name__ == "__main__":
    minimal()
