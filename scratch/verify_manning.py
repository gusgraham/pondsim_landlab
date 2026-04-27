import numpy as np
from swesim.engine import SimulationConfig
from swesim.backends.base import BackendRegistry, extract_grid_arrays
from landlab import RasterModelGrid

def test_manning_array():
    print("Testing Manning's n array support...")
    grid = RasterModelGrid((10, 10), xy_spacing=1.0)
    grid.add_field("topographic__elevation", np.zeros(100), at="node")
    grid.add_field("surface_water__depth", np.zeros(100), at="node")
    
    # Spatially varying n
    n_arr = np.linspace(0.01, 0.05, 100)
    config = SimulationConfig(output_dir=".", simulation_duration_s=10.0, manning_n=n_arr)
    
    from swesim import backends # noqa
    
    # Test Numba CPU
    BackendClass = BackendRegistry._backends["numba_cpu"]
    grid_data = extract_grid_arrays(grid)
    solver = BackendClass(grid_data, config)
    
    print(f"Solver Manning array shape: {solver.n_nodes_arr.shape}")
    print(f"Min n: {solver.n_nodes_arr.min()}, Max n: {solver.n_nodes_arr.max()}")
    
    assert solver.n_nodes_arr.shape == (100,)
    assert np.allclose(solver.n_nodes_arr, n_arr)
    print("SUCCESS: Manning array correctly passed to Numba CPU backend.")

if __name__ == "__main__":
    test_manning_array()
