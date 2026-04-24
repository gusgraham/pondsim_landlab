import numpy as np
import pandas as pd
from pathlib import Path
from pondsim.engine import run_simulation, SimulationConfig, SimulationResult
from pondsim.raster import read_dem
from pondsim.sources import load_sources
from pondsim.hydrographs import HydrographSet, make_synthetic_hydrograph
from landlab import RasterModelGrid
from landlab.components.overland_flow import OverlandFlow
import matplotlib.pyplot as plt

def run_landlab_native(dem, sources, hydrographs, config):
    nrows, ncols = dem.shape
    grid = RasterModelGrid((nrows, ncols), xy_spacing=(dem.dx, dem.dy), xy_of_lower_left=dem.xy_of_lower_left)
    
    elev = dem.elevation_landlab_order()
    grid.add_field("topographic__elevation", elev, at="node")
    grid.add_field("surface_water__depth", np.zeros(nrows * ncols), at="node")
    grid.add_field("mannings_n", np.full(nrows * ncols, config.manning_n), at="node")
    
    of = OverlandFlow(grid, steep_slopes=config.steep_slopes)
    
    elapsed_time = 0.0
    duration = config.simulation_duration_s
    max_depth = np.zeros(nrows * ncols)
    
    while elapsed_time < duration:
        dt = of.calc_time_step()
        if elapsed_time + dt > duration:
            dt = duration - elapsed_time
        
        # Inflow
        for nid, gn in zip(sources.node_ids, sources.grid_nodes):
            avg_flow = hydrographs.flow_average(nid, elapsed_time, elapsed_time + dt)
            grid.at_node["surface_water__depth"][gn] += avg_flow * dt / (grid.dx * grid.dy)
            
        of.run_one_step(dt)
        
        max_depth = np.maximum(max_depth, grid.at_node["surface_water__depth"])
        elapsed_time += dt
        
    return max_depth.reshape((nrows, ncols))[::-1, :]

def main():
    # Use real test data
    dem_path = "_test_data/Extract_5m_Float32.asc"
    sources_path = "_test_data/overalls.gpkg"
    
    dem = read_dem(dem_path)
    print(f"DEM Extent: {dem.xy_extent()}")
    
    # Grid for source mapping
    grid_temp = RasterModelGrid(dem.shape, xy_spacing=(dem.dx, dem.dy), xy_of_lower_left=dem.xy_of_lower_left)
    sources = load_sources(sources_path, grid_temp)
    print(f"Loaded {len(sources)} sources.")
    
    duration = 3600.0 # 1 hour
    vol = 100.0 # 100 m3 per source
    hydrographs = make_synthetic_hydrograph(sources.node_ids, duration, vol, dt_s=60.0)
    
    config = SimulationConfig(
        output_dir="comparison_results",
        simulation_duration_s=duration,
        manning_n=0.03,
        backend="numpy",
        fill_sinks=False
    )
    
    print("Running Pondsim (NumPy)...")
    res_pondsim = run_simulation(dem, sources, hydrographs, config)
    
    print("Running Landlab Native...")
    res_landlab = run_landlab_native(dem, sources, hydrographs, config)
    
    diff = res_pondsim.max_depth - res_landlab
    print(f"Comparison Summary:")
    print(f"  Pondsim Peak Depth: {res_pondsim.max_depth.max():.4f} m")
    print(f"  Landlab Peak Depth: {res_landlab.max():.4f} m")
    print(f"  Max Diff:           {np.abs(diff).max():.4f} m")
    print(f"  Mean Diff:          {np.abs(diff).mean():.4f} m")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(res_pondsim.max_depth, cmap="Blues")
    plt.title("Pondsim (NumPy)")
    plt.colorbar(label="Depth (m)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(res_landlab, cmap="Blues")
    plt.title("Landlab Native")
    plt.colorbar(label="Depth (m)")
    
    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    print("Saved comparison_plot.png")

if __name__ == "__main__":
    main()
