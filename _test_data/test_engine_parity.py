"""
Comprehensive engine comparison script.
Runs Landlab, Numba CPU, and Numba CUDA backends on the same test case
 and compares their results.
"""
import sys
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landlab import RasterModelGrid

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swesim.raster import read_dem
from swesim.sources import load_sources
from swesim.hydrographs import make_synthetic_hydrograph
from swesim.engine import SimulationConfig
from swesim.backends.base import extract_grid_arrays, BackendRegistry
from swesim.backends import landlab_backend, numba_cpu, numba_cuda  # noqa: ensure registered

ROOT = Path(__file__).parent
# DEM_PATH = ROOT / "test_deep_1m.tif"
DEM_PATH = ROOT / "test_deep_2m.tif"
SRC_PATH = ROOT / "test_deep_point.geojson"
OUTPUT_DIR = ROOT / "comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

STOP_AT_S = 300  # Initial comparison duration
FULL_DURATION = 3600.0 # Full duration for some metrics

def build_grid(dem):
    import richdem as rd
    nrows, ncols = dem.shape
    elev_ll = dem.elevation_landlab_order()
    nan_mask = np.isnan(elev_ll)

    g = RasterModelGrid(
        (nrows, ncols),
        xy_spacing=(dem.dx, dem.dy),
        xy_of_lower_left=dem.xy_of_lower_left,
    )

    if nan_mask.any():
        valid_min = float(np.nanmin(elev_ll))
        valid_mean = float(np.nanmean(elev_ll))
        boundary_nodes = np.concatenate([
            np.arange(ncols),
            np.arange((nrows-1)*ncols, nrows*ncols),
            np.arange(ncols, (nrows-1)*ncols, ncols),
            np.arange(2*ncols-1, (nrows-1)*ncols, ncols),
        ])
        edge_mask = np.zeros(len(elev_ll), dtype=bool)
        edge_mask[boundary_nodes] = True
        elev_ll[nan_mask & edge_mask] = valid_min
        elev_ll[nan_mask & ~edge_mask] = valid_mean
        g.status_at_node[nan_mask & ~edge_mask] = g.BC_NODE_IS_CLOSED
        g.status_at_node[nan_mask & edge_mask] = g.BC_NODE_IS_FIXED_VALUE
    else:
        valid_mean = None

    g.set_status_at_node_on_edges(right=1, top=1, left=1, bottom=1)

    # fill sinks
    elev_2d = dem.elevation.copy()
    rd_nodata = -9999.0
    elev_2d[np.isnan(elev_2d)] = rd_nodata
    rd_arr = rd.rdarray(elev_2d, no_data=rd_nodata)
    rd.fill_depressions(rd_arr, in_place=True)
    elev_filled = np.flipud(np.array(rd_arr)).ravel().astype(np.float64)
    if nan_mask.any():
        elev_filled[nan_mask] = valid_mean

    g.add_field("topographic__elevation", elev_filled.copy(), at="node")
    g.add_field("surface_water__depth", np.zeros(nrows*ncols), at="node")
    return g, elev_filled

def run_engine(backend_id, grid, dem, sources, hydrographs, duration):
    print(f"Running engine: {backend_id} for {duration}s...")
    
    config = SimulationConfig(
        output_dir=OUTPUT_DIR / backend_id,
        simulation_duration_s=duration,
        manning_n=0.03,
        fill_sinks=True,
        backend=backend_id,
    )
    
    grid_data = extract_grid_arrays(grid)
    # reset depth
    grid.at_node["surface_water__depth"][:] = 0.0
    grid_data["depth"][:] = 0.0

    BackendClass = BackendRegistry._backends[backend_id]
    solver = BackendClass(grid_data, config, grid=grid)
    
    area = dem.dx * dem.dy
    nid = sources.node_ids[0]
    src_gn = sources.grid_nodes[0]
    source_nodes_arr = np.array([src_gn], dtype=np.int32)
    source_deltas_arr = np.zeros(1, dtype=np.float64)
    
    rows = []
    elapsed = 0.0
    step = 0
    start_time = time.time()
    total_injected = 0.0
    
    while elapsed < duration:
        raw_dt = solver.calc_time_step()
        if not math.isfinite(raw_dt) or raw_dt <= 0:
            raw_dt = 1.0
        dt = min(raw_dt, config.max_adaptive_dt_s)
        
        remaining = duration - elapsed
        dt = min(dt, remaining)
        if dt <= 0: break
            
        # Injection CFL cap
        q_check = hydrographs.flow_at(nid, elapsed + 0.5 * dt)
        if q_check > 0.0:
            h_inj = q_check * dt / area
            dt_inj = 0.7 * min(dem.dx, dem.dy) / math.sqrt(9.80665 * h_inj)
            if dt_inj < dt:
                dt = dt_inj
        dt = min(dt, remaining)
        if dt <= 0: break

        # inject
        flow = hydrographs.flow_at(nid, elapsed + 0.5 * dt)
        total_injected += flow * dt
        source_deltas_arr[0] = flow * dt / area
        solver.add_to_depths(source_nodes_arr, source_deltas_arr)

        # step
        solver.run_one_step(dt)
        
        # Record stats (sparse recording if long duration)
        if step % 10 == 0 or elapsed + dt >= duration:
            rows.append({
                "elapsed_s": elapsed + dt,
                "dt_s": dt,
                "src_depth": float(solver.depth[src_gn]),
                "total_vol": solver.get_total_volume() if hasattr(solver, 'get_total_volume') else np.sum(solver.depth) * area
            })

        elapsed += dt
        step += 1
        
        if step % 1000 == 0:
            print(f"  [{backend_id}] t={elapsed:.1f}s step={step}")

    total_time = time.time() - start_time
    print(f"Finished {backend_id} in {total_time:.2f}s, Injected: {total_injected:.4f} m3")
    
    return pd.DataFrame(rows), solver.depth.copy(), total_time, total_injected

def main():
    dem = read_dem(str(DEM_PATH))
    grid, elev_filled = build_grid(dem)
    sources = load_sources(str(SRC_PATH), grid)
    hydrographs = make_synthetic_hydrograph(sources.node_ids, duration_s=3600.0, volumes_m3=1000.0)
    
    engines = ["landlab", "numba_cpu", "numba_cuda"]
    results = {}
    
    for eng in engines:
        try:
            df, final_depth, wall_time, injected = run_engine(eng, grid, dem, sources, hydrographs, STOP_AT_S)
            results[eng] = {
                "df": df,
                "depth": final_depth,
                "wall_time": wall_time,
                "injected": injected
            }
        except Exception as e:
            import traceback
            print(f"Failed to run engine {eng}: {e}")
            traceback.print_exc()

    if len(results) < 2:
        print("Not enough engines succeeded to compare.")
        return

    # Comparison
    print("\n=== Comparison Summary ===")
    
    # 1. Wall time
    print("\nExecution Time:")
    for eng, data in results.items():
        print(f"  {eng:10}: {data['wall_time']:>8.2f}s")

    # 2. Final Depth Differences (Spatial)
    ref_eng = "landlab"
    if ref_eng not in results:
        ref_eng = list(results.keys())[0]
        
    print(f"\nDifferences relative to {ref_eng}:")
    ref_depth = results[ref_eng]["depth"]
    
    for eng in results:
        if eng == ref_eng: continue
        diff = results[eng]["depth"] - ref_depth
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))
        print(f"  {eng:10}: MAE={mae:.6e}, RMSE={rmse:.6e}, MaxDiff={max_diff:.6e}")

    # 2b. Direct CPU vs CUDA Parity
    if "numba_cpu" in results and "numba_cuda" in results:
        diff_p = results["numba_cuda"]["depth"] - results["numba_cpu"]["depth"]
        mae_p = np.mean(np.abs(diff_p))
        rmse_p = np.sqrt(np.mean(diff_p**2))
        max_p = np.max(np.abs(diff_p))
        print(f"\nNumba CPU vs CUDA Parity:")
        print(f"  MAE={mae_p:.6e}, RMSE={rmse_p:.6e}, MaxDiff={max_p:.6e}")

    # 3. Time Series comparison (at source)
    plt.figure(figsize=(10, 6))
    for eng, data in results.items():
        plt.plot(data["df"]["elapsed_s"], data["df"]["src_depth"], label=eng)
    plt.title(f"Source Node Depth Comparison (T={STOP_AT_S}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "src_depth_comparison.png")
    print(f"\nPlot saved to {OUTPUT_DIR}/src_depth_comparison.png")

    # 4. Volume Conservation
    print("\nVolume Conservation (Final):")
    for eng, data in results.items():
        vol = data["df"]["total_vol"].iloc[-1]
        injected = data["injected"]
        diff = vol - injected
        print(f"  {eng:10}: Final={vol:>12.4f} m3, Injected={injected:>12.4f} m3, Delta={diff:>10.4f} m3")

if __name__ == "__main__":
    main()
