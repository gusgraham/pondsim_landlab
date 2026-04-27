"""
Step-level diagnostic: runs both backends on the 1m test case and records
depth at the source node + the 4 adjacent nodes every timestep, plus the
dt used and q on the outgoing links.  Stops after 300s of sim time so it
runs quickly.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from landlab import RasterModelGrid

from pondsim.raster import read_dem
from pondsim.sources import load_sources
from pondsim.hydrographs import make_synthetic_hydrograph
from pondsim.engine import SimulationConfig
from pondsim.backends.base import extract_grid_arrays
from pondsim.backends import landlab_backend, numba_cpu   # noqa: ensure registered

ROOT = Path(__file__).parent

DEM_PATH    = ROOT / "test_deep_1m.tif"
SRC_PATH    = ROOT / "test_deep_point.geojson"
STOP_AT_S   = 300   # only simulate first 5 minutes

# ── load DEM & sources ──────────────────────────────────────────────────────
dem  = read_dem(str(DEM_PATH))
nrows, ncols = dem.shape

grid_tmp = RasterModelGrid(
    (nrows, ncols),
    xy_spacing=(dem.dx, dem.dy),
    xy_of_lower_left=dem.xy_of_lower_left,
)
sources = load_sources(str(SRC_PATH), grid_tmp)
print(f"Source nodes: {sources.node_ids}  grid nodes: {sources.grid_nodes}")

hydrographs = make_synthetic_hydrograph(
    sources.node_ids,
    duration_s=3600.0,
    volumes_m3=1000.0,
)

config = SimulationConfig(
    output_dir=ROOT / "_diag_out",
    simulation_duration_s=STOP_AT_S,
    manning_n=0.03,
    fill_sinks=True,
    backend="landlab",
)

# ── replicate the engine setup (fill sinks, BCs, grid fields) ──────────────
def build_grid(dem, config):
    import richdem as rd
    from landlab.components.sink_fill import SinkFiller

    nrows, ncols = dem.shape
    elev_ll = dem.elevation_landlab_order()
    nan_mask = np.isnan(elev_ll)

    g = RasterModelGrid(
        (nrows, ncols),
        xy_spacing=(dem.dx, dem.dy),
        xy_of_lower_left=dem.xy_of_lower_left,
    )

    if nan_mask.any():
        valid_min  = float(np.nanmin(elev_ll))
        valid_mean = float(np.nanmean(elev_ll))
        boundary_nodes = np.concatenate([
            np.arange(ncols),
            np.arange((nrows-1)*ncols, nrows*ncols),
            np.arange(ncols, (nrows-1)*ncols, ncols),
            np.arange(2*ncols-1, (nrows-1)*ncols, ncols),
        ])
        edge_mask = np.zeros(len(elev_ll), dtype=bool)
        edge_mask[boundary_nodes] = True
        elev_ll[nan_mask & edge_mask]  = valid_min
        elev_ll[nan_mask & ~edge_mask] = valid_mean
        g.status_at_node[nan_mask & ~edge_mask] = g.BC_NODE_IS_CLOSED
        g.status_at_node[nan_mask & edge_mask]  = g.BC_NODE_IS_FIXED_VALUE
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

    elev_ll = np.nan_to_num(elev_ll, nan=valid_mean if valid_mean else 0.0)
    g.add_field("topographic__elevation", elev_filled.copy(), at="node")
    g.add_field("surface_water__depth",   np.zeros(nrows*ncols),     at="node")
    return g, elev_filled

print("Building grid…")
grid, elev_filled = build_grid(dem, config)

# source grid node index
nid    = sources.node_ids[0]
src_gn = sources.grid_nodes[0]
print(f"Source '{nid}'  grid-node index: {src_gn}  "
      f"(row={src_gn//ncols}, col={src_gn%ncols})")

# 4 neighbours of source node
neighbours = grid.adjacent_nodes_at_node[src_gn]
neighbours = neighbours[neighbours != -1]
print(f"Neighbour nodes: {neighbours}")

valid_sources = [(nid, src_gn)]
area = dem.dx * dem.dy
source_nodes_arr  = np.array([src_gn], dtype=np.int32)
source_deltas_arr = np.zeros(1, dtype=np.float64)

# ── run one backend, recording per-step data ────────────────────────────────
def run_backend(backend_id, grid, elev_filled):
    from pondsim.backends.base import BackendRegistry
    grid_data = extract_grid_arrays(grid)

    # reset depth to zero
    grid.at_node["surface_water__depth"][:] = 0.0
    grid_data["depth"][:] = 0.0

    BackendClass = BackendRegistry._backends[backend_id]
    solver = BackendClass(grid_data, config, grid=grid)

    rows = []
    elapsed = 0.0
    step = 0

    while elapsed < STOP_AT_S:
        # timestep
        raw_dt = solver.calc_time_step()
        if not math.isfinite(raw_dt) or raw_dt <= 0:
            raw_dt = 1.0
        dt = min(raw_dt, config.max_adaptive_dt_s)

        # Clamp to sim end
        remaining = STOP_AT_S - elapsed
        dt = min(dt, remaining)
        if dt <= 0:
            break

        # Injection CFL cap — mirrors engine.py logic.
        # calc_time_step() uses existing depths; on a dry grid it returns a large dt
        # (based on h_min) but the injected water can be much deeper.  Without this
        # cap the explicit Numba solver oscillates wildly on the first wet step.
        q_check = hydrographs.flow_at(nid, elapsed + 0.5 * dt)
        if q_check > 0.0:
            h_inj = q_check * dt / area
            dt_inj = 0.7 * min(dem.dx, dem.dy) / math.sqrt(9.80665 * h_inj)
            if dt_inj < dt:
                dt = dt_inj
        dt = min(dt, remaining)
        if dt <= 0:
            break

        # inject
        flow = hydrographs.flow_at(nid, elapsed + 0.5 * dt)
        source_deltas_arr[0] = flow * dt / area
        solver.add_to_depths(source_nodes_arr, source_deltas_arr)

        depth = solver.depth
        src_depth_post_inject = float(depth[src_gn])
        nbr_depths = [float(depth[n]) for n in neighbours]

        # q on links from source (before step)
        q_src_links = []
        if hasattr(solver, 'q'):
            for link in grid.links_at_node[src_gn]:
                if link != -1:
                    q_src_links.append(float(solver.q[link]))

        # step
        solver.run_one_step(dt)

        depth_after = solver.depth
        src_depth_after = float(depth_after[src_gn])

        rows.append({
            "step":              step,
            "elapsed_s":         elapsed,
            "dt_s":              dt,
            "raw_dt_s":          raw_dt,
            "inflow_m3s":        flow,
            "delta_depth_m":     float(source_deltas_arr[0]),
            "src_depth_post_inj":src_depth_post_inject,
            "src_depth_post_step":src_depth_after,
            "nbr_mean_depth":    float(np.mean(nbr_depths)),
            "nbr_max_depth":     float(np.max(nbr_depths)) if nbr_depths else 0.0,
            "q_max_src_link":    max(abs(q) for q in q_src_links) if q_src_links else float("nan"),
        })

        elapsed += dt
        step    += 1

        if step % 500 == 0:
            print(f"  [{backend_id}] t={elapsed:.1f}s  src_depth={src_depth_after:.4f}m  dt={dt:.4f}s")

    return pd.DataFrame(rows)


print("\n--- Landlab ---")
df_ll  = run_backend("landlab",   grid, elev_filled)
print("\n--- Numba CPU ---")
df_nb  = run_backend("numba_cpu", grid, elev_filled)

# ── summary comparison ───────────────────────────────────────────────────────
print("\n=== First 20 steps ===")
print("\nLandlab:")
print(df_ll[["step","elapsed_s","dt_s","delta_depth_m",
             "src_depth_post_inj","src_depth_post_step","nbr_mean_depth","q_max_src_link"]].head(20).to_string(index=False))

print("\nNumba CPU:")
print(df_nb[["step","elapsed_s","dt_s","delta_depth_m",
             "src_depth_post_inj","src_depth_post_step","nbr_mean_depth","q_max_src_link"]].head(20).to_string(index=False))

# peak source depth per engine
print(f"\nMax source depth — Landlab:   {df_ll['src_depth_post_step'].max():.4f} m  "
      f"at t={df_ll.loc[df_ll['src_depth_post_step'].idxmax(),'elapsed_s']:.1f}s")
print(f"Max source depth — Numba CPU: {df_nb['src_depth_post_step'].max():.4f} m  "
      f"at t={df_nb.loc[df_nb['src_depth_post_step'].idxmax(),'elapsed_s']:.1f}s")

# check for steps where numba source depth is growing despite having neighbours
diverge = df_nb[df_nb["src_depth_post_step"] > df_nb["src_depth_post_inj"] * 1.01]
print(f"\nNumba steps where depth GROWS through the solve: {len(diverge)} / {len(df_nb)}")

out = ROOT / "_diag_out"
out.mkdir(exist_ok=True)
df_ll.to_csv(out / "diag_landlab.csv",   index=False)
df_nb.to_csv(out / "diag_numba_cpu.csv", index=False)
print(f"\nFull traces -> {out}/diag_landlab.csv  /  diag_numba_cpu.csv")
