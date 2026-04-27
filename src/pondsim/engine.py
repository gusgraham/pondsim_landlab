"""
Simulation engine.

Design goals
------------
* No Qt / GUI imports — runs headless from a script.
* Emits progress via a callback so a QThread wrapper can forward to the UI.
* SolverBackend protocol keeps Landlab swappable for a numba kernel later.
* Fixes three bugs from the original code:
    1. Filled grid actually used for simulation.
    2. NetCDF snapshot uses elapsed-time threshold, not floating-point modulo.
    3. SinkFiller called before OverlandFlow is set up.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .hydrographs import HydrographSet
from .raster import DEM, transform_from_landlab, write_raster
from .sources import PointSources
from .backends.base import BackendRegistry, extract_grid_arrays

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solver protocol — lets us swap backends without changing the sim loop
# ---------------------------------------------------------------------------

class SolverBackend(Protocol):
    """Minimal interface any hydrodynamic backend must satisfy."""

    def run_one_step(self, dt: float) -> None: ...
    def calc_time_step(self) -> float: ...
    def sync_to_grid(self) -> None: ...
    def add_to_depth(self, node_idx: int, value: float) -> None: ...
    def add_to_depths(self, node_indices: np.ndarray, values: np.ndarray) -> None: ...
    @property
    def depth(self) -> np.ndarray: ...
    @property
    def NAME(self) -> str: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    output_dir: str | Path
    simulation_duration_s: float       # total wall-clock time to simulate
    manning_n: float | np.ndarray = 0.03    # scalar or 2D array (normalized to array by backend)
    backend: str = "auto"              # auto, numpy, numba_cpu, numba_cuda
    fixed_timestep_s: float | None = None   # None → adaptive
    max_adaptive_dt_s: float = 120.0   # cap on adaptive step
    snapshot_interval_s: float = 300.0 # NetCDF snapshot cadence
    export_netcdf: bool = False
    steep_slopes: bool = True
    fill_sinks: bool = True
    max_depth_threshold: float = 5e-5  # depths below this treated as dry


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    max_depth: np.ndarray          # 2D, top-down (rasterio order)
    max_level: np.ndarray          # 2D, top-down
    max_q: np.ndarray              # 2D, top-down unit discharge (m^2/s)
    node_hydrographs: pd.DataFrame
    dem: DEM
    output_dir: Path
    timestamp_started: str = ""
    timestamp_finished: str = ""
    wall_clock_duration_s: float = 0.0
    backend_name: str = ""


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[float, str], None]   # (fraction_0_to_1, message)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_simulation(
    dem: DEM,
    sources: PointSources,
    hydrographs: HydrographSet,
    config: SimulationConfig,
    progress_cb: ProgressCallback | None = None,
    cancel_flag: list[bool] | None = None,
) -> SimulationResult:
    """
    Run the overland-flow simulation and write output files.
    """
    import time
    from datetime import datetime
    _start_perf = time.perf_counter()
    timestamp_started = datetime.now().isoformat()
    from landlab import RasterModelGrid
    from landlab.components.overland_flow import OverlandFlow
    from landlab.components.sink_fill import SinkFiller
    from landlab.grid.raster_mappers import (
        map_mean_of_horizontal_links_to_node,
        map_mean_of_vertical_links_to_node,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elapsed_time = 0.0
    _last_wall_report = time.perf_counter()
    last_report_frac  = -1.0
    _REPORT_INTERVAL  = 0.1   # wall-clock seconds between GUI updates (~10 fps)

    def _report(frac: float, msg: str, force: bool = False) -> None:
        nonlocal _last_wall_report, last_report_frac
        now_wall = time.perf_counter()
        if force or (now_wall - _last_wall_report >= _REPORT_INTERVAL) or (frac - last_report_frac > 0.05):
            logger.debug(msg)
            if progress_cb:
                progress_cb(frac, msg)
            _last_wall_report = now_wall
            last_report_frac  = frac

    nrows, ncols = dem.shape
    elev_ll = dem.elevation_landlab_order()   # bottom-up, flattened

    grid = RasterModelGrid(
        (nrows, ncols),
        xy_spacing=(dem.dx, dem.dy),
        xy_of_lower_left=dem.xy_of_lower_left,
    )

    # Replace NaNs before building the grid.
    nan_mask = np.isnan(elev_ll)
    if nan_mask.any():
        valid_min = float(np.nanmin(elev_ll))
        valid_mean = float(np.nanmean(elev_ll))
        boundary_nodes = np.concatenate([
            np.arange(ncols),                              # bottom row
            np.arange((nrows - 1) * ncols, nrows * ncols),  # top row
            np.arange(ncols, (nrows - 1) * ncols, ncols),  # left column
            np.arange(2 * ncols - 1, (nrows - 1) * ncols, ncols),  # right column
        ])
        
        # Set all NaN nodes to closed boundary status initially
        grid.status_at_node[nan_mask] = grid.BC_NODE_IS_CLOSED
        
        # Perimeter nodes (NaN or not) should be outlets if on the edge
        # unless we specifically want to close them.
        edge_mask = np.zeros(len(elev_ll), dtype=bool)
        edge_mask[boundary_nodes] = True
        
        # Perimeter NaNs -> fixed value (outlet)
        boundary_nan = nan_mask & edge_mask
        elev_ll[boundary_nan] = valid_min
        grid.status_at_node[boundary_nan] = grid.BC_NODE_IS_FIXED_VALUE
        
        # Interior NaNs -> closed boundary
        interior_nan = nan_mask & ~edge_mask
        elev_ll[interior_nan] = valid_mean
        grid.status_at_node[interior_nan] = grid.BC_NODE_IS_CLOSED

        logger.warning(
            "%d NaN cells replaced: %d boundary→outlet (%.2f m), %d interior→closed boundary",
            nan_mask.sum(), boundary_nan.sum(), valid_min, interior_nan.sum()
        )
    
    # All perimeter nodes are free-drainage outlets (depth forced to 0 each step).
    # FIXED_GRADIENT (status=2) is intentionally avoided: the Landlab backend's
    # apply_bc() copies interior depth to boundary nodes, which creates zero water-
    # surface gradient on the next OverlandFlow step and blocks outflow entirely.
    grid.set_status_at_node_on_edges(right=1, top=1, left=1, bottom=1)

    # Re-apply NaN boundary status (in case set_status_at_node_on_edges overwrote them)
    if nan_mask.any():
        grid.status_at_node[nan_mask & ~edge_mask] = grid.BC_NODE_IS_CLOSED
        grid.status_at_node[nan_mask & edge_mask] = grid.BC_NODE_IS_FIXED_VALUE

    n_core = np.sum(grid.status_at_node == 0)
    n_outlets = np.sum(grid.status_at_node == 1)
    logger.info("Grid status: %d core nodes, %d outlets", n_core, n_outlets)

    # Final safety check: ensure no NaNs in topography
    elev_ll = np.nan_to_num(elev_ll, nan=valid_mean if 'valid_mean' in locals() else 0.0)
    
    # ------------------------------------------------------------------
    # 2. Fill sinks (Use RichDEM for high performance)
    # ------------------------------------------------------------------
    grid.add_field("topographic__elevation", elev_ll.copy(), at="node")
    grid.add_field("surface_water__depth", np.zeros(nrows * ncols), at="node")

    if config.fill_sinks:
        _report(0.0, "Filling sinks (RichDEM) …")
        try:
            import richdem as rd
            # RichDEM works on 2D top-down arrays. 
            # We fill the 2D elevation before flattening to Landlab order.
            elev_2d = dem.elevation.copy()
            # Replace NaNs with a value RichDEM understands as NoData
            rd_nodata = -9999.0
            elev_2d[np.isnan(elev_2d)] = rd_nodata
            
            rd_arr = rd.rdarray(elev_2d, no_data=rd_nodata)
            rd.fill_depressions(rd_arr, in_place=True)
            
            # Map back to Landlab order
            elev_filled = np.flipud(np.array(rd_arr)).ravel().astype(np.float64)
            # Restore NaNs for consistency with the rest of the engine
            elev_filled[nan_mask] = valid_mean if 'valid_mean' in locals() else 0.0
            
            logger.info("RichDEM sink filling complete.")
        except Exception as exc:
            logger.warning("RichDEM sink filling failed (%s) — falling back to unfilled terrain.", exc)
            _report(0.0, "Sink fill failed — continuing …")
            elev_filled = elev_ll.copy()
    else:
        elev_filled = elev_ll.copy()

    # Store clean initial elevation for max-level calculation
    elev_initial = elev_filled.copy()
    grid.at_node["topographic__elevation"][:] = elev_filled

    # ------------------------------------------------------------------
    # 3. Initialise result tracking fields
    # ------------------------------------------------------------------
    grid.add_field("surface_water__maxdepth",
                   np.full(nrows * ncols, config.max_depth_threshold, dtype=np.float32), 
                   at="node")
    grid.add_field("surface_water__maxlevel", elev_initial.copy().astype(np.float32), at="node")
    grid.add_field("surface_water__max_q", np.zeros(nrows * ncols, dtype=np.float32), at="node")

    # ------------------------------------------------------------------
    # 4. Set up Solver Backend
    # ------------------------------------------------------------------
    # Ensure all backends are registered
    from . import backends  # noqa: F401
    
    # Extract data for the backend
    grid_data = extract_grid_arrays(grid)
    
    # Select best engine
    BackendClass = BackendRegistry.get_best_backend(grid, config)
    solver = BackendClass(grid_data, config, grid=grid)

    # Numba backends don't create the discharge link field that Landlab does
    # automatically; create it here so NetCDF export works for all backends.
    if "surface_water__discharge" not in grid.at_link:
        grid.add_field("surface_water__discharge",
                       np.zeros(grid.number_of_links), at="link")

    # ------------------------------------------------------------------
    # 5. Optional NetCDF setup
    # ------------------------------------------------------------------
    nc_path = output_dir / "temporal_results.nc"
    netcdf = None
    nc_vars: dict = {}

    if config.export_netcdf:
        netcdf, nc_vars = _open_netcdf(nc_path, grid, nrows, ncols)

    # ------------------------------------------------------------------
    # 6. Simulation loop
    # ------------------------------------------------------------------
    _report(0.0, "Starting simulation …")
    next_snapshot_time = 0.0     # Bug fix #2 — threshold not modulo
    snapshot_index = 0
    inflow_volume_total = 0.0

    # Pre-calculate sources that actually have hydrographs
    valid_sources = []
    for nid, _x, _y, gn in sources.iter():
        if nid in hydrographs.flows:
            valid_sources.append((nid, gn))
    
    source_nodes_arr = np.array([s[1] for s in valid_sources], dtype=np.int32)
    source_deltas_arr = np.zeros(len(valid_sources), dtype=np.float32)
    dx, dy = grid.dx, grid.dy
    area = dx * dy

    # Pre-calculate links for max_q mapping (replaces -1 with a safe index for np.zeros array)
    q_map_links = grid.links_at_node.copy()
    q_map_links[q_map_links == -1] = grid.number_of_links

    node_hyd_rows: list[dict] = []

    while elapsed_time < config.simulation_duration_s:
        if cancel_flag and cancel_flag[0]:
            logger.warning("Simulation cancelled at t=%.1f s", elapsed_time)
            break

        # Choose timestep
        if config.fixed_timestep_s is not None:
            dt = config.fixed_timestep_s
        else:
            dt = solver.calc_time_step()
            if np.isnan(dt) or dt <= 0:
                dt = 1.0   # fallback for dry-grid startup
            dt = min(dt, config.max_adaptive_dt_s)

        # Clamp dt so we hit snapshot boundaries cleanly
        if elapsed_time < next_snapshot_time <= elapsed_time + dt:
            dt = next_snapshot_time - elapsed_time

        # Clamp dt to sim end
        remaining = config.simulation_duration_s - elapsed_time
        dt = min(dt, remaining)
        if dt <= 0:
            break

        # Prevent point-source injection from violating CFL.
        # calc_time_step() uses existing depths; on a dry grid it returns a large
        # dt (based on h_min), but the injected water can be much deeper.  Without
        # this cap the explicit Numba solver oscillates wildly on the first step.
        # We evaluate Q at the midpoint of the CURRENT tentative dt so that a
        # zero-value hydrograph at t=0 doesn't silently bypass the check.
        if config.fixed_timestep_s is None and valid_sources:
            for nid, gn in valid_sources:
                q_flow = hydrographs.flow_at(nid, elapsed_time + 0.5 * dt)
                if q_flow > 0.0:
                    h_inj = q_flow * dt / area
                    dt_inj = 0.7 * min(dx, dy) / math.sqrt(9.80665 * h_inj)
                    if dt_inj < dt:
                        dt = dt_inj
            dt = min(dt, remaining)
            if dt <= 0:
                break

        # Inject inflows from hydrographs
        row: dict = {"time_s": elapsed_time}
        if valid_sources:
            t_mid = elapsed_time + 0.5 * dt
            for idx, (nid, gn) in enumerate(valid_sources):
                flow = hydrographs.flow_at(nid, t_mid)
                vol_step = flow * dt
                source_deltas_arr[idx] = vol_step / area
                inflow_volume_total += vol_step
                row[nid] = flow
            
            solver.add_to_depths(source_nodes_arr, source_deltas_arr)

        node_hyd_rows.append(row)
        elapsed_time += dt

        # Snapshot before stepping (so t=0 is captured)
        if elapsed_time >= next_snapshot_time and config.export_netcdf and netcdf is not None:
            # Sync solver state to grid fields so _write_nc_snapshot sees
            # current data regardless of which backend is active.
            solver.sync_to_grid()
            if hasattr(solver, "q"):
                grid.at_link["surface_water__discharge"][:] = solver.q
            _write_nc_snapshot(nc_vars, snapshot_index, elapsed_time,
                               grid, nrows, ncols)
            snapshot_index += 1
            next_snapshot_time += config.snapshot_interval_s

        # Hydrodynamic step
        solver.run_one_step(dt)

        # Update max depth / level / discharge
        if not getattr(solver, "TRACKS_MAX_INTERNALLY", False):
            depth = solver.depth
            # Max Depth
            maxd = grid.at_node["surface_water__maxdepth"]
            np.maximum(depth, maxd, out=maxd)
            
            # Max Level
            maxl = grid.at_node["surface_water__maxlevel"]
            # Optimization: only compute level where depth > 0
            np.maximum(depth + elev_initial, maxl, out=maxl)
            
            # Track max discharge (q) - map link values to nodes
            if hasattr(solver, "q"):
                q_mag = np.abs(solver.q)
            else: # Landlab
                q_mag = np.abs(grid.at_link["surface_water__discharge"])
                
            # Map link maxes to node-based max_q (max magnitude of any connected link)
            mqn = grid.at_node["surface_water__max_q"]
            q_with_zero = np.zeros(grid.number_of_links + 1, dtype=q_mag.dtype)
            q_with_zero[:-1] = q_mag
            
            # Gather and take max using pre-calculated mapping
            node_q_current = np.max(q_with_zero[q_map_links], axis=1)
            np.maximum(mqn, node_q_current, out=mqn)

        # Reporting and Volume
        if hasattr(solver, "get_total_volume"):
            current_vol = solver.get_total_volume()
        else:
            current_vol = np.sum(solver.depth) * area

        _report(elapsed_time / config.simulation_duration_s,
                f"t = {elapsed_time:.1f} / {config.simulation_duration_s:.0f} s  "
                f"inflow = {inflow_volume_total:.1f} m3  "
                f"stored = {current_vol:.1f} m3")

    # Final sync back to grid nodes
    if getattr(solver, "TRACKS_MAX_INTERNALLY", False):
        # Sync max fields from solver to grid
        grid.at_node["surface_water__maxdepth"][:] = solver.max_depth
        grid.at_node["surface_water__maxlevel"][:] = solver.max_level
        grid.at_node["surface_water__max_q"][:] = solver.max_q
    
    solver.sync_to_grid()

    # Final max level
    grid.at_node["surface_water__maxlevel"][:] = (
        grid.at_node["surface_water__maxdepth"] + elev_initial
    )

    _report(1.0, f"Sim complete. Inflow: {inflow_volume_total:.1f} m3", force=True)

    if netcdf is not None:
        netcdf.close()

    # ------------------------------------------------------------------
    # 7. Write output rasters (GeoTIFF — readable by anything)
    # ------------------------------------------------------------------
    transform = transform_from_landlab(dem.xy_of_lower_left, dem.dx, dem.dy, nrows)

    def _ll_to_topdown(flat: np.ndarray) -> np.ndarray:
        return np.flipud(flat.reshape(nrows, ncols))

    # Final data cleaning for visualization/save
    max_depth_raw = grid.at_node["surface_water__maxdepth"]
    max_depth_raw[~np.isfinite(max_depth_raw)] = 0.0
    
    max_level_raw = grid.at_node["surface_water__maxlevel"]
    max_level_raw[~np.isfinite(max_level_raw)] = 0.0

    max_depth_2d = _ll_to_topdown(max_depth_raw)
    max_level_2d = _ll_to_topdown(max_level_raw)

    max_q_raw = grid.at_node["surface_water__max_q"]
    max_q_raw[~np.isfinite(max_q_raw)] = 0.0
    max_q_2d = _ll_to_topdown(max_q_raw)

    write_raster(output_dir / "max_depth.tif", max_depth_2d, transform, dem.crs)
    write_raster(output_dir / "max_level.tif", max_level_2d, transform, dem.crs)
    write_raster(output_dir / "max_q.tif", max_q_2d, transform, dem.crs)
    logger.info("Wrote max_depth.tif, max_level.tif, and max_q.tif to %s", output_dir)

    node_hydrographs = pd.DataFrame(node_hyd_rows)
    hydro_path = output_dir / "node_hydrographs.csv"
    node_hydrographs.to_csv(hydro_path, index=False)
    logger.info("Wrote node_hydrographs.csv")

    _report(1.0, "Done.")
    return SimulationResult(
        max_depth=max_depth_2d,
        max_level=max_level_2d,
        max_q=max_q_2d,
        node_hydrographs=node_hydrographs,
        dem=dem,
        output_dir=output_dir,
        timestamp_started=timestamp_started,
        timestamp_finished=datetime.now().isoformat(),
        wall_clock_duration_s=time.perf_counter() - _start_perf,
        backend_name=solver.NAME
    )


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------

def _open_netcdf(path: Path, grid, nrows: int, ncols: int):
    import netCDF4  # type: ignore

    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", None)     # unlimited
    ds.createDimension("y", nrows)
    ds.createDimension("x", ncols)

    v_time = ds.createVariable("time", "f8", ("time",))
    v_time.units = "seconds"
    v_time.long_name = "elapsed simulation time"

    v_x = ds.createVariable("x", "f4", ("x",))
    v_x.units = "m"; v_x.axis = "X"
    v_x[:] = np.unique(grid.node_x)

    v_y = ds.createVariable("y", "f4", ("y",))
    v_y.units = "m"; v_y.axis = "Y"
    v_y[:] = np.unique(grid.node_y)[::-1]   # top-down for viewers

    depth = ds.createVariable("surface_water_depth", "f4", ("time", "y", "x"))
    depth.units = "m"
    depth.long_name = "Surface Water Depth"

    u_swd = ds.createVariable("u-swd", "f4", ("time", "y", "x"))
    u_swd.units = "m**3 s**-1"
    u_swd.long_name = "Surface Water Discharge u-component"

    v_swd = ds.createVariable("v-swd", "f4", ("time", "y", "x"))
    v_swd.units = "m**3 s**-1"
    v_swd.long_name = "Surface Water Discharge v-component"

    return ds, {"time": v_time, "depth": depth, "u_swd": u_swd, "v_swd": v_swd}


def _write_nc_snapshot(nc_vars: dict, i: int, elapsed_s: float,
                       grid, nrows: int, ncols: int) -> None:
    from landlab.grid.raster_mappers import (
        map_mean_of_horizontal_links_to_node,
        map_mean_of_vertical_links_to_node,
    )

    nc_vars["time"][i] = elapsed_s

    depth_2d = np.flipud(
        grid.at_node["surface_water__depth"].reshape(nrows, ncols)
    )
    nc_vars["depth"][i, :, :] = depth_2d.astype(np.float32)

    u_node = map_mean_of_horizontal_links_to_node(grid, "surface_water__discharge")
    v_node = map_mean_of_vertical_links_to_node(grid, "surface_water__discharge")

    nc_vars["u_swd"][i, :, :] = np.flipud(u_node.reshape(nrows, ncols)).astype(np.float32)
    nc_vars["v_swd"][i, :, :] = np.flipud(v_node.reshape(nrows, ncols)).astype(np.float32)
