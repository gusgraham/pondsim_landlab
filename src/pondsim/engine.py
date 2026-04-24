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
    node_hydrographs: pd.DataFrame
    dem: DEM
    output_dir: Path


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

    Parameters
    ----------
    dem:          loaded DEM (rasterio top-down order)
    sources:      snapped point sources
    hydrographs:  time-varying inflows per node_id
    config:       simulation settings
    progress_cb:  optional (fraction, message) callback for UI integration
    cancel_flag:  mutable list[bool]; set cancel_flag[0]=True to abort
    """
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
    last_report_time = -100.0
    last_report_frac = -1.0

    def _report(frac: float, msg: str, force: bool = False) -> None:
        nonlocal last_report_time, last_report_frac
        now = elapsed_time # Use simulation time as a proxy or real time
        if force or (now - last_report_time > 1.0) or (frac - last_report_frac > 0.01):
            logger.info(msg)
            if progress_cb:
                progress_cb(frac, msg)
            last_report_time = now
            last_report_frac = frac

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
    
    # Ensure we have outlets for SinkFiller (Landlab default + our NaN logic)
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
    
    grid.add_field("topographic__elevation", elev_ll.copy(), at="node")
    grid.add_field("surface_water__depth",
                   np.zeros(nrows * ncols), at="node")

    # ------------------------------------------------------------------
    # 2. Fill sinks (Bug fix #1 — actually run SinkFiller and USE it)
    # ------------------------------------------------------------------
    if config.fill_sinks:
        _report(0.0, "Filling sinks …")
        try:
            sf = SinkFiller(grid, apply_slope=True)
            sf.run_one_step()
            elev_filled = grid.at_node["topographic__elevation"].copy()
            logger.info("Sink filling complete. Max fill depth: %.3f m",
                        (elev_filled - elev_ll).max())
        except ValueError as exc:
            logger.warning("SinkFiller failed (%s) — continuing without sink filling.", exc)
            _report(0.0, "Sink fill skipped — continuing …")
            elev_filled = elev_ll.copy()
            grid.at_node["topographic__elevation"] = elev_filled.copy()
    else:
        elev_filled = elev_ll.copy()

    # Store clean initial elevation for max-level calculation
    elev_initial = elev_filled.copy()

    # ------------------------------------------------------------------
    # 3. Initialise result tracking fields
    # ------------------------------------------------------------------
    grid.add_field("surface_water__maxdepth",
                   np.full(nrows * ncols, config.max_depth_threshold, dtype=np.float32), 
                   at="node")
    grid.add_field("surface_water__maxlevel", elev_initial.copy().astype(np.float32), at="node")

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

    node_hyd_rows: list[dict] = []
    active_node_ids = [nid for nid in sources.node_ids
                       if nid in hydrographs.flows]

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

        # Inject inflows from hydrographs
        row: dict = {"time_s": elapsed_time}
        for nid, _x, _y, gn in sources.iter():
            if nid not in hydrographs.flows:
                continue
            avg_flow = hydrographs.flow_average(nid, elapsed_time, elapsed_time + dt)
            dx, dy = grid.dx, grid.dy
            delta_depth = avg_flow * dt / (dx * dy)
            solver.depth[gn] += delta_depth
            inflow_volume_total += avg_flow * dt
            row[nid] = avg_flow

        node_hyd_rows.append(row)

        # Snapshot before stepping (so t=0 is captured)
        if elapsed_time >= next_snapshot_time and config.export_netcdf and netcdf is not None:
            _write_nc_snapshot(nc_vars, snapshot_index, elapsed_time,
                               grid, nrows, ncols)
            snapshot_index += 1
            next_snapshot_time += config.snapshot_interval_s

        # Hydrodynamic step
        solver.run_one_step(dt)

        # Update max depth / level
        # Get current depth from solver for max_depth tracking
        depth = solver.depth
        maxd = grid.at_node["surface_water__maxdepth"]
        np.maximum(depth, maxd, out=maxd)

        elapsed_time += dt
        _report(elapsed_time / config.simulation_duration_s,
                f"t = {elapsed_time:.0f} / {config.simulation_duration_s:.0f} s  "
                f"  inflow = {inflow_volume_total:.1f} m³")

    # Final sync back to grid nodes
    solver.sync_to_grid()

    # Final max level
    grid.at_node["surface_water__maxlevel"][:] = (
        grid.at_node["surface_water__maxdepth"] + elev_initial
    )

    _report(1.0, f"Sim complete. Inflow: {inflow_volume_total:.1f} m³", force=True)

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

    write_raster(output_dir / "max_depth.tif", max_depth_2d, transform, dem.crs)
    write_raster(output_dir / "max_level.tif", max_level_2d, transform, dem.crs)
    logger.info("Wrote max_depth.tif and max_level.tif to %s", output_dir)

    node_hydrographs = pd.DataFrame(node_hyd_rows)
    hydro_path = output_dir / "node_hydrographs.csv"
    node_hydrographs.to_csv(hydro_path, index=False)
    logger.info("Wrote node_hydrographs.csv")

    _report(1.0, "Done.")
    return SimulationResult(
        max_depth=max_depth_2d,
        max_level=max_level_2d,
        node_hydrographs=node_hydrographs,
        dem=dem,
        output_dir=output_dir,
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
