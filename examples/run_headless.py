"""
Headless (no GUI) example — validates the full pipeline without Qt.

Usage:
    python examples/run_headless.py --dem path/to/dem.asc --output path/to/out/

The script synthesises a small hydrograph and places a single point source at
the centroid of the DTM so it runs self-contained with any DEM.

For ICM-driven runs, replace the synthetic hydrograph with:
    hydrographs = load_hydrographs("my_icm_export.csv")

and point sources with:
    sources = load_sources("my_nodes.geojson", grid)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from pondsim import (
    read_dem,
    run_simulation,
    SimulationConfig,
    HydrographSet,
    PointSources,
)
from pondsim.hydrographs import make_synthetic_hydrograph
from pondsim.sources import sources_from_xy
from pondsim.viz import plot_dem, plot_overlay, plot_sources, plot_hydrographs


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Pondsim headless run")
    p.add_argument("--dem", required=True, help="Path to DEM (.asc or .tif)")
    p.add_argument("--output", default="./pondsim_output", help="Output folder")
    p.add_argument("--sources", default=None,
                   help="GeoJSON/GPKG/SHP of ICM overflow points (optional)")
    p.add_argument("--hydrographs", default=None,
                   help="CSV hydrograph file (optional; uses synthetic if absent)")
    p.add_argument("--duration", type=float, default=7200,
                   help="Simulation duration in seconds (default 7200 = 2 h)")
    p.add_argument("--fixed-dt", type=float, default=None,
                   help="Fixed timestep in seconds (default: adaptive)")
    p.add_argument("--netcdf", action="store_true", help="Export temporal NetCDF")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load DEM
    # ------------------------------------------------------------------
    logging.info("Loading DEM: %s", args.dem)
    dem = read_dem(args.dem)
    nrows, ncols = dem.shape
    logging.info("  Grid: %d rows × %d cols, %.1f m resolution", nrows, ncols, dem.dx)

    # ------------------------------------------------------------------
    # 2. Build a minimal RasterModelGrid just for source snapping
    #    (engine.py creates its own; this one is for sources_from_xy)
    # ------------------------------------------------------------------
    from landlab import RasterModelGrid
    _snap_grid = RasterModelGrid(
        (nrows, ncols),
        xy_spacing=(dem.dx, dem.dy),
        xy_of_lower_left=dem.xy_of_lower_left,
    )

    # ------------------------------------------------------------------
    # 3. Point sources
    # ------------------------------------------------------------------
    if args.sources:
        from pondsim.sources import load_sources
        logging.info("Loading point sources: %s", args.sources)
        sources = load_sources(args.sources, _snap_grid)
        node_ids = sources.node_ids
    else:
        # Place a single synthetic source at the DTM centroid
        x0, x1, y0, y1 = dem.xy_extent()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        logging.info("No sources file — using synthetic source at centroid (%.1f, %.1f)", cx, cy)
        sources = sources_from_xy(["SYN_001"], [cx], [cy], _snap_grid)
        node_ids = ["SYN_001"]

    # ------------------------------------------------------------------
    # 4. Hydrographs
    # ------------------------------------------------------------------
    if args.hydrographs:
        from pondsim.hydrographs import load_hydrographs
        logging.info("Loading hydrographs: %s", args.hydrographs)
        hydrographs = load_hydrographs(args.hydrographs)
        duration_s = hydrographs.duration_s
    else:
        duration_s = args.duration
        total_vol_per_node = 500.0   # m³ — adjust for realistic runs
        logging.info(
            "No hydrograph file — synthetic UH: %.0f m³ per node, %.0f s duration",
            total_vol_per_node, duration_s,
        )
        hydrographs = make_synthetic_hydrograph(
            node_ids=node_ids,
            total_volume_m3=total_vol_per_node,
            duration_s=duration_s,
        )

    # ------------------------------------------------------------------
    # 5. Run
    # ------------------------------------------------------------------
    config = SimulationConfig(
        output_dir=args.output,
        simulation_duration_s=duration_s,
        fixed_timestep_s=args.fixed_dt,
        export_netcdf=args.netcdf,
        fill_sinks=True,
    )

    logging.info("Starting simulation …")
    result = run_simulation(dem, sources, hydrographs, config)

    # ------------------------------------------------------------------
    # 6. Quick summary plots
    # ------------------------------------------------------------------
    output_dir = Path(args.output)

    # Map: DEM + depth overlay + sources
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_dem(dem, ax=ax)
    if result.max_depth.max() > 0:
        plot_overlay(result.max_depth, dem, ax=ax,
                     label="Max Water Depth (m)", cmap="Blues")
    plot_sources(sources, ax=ax)
    ax.set_title("Maximum Surface Water Depth")
    fig.tight_layout()
    map_png = output_dir / "map_max_depth.png"
    fig.savefig(map_png, dpi=150, bbox_inches="tight")
    logging.info("Saved map: %s", map_png)
    plt.close(fig)

    # Hydrographs
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_hydrographs(result.node_hydrographs, ax=ax2)
    ax2.set_title("Node Inflow Hydrographs")
    hyd_png = output_dir / "hydrographs.png"
    fig2.savefig(hyd_png, dpi=150, bbox_inches="tight")
    logging.info("Saved hydrograph plot: %s", hyd_png)
    plt.close(fig2)

    # Summary stats
    peak_depth = float(result.max_depth.max())
    wet_cells = int((result.max_depth > 0.01).sum())
    wet_area_ha = wet_cells * dem.dx * dem.dy / 10_000
    logging.info("Peak max depth:    %.3f m", peak_depth)
    logging.info("Wet area (>1 cm):  %.2f ha", wet_area_ha)
    logging.info("Output folder:     %s", output_dir.resolve())


if __name__ == "__main__":
    main()
