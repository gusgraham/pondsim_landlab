from __future__ import annotations
import os
import sys
import json
import logging
import platform
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Callable

from .engine import run_simulation, SimulationConfig, SimulationResult, ProgressCallback
from .raster import DEM, identify_flood_clusters, clip_dem_to_bbox, write_raster, transform_from_landlab
from .sources import PointSources, load_sources
from .hydrographs import HydrographSet
from .backends.base import BackendRegistry

logger = logging.getLogger(__name__)

class SimulationWorkflow:
    """
    Headless orchestrator for Pondsim simulations.
    Manages input data, execution (single or two-pass), results stitching, and metadata export.
    """

    def __init__(
        self,
        dem: DEM,
        sources: PointSources,
        hydrographs: HydrographSet,
        roughness: Optional[np.ndarray] = None,
    ):
        self.dem = dem
        self.sources = sources
        self.hydrographs = hydrographs
        self.roughness = roughness

    @classmethod
    def from_paths(
        cls,
        dem_path: str | Path,
        sources_path: str | Path,
        roughness_path: Optional[str | Path] = None,
        hydrograph_path: Optional[str | Path] = None,
        synthetic_vol: Optional[float] = None,
        synthetic_duration: Optional[float] = None,
    ) -> SimulationWorkflow:
        """Utility to load all data from disk."""
        from .raster import read_dem
        from .sources import load_sources
        from .hydrographs import HydrographSet, make_synthetic_hydrograph
        from landlab import RasterModelGrid

        logger.info(f"Loading DEM: {dem_path}")
        dem = read_dem(dem_path)
        
        # We need a temporary grid just to snap sources
        grid = RasterModelGrid(dem.shape, xy_spacing=(dem.dx, dem.dy), 
                               xy_of_lower_left=dem.xy_of_lower_left)
        
        logger.info(f"Loading sources: {sources_path}")
        sources = load_sources(sources_path, grid)
        
        roughness = None
        if roughness_path:
            logger.info(f"Loading roughness: {roughness_path}")
            rough_data = read_dem(roughness_path)
            if rough_data.shape != dem.shape:
                raise ValueError(f"Roughness grid {rough_data.shape} mismatch with DEM {dem.shape}")
            roughness = rough_data.elevation
            
        hydrographs = None
        if hydrograph_path:
            logger.info(f"Loading hydrographs: {hydrograph_path}")
            hydrographs = HydrographSet.from_csv(hydrograph_path)
        elif synthetic_vol is not None and synthetic_duration is not None:
            logger.info(f"Generating synthetic hydrographs ({synthetic_vol} m3 per node)")
            hydrographs = make_synthetic_hydrograph(
                node_ids=sources.node_ids,
                volumes_m3=synthetic_vol,
                duration_s=synthetic_duration
            )
        else:
            raise ValueError("Either hydrograph_path or synthetic parameters must be provided.")
            
        return cls(dem, sources, hydrographs, roughness)

    def run_one_pass(
        self, 
        config: SimulationConfig, 
        progress_cb: Optional[ProgressCallback] = None,
        cancel_flag: Optional[list[bool]] = None
    ) -> SimulationResult:
        """Execute a standard single-pass simulation."""
        # Inject roughness into config if it's not already there
        if self.roughness is not None:
            config.manning_n = self.roughness
            
        result = run_simulation(
            self.dem,
            self.sources,
            self.hydrographs,
            config,
            progress_cb=progress_cb,
            cancel_flag=cancel_flag
        )
        self.save_results(result, config)
        return result

    def run_two_pass(
        self,
        config: SimulationConfig,
        coarse_factor: int = 5,
        buffer_m: float = 200.0,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_flag: Optional[list[bool]] = None,
        interactive_cb: Optional[Callable[[int, int, float], bool]] = None
    ) -> SimulationResult:
        """
        Execute the coarse-to-fine workflow.
        
        If interactive_cb is provided (e.g. from GUI), it will be called with
        (n_clusters, total_pixels, percentage) to ask whether to proceed.
        Otherwise, it auto-proceeds (headless behavior).
        """
        from landlab import RasterModelGrid
        from .raster import resample_dem
        import tempfile

        # 1. Coarse Pass
        logger.info(f"Starting coarse pass (factor {coarse_factor}x) ...")
        coarse_dem = resample_dem(self.dem, coarse_factor)
        cr, cc = coarse_dem.shape
        
        _grid = RasterModelGrid((cr, cc), xy_spacing=(coarse_dem.dx, coarse_dem.dy), 
                                xy_of_lower_left=coarse_dem.xy_of_lower_left)
        coarse_sources = load_sources(None, _grid, hydrograph_ids=self.hydrographs.node_ids, 
                                      existing_sources=self.sources)

        coarse_out = Path(tempfile.mkdtemp(prefix="pondsim_coarse_"))
        coarse_config = SimulationConfig(
            output_dir=coarse_out,
            simulation_duration_s=config.simulation_duration_s,
            manning_n=config.manning_n,
            backend=config.backend,
            fixed_timestep_s=config.fixed_timestep_s,
            snapshot_interval_s=config.snapshot_interval_s,
            export_netcdf=False,
            fill_sinks=config.fill_sinks
        )

        coarse_result = run_simulation(coarse_dem, coarse_sources, self.hydrographs, coarse_config, 
                                       progress_cb=progress_cb, cancel_flag=cancel_flag)

        if cancel_flag and cancel_flag[0]:
            return coarse_result

        # 2. Clustering & Clipping
        bboxes = identify_flood_clusters(coarse_result.max_depth, coarse_result.dem, threshold=0.01, buffer_m=buffer_m)
        n_clusters = len(bboxes)
        
        if n_clusters == 0:
            logger.warning("No flooded clusters identified in coarse pass. Returning coarse result.")
            return coarse_result

        # Calculate efficiency
        total_pixels = 0
        tasks_data = []
        for bbox in bboxes:
            clipped_dem = clip_dem_to_bbox(self.dem, *bbox)
            total_pixels += clipped_dem.shape[0] * clipped_dem.shape[1]
            tasks_data.append(clipped_dem)

        orig_pixels = self.dem.shape[0] * self.dem.shape[1]
        pct = 100.0 * total_pixels / orig_pixels
        logger.info(f"Identified {n_clusters} clusters. Total fine area: {total_pixels:,} cells ({pct:.1f}% of full DEM)")

        # 3. Interactive Check
        if interactive_cb is not None:
            if not interactive_cb(n_clusters, total_pixels, pct):
                logger.info("Fine pass cancelled by user.")
                return coarse_result

        if cancel_flag and cancel_flag[0]:
            return coarse_result

        # 4. Fine Pass(es)
        results = []
        for i, clipped_dem in enumerate(tasks_data):
            if cancel_flag and cancel_flag[0]:
                break
            logger.info(f"Running fine pass for cluster {i+1}/{n_clusters} ...")
            cr, cc = clipped_dem.shape
            _grid_f = RasterModelGrid((cr, cc), xy_spacing=(clipped_dem.dx, clipped_dem.dy),
                                    xy_of_lower_left=clipped_dem.xy_of_lower_left)
            clipped_sources = load_sources(None, _grid_f, hydrograph_ids=self.hydrographs.node_ids,
                                          existing_sources=self.sources)
            
            cluster_out = Path(config.output_dir) / f"cluster_{i+1}"
            cluster_config = SimulationConfig(**{**asdict(config), "output_dir": cluster_out})
            
            res = run_simulation(clipped_dem, clipped_sources, self.hydrographs, cluster_config, 
                                 progress_cb=progress_cb, cancel_flag=cancel_flag)
            results.append(res)

        # 5. Stitching
        final_result = self.stitch_results(results, config.output_dir)
        self.save_results(final_result, config)
        return final_result

    def stitch_results(self, results: list[SimulationResult], output_dir: Path | str) -> SimulationResult:
        """Merge multiple disjoint simulation results into a global one."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nrows, ncols = self.dem.shape
        global_max_depth = np.zeros((nrows, ncols), dtype=np.float32)
        global_max_level = self.dem.elevation.copy().astype(np.float32)

        all_hyds = []
        for res in results:
            gx0, gy_top = self.dem.transform.c, self.dem.transform.f
            cx0, cy_top = res.dem.transform.c, res.dem.transform.f

            col_off = int(round((cx0 - gx0) / self.dem.dx))
            row_off = int(round((gy_top - cy_top) / self.dem.dy))
            rh, rw = res.max_depth.shape

            target_depth = global_max_depth[row_off:row_off + rh, col_off:col_off + rw]
            np.maximum(target_depth, res.max_depth, out=target_depth)

            target_level = global_max_level[row_off:row_off + rh, col_off:col_off + rw]
            np.maximum(target_level, res.max_level, out=target_level)
            all_hyds.append(res.node_hydrographs)

        if all_hyds:
            final_hyd = all_hyds[0]
            for next_hyd in all_hyds[1:]:
                final_hyd = pd.merge(final_hyd, next_hyd, on="time_s", how="outer").sort_values("time_s")
            final_hyd = final_hyd.fillna(0.0)
        else:
            final_hyd = pd.DataFrame(columns=["time_s"])

        # Write stitched files
        transform = transform_from_landlab(self.dem.xy_of_lower_left, self.dem.dx, self.dem.dy, nrows)
        write_raster(output_dir / "max_depth.tif", global_max_depth, transform, self.dem.crs)
        write_raster(output_dir / "max_level.tif", global_max_level, transform, self.dem.crs)
        final_hyd.to_csv(output_dir / "node_hydrographs.csv", index=False)

        return SimulationResult(
            max_depth=global_max_depth,
            max_level=global_max_level,
            node_hydrographs=final_hyd,
            dem=self.dem,
            output_dir=output_dir
        )

    def save_results(self, result: SimulationResult, config: SimulationConfig):
        """Export JSON summary stats and metadata for the run."""
        out_dir = Path(result.output_dir)
        
        # 1. Summary Statistics
        inundated_area = np.count_nonzero(result.max_depth > 0.01) * (self.dem.dx * self.dem.dy)
        stats = {
            "peak_depth_m": float(np.max(result.max_depth)),
            "inundated_area_sqm": float(inundated_area),
            "simulation_duration_s": config.simulation_duration_s,
            "backend_used": config.backend,
            "timestamp_finished": datetime.now().isoformat(),
        }
        with open(out_dir / "summary_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # 2. Metadata & Reproducibility
        meta = {
            "pondsim_version": "1.0.0",
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_cores": os.cpu_count(),
            "config": {k: str(v) for k, v in asdict(config).items()},
        }
        
        try:
            import numba
            meta["numba_version"] = numba.__version__
            from numba import cuda
            if cuda.is_available():
                dev = cuda.get_current_device()
                meta["gpu"] = {
                    "name": dev.name.decode('utf-8') if isinstance(dev.name, bytes) else dev.name,
                    "vram_total_bytes": cuda.current_context().get_memory_info()[1]
                }
        except ImportError:
            pass

        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        logger.info(f"Results saved to {out_dir}")
