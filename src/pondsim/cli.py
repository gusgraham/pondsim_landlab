#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from .workflow import SimulationWorkflow
from .engine import SimulationConfig

def main():
    parser = argparse.ArgumentParser(description="Pondsim Headless CLI - Automated flood simulation orchestrator.")
    
    # Input group
    input_grp = parser.add_argument_group("Input Data")
    input_grp.add_argument("--dem", required=True, help="Path to topographic DEM (GeoTIFF / ASC).")
    input_grp.add_argument("--sources", required=True, help="Path to point sources (GeoJSON / SHP / GPKG).")
    input_grp.add_argument("--roughness", help="Optional: Path to Manning's n roughness raster.")
    input_grp.add_argument("--hydrographs", help="Optional: Path to inflows CSV. If omitted, specify --syn-vol and --syn-dur.")
    
    # Parameter group
    param_grp = parser.add_argument_group("Simulation Parameters")
    param_grp.add_argument("--out-dir", required=True, help="Directory to write results.")
    param_grp.add_argument("--duration", type=float, help="Simulation duration in seconds. Overrides hydrograph duration if provided.")
    param_grp.add_argument("--manning", type=float, default=0.03, help="Default Manning's n (used if no roughness raster). Default: 0.03.")
    param_grp.add_argument("--backend", default="auto", choices=["auto", "numpy", "numba_cpu", "numba_cuda"], 
                           help="Solver backend. 'auto' chooses the best available Tier. Default: auto.")
    param_grp.add_argument("--syn-vol", type=float, help="Synthetic inflow volume per node (m3). Required if no --hydrographs.")
    param_grp.add_argument("--syn-dur", type=float, help="Synthetic inflow duration (s). Required if no --hydrographs.")
    
    # Workflow group
    flow_grp = parser.add_argument_group("Workflow Options")
    flow_grp.add_argument("--two-pass", action="store_true", help="Execute coarse-to-fine two-pass analysis.")
    flow_grp.add_argument("--coarse-factor", type=int, default=5, help="Coarsening factor for two-pass analysis (e.g. 5x). Default: 5.")
    flow_grp.add_argument("--buffer", type=float, default=200.0, help="Buffer (m) around active flood extent for clipping. Default: 200.")
    
    # General
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only log warnings and errors.")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
        
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr
    )
    
    # Validation
    if not args.hydrographs and (args.syn_vol is None or args.syn_dur is None):
        logging.error("Incomplete input: Specify either --hydrographs path or both --syn-vol and --syn-dur for synthetic inflows.")
        sys.exit(1)

    try:
        # Initialize workflow (loads data from disk)
        wf = SimulationWorkflow.from_paths(
            dem_path=args.dem,
            sources_path=args.sources,
            roughness_path=args.roughness,
            hydrograph_path=args.hydrographs,
            synthetic_vol=args.syn_vol,
            synthetic_duration=args.syn_dur
        )
        
        # Determine duration
        duration = args.duration
        if duration is None:
            duration = wf.hydrographs.duration_s
            
        config = SimulationConfig(
            output_dir=Path(args.out_dir),
            simulation_duration_s=duration,
            manning_n=args.manning,
            backend=args.backend,
            fill_sinks=True
        )

        # Execute
        if args.two_pass:
            wf.run_two_pass(config, coarse_factor=args.coarse_factor, buffer_m=args.buffer)
        else:
            wf.run_one_pass(config)
            
        logging.info("Headless simulation run completed successfully.")

    except Exception as e:
        if args.verbose:
            logging.exception("Fatal error during simulation execution.")
        else:
            logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
