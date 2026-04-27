# Swesim — Overland Flow Simulation Engine

Swesim is a high-performance overland flow simulation tool designed to model flood extents and depths using the Shallow Water Equations (SWE). It features a "Headless-First" architecture, making it equally suitable for interactive desktop use, automated command-line pipelines, or as a backend library for web applications.

## 🚀 Key Features

- **Coarse-to-Fine "Two-Pass" Analysis**: Automatically identifies active flood clusters using a fast coarse pass, then runs high-resolution simulations only on the relevant areas. This typically reduces compute time and VRAM usage by 80-90%.
- **Tiered Solver Backends**:
    - **Landlab Native**: The original Landlab `OverlandFlow` component; highly stable and significantly faster than the baseline NumPy implementation.
    - **NumPy**: Absolute baseline solver; primarily for architectural reference or environments where Landlab is unavailable.
    - **Numba CPU**: Parallelized implementation for multi-core CPU performance.
    - **Numba CUDA**: Massively parallel GPU solver for NVIDIA hardware, capable of handling millions of cells in minutes.
- **Spatially Varying Roughness**: Support for Manning's n rasters to account for different land-use types (e.g., pavement vs. vegetation).
- **Automated Pipeline Integration**: Standardized JSON outputs for metadata and statistics enable seamless integration into larger data workflows.

## 🛠 Usage

This document describes the primary ways to interact with Swesim.

### 1. Unified Startup Script
The easiest way to run Swesim is via the `run.sh` script in the project root. It handles environment configuration and virtual environment detection automatically.

- **Start the GUI**:
  ```bash
  ./run.sh gui
  ```
- **Run the Headless CLI**:
  ```bash
  ./run.sh cli --dem path/to/dem.tif --sources path/to/src.geojson --out-dir ./results
  ```
- **Run Unit Tests**:
  ```bash
  ./run.sh test
  ```

### 2. Python API (Library Mode)
For deeper integration (e.g., in a web backend), you can import the orchestrator directly:

```python
from swesim.workflow import SimulationWorkflow
from swesim.engine import SimulationConfig

# Initialize the workflow orchestrator
wf = SimulationWorkflow.from_paths(
    dem_path="grid.tif",
    sources_path="manholes.geojson",
    synthetic_vol=500.0,
    synthetic_duration=7200.0
)

# Configure the simulation settings
config = SimulationConfig(
    output_dir="./results",
    backend="auto",  # Automatically selects the best available hardware
    simulation_duration_s=7200
)

# Execute the full two-pass workflow
result = wf.run_two_pass(config)
```

## 🔌 API Reference (Headless Core)

For programmatic integration, the following objects in `swesim.engine` are the primary interfaces:

### `SimulationConfig` (Input)
A dataclass defining the simulation environment:
- `output_dir`: (Path/str) Where to write rasters and JSONs.
- `simulation_duration_s`: (float) Total time to simulate.
- `manning_n`: (float | ndarray) Manning's n roughness.
- `backend`: (str) `'auto'`, `'numba_cuda'`, `'numba_cpu'`, `'landlab'`, or `'numpy'`.
- `fixed_timestep_s`: (float | None) If set, disables adaptive timestepping.
- `snapshot_interval_s`: (float) Frequency of hydrograph recording.
- `export_netcdf`: (bool) If true, exports full depth history (memory intensive).
- `fill_sinks`: (bool) Pre-process DEM to remove depressions.

### `SimulationResult` (Output)
The object returned by the workflow methods:
- `max_depth`: (ndarray) 2D NumPy array of peak depths.
- `max_level`: (ndarray) 2D NumPy array of peak water levels.
- `node_hydrographs`: (DataFrame) Time-series data for point sources.
- `dem`: (DEM) The grid geometry object used for the run.
- `output_dir`: (Path) Final location of all saved artifacts.

### `SimulationWorkflow` (Orchestrator)
The recommended entry point:
- `run_one_pass(config)`: Standard simulation.
- `run_two_pass(config, coarse_factor=5, buffer_m=200)`: Efficient multi-pass analysis.
- `save_results(result, config)`: Manually trigger JSON metadata/stats generation.

## 📂 Project Architecture

Swesim is designed around the **Orchestrator Pattern**. The GUI and CLI are both thin "consumers" of the unified `SimulationWorkflow` core.

- **`src/swesim/workflow.py`**: The central orchestrator. Manages data loading, pass management, results stitching, and metadata export.
- **`src/swesim/engine.py`**: The solver infrastructure and time-stepping logic.
- **`src/swesim/backends/`**: Individual implementations of the de Almeida SWE solver for different hardware tiers.
- **`src/swesim/app.py`**: The Qt-based graphical desktop interface.
- **`src/swesim/cli.py`**: The command-line interface entry point.

## 📊 Outputs

Every simulation run produces a comprehensive output package in the specified `--out-dir`:

- **`max_depth.tif`**: GeoTIFF raster of maximum flood depths (meters).
- **`max_level.tif`**: GeoTIFF raster of maximum water surface levels (meters).
- **`node_hydrographs.csv`**: Time-series depth/flow data for every point source.
- **`summary_stats.json`**: High-level results (peak depth, total inundated area, run duration).
- **`metadata.json`**: A reproducibility record containing hardware specs (CPU/GPU), software versions (Python/Numba), and the exact configuration parameters used.

## 🧪 Testing

To ensure stability across all backends and interfaces, Swesim includes a test suite that can be run via:
```bash
./run.sh test
```
This is particularly important when developing new solver kernels or modifying the `SimulationWorkflow` core.
