"""3-way comparison: Landlab vs Numba CPU vs Numba CUDA"""
import logging
import numpy as np
from pondsim.raster import DEM
from pondsim.engine import run_simulation, SimulationConfig
from pondsim.sources import PointSources
from pondsim.hydrographs import HydrographSet
from rasterio.transform import from_origin

logging.basicConfig(level=logging.WARNING, format="%(message)s")

dx = 5.0
nrows, ncols = 40, 40
dem_arr = np.zeros((nrows, ncols))
for c in range(ncols):
    dem_arr[:, c] = 10.0 - (c * 0.01)

transform = from_origin(0, nrows*dx, dx, dx)
dem = DEM(dem_arr, transform, None, None)

gx = (ncols // 2 + 0.5) * dx
gy = (nrows // 2 - 0.5) * dx
gn = (nrows // 2) * ncols + (ncols // 2)
sources = PointSources(
    node_ids=["S1"],
    xy=np.array([[gx, gy]]),
    grid_nodes=np.array([gn], dtype=np.int32)
)

duration_s = 10.0
hydrographs = HydrographSet(
    times_s=np.array([0.0, 1000.0]),
    flows={"S1": np.array([2.0, 2.0])}
)

results = {}
for backend in ["landlab", "numba_cpu", "numba_cuda"]:
    cfg = SimulationConfig(
        output_dir=f"cmp3_{backend}",
        simulation_duration_s=duration_s,
        backend=backend,
        manning_n=0.03
    )
    res = run_simulation(dem, sources, hydrographs, cfg)
    peak_d = float(res.max_depth.max())
    peak_q = float(res.max_q.max())
    vol = float(np.sum(res.max_depth) * dx * dx)
    results[backend] = (peak_d, peak_q, vol)
    print(f"{backend:>15s}:  depth={peak_d:.6f}  q={peak_q:.6f}  vol={vol:.2f} m3")

# Show ratios vs Landlab
ll = results["landlab"]
for name in ["numba_cpu", "numba_cuda"]:
    r = results[name]
    print(f"{name:>15s} ratio:  depth={r[0]/ll[0]:.4f}  q={r[1]/ll[1]:.4f}  vol={r[2]/ll[2]:.4f}")
