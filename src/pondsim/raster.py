from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin


@dataclass
class DEM:
    elevation: np.ndarray          # 2D, rows top-to-bottom
    transform: Affine              # rasterio geotransform
    crs: rasterio.crs.CRS | None
    nodata: float | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def dx(self) -> float:
        return abs(self.transform.a)

    @property
    def dy(self) -> float:
        return abs(self.transform.e)

    @property
    def xy_of_lower_left(self) -> tuple[float, float]:
        nrows = self.elevation.shape[0]
        x0 = self.transform.c
        y_top = self.transform.f
        return (x0, y_top - nrows * self.dy)

    def xy_extent(self) -> tuple[float, float, float, float]:
        nrows, ncols = self.elevation.shape
        x0 = self.transform.c
        y_top = self.transform.f
        return (x0, x0 + ncols * self.dx, y_top - nrows * self.dy, y_top)

    def world_to_rowcol(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.transform.c) / self.dx)
        row = int((self.transform.f - y) / self.dy)
        return row, col

    def elevation_landlab_order(self) -> np.ndarray:
        # Landlab rasters index rows bottom-up; rasterio is top-down.
        return np.flipud(self.elevation).ravel()


def read_dem(path: str | Path) -> DEM:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return DEM(elevation=arr, transform=ds.transform, crs=ds.crs, nodata=nodata)


def write_raster(
    path: str | Path,
    data: np.ndarray,
    transform: Affine,
    crs: rasterio.crs.CRS | None,
    nodata: float | None = -9999.0,
) -> None:
    out = data.astype(np.float32)
    if nodata is not None:
        out = np.where(np.isnan(out), nodata, out)
    with rasterio.open(
        path,
        "w",
        driver="GTiff" if str(path).lower().endswith(".tif") else "AAIGrid",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(out, 1)


def transform_from_landlab(xy_of_lower_left: tuple[float, float], dx: float, dy: float,
                           nrows: int) -> Affine:
    x0, y0 = xy_of_lower_left
    y_top = y0 + nrows * dy
    return from_origin(x0, y_top, dx, dy)


def resample_dem(dem: DEM, factor: int) -> DEM:
    """Downsample DEM by integer factor using block-mean resampling.

    Useful for a fast coarse-pass analysis before clipping to flood extent.
    NaN cells are filled with the array mean before averaging so they don't
    propagate into neighbouring blocks.
    """
    arr = dem.elevation.copy()
    nrows, ncols = arr.shape
    new_rows = nrows // factor
    new_cols = ncols // factor
    arr_trim = arr[:new_rows * factor, :new_cols * factor]
    mean_val = float(np.nanmean(arr_trim))
    arr_trim = np.where(np.isnan(arr_trim), mean_val, arr_trim)
    coarse = arr_trim.reshape(new_rows, factor, new_cols, factor).mean(axis=(1, 3))
    new_transform = from_origin(
        dem.transform.c, dem.transform.f,
        dem.dx * factor, dem.dy * factor,
    )
    return DEM(elevation=coarse, transform=new_transform, crs=dem.crs, nodata=dem.nodata)


def clip_dem_to_bbox(dem: DEM,
                     x_min: float, x_max: float,
                     y_min: float, y_max: float) -> DEM:
    """Clip DEM to a geographic bounding box, returning a new DEM with an
    updated transform. Coordinates are in the DEM's CRS (metres or degrees).
    """
    dx, dy = dem.dx, dem.dy
    x0 = dem.transform.c
    y_top = dem.transform.f
    nrows, ncols = dem.shape

    col_left  = max(0,     int((x_min - x0)    / dx))
    col_right = min(ncols, int((x_max - x0)    / dx) + 1)
    row_top   = max(0,     int((y_top - y_max) / dy))
    row_bot   = min(nrows, int((y_top - y_min) / dy) + 1)

    clipped = dem.elevation[row_top:row_bot, col_left:col_right]
    new_transform = from_origin(
        x0 + col_left * dx,
        y_top - row_top * dy,
        dx, dy,
    )
    return DEM(elevation=clipped, transform=new_transform, crs=dem.crs, nodata=dem.nodata)


def flood_extent_bbox(max_depth: np.ndarray, dem: DEM,
                      threshold: float = 0.01,
                      buffer_m: float = 200.0,
                      ) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) bounding box of flooded cells
    (depth > threshold) with buffer_m added on all sides.

    Falls back to the full DEM extent if no cell exceeds threshold.
    The returned bbox is clamped to the DEM extent.
    """
    flooded = max_depth > threshold
    if not flooded.any():
        return dem.xy_extent()

    rows, cols = np.where(flooded)
    x0, x1, y0, y_top = dem.xy_extent()
    dx, dy = dem.dx, dem.dy

    x_min = x0 + cols.min() * dx - buffer_m
    x_max = x0 + (cols.max() + 1) * dx + buffer_m
    y_min = y_top - (rows.max() + 1) * dy - buffer_m
    y_max = y_top - rows.min() * dy + buffer_m

    return (
        max(x_min, x0), min(x_max, x1),
        max(y_min, y0), min(y_max, y_top),
    )
