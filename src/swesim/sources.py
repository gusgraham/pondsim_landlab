"""
Point source data model.

A PointSources object ties ICM node IDs to grid node indices on the DTM.
It can be loaded from a GeoJSON/GeoPackage/Shapefile (must have a 'node_id'
attribute and point geometry in the DTM's CRS), or built programmatically.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PointSources:
    """
    Mapping from ICM node_id strings to Landlab grid node indices.

    node_ids:   list of source identifiers (matches HydrographSet keys)
    xy:         (N, 2) array of source coordinates (same CRS as DTM)
    grid_nodes: (N,) array of Landlab RasterModelGrid node indices
    volumes_m3: optional per-node volumes read from source file (e.g. vol_m3)
    skipped:    node_ids that fell outside the DTM extent
    """
    node_ids: list[str]
    xy: np.ndarray                     # shape (N, 2)
    grid_nodes: np.ndarray             # shape (N,) dtype int
    volumes_m3: dict[str, float] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.node_ids)

    def iter(self):
        """Yield (node_id, x, y, grid_node) tuples."""
        for nid, xy, gn in zip(self.node_ids, self.xy, self.grid_nodes):
            yield nid, xy[0], xy[1], int(gn)


def load_sources(
    path: str | Path,
    grid,                    # landlab.RasterModelGrid
    hydrograph_ids: Optional[list[str]] = None,
) -> PointSources:
    """
    Load point sources from a spatial file and snap to the nearest grid node.

    Parameters
    ----------
    path:
        GeoJSON, GeoPackage, or Shapefile.  Must contain a 'node_id' column
        and point geometry in the DTM's CRS (no on-the-fly reprojection here —
        make sure the file matches).
    grid:
        The RasterModelGrid the sources will be injected into.
    hydrograph_ids:
        If provided, warn about any sources whose node_id has no matching
        hydrograph, and vice versa.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required to load spatial source files") from exc

    gdf = gpd.read_file(path)

    if "node_id" not in gdf.columns:
        raise ValueError(f"Source file {path} must contain a 'node_id' column")

    gdf["node_id"] = gdf["node_id"].astype(str)

    # Grid bounding box (Landlab stores node_x / node_y as flattened arrays)
    x_min, x_max = grid.node_x.min(), grid.node_x.max()
    y_min, y_max = grid.node_y.min(), grid.node_y.max()

    # Detect volume column — accept vol_m3 or col_m3 (legacy typo in old data)
    vol_col = None
    for candidate in ("vol_m3", "col_m3"):
        if candidate in gdf.columns:
            vol_col = candidate
            if candidate == "col_m3":
                logger.warning("Reading volumes from 'col_m3' column (legacy name)")
            break

    node_ids: list[str] = []
    xy_list: list[tuple[float, float]] = []
    grid_nodes: list[int] = []
    volumes_m3: dict[str, float] = {}
    skipped: list[str] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            logger.warning("Skipping source %s — no geometry", row["node_id"])
            skipped.append(str(row["node_id"]))
            continue

        x, y = geom.x, geom.y

        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            logger.warning(
                "Skipping source %s at (%.1f, %.1f) — outside DTM extent "
                "(%.1f–%.1f, %.1f–%.1f)",
                row["node_id"], x, y, x_min, x_max, y_min, y_max,
            )
            skipped.append(str(row["node_id"]))
            continue

        nid = str(row["node_id"])
        gn = grid.find_nearest_node((x, y))
        node_ids.append(nid)
        xy_list.append((x, y))
        grid_nodes.append(int(gn))
        if vol_col is not None:
            volumes_m3[nid] = float(row[vol_col])

    if skipped:
        warnings.warn(f"{len(skipped)} source(s) skipped (outside DTM): {skipped}")

    if hydrograph_ids is not None:
        source_set = set(node_ids)
        hydro_set = set(hydrograph_ids)
        no_hydro = source_set - hydro_set
        no_source = hydro_set - source_set
        if no_hydro:
            warnings.warn(f"Sources with no hydrograph (will inject zero): {sorted(no_hydro)}")
        if no_source:
            warnings.warn(f"Hydrographs with no source location (ignored): {sorted(no_source)}")

    return PointSources(
        node_ids=node_ids,
        xy=np.array(xy_list) if xy_list else np.empty((0, 2)),
        grid_nodes=np.array(grid_nodes, dtype=int),
        volumes_m3=volumes_m3,
        skipped=skipped,
    )


def sources_from_xy(
    node_ids: list[str],
    x: list[float],
    y: list[float],
    grid,
) -> PointSources:
    """Convenience builder for scripting / testing — no file needed."""
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    grid_nodes = np.array(
        [grid.find_nearest_node((xi, yi)) for xi, yi in zip(xs, ys)], dtype=int
    )
    return PointSources(
        node_ids=[str(n) for n in node_ids],
        xy=np.column_stack([xs, ys]),
        grid_nodes=grid_nodes,
    )
