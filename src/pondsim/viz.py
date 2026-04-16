"""
Matplotlib-based visualisation — no QGIS dependency.

MapCanvas is a QWidget containing an embedded matplotlib figure.
It can display:
  - A DEM hillshade as base layer
  - Raster overlays (depth, level) with pseudocolor + transparent zeros
  - Point source markers
  - Optional basemap tiles via contextily (requires internet)

Usage in Qt:
    canvas = MapCanvas(parent)
    canvas.show_dem(dem)
    canvas.add_overlay(max_depth_array, dem, name="Max Depth", cmap="Blues")
    canvas.add_sources(sources)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PyQt5 import QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from .raster import DEM
from .sources import PointSources


# ---------------------------------------------------------------------------
# Standalone plot functions (no Qt required — usable from scripts)
# ---------------------------------------------------------------------------

def plot_dem(dem: DEM, ax: Optional[plt.Axes] = None,
             cmap: str = "terrain") -> plt.Axes:
    """Render DEM with hillshading onto ax (or a new figure)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    ls = mcolors.LightSource(azdeg=315, altdeg=45)
    elev = dem.elevation
    vmin, vmax = np.nanpercentile(elev, [2, 98])
    hs = ls.shade(elev, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax,
                  blend_mode="overlay", vert_exag=2.0)

    x0, x1, y0, y1 = dem.xy_extent()
    ax.imshow(hs, extent=[x0, x1, y0, y1], origin="upper", aspect="equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return ax


def plot_overlay(data: np.ndarray, dem: DEM,
                 ax: Optional[plt.Axes] = None,
                 cmap: str = "Blues",
                 label: str = "Depth (m)",
                 alpha: float = 0.75,
                 zero_transparent: bool = True) -> plt.Axes:
    """
    Overlay a 2D result array (top-down) on ax with a pseudocolor scheme.
    Values at or below zero are rendered fully transparent.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    masked = np.where(data > 0, data, np.nan) if zero_transparent else data
    vmax = np.nanpercentile(data[data > 0], 98) if (data > 0).any() else 1.0

    x0, x1, y0, y1 = dem.xy_extent()
    img = ax.imshow(
        masked, extent=[x0, x1, y0, y1], origin="upper", aspect="equal",
        cmap=cmap, vmin=0, vmax=vmax, alpha=alpha,
        interpolation="nearest",
    )
    plt.colorbar(img, ax=ax, label=label, shrink=0.7)
    return ax


def plot_sources(sources: PointSources, ax: plt.Axes,
                 color: str = "red", zorder: int = 5) -> plt.Axes:
    if len(sources) == 0:
        return ax
    xs, ys = sources.xy[:, 0], sources.xy[:, 1]
    ax.scatter(xs, ys, c=color, s=30, zorder=zorder, label="Point sources",
               edgecolors="white", linewidths=0.5)
    for nid, x, y in zip(sources.node_ids, xs, ys):
        ax.annotate(nid, (x, y), fontsize=6, color=color,
                    xytext=(3, 3), textcoords="offset points")
    return ax


def plot_hydrographs(df, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot all node hydrographs from the output CSV DataFrame."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    time_col = "time_s"
    for col in df.columns:
        if col == time_col:
            continue
        ax.plot(df[time_col] / 3600, df[col], label=col)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Flow (m³/s)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    return ax


def add_basemap(ax: plt.Axes, crs_epsg: int = 27700, zoom: str = "auto") -> None:
    """Overlay contextily basemap tiles — requires internet access."""
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs=f"EPSG:{crs_epsg}", source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=zoom, alpha=0.5)
    except ImportError:
        pass   # contextily not installed — silently skip
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Basemap unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Qt-embedded canvas widget
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:

    class MapCanvas(QtWidgets.QWidget):
        """
        A self-contained Qt widget with an embedded matplotlib map.

        Typical use:
            canvas = MapCanvas(parent)
            layout.addWidget(canvas)
            canvas.show_dem(dem)
            canvas.add_result_overlay(max_depth, dem, "Max Depth (m)")
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self._fig = Figure(tight_layout=True)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._toolbar = NavToolbar(self._canvas, self)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas)

            self._dem: Optional[DEM] = None

        @property
        def ax(self) -> plt.Axes:
            return self._ax

        def show_dem(self, dem: DEM, cmap: str = "terrain",
                     add_basemap: bool = False) -> None:
            self._ax.clear()
            self._dem = dem
            plot_dem(dem, ax=self._ax, cmap=cmap)
            if add_basemap and dem.crs is not None:
                epsg = dem.crs.to_epsg() or 27700
                add_basemap(self._ax, crs_epsg=epsg)
            self._canvas.draw()

        def add_overlay(self, data: np.ndarray, label: str = "Depth (m)",
                        cmap: str = "Blues") -> None:
            if self._dem is None:
                raise RuntimeError("Call show_dem() before add_overlay()")
            plot_overlay(data, self._dem, ax=self._ax, cmap=cmap, label=label)
            self._canvas.draw()

        def add_sources(self, sources: PointSources) -> None:
            if len(sources):
                plot_sources(sources, self._ax)
            self._canvas.draw()

        def clear(self) -> None:
            self._ax.clear()
            self._canvas.draw()

        def save(self, path: str | Path, dpi: int = 150) -> None:
            self._fig.savefig(path, dpi=dpi, bbox_inches="tight")

    class HydrographCanvas(QtWidgets.QWidget):
        """Embedded hydrograph time-series panel."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._fig = Figure(tight_layout=True)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._toolbar = NavToolbar(self._canvas, self)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas)

        def update(self, df) -> None:
            self._ax.clear()
            plot_hydrographs(df, ax=self._ax)
            self._canvas.draw()
