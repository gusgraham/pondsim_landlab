"""
Hydrograph loading and time-interpolation.

Supports two CSV layouts:

  Long format (preferred, easy to diff):
      node_id,time_s,flow_m3s
      MH_001,0,0.00
      MH_001,60,0.15

  Wide format (ICM default export):
      time_s,MH_001,MH_002,...
      0,0.00,0.00
      60,0.15,0.03

Both are normalised to a dict[node_id -> (times_array, flows_array)].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class HydrographSet:
    """
    Time-varying inflow for one or more ICM overflow nodes.

    times_s:  sorted 1-D array of sample times in seconds
    flows:    dict mapping node_id (str) -> 1-D flow array (m³/s), same length
    """

    times_s: np.ndarray
    flows: dict[str, np.ndarray]

    @property
    def node_ids(self) -> list[str]:
        return list(self.flows.keys())

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    def flow_at(self, node_id: str, t: float) -> float:
        """Linearly interpolate flow for node_id at time t (seconds)."""
        arr = self.flows[node_id]
        return float(np.interp(t, self.times_s, arr, left=0.0, right=0.0))

    def flow_average(self, node_id: str, t_start: float, t_end: float,
                     n_points: int = 4) -> float:
        """Average flow over [t_start, t_end] using n_points quadrature."""
        ts = np.linspace(t_start, t_end, n_points)
        return float(np.mean([self.flow_at(node_id, t) for t in ts]))


def load_hydrographs(path: str | Path) -> HydrographSet:
    """Auto-detect long vs wide format and return a HydrographSet."""
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "node_id" in df.columns:
        return _from_long(df)
    else:
        return _from_wide(df)


def _from_long(df: pd.DataFrame) -> HydrographSet:
    df["time_s"] = pd.to_numeric(df["time_s"])
    df["flow_m3s"] = pd.to_numeric(df["flow_m3s"])
    df["node_id"] = df["node_id"].astype(str)

    times_s = np.sort(df["time_s"].unique())
    flows: dict[str, np.ndarray] = {}
    for node_id, grp in df.groupby("node_id"):
        grp = grp.sort_values("time_s")
        # Reindex to the common time axis so every node covers the full period
        flows[str(node_id)] = np.interp(times_s, grp["time_s"].values,
                                        grp["flow_m3s"].values,
                                        left=0.0, right=0.0)
    return HydrographSet(times_s=times_s, flows=flows)


def _from_wide(df: pd.DataFrame) -> HydrographSet:
    time_col = df.columns[0]   # first column is time
    node_cols = df.columns[1:]
    times_s = df[time_col].astype(float).values
    flows = {str(col): df[col].astype(float).values for col in node_cols}
    return HydrographSet(times_s=times_s, flows=flows)


def make_synthetic_hydrograph(
    node_ids: list[str],
    duration_s: float,
    volumes_m3: dict[str, float] | float = 500.0,
    time_to_peak_s: float | None = None,
    dt_s: float = 60.0,
) -> HydrographSet:
    """
    Generate a unit-hydrograph shaped inflow for testing / fallback.
    Uses the same NRCS shape as the original code.

    volumes_m3 can be:
      - a dict mapping node_id -> volume  (uses per-node volumes)
      - a float applied to all nodes
    """
    if time_to_peak_s is None:
        time_to_peak_s = duration_s / 5.0

    times_s = np.arange(0.0, duration_s + dt_s, dt_s)

    def _unit_shape(Qp: float) -> np.ndarray:
        def _flow(t: float) -> float:
            if t <= 0:
                return 0.0
            if t <= 1.25 * time_to_peak_s:
                return (Qp / 2) * (1 - np.cos(np.pi * t / time_to_peak_s))
            return 4.34 * Qp * np.exp(-1.3 * t / time_to_peak_s)
        return np.array([_flow(t) for t in times_s])

    flows: dict[str, np.ndarray] = {}
    for nid in node_ids:
        vol = volumes_m3[nid] if isinstance(volumes_m3, dict) else float(volumes_m3)
        Qp = vol / (time_to_peak_s * 1.39)
        flows[nid] = _unit_shape(Qp)

    return HydrographSet(times_s=times_s, flows=flows)
