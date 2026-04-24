"""
Qt GUI application — no QGIS.

Layout
------
Left panel:  controls (load DEM, load sources, load hydrographs, parameters, run)
Right panel: tabbed MapCanvas (map) + HydrographCanvas (hydrographs)
Bottom:      progress bar + status line
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore, QtGui

from .engine import SimulationConfig, SimulationResult, run_simulation
from .hydrographs import HydrographSet, load_hydrographs, make_synthetic_hydrograph
from .project import (LastRun, Project, ProjectParameters, load_project,
                      save_project, PROJECT_EXTENSION)
from .raster import DEM, read_dem, identify_flood_clusters, clip_dem_to_bbox
from .sources import PointSources, load_sources, sources_from_xy
from .viz import HydrographCanvas, MapCanvas

logger = logging.getLogger(__name__)


class QtStatusLogHandler(logging.Handler):
    """Bridge between standard logging and Qt status label."""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        if record.levelno >= logging.INFO:
            msg = self.format(record)
            # Use QMetaObject to ensure thread-safe UI update
            self.callback(msg)


@dataclass
class SimulationTask:
    dem: DEM
    sources: PointSources
    hydrographs: HydrographSet
    config: SimulationConfig
    label: str = "simulation"


# ---------------------------------------------------------------------------
# Worker thread — keeps the GUI responsive
# ---------------------------------------------------------------------------

class SimWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)   # (fraction, message)
    finished = QtCore.pyqtSignal(object)        # list[SimulationResult] or single SimulationResult
    errored = QtCore.pyqtSignal(str)            # traceback string
    ask_proceed = QtCore.pyqtSignal(int, int, float) # (n_clusters, pixels, pct)

    def __init__(self, work_func: Callable[[list[bool], ProgressCallback, Callable], Any]):
        super().__init__()
        self.work_func = work_func
        self._cancel = [False]
        self._mutex = QtCore.QMutex()
        self._wait_cond = QtCore.QWaitCondition()
        self._proceed_res = False

    def cancel(self):
        self._cancel[0] = True
        self._wait_cond.wakeAll()

    def set_proceed_response(self, res: bool):
        self._mutex.lock()
        self._proceed_res = res
        self._wait_cond.wakeAll()
        self._mutex.unlock()

    def _interactive_cb(self, n, px, pct):
        """Called by the workflow from the worker thread."""
        self.ask_proceed.emit(n, px, pct)
        self._mutex.lock()
        self._wait_cond.wait(self._mutex)
        res = self._proceed_res
        self._mutex.unlock()
        return res

    def run(self):
        try:
            def _proxied_cb(frac, msg):
                self.progress.emit(frac, msg)

            result = self.work_func(self._cancel, _proxied_cb, self._interactive_cb)
            
            if not self._cancel[0]:
                self.finished.emit(result)
        except Exception:
            self.errored.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pondsim — Overland Flow Simulation")
        self.resize(1280, 800)

        self._dem: Optional[DEM] = None
        self._roughness: Optional[np.ndarray] = None
        self._sources: Optional[PointSources] = None
        self._hydrographs: Optional[HydrographSet] = None
        self._worker: Optional[SimWorker] = None

        self._project: Project = Project(name="Untitled")
        self._project_path: Optional[Path] = None
        self._unsaved_changes: bool = False
        self._coarse_result: Optional[SimulationResult] = None

        self._setup_logging()
        self._build_ui()
        self._connect_signals()
        self._update_title()

    def _setup_logging(self):
        # Attach a handler that pipes 'pondsim' info logs to the status label
        self._log_handler = QtStatusLogHandler(self._on_log_emitted)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("pondsim").addHandler(self._log_handler)
        logging.getLogger("pondsim").setLevel(logging.INFO)

    @QtCore.pyqtSlot(str)
    def _on_log_emitted(self, msg: str):
        self.lbl_status.setText(msg)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Menu bar ----
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        act_new = QtWidgets.QAction("New Project", self)
        act_new.setShortcut("Ctrl+N")
        file_menu.addAction(act_new)

        act_open = QtWidgets.QAction("Open Project…", self)
        act_open.setShortcut("Ctrl+O")
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_save = QtWidgets.QAction("Save Project", self)
        act_save.setShortcut("Ctrl+S")
        file_menu.addAction(act_save)

        act_save_as = QtWidgets.QAction("Save Project As…", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        file_menu.addAction(act_save_as)

        self._act_new = act_new
        self._act_open = act_open
        self._act_save = act_save
        self._act_save_as = act_save_as

        # ---- Central layout ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_h = QtWidgets.QHBoxLayout(central)
        main_h.setContentsMargins(8, 8, 8, 8)

        # ---- Left control panel (in a scroll area) ----
        scroll = QtWidgets.QScrollArea()
        scroll.setFixedWidth(340)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        ctrl_widget = QtWidgets.QWidget()
        ctrl_v = QtWidgets.QVBoxLayout(ctrl_widget)
        ctrl_v.setSpacing(6)
        ctrl_v.setContentsMargins(0, 0, 10, 0) # room for scrollbar

        # DEM
        grp_dem = QtWidgets.QGroupBox("Ground Model (DEM)")
        dem_layout = QtWidgets.QVBoxLayout(grp_dem)
        self.lbl_dem = QtWidgets.QLabel("(none loaded)")
        self.lbl_dem.setWordWrap(True)
        self.btn_load_dem = QtWidgets.QPushButton("Load DEM (ASC / TIF)…")
        self.btn_load_roughness = QtWidgets.QPushButton("Load Roughness (Optional)…")
        dem_layout.addWidget(self.lbl_dem)
        dem_layout.addWidget(self.btn_load_dem)
        dem_layout.addWidget(self.btn_load_roughness)
        ctrl_v.addWidget(grp_dem)

        # Point sources
        grp_src = QtWidgets.QGroupBox("Point Sources")
        src_layout = QtWidgets.QVBoxLayout(grp_src)
        self.lbl_sources = QtWidgets.QLabel("(none loaded)")
        self.lbl_sources.setWordWrap(True)
        self.btn_load_sources = QtWidgets.QPushButton("Load Sources (GeoJSON / GPKG / SHP)…")
        src_layout.addWidget(self.lbl_sources)
        src_layout.addWidget(self.btn_load_sources)
        ctrl_v.addWidget(grp_src)

        # Hydrographs
        grp_hyd = QtWidgets.QGroupBox("Hydrographs (ICM overflow CSV) — optional")
        hyd_layout = QtWidgets.QVBoxLayout(grp_hyd)
        self.lbl_hydro = QtWidgets.QLabel("(none — synthetic UH will be used)")
        self.lbl_hydro.setWordWrap(True)
        self.btn_load_hydro = QtWidgets.QPushButton("Load Hydrographs (CSV)…")

        syn_row = QtWidgets.QHBoxLayout()
        syn_row.addWidget(QtWidgets.QLabel("Synthetic vol/node:"))
        self.spn_syn_vol = QtWidgets.QDoubleSpinBox()
        self.spn_syn_vol.setRange(1, 1e6); self.spn_syn_vol.setValue(500)
        self.spn_syn_vol.setSuffix(" m³")
        syn_row.addWidget(self.spn_syn_vol)

        dur_row = QtWidgets.QHBoxLayout()
        dur_row.addWidget(QtWidgets.QLabel("Duration:"))
        self.spn_duration = QtWidgets.QDoubleSpinBox()
        self.spn_duration.setRange(60, 86400); self.spn_duration.setValue(7200)
        self.spn_duration.setSuffix(" s")
        dur_row.addWidget(self.spn_duration)

        hyd_layout.addWidget(self.lbl_hydro)
        hyd_layout.addWidget(self.btn_load_hydro)
        hyd_layout.addLayout(syn_row)
        hyd_layout.addLayout(dur_row)
        ctrl_v.addWidget(grp_hyd)

        # Parameters
        grp_par = QtWidgets.QGroupBox("Simulation Parameters")
        par_form = QtWidgets.QFormLayout(grp_par)

        self.chk_adaptive = QtWidgets.QCheckBox("Use adaptive timestep")
        self.chk_adaptive.setChecked(True)
        par_form.addRow(self.chk_adaptive)

        self.spn_fixed_dt = QtWidgets.QDoubleSpinBox()
        self.spn_fixed_dt.setRange(1, 3600); self.spn_fixed_dt.setValue(60)
        self.spn_fixed_dt.setSuffix(" s"); self.spn_fixed_dt.setEnabled(False)
        par_form.addRow("Fixed timestep:", self.spn_fixed_dt)

        self.spn_snapshot = QtWidgets.QDoubleSpinBox()
        self.spn_snapshot.setRange(30, 3600); self.spn_snapshot.setValue(300)
        self.spn_snapshot.setSuffix(" s")
        par_form.addRow("Snapshot interval:", self.spn_snapshot)

        self.chk_netcdf = QtWidgets.QCheckBox("Export temporal NetCDF")
        par_form.addRow(self.chk_netcdf)

        self.chk_fill = QtWidgets.QCheckBox("Fill sinks (recommended)")
        self.chk_fill.setChecked(True)
        par_form.addRow(self.chk_fill)

        self.spn_manning = QtWidgets.QDoubleSpinBox()
        self.spn_manning.setRange(0.01, 1.0); self.spn_manning.setValue(0.03)
        self.spn_manning.setSingleStep(0.005); self.spn_manning.setDecimals(3)
        par_form.addRow("Manning's n (default):", self.spn_manning)

        self.cmb_backend = QtWidgets.QComboBox()
        from .backends.base import BackendRegistry
        backends = ["auto"] + [info.id for info in BackendRegistry.get_backend_info()]
        self.cmb_backend.addItems(backends)
        par_form.addRow("Solver Backend:", self.cmb_backend)

        ctrl_v.addWidget(grp_par)

        # Two-pass analysis
        grp_twopass = QtWidgets.QGroupBox("Two-pass Analysis (coarse → fine)")
        grp_twopass.setCheckable(True)
        grp_twopass.setChecked(False)
        tp_form = QtWidgets.QFormLayout(grp_twopass)

        self.spn_coarse_factor = QtWidgets.QSpinBox()
        self.spn_coarse_factor.setRange(2, 20)
        self.spn_coarse_factor.setValue(5)
        self.spn_coarse_factor.setSuffix("×")
        tp_form.addRow("Coarsening factor:", self.spn_coarse_factor)

        self.spn_buffer = QtWidgets.QDoubleSpinBox()
        self.spn_buffer.setRange(50, 5000)
        self.spn_buffer.setValue(200)
        self.spn_buffer.setSuffix(" m")
        tp_form.addRow("Clip buffer:", self.spn_buffer)

        self.grp_twopass = grp_twopass
        ctrl_v.addWidget(grp_twopass)

        # Output folder
        grp_out = QtWidgets.QGroupBox("Output")
        out_layout = QtWidgets.QHBoxLayout(grp_out)
        self.txt_output = QtWidgets.QLineEdit()
        self.txt_output.setPlaceholderText("Select output folder…")
        self.btn_output = QtWidgets.QPushButton("…")
        self.btn_output.setFixedWidth(30)
        out_layout.addWidget(self.txt_output)
        out_layout.addWidget(self.btn_output)
        ctrl_v.addWidget(grp_out)

        # Sim stats read-outs
        grp_stats = QtWidgets.QGroupBox("Info")
        stats_form = QtWidgets.QFormLayout(grp_stats)
        self.lbl_duration = QtWidgets.QLabel("—")
        self.lbl_nodes = QtWidgets.QLabel("—")
        self.lbl_grid = QtWidgets.QLabel("—")
        stats_form.addRow("Hydrograph duration:", self.lbl_duration)
        stats_form.addRow("Active nodes:", self.lbl_nodes)
        stats_form.addRow("Grid size:", self.lbl_grid)
        ctrl_v.addWidget(grp_stats)

        ctrl_v.addStretch()

        # Run / Cancel
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_run.setEnabled(False)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        ctrl_v.addLayout(btn_row)

        scroll.setWidget(ctrl_widget)
        main_h.addWidget(scroll)

        # ---- Right visualisation panel ----
        right_v = QtWidgets.QVBoxLayout()

        self.tabs = QtWidgets.QTabWidget()
        self.map_canvas = MapCanvas()
        self.hyd_canvas = HydrographCanvas()
        self.tabs.addTab(self.map_canvas, "Map")
        self.tabs.addTab(self.hyd_canvas, "Hydrographs")
        right_v.addWidget(self.tabs)

        # Progress bar + status
        prog_row = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status = QtWidgets.QLabel("Ready.")
        prog_row.addWidget(self.progress_bar)
        prog_row.addWidget(self.lbl_status, 1)
        right_v.addLayout(prog_row)

        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_v)
        main_h.addWidget(right_container, 1)

    def _connect_signals(self):
        self._act_new.triggered.connect(self._new_project)
        self._act_open.triggered.connect(self._open_project)
        self._act_save.triggered.connect(self._save_project)
        self._act_save_as.triggered.connect(self._save_project_as)
        self.btn_load_dem.clicked.connect(self._load_dem)
        self.btn_load_sources.clicked.connect(self._load_sources)
        self.btn_load_roughness.clicked.connect(self._load_roughness)
        self.btn_load_hydro.clicked.connect(self._load_hydrographs)
        self.btn_output.clicked.connect(self._select_output)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self._cancel)
        self.chk_adaptive.toggled.connect(
            lambda checked: self.spn_fixed_dt.setEnabled(not checked)
        )

    # ------------------------------------------------------------------
    # Project methods
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._unsaved_changes:
            reply = QtWidgets.QMessageBox.question(
                self, "Unsaved changes",
                "Save project before closing?",
                QtWidgets.QMessageBox.Save |
                QtWidgets.QMessageBox.Discard |
                QtWidgets.QMessageBox.Cancel,
            )
            if reply == QtWidgets.QMessageBox.Save:
                if not self._save_project():
                    event.ignore()
                    return
            elif reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()

    def _update_title(self):
        name = self._project.name if self._project else "Untitled"
        marker = " *" if self._unsaved_changes else ""
        self.setWindowTitle(f"Pondsim — {name}{marker}")

    def _mark_changed(self):
        self._unsaved_changes = True
        self._update_title()

    def _new_project(self):
        if self._unsaved_changes:
            reply = QtWidgets.QMessageBox.question(
                self, "Unsaved changes", "Save current project first?",
                QtWidgets.QMessageBox.Save |
                QtWidgets.QMessageBox.Discard |
                QtWidgets.QMessageBox.Cancel,
            )
            if reply == QtWidgets.QMessageBox.Save:
                if not self._save_project():
                    return
            elif reply == QtWidgets.QMessageBox.Cancel:
                return

        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Project", "Project name:", text="Untitled"
        )
        if not ok or not name.strip():
            return

        self._project = Project(name=name.strip())
        self._project_path = None
        self._dem = None
        self._sources = None
        self._hydrographs = None
        self._unsaved_changes = False

        self.lbl_dem.setText("(none loaded)")
        self.lbl_sources.setText("(none loaded)")
        self.lbl_hydro.setText("(none — synthetic UH will be used)")
        self.txt_output.setText("")
        self.map_canvas.clear()
        self._check_can_run()
        self._update_title()

    def _open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", "",
            f"Pondsim projects (*{PROJECT_EXTENSION});;All files (*)"
        )
        if not path:
            return
        try:
            project = load_project(path)
            self._project = project
            self._project_path = Path(path)
            self._unsaved_changes = False
            self._restore_from_project()
            self._update_title()
        except Exception:
            self._show_error("Failed to open project", traceback.format_exc())

    def _save_project(self) -> bool:
        if self._project_path is None:
            return self._save_project_as()
        self._write_project(self._project_path)
        return True

    def _save_project_as(self) -> bool:
        default_name = self._project.name + PROJECT_EXTENSION
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project As", default_name,
            f"Pondsim projects (*{PROJECT_EXTENSION});;All files (*)"
        )
        if not path:
            return False
        self._project_path = Path(path)
        self._write_project(self._project_path)
        return True

    def _write_project(self, path: Path):
        self._snapshot_project()
        save_project(self._project, path)
        self._unsaved_changes = False
        self._update_title()

    def _snapshot_project(self):
        """Copy current UI state into self._project."""
        p = self._project
        p.dem_path = self.lbl_dem.toolTip() or None
        p.sources_path = self.lbl_sources.toolTip() or None
        p.hydrographs_path = self.lbl_hydro.toolTip() or None
        p.output_dir = self.txt_output.text() or None
        p.parameters = ProjectParameters(
            simulation_duration_s=self.spn_duration.value(),
            use_adaptive_timestep=self.chk_adaptive.isChecked(),
            fixed_timestep_s=self.spn_fixed_dt.value(),
            snapshot_interval_s=self.spn_snapshot.value(),
            export_netcdf=self.chk_netcdf.isChecked(),
            fill_sinks=self.chk_fill.isChecked(),
            synthetic_volume_m3=self.spn_syn_vol.value(),
            use_two_pass=self.grp_twopass.isChecked(),
            coarse_factor=self.spn_coarse_factor.value(),
            manning_n=self.spn_manning.value(),
            backend=self.cmb_backend.currentText(),
        )
        p.roughness_path = self.btn_load_roughness.toolTip() if self._roughness is not None else None

    def _restore_from_project(self):
        """Reload all UI state and data from self._project."""
        p = self._project

        # Parameters
        params = p.parameters
        self.chk_adaptive.setChecked(params.use_adaptive_timestep)
        self.spn_fixed_dt.setValue(params.fixed_timestep_s)
        self.spn_snapshot.setValue(params.snapshot_interval_s)
        self.chk_netcdf.setChecked(params.export_netcdf)
        self.chk_fill.setChecked(params.fill_sinks)
        self.spn_syn_vol.setValue(params.synthetic_volume_m3)
        self.spn_duration.setValue(params.simulation_duration_s)
        self.grp_twopass.setChecked(params.use_two_pass)
        self.spn_coarse_factor.setValue(params.coarse_factor)
        self.spn_buffer.setValue(params.buffer_m)
        self.spn_manning.setValue(params.manning_n)
        
        idx = self.cmb_backend.findText(params.backend)
        if idx >= 0:
            self.cmb_backend.setCurrentIndex(idx)

        if p.output_dir:
            self.txt_output.setText(p.output_dir)

        # Reload data files
        if p.dem_path and Path(p.dem_path).exists():
            self._load_dem_from_path(p.dem_path)
        else:
            self._dem = None
            self.lbl_dem.setText("(file not found)" if p.dem_path else "(none loaded)")

        if p.sources_path and Path(p.sources_path).exists() and self._dem is not None:
            self._load_sources_from_path(p.sources_path)
        elif p.sources_path:
            self.lbl_sources.setText("(file not found)")

        if p.hydrographs_path and Path(p.hydrographs_path).exists():
            self._load_hydrographs_from_path(p.hydrographs_path)

        if p.roughness_path and Path(p.roughness_path).exists() and self._dem is not None:
            self._load_roughness_from_path(p.roughness_path)

        # Reload last run results onto map
        if p.last_run and p.last_run.exists() and self._dem is not None:
            try:
                result_depth = read_dem(p.last_run.max_depth_path)
                self.map_canvas.add_overlay(result_depth.elevation,
                                            label="Max Depth (m) — previous run")
                self.lbl_status.setText("Previous run results loaded.")
            except Exception:
                pass

        self._check_can_run()

    # ------------------------------------------------------------------
    # Load handlers
    # ------------------------------------------------------------------

    def _load_dem(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open DEM", "", "Raster files (*.asc *.tif *.tiff);;All files (*)"
        )
        if path:
            self._load_dem_from_path(path)

    def _load_dem_from_path(self, path: str):
        try:
            self._dem = read_dem(path)
            nrows, ncols = self._dem.shape
            self.lbl_dem.setText(f"{Path(path).name}\n{ncols}×{nrows} cells, "
                                 f"{self._dem.dx:.1f} m resolution")
            self.lbl_dem.setToolTip(str(path))
            self.lbl_grid.setText(f"{ncols}×{nrows}  ({self._dem.dx:.1f} m)")
            self.map_canvas.show_dem(self._dem)
            self._mark_changed()
            self._check_can_run()
        except Exception:
            self._show_error("Failed to load DEM", traceback.format_exc())

    def _load_sources(self):
        if self._dem is None:
            QtWidgets.QMessageBox.warning(self, "Load DEM first",
                                          "Please load a DEM before loading point sources.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Point Sources", "",
            "Spatial files (*.geojson *.gpkg *.shp);;All files (*)"
        )
        if path:
            self._load_sources_from_path(path)

    def _load_sources_from_path(self, path: str):
        try:
            from landlab import RasterModelGrid
            nrows, ncols = self._dem.shape
            _grid = RasterModelGrid(
                (nrows, ncols),
                xy_spacing=(self._dem.dx, self._dem.dy),
                xy_of_lower_left=self._dem.xy_of_lower_left,
            )
            hydro_ids = self._hydrographs.node_ids if self._hydrographs else None
            self._sources = load_sources(path, _grid, hydrograph_ids=hydro_ids)
            n = len(self._sources)
            has_vols = bool(self._sources.volumes_m3)
            vol_note = f"\nVolumes read from file ({len(self._sources.volumes_m3)} nodes)" if has_vols else ""
            skip_note = f"\n⚠ {len(self._sources.skipped)} skipped (outside DTM)" if self._sources.skipped else ""
            self.lbl_sources.setText(f"{Path(path).name}\n{n} source(s) snapped to grid{vol_note}{skip_note}")
            self.lbl_sources.setToolTip(str(path))
            self.map_canvas.add_sources(self._sources)
            self.lbl_nodes.setText(str(n))
            self._mark_changed()
            self._check_can_run()
        except Exception:
            self._show_error("Failed to load sources", traceback.format_exc())

    def _load_hydrographs(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Hydrograph CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._load_hydrographs_from_path(path)

    def _load_hydrographs_from_path(self, path: str):
        try:
            self._hydrographs = load_hydrographs(path)
            dur_h = self._hydrographs.duration_s / 3600
            n = len(self._hydrographs.node_ids)
            self.lbl_hydro.setText(f"{Path(path).name}\n{n} node(s), {dur_h:.1f} h duration")
            self.lbl_hydro.setToolTip(str(path))
            self.lbl_duration.setText(f"{dur_h:.2f} h")
            self.hyd_canvas.update(
                pd.DataFrame({
                    "time_s": self._hydrographs.times_s,
                    **self._hydrographs.flows,
                })
            )
            self.tabs.setCurrentIndex(1)
            self._mark_changed()
            self._check_can_run()
        except Exception:
            self._show_error("Failed to load hydrographs", traceback.format_exc())

    def _load_roughness(self):
        if self._dem is None:
            QtWidgets.QMessageBox.warning(self, "Load DEM first", "Please load a DEM before loading a roughness map.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Roughness Map", "", "Raster files (*.asc *.tif *.tiff);;All files (*)"
        )
        if path:
            self._load_roughness_from_path(path)

    def _load_roughness_from_path(self, path: str):
        try:
            from .raster import read_dem
            rough_data = read_dem(path)
            if self._dem and rough_data.shape != self._dem.shape:
                QtWidgets.QMessageBox.warning(self, "Grid mismatch", 
                    f"Roughness grid {rough_data.shape} does not match DEM {self._dem.shape}.")
                return
            self._roughness = rough_data.elevation
            self.btn_load_roughness.setText(f"Roughness: {Path(path).name}")
            self.btn_load_roughness.setToolTip(str(path))
            self._mark_changed()
        except Exception:
            self._show_error("Failed to load roughness", traceback.format_exc())

    def _select_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.txt_output.setText(folder)
            self._mark_changed()
            self._check_can_run()

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------

    def _check_can_run(self):
        ready = (
            self._dem is not None
            and self._sources is not None
            and bool(self.txt_output.text())
        )
        self.btn_run.setEnabled(ready)
        # Synthetic controls only active when no CSV loaded
        has_csv = self._hydrographs is not None
        has_shp_vols = bool(self._sources and self._sources.volumes_m3)
        self.spn_syn_vol.setEnabled(not has_csv and not has_shp_vols)
        self.spn_duration.setEnabled(not has_csv)

    def _run(self):
        from .workflow import SimulationWorkflow
        
        # Prepare inputs
        hydrographs = self._hydrographs
        if hydrographs is None:
            duration_s = self.spn_duration.value()
            volumes = self._sources.volumes_m3 if self._sources.volumes_m3 else self.spn_syn_vol.value()
            hydrographs = make_synthetic_hydrograph(self._sources.node_ids, duration_s, volumes)
        
        config = SimulationConfig(
            output_dir=self.txt_output.text(),
            simulation_duration_s=hydrographs.duration_s,
            manning_n=self._roughness if self._roughness is not None else self.spn_manning.value(),
            backend=self.cmb_backend.currentText(),
            fixed_timestep_s=None if self.chk_adaptive.isChecked() else self.spn_fixed_dt.value(),
            snapshot_interval_s=self.spn_snapshot.value(),
            export_netcdf=self.chk_netcdf.isChecked(),
            fill_sinks=self.chk_fill.isChecked(),
        )

        wf = SimulationWorkflow(self._dem, self._sources, hydrographs, self._roughness)
        
        if self.grp_twopass.isChecked():
            factor = self.spn_coarse_factor.value()
            buffer_m = self.spn_buffer.value()
            work = lambda cancel, prog, inter: wf.run_two_pass(config, factor, buffer_m, prog, cancel, inter)
        else:
            work = lambda cancel, prog, _inter: wf.run_one_pass(config, prog, cancel)
            
        self._start_worker(work)

    def _start_worker(self, work_func):
        """Launch a SimWorker for the given workflow function."""
        self._worker = SimWorker(work_func)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.errored.connect(self._on_error)
        self._worker.ask_proceed.connect(self._on_ask_proceed)
        
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self._worker.start()

    def _on_ask_proceed(self, n, px, pct):
        msg = (
            f"Active flood extent identified ({n} disjoint cluster(s)).\n\n"
            f"Total fine-pass area: {px:,} cells ({pct:.1f}% of full DEM)\n"
            f"Proceed with fine-resolution simulation?"
        )
        reply = QtWidgets.QMessageBox.question(
            self, "Coarse pass complete", msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        self._worker.set_proceed_response(reply == QtWidgets.QMessageBox.Yes)

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Cancelling…")

    def _on_progress(self, frac: float, msg: str):
        if frac == 0.0 and "sinks" in msg.lower():
            self.progress_bar.setRange(0, 0)   # indeterminate / pulsing
        else:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)  # restore determinate
            self.progress_bar.setValue(int(frac * 100))
        self.lbl_status.setText(msg)

    def _on_finished(self, result: SimulationResult):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        # Show results on map (the workflow returns the final stitched result)
        self.map_canvas.show_dem(result.dem)
        if self._sources is not None:
            self.map_canvas.add_sources(self._sources)
        self.map_canvas.add_overlay(result.max_depth, label="Max Depth (m)")
        self.tabs.setCurrentIndex(0)

        # Show hydrographs
        self.hyd_canvas.update(result.node_hydrographs)

        # Record last run in project
        self._project.last_run = LastRun(
            max_depth_path=str(result.output_dir / "max_depth.tif"),
            max_level_path=str(result.output_dir / "max_level.tif"),
            node_hydrographs_path=str(result.output_dir / "node_hydrographs.csv"),
        )
        if self._project_path:
            self._write_project(self._project_path)
        else:
            self._unsaved_changes = True

        QtWidgets.QMessageBox.information(
            self, "Simulation complete",
            f"Results written to:\n{result.output_dir}"
        )

    def _on_error(self, tb: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Error — see details.")
        self._show_error("Simulation error", tb)

    def _show_error(self, title: str, detail: str):
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(title)
        dlg.setDetailedText(detail)
        dlg.setIcon(QtWidgets.QMessageBox.Critical)
        dlg.exec_()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    # Silence Numba internal logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Pondsim")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
