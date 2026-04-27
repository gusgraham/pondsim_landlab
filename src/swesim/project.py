"""
Project save/load — persists the full simulation setup to a .swesim JSON file.

A project stores:
  - Input file paths (absolute)
  - All simulation parameters
  - Output folder path (defaults to <project_dir>/<project_name>/)
  - Reference to last completed run's outputs (for reloading results)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


SWESIM_VERSION = "1.0"
PROJECT_EXTENSION = ".swesim"


@dataclass
class LastRun:
    max_depth_path: str
    max_level_path: str
    node_hydrographs_path: str

    def exists(self) -> bool:
        return Path(self.max_depth_path).exists()


@dataclass
class ProjectParameters:
    simulation_duration_s: float = 7200.0
    use_adaptive_timestep: bool = True
    fixed_timestep_s: float = 60.0
    snapshot_interval_s: float = 300.0
    export_netcdf: bool = False
    fill_sinks: bool = True
    synthetic_volume_m3: float = 500.0
    use_two_pass: bool = False
    coarse_factor: int = 5
    buffer_m: float = 200.0
    manning_n: float = 0.03
    backend: str = "auto"


@dataclass
class Project:
    name: str
    dem_path: Optional[str] = None
    sources_path: Optional[str] = None
    hydrographs_path: Optional[str] = None
    roughness_path: Optional[str] = None
    output_dir: Optional[str] = None
    parameters: ProjectParameters = field(default_factory=ProjectParameters)
    last_run: Optional[LastRun] = None

    def derive_output_dir(self, project_file_path: str | Path) -> str:
        """Default output dir: sibling folder named after the project."""
        return str(Path(project_file_path).parent / self.name)

    @property
    def is_complete(self) -> bool:
        """True if the minimum inputs needed to run are set."""
        return bool(self.dem_path and self.sources_path and self.output_dir)


def save_project(project: Project, path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() != PROJECT_EXTENSION:
        path = path.with_suffix(PROJECT_EXTENSION)

    data = {
        "version": SWESIM_VERSION,
        "name": project.name,
        "dem_path": project.dem_path,
        "sources_path": project.sources_path,
        "hydrographs_path": project.hydrographs_path,
        "roughness_path": project.roughness_path,
        "output_dir": project.output_dir,
        "parameters": asdict(project.parameters),
        "last_run": asdict(project.last_run) if project.last_run else None,
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> Project:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    params_data = data.get("parameters", {})
    parameters = ProjectParameters(**{
        k: v for k, v in params_data.items()
        if k in ProjectParameters.__dataclass_fields__
    })

    last_run = None
    if data.get("last_run"):
        lr = data["last_run"]
        last_run = LastRun(
            max_depth_path=lr["max_depth_path"],
            max_level_path=lr["max_level_path"],
            node_hydrographs_path=lr["node_hydrographs_path"],
        )

    return Project(
        name=data.get("name", path.stem),
        dem_path=data.get("dem_path"),
        sources_path=data.get("sources_path"),
        hydrographs_path=data.get("hydrographs_path"),
        roughness_path=data.get("roughness_path"),
        output_dir=data.get("output_dir"),
        parameters=parameters,
        last_run=last_run,
    )
