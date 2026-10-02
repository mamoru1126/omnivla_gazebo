"""シミュレーション系ノード (explorer / topomap / eval) の共通処理."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import ros_utils  # noqa: F401  (sys.path 設定のため)
from omnivla_nav.sim_map import OccupancyGrid, PathPlanner, build_occupancy_from_sdf, robot_spawn_pose, world_name


def resolve_world_sdf(world: str, world_sdf: str = "") -> str:
    if world_sdf:
        return os.path.abspath(world_sdf)
    from ament_index_python.packages import get_package_share_directory

    path = os.path.join(get_package_share_directory("omnivla_gazebo"), "worlds", f"{world}.sdf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"world SDF not found: {path}")
    return path


@dataclass
class SimMap:
    sdf_path: str
    world: str
    grid: OccupancyGrid
    planner: PathPlanner
    spawn: Optional[Tuple[float, float, float]]
    main_component: int
    robot_radius: float

    def clearance(self, x: float, y: float) -> float:
        return self.grid.clearance(x, y)

    def in_collision(self, x: float, y: float, factor: float = 0.9) -> bool:
        """ロボット中心から最寄り障害物までの距離が robot_radius*factor 未満なら衝突とみなす."""
        return self.clearance(x, y) < self.robot_radius * factor

    def sample_pose(self, rng: np.random.Generator, min_clearance: float) -> Tuple[float, float, float]:
        x, y = self.planner.sample_free(rng, component=self.main_component, min_clearance=min_clearance)
        return x, y, float(rng.uniform(-math.pi, math.pi))


def load_sim_map(world: str, world_sdf: str = "", resolution: float = 0.05, robot_radius: float = 0.25,
                 safety_margin: float = 0.15, robot_name: str = "omnivla_robot") -> SimMap:
    path = resolve_world_sdf(world, world_sdf)
    grid = build_occupancy_from_sdf(path, resolution=resolution, exclude_models=(robot_name,))
    planner = PathPlanner(grid, inflate=robot_radius + safety_margin)
    spawn = robot_spawn_pose(path, robot_name)
    comp = planner.component(*spawn[:2]) if spawn is not None else 0
    if comp == 0:
        # スポーン位置が使えない場合は最大の連結成分
        labels = planner.labels
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        comp = int(np.argmax(counts))
    return SimMap(path, world_name(path), grid, planner, spawn, comp, robot_radius)
