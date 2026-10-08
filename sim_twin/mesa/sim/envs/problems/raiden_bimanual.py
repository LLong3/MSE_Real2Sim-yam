"""Digital clone of the bimanual raiden rig: two YAM followers on a rail, the desk, the room exported from the
AHa-3D Blender twin, the ZED X scene camera and a ZED X One on each wrist.

Same machinery as ``Raiden_Lab_Tabletop_Manipulation`` (BDDL regions, predicates, samplers, the YAM motor loop);
the arena, the arm bases and the cameras come from ``rigs/raiden_bimanual``. Robots are robosuite's multi-robot
list, as in BiMESA: ``robot0`` = left arm, ``robot1`` = right arm.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

from mesa.sim.envs.arenas.raiden_bimanual_arena import RaidenBimanualArena, desk_geometry, world_from_metric
from mesa.sim.envs.rigs import cv_pose_to_mujoco, load_rig, mujoco_camera_attribs

from ..bddl_base_domain import TASK_MAPPING, register_problem
from . import raiden_lab  # noqa: F401  (registers the single-arm problem)
from .mimiclabs_tabletop_manipulation import MimicLabs_Tabletop_Manipulation_Base

# register_problem returns None, so the class comes from the registry
Raiden_Lab_Tabletop_Manipulation = TASK_MAPPING["raiden_lab_tabletop_manipulation"]

BIMANUAL_ROBOTS = ["RaidenYamBimanualLeft", "RaidenYamBimanualRight"]


@register_problem
class Raiden_Bimanual_Tabletop_Manipulation(Raiden_Lab_Tabletop_Manipulation):
    rig_name = "raiden_bimanual"

    def __init__(self, **kwargs):
        rig = load_rig(self.rig_name)
        size, offset = desk_geometry(rig)
        self.table_full_size = size
        self.table_offset = offset
        kwargs.setdefault("robots", list(BIMANUAL_ROBOTS))
        kwargs["workspace_offset"] = offset
        # skip Raiden_Lab's __init__: its table comes from the single-arm layout atlas
        MimicLabs_Tabletop_Manipulation_Base.__init__(self, **kwargs)

    def _setup_camera(self, mujoco_arena):
        super(Raiden_Lab_Tabletop_Manipulation, self)._setup_camera(mujoco_arena)
        rig = self.rig
        W = world_from_metric(rig)
        for name, cam in rig["cameras"].items():
            if "T_metric_cam" not in cam:
                continue  # the wrist cameras live in the robot models
            pos, quat = cv_pose_to_mujoco(W @ np.array(cam["T_metric_cam"]))
            attribs = mujoco_camera_attribs(cam)
            attribs.pop("fovy", None)  # the intrinsic model supersedes fovy
            mujoco_arena.set_camera(camera_name=name, pos=pos, quat=quat, camera_attribs=attribs)

    def _load_model(self):
        ManipulationEnv._load_model(self)
        for robot in self.robots:
            robot.robot_model.set_base_xpos(tuple(robot.robot_model.world_base_pos))
            robot.robot_model.set_base_ori(tuple(robot.robot_model.world_base_euler))

        mujoco_arena = RaidenBimanualArena(rig_name=self.rig_name)
        mujoco_arena.set_origin([0, 0, 0])
        self._setup_camera(mujoco_arena)
        self._load_fixtures_in_arena(mujoco_arena)
        self._load_objects_in_arena(mujoco_arena)
        self._load_sites_in_arena(mujoco_arena)
        self._generate_object_state_wrapper()
        self._setup_placement_initializer(mujoco_arena)
        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())
        # no _randomize_lighting_dir: it rewrites the first light (castshadow, direction); the rig sets its own
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.fixtures,
        )
        head = self.rig["layout"].get("lights", {}).get("headlight")
        if head:
            ET.SubElement(self.model.root.find("visual"), "headlight", {k: array_to_string(v) for k, v in head.items()})
        for fixture in self.fixtures:
            self.model.merge_assets(fixture)
