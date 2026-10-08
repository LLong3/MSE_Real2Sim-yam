"""The two YAM followers of the bimanual raiden rig (``rigs/raiden_bimanual``).

Each arm is the single-arm ``RaidenYam`` model (i2rt link offsets and inertials, the fitted dynamics, the wrist
camera from the rig) with its own wrist camera, base pose and joint zero offsets from ``rig.json`` ``arms``.
``rig.json`` ``robot.gripper`` picks the gripper of both arms: ``RaidenYamGripper`` (the stock linear_4310 claws, as
in the single-arm twin) or ``RaidenYamUmiFtGripper``, the same housing, finger joints, actuator and coupling with the
UMI-FT claw geometry and mass in the finger bodies; with it the left arm's claw pair is rolled on its mount
(``RaidenYamUmiFtGripperLeft``). ``robot.geoms`` = ``"i2rt"`` puts the link meshes and collision geoms on i2rt's
meshes (``I2RT_GEOM_SHIFT``); the single-arm twin keeps yam_robot.xml's geom offsets.
"""

from __future__ import annotations

import os
import threading
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from scipy.spatial.transform import Rotation

from mesa.sim.envs.rigs import load_rig

from .raiden_yam import RaidenYam, _write_rig_robot_xml
from .yam_gripper import RaidenYamGripper

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "yam")

# UMI-FT claws (AHa-3D scenes/yam_bimanual_rig/assets/yam_arm/source/umi_ft/mjcf_v4, umi_ft_fingers_snippet.xml; placement
# claw_v4.json, 09-30): meshes in the i2rt finger body frames, which are the tip_left / tip_right frames of raiden_yam_gripper.xml.
# Colours: MuJoCo works in display (sRGB) values, the snippet and Blender in linear ones. Pad and clamp are the
# ZED camera-space fits of the Blender build (delivery_v3.json arm_material_overrides), the rest the snippet's
# colours; all converted linear -> sRGB.
UMI_VISUAL = {  # part: rgba (sRGB)
    "pad": (0.907, 0.507, 0.143, 1), "face": (0.19, 0.19, 0.19, 1), "clamp": (0.835, 0.771, 0.229, 1),
    "white": (0.955, 0.955, 0.945, 1), "black": (0.221, 0.221, 0.235, 1), "carriage": (0.293, 0.297, 0.304, 1),
}
UMI_FINGER = {
    "left": dict(body="tip_left", p="lf", inertial=dict(
        pos="-0.03652 0.01392 -0.05058", mass="0.1016",
        fullinertia="9.0189e-05 2.1182e-04 1.5839e-04 2.9657e-05 -6.0407e-05 2.4805e-05"),
        boxes={"body_outer": ("-0.06336 0.02390 -0.07721", "0.04892 0.01450 0.01303"),
               "body_inner": ("-0.02674 0.02390 -0.05591", "0.01230 0.01450 0.00828")}),
    "right": dict(body="tip_right", p="rf", inertial=dict(
        pos="-0.01410 -0.03652 -0.05058", mass="0.1016",
        fullinertia="2.1181e-04 9.0550e-05 1.5875e-04 -2.9544e-05 -2.5184e-05 -6.0406e-05"),
        boxes={"body_outer": ("-0.02390 -0.06336 -0.07721", "0.01450 0.04892 0.01303"),
               "body_inner": ("-0.02390 -0.02674 -0.05591", "0.01450 0.01230 0.00828")}),
}
# Per-arm claw mount (claw_v4.json arm_mount): both finger bodies rolled about the link_6 z axis (the gripper centre line;
# the linear_4310 body is the link_6 frame). Left arm -4.27 deg from its wrist image (clamp screws), right arm none.
UMI_MOUNT_ROLL_RAD = {"left": -0.074526, "right": 0.0}
UMI_ARM_GRIPPER = {"left": "RaidenYamUmiFtGripperLeft", "right": "RaidenYamUmiFtGripper"}


def _write_umi_ft_gripper_xml(arm: str = "right") -> str:
    """raiden_yam_gripper.xml with the stock claw (visual mesh, hull, tip box, inertial) replaced by the UMI-FT claw.
    The carriage box, joints, damping, friction class, actuator and coupling are kept as they are. With a mount roll
    (UMI_MOUNT_ROLL_RAD), tip_left and tip_right move into a body 'umi_claw_mount' rolled about the link_6 z axis."""
    roll = UMI_MOUNT_ROLL_RAD[arm]
    suffix = f"_{arm}" if roll else ""
    tree = ET.parse(os.path.join(_ASSETS, "raiden_yam_gripper.xml"))
    root = tree.getroot()
    root.set("model", f"raiden_yam_gripper_umi_ft{suffix}")
    asset = root.find("asset")
    for side, f in UMI_FINGER.items():
        for part in list(UMI_VISUAL) + ["pad_collision"]:
            ET.SubElement(asset, "mesh", {"name": f"umi_tip_{side}_{part}", "file": f"meshes/umi_ft/tip_{side}_{part}.stl"})
        body = root.find(f".//body[@name='{f['body']}']")
        inertial = body.find("inertial")
        inertial.attrib.clear()
        inertial.attrib.update(f["inertial"])
        for g in list(body.findall("geom")):  # stock claw: visual mesh, hull, tip box
            if g.get("mesh", "").startswith("linear_4310_tip") or g.get("name") in (f"{f['p']}_finger", f"{f['p']}_tip"):
                body.remove(g)
        for part, rgba in UMI_VISUAL.items():
            ET.SubElement(body, "geom", {"name": f"{f['p']}_umi_{part}_visual", "type": "mesh", "mesh": f"umi_tip_{side}_{part}",
                                         "rgba": " ".join(f"{v:g}" for v in rgba), "contype": "0", "conaffinity": "0", "group": "1"})
        # the pad keeps the stock tip's name, which MESA's jaw-width observable looks up
        ET.SubElement(body, "geom", {"class": "yam_finger", "type": "mesh", "mesh": f"umi_tip_{side}_pad_collision",
                                     "name": f"{f['p']}_tip"})
        for name, (pos, size) in f["boxes"].items():
            ET.SubElement(body, "geom", {"class": "yam_finger", "type": "box", "pos": pos, "size": size, "name": f"{f['p']}_{name}"})
    if roll:
        link6 = root.find(".//body[@name='linear_4310']")
        mount = ET.Element("body", {"name": "umi_claw_mount", "quat": f"{np.cos(roll / 2):.8f} 0 0 {np.sin(roll / 2):.8f}"})
        for name in ("tip_left", "tip_right"):
            body = link6.find(f"body[@name='{name}']")
            link6.remove(body)
            mount.append(body)
        link6.append(mount)
    out = os.path.join(_ASSETS, f"raiden_yam_gripper_umi_ft{suffix}.xml")
    tmp = f"{out}.{os.getpid()}.{threading.get_ident()}.tmp"
    tree.write(tmp)
    os.replace(tmp, out)
    return out


class RaidenYamUmiFtGripper(RaidenYamGripper):
    """The raiden linear_4310 gripper with the bimanual rig's UMI-FT claws (fin-ray pad as a rigid hull)."""

    xml_file = "raiden_yam_gripper_umi_ft.xml"
    arm = "right"

    def __init__(self, idn=0):
        _write_umi_ft_gripper_xml(self.arm)
        super().__init__(idn=idn)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["lf_tip", "lf_body_outer", "lf_body_inner", "lf_carriage"],
            "right_finger": ["rf_tip", "rf_body_outer", "rf_body_inner", "rf_carriage"],
            "left_fingerpad": ["lf_tip"],
            "right_fingerpad": ["rf_tip"],
        }


class RaidenYamUmiFtGripperLeft(RaidenYamUmiFtGripper):
    """The left arm's UMI-FT gripper: the claw pair rolled on its mount (UMI_MOUNT_ROLL_RAD)."""

    xml_file = "raiden_yam_gripper_umi_ft_left.xml"
    arm = "left"


def world_from_metric(rig: dict) -> np.ndarray:
    return np.array(rig["world_from_metric"]["matrix"], dtype=float)


def metric_pose_to_world(rig: dict, T_metric) -> np.ndarray:
    return world_from_metric(rig) @ np.asarray(T_metric, dtype=float)


class RaidenYamBimanual(RaidenYam):
    """One follower of the bimanual rig; ``arm`` selects its entry in ``rig.json`` ``arms``."""

    rig_name = "raiden_bimanual"
    arm = None

    def __init__(self, idn=0):
        rig = load_rig(self.rig_name)
        cfg = rig["arms"][self.arm]
        offsets = cfg["home_joint_offsets_rad"] if rig["robot"]["home_joint_offsets"]["apply"] else None
        xml = _write_rig_robot_xml(self.rig_name, wrist_camera=cfg["wrist_camera"], suffix=f"_{self.arm}",
                                   joint_zero_offsets=offsets)
        ManipulatorModel.__init__(self, xml, idn=idn)
        self._rig = rig

    @property
    def default_gripper(self):
        name = load_rig(self.rig_name)["robot"].get("gripper", "RaidenYamGripper")
        if name == "RaidenYamUmiFtGripper":  # per-arm claw mount
            name = UMI_ARM_GRIPPER[self.arm]
        return {"right": name}

    @property
    def world_from_base(self) -> np.ndarray:
        return metric_pose_to_world(self._rig, self._rig["arms"][self.arm]["metric_from_base"])

    @property
    def world_base_pos(self) -> np.ndarray:
        return self.world_from_base[:3, 3].copy()

    @property
    def world_base_euler(self) -> np.ndarray:
        """xyz Euler angles (robosuite ``set_base_ori``) of the base in the world."""
        return Rotation.from_matrix(self.world_from_base[:3, :3]).as_euler("xyz")


class RaidenYamBimanualLeft(RaidenYamBimanual):
    arm = "left"


class RaidenYamBimanualRight(RaidenYamBimanual):
    arm = "right"
