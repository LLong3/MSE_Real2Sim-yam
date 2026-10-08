"""Objects of the bimanual raiden rig: an open vial (a wide-mouth bottle) and its screw cap, MJCF primitives sized by
``VIAL_CAPPING``.

The thread is not modelled as such. A hidden chain inside the cap stands for it, with joints and contacts only (no
equality, which robosuite drops from objects):
- ``thread``: a hinge about the cap axis. Its body carries the rest of the chain, whose three balls touch only the
  rim of the vial's neck, where friction holds them. Turning the cap while it sits on the neck turns the hinge, so the
  hinge angle counts the turns made on the neck. It stops at ``turns_to_tight``; past the stop the cap slips on the rim.
- ``thread_descent``: a slide with a follower ball that rides under the cap's thread flank, a helix of plates fixed in
  the cap. The flank rises ``descent`` over the hinge range, so the cap seats ``descent`` lower when turned to tight.
- ``thread_play``: a slide with the rim balls and a spring (below the cap's weight) that pushes them onto the rim, so
  they stay there while the cap is held up to ``_THREAD_PLAY`` above its seat (the thread's axial play).
All of it is plain qpos: replay, saved models and MimicGen carry it with no extra state.

Collision bits: 1 the world, 2 the rim balls, 4 the cap's inner skirt, 8 the cap's outside, 16 the follower. The neck
touches the world, the rim balls and the skirt; the cap's outside touches the world only, so the cap can sit over the
neck; the flank touches the follower only.
"""

import math
import os
import threading
import xml.etree.ElementTree as ET

import numpy as np

from .base_object import BaseObject, register_graspable_object, register_object

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# The user's "vial" (photos IMG_5416-5419, 09-30): a wide-mouth HDPE bottle of sea sand with a black screw cap
# (53-400 or 58-400 type). Measured by the user (09-30): body diameter 75 mm, lid diameter 60 mm, lid height 10 mm,
# 140 mm from the bottom to the top of the lid screwed tight, and the cap moves down about 5 mm from first engagement
# to tight. The rest are estimates: photo proportions, the neck from the lid, guessed masses (lengths m, masses kg).
_PITCH = 0.0254 / 6  # estimate: a 400 finish of 38 mm and up has 6 threads per inch
VIAL_CAPPING = {
    "vial": dict(
        body_radius=0.0375,  # measured
        body_height=0.110, shoulder_height=0.0105,  # photo estimates: bottom to the shoulder; the rounded shoulder
        neck_radius=0.02725,  # estimate: at the thread crest, 5.5 mm under the lid's diameter
        neck_height=0.017,  # shoulder to rim: the measured 140 mm minus the cap's top wall, body and shoulder
        mass=0.30,  # guess: the bottle part-filled with sand
        rgba=(0.92, 0.92, 0.89, 1),
    ),
    "cap": dict(
        radius=0.030, height=0.010,  # measured
        top=0.0025,  # estimate: top wall with the liner
        clearance=0.0008,  # estimate: skirt inside radius minus neck radius (thread slop)
        mass=0.010,  # guess
        rgba=(0.06, 0.06, 0.06, 1),
        descent=0.005,  # measured: how far the cap moves down from first engagement to tight
        turns_to_tight=0.005 / _PITCH,  # estimate from the descent and the pitch: 1.18 turns
    ),
    # screwed on: the hinge within tight_tol_deg of its stop, the cap axis within tilt_deg and xy (m) of the neck axis,
    # and the cap at its tight seat (height above the rim within the thread's play, +-z)
    "screwed_on": dict(tight_tol_deg=10.0, tilt_deg=5.0, xy=0.002, z=0.001),
}

# the rig objects' contact (white cubes, croissant): stiff, so the 50 N claws do not sink into light objects. The
# priority keeps it from being mixed with the claws' default contact, under which the claws sink 7 mm into the cap.
_CONTACT = dict(solimp="0.998 0.998 0.001", solref="0.001 1", friction="0.95 0.3 0.1", priority="1")
# the same stiffness for the thread chain's joint limits: the default soft limit of such light bodies gives way by
# centimetres under the claws' press (slides) and by tens of degrees under the rim's friction (hinge)
_HARD_LIMIT = dict(solreflimit=_CONTACT["solref"], solimplimit=_CONTACT["solimp"])
_THREAD_BALL = 0.0015  # radius of the three rim contacts
_FOLLOWER = 0.001  # radius of the follower under the flank
_HIDDEN_MASS = 0.0001  # each part of the hidden chain: the hinge body, the follower and the three rim balls
_THREAD_PLAY = 0.002  # axial travel of the rim balls below their seat under the top wall
_FLANK_SEGMENT = math.radians(15)  # the flank helix in plates of about this angle
_FLANK_PLATE = 0.001  # their thickness: with the follower, it leaves 1.2 mm between the helix turns
_SKIRT_BOXES = 12
_LEAD_IN = 0.003  # height of the skirt's lead-in
_LEAD_IN_PLATE = 0.0005  # thickness of the lead-in plates
# where the claws hold (the vial's body, the cap's outside): torsional friction stands for the soft pads' contact
# patch; with one point contact per pad, a held object spins freely about the line between the pads
_HELD = dict(condim="4", friction="0.95 0.005 0.0001")
_GLIDE_FRICTION = "0.2 0.005 0.0001"  # skirt on neck: a guide, not a grip
_RIM_FRICTION = "1.0 0.005 0.0001"  # thread body on rim: holds the thread body while the cap turns


def _geom(parent, name, collision=True, **attrs):
    if collision:
        attrs = {**_CONTACT, "group": "0", **attrs}
    else:
        attrs = {"group": "1", "contype": "0", "conaffinity": "0", "mass": "0", **attrs}
    return ET.SubElement(parent, "geom", {"name": name, **{k: _fmt(v) for k, v in attrs.items()}})


def _fmt(v):
    if isinstance(v, str):
        return v
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _write(model: str, build) -> str:
    """Write the object's model.xml: ``build(object_body)`` fills the body; the outer body keeps the placement sites."""
    root = ET.Element("mujoco", model=model)
    outer = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    half = build(ET.SubElement(outer, "body", name="object"))  # (radius, half height) of the bounding cylinder
    for name, pos in (("bottom_site", (0, 0, -half[1])), ("top_site", (0, 0, half[1])),
                      ("horizontal_radius_site", (half[0], half[0], 0))):
        ET.SubElement(outer, "site", name=name, pos=_fmt(pos), size="0.002", rgba="0 0 0 0")
    out_dir = os.path.join(_ASSETS, model)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "model.xml")
    tmp = f"{out}.{os.getpid()}.{threading.get_ident()}.tmp"
    ET.ElementTree(root).write(tmp)
    os.replace(tmp, out)
    return out


def _vial_body(body):
    v = VIAL_CAPPING["vial"]
    h = v["body_height"] + v["shoulder_height"] + v["neck_height"]
    z0 = -h / 2  # the origin is the centre of the vial's height
    top = z0 + v["body_height"]  # where the shoulder starts
    foot = v["body_radius"] / math.sqrt(2) * 0.98
    # a square foot stands flat (box-box contacts); the round side starts 0.5 mm up and takes the claws
    _geom(body, "foot", type="box", size=(foot, foot, 0.0015), pos=(0, 0, z0 + 0.0015), mass=0.2 * v["mass"])
    _geom(body, "body", type="cylinder", size=(v["body_radius"], (v["body_height"] - 0.0005) / 2),
          pos=(0, 0, z0 + 0.0005 + (v["body_height"] - 0.0005) / 2), mass=0.7 * v["mass"], **_HELD)
    # the neck runs down through the shoulder, which has no collision (it is below the cap and the claws)
    neck_len = v["shoulder_height"] + v["neck_height"]
    _geom(body, "neck", type="cylinder", size=(v["neck_radius"], neck_len / 2), pos=(0, 0, h / 2 - neck_len / 2),
          mass=0.1 * v["mass"], contype="0", conaffinity="7", friction=_GLIDE_FRICTION)
    _geom(body, "body_visual", collision=False, type="cylinder", size=(v["body_radius"], v["body_height"] / 2),
          pos=(0, 0, z0 + v["body_height"] / 2), rgba=v["rgba"])
    # a dome that meets the neck at the shoulder height
    dome = v["shoulder_height"] / math.sqrt(1 - (v["neck_radius"] / v["body_radius"]) ** 2)
    _geom(body, "shoulder_visual", collision=False, type="ellipsoid", size=(v["body_radius"], v["body_radius"], dome),
          pos=(0, 0, top), rgba=v["rgba"])
    _geom(body, "neck_visual", collision=False, type="cylinder", size=(v["neck_radius"], neck_len / 2),
          pos=(0, 0, h / 2 - neck_len / 2), rgba=v["rgba"])
    _geom(body, "mouth_visual", collision=False, type="cylinder", size=(v["neck_radius"] - 0.002, 0.0002),
          pos=(0, 0, h / 2), rgba=(0.35, 0.33, 0.28, 1))
    return v["body_radius"], h / 2


def _cap_body(body):
    c = VIAL_CAPPING["cap"]
    r_in = VIAL_CAPPING["vial"]["neck_radius"] + c["clearance"]
    wall = c["radius"] - r_in
    # the outside: a solid cylinder that the claws hold; it does not touch the neck (bit 8). It starts 0.5 mm up: three
    # feet under the wall stand the cap flat (box-box contacts; a cylinder on the desk box gets one contact and sinks)
    _geom(body, "outside", type="cylinder", size=(c["radius"], (c["height"] - 0.0005) / 2), pos=(0, 0, 0.00025),
          mass=c["mass"] - 5 * _HIDDEN_MASS, contype="8", conaffinity="1", **_HELD)
    for i in range(3):
        a = 2 * math.pi * (i + 0.5) / 3
        r = c["radius"] - 0.0004
        _geom(body, f"foot{i}", type="box", size=(0.0003, 0.001, 0.0015), mass=0, contype="8", conaffinity="1",
              pos=(r * math.cos(a), r * math.sin(a), 0.0015 - c["height"] / 2), euler=(0, 0, a))
    _geom(body, "outside_visual", collision=False, type="cylinder", size=(c["radius"], c["height"] / 2), rgba=c["rgba"])
    # the inner skirt: flat plates around the neck, touching only the neck (bit 4). At its foot a lead-in (plates
    # tilted into a funnel, within the wall) centres a cap set down up to clearance + depth off the neck's axis.
    bottom, top = -c["height"] / 2, c["height"] / 2 - c["top"]
    r_mid = r_in + wall / 2
    half_w = 1.05 * r_mid * math.tan(math.pi / _SKIRT_BOXES)
    depth = wall - 0.0003
    tilt = math.atan2(depth, _LEAD_IN)
    r_lead = r_in + depth / 2 + _LEAD_IN_PLATE / 2 * math.cos(tilt)
    z_lead = bottom + _LEAD_IN / 2 + _LEAD_IN_PLATE / 2 * math.sin(tilt)
    for i in range(_SKIRT_BOXES):
        a = 2 * math.pi * i / _SKIRT_BOXES
        ca, sa = math.cos(a), math.sin(a)
        _geom(body, f"skirt{i}", type="box", size=(wall / 2, half_w, (top - bottom - _LEAD_IN) / 2),
              pos=(r_mid * ca, r_mid * sa, (top + bottom + _LEAD_IN) / 2), euler=(0, 0, a), mass=0,
              contype="4", conaffinity="0", friction=_GLIDE_FRICTION)
        _geom(body, f"lead_in{i}", type="box", size=(_LEAD_IN_PLATE / 2, half_w, math.hypot(depth, _LEAD_IN) / 2),
              pos=(r_lead * ca, r_lead * sa, z_lead), mass=0, contype="4", conaffinity="0", friction=_GLIDE_FRICTION,
              xyaxes=(ca * math.cos(tilt), sa * math.cos(tilt), math.sin(tilt), -sa, ca, 0))
    # the hidden thread chain (module docstring). Every joint is at 0 at the start of the thread, the cap on its seat;
    # screwed tight, the descent slide is at ``descent`` and the rim balls' bottoms are at the top wall's underside.
    neck_r = VIAL_CAPPING["vial"]["neck_radius"]
    tight = 2 * math.pi * c["turns_to_tight"]
    thread = ET.SubElement(body, "body", name="thread")  # the hinge body; the follower must not be the cap's child
    ET.SubElement(thread, "inertial", pos="0 0 0", mass=_fmt(_HIDDEN_MASS), diaginertia="1e-08 1e-08 1e-08")
    ET.SubElement(thread, "joint", name="thread", type="hinge", axis="0 0 1", limited="true", range=_fmt((0, tight)),
                  damping="0.0001", frictionloss="0.0001", armature="0.000001", **_HARD_LIMIT)
    follower = ET.SubElement(thread, "body", name="thread_follower")
    ET.SubElement(follower, "joint", name="thread_descent", type="slide", axis="0 0 1", limited="true",
                  range=_fmt((-0.0005, c["descent"] + 0.0005)), damping="0.5", armature="0.002", **_HARD_LIMIT)
    r_f = 0.5 * neck_r
    _geom(follower, "follower", type="sphere", size=_FOLLOWER, mass=_HIDDEN_MASS,
          pos=(r_f, 0, top - c["descent"] - _FOLLOWER),
          contype="16", conaffinity="0", friction=_GLIDE_FRICTION)
    # the spring pushes the rim balls down their play with 0.6-0.9 x the cap's weight: left alone, the cap sits on
    # its seat; held up to the play above it, the balls stay on the rim
    rim = ET.SubElement(follower, "body", name="thread_rim")
    ET.SubElement(rim, "joint", name="thread_play", type="slide", axis="0 0 1", limited="true",
                  range=_fmt((-_THREAD_PLAY, 0)), stiffness=_fmt(0.3 * 9.81 * c["mass"] / _THREAD_PLAY),
                  springref=_fmt(-3 * _THREAD_PLAY), damping="0.5", armature="0.002", **_HARD_LIMIT)
    r = 0.6 * neck_r
    for i in range(3):
        a = 2 * math.pi * i / 3
        _geom(rim, f"thread{i}", type="sphere", size=_THREAD_BALL, mass=_HIDDEN_MASS,
              pos=(r * math.cos(a), r * math.sin(a), top - c["descent"] + _THREAD_BALL), contype="2", conaffinity="0",
              friction=_RIM_FRICTION)
    # the flank: plates along the follower's path (hinge angle = its angle in the cap), their underside rising from
    # descent below the top wall at the hinge's 0 to the top wall at its stop; they touch the follower only (bit 16)
    n = math.ceil((tight + 2 * _FOLLOWER / r_f) / _FLANK_SEGMENT)
    ends = np.linspace(-_FOLLOWER / r_f, tight + _FOLLOWER / r_f, n + 1)
    under = lambda a: np.array([r_f * math.cos(a), r_f * math.sin(a), top - c["descent"] * (1 - a / tight)])
    for i in range(n):
        p0, p1 = under(ends[i]), under(ends[i + 1])
        x = (p1 - p0) / np.linalg.norm(p1 - p0)
        y = np.cross([0, 0, 1], x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        _geom(body, f"flank{i}", type="box", size=(0.525 * np.linalg.norm(p1 - p0), 0.0015, _FLANK_PLATE / 2),
              pos=(p0 + p1) / 2 + _FLANK_PLATE / 2 * z, xyaxes=np.r_[x, y], mass=0, contype="0", conaffinity="16",
              friction=_GLIDE_FRICTION)
    return c["radius"], c["height"] / 2


class RaidenBimanualObject(BaseObject):
    yaw_symmetry = 360  # round

    def __init__(self, name, xml, joints=[dict(type="free", damping="0.0005")]):
        super().__init__(xml, name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
        self.category_name = os.path.basename(os.path.dirname(xml))
        self.rotation = None
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
@register_graspable_object
class RaidenVial(RaidenBimanualObject):
    """The open vial, upright; the origin is the centre of its height."""

    def __init__(self, name="raiden_vial", joints=[dict(type="free", damping="0.0005")]):
        super().__init__(name, _write("raiden_vial", _vial_body), joints)

    @property
    def rim_height(self) -> float:
        """Height of the neck's rim above the origin."""
        v = VIAL_CAPPING["vial"]
        return (v["body_height"] + v["shoulder_height"] + v["neck_height"]) / 2


@register_object
@register_graspable_object
class RaidenVialCap(RaidenBimanualObject):
    """The vial's screw cap, opening down; the origin is the centre of its height. ``thread`` counts its turns on the neck."""

    def __init__(self, name="raiden_vial_cap", joints=[dict(type="free", damping="0.0005")]):
        super().__init__(name, _write("raiden_vial_cap", _cap_body), joints)

    @property
    def thread_joint(self) -> str:
        return self.naming_prefix + "thread"

    @property
    def seated_height(self) -> float:
        """Height of the origin above the rim when the cap is screwed tight and sits on its seat (the top wall on the
        rim). Held, it may ride up to ``_THREAD_PLAY`` higher; at the start of the thread it sits ``descent`` higher."""
        c = VIAL_CAPPING["cap"]
        return -(c["height"] / 2 - c["top"])

    def is_screwed_on(self, sim, vial: RaidenVial) -> bool:
        tol = VIAL_CAPPING["screwed_on"]
        tight = 2 * math.pi * VIAL_CAPPING["cap"]["turns_to_tight"]
        if sim.data.get_joint_qpos(self.thread_joint) < tight - math.radians(tol["tight_tol_deg"]):
            return False
        axis = sim.data.get_body_xmat(vial.root_body)[:, 2]
        d = sim.data.get_body_xpos(self.root_body) - (sim.data.get_body_xpos(vial.root_body) + vial.rim_height * axis)
        along = d @ axis
        tilt = math.degrees(math.acos(np.clip(sim.data.get_body_xmat(self.root_body)[:, 2] @ axis, -1.0, 1.0)))
        return (tilt <= tol["tilt_deg"] and np.linalg.norm(d - along * axis) <= tol["xy"]
                and -tol["z"] <= along - self.seated_height <= _THREAD_PLAY + tol["z"])
