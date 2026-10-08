"""Arena of the bimanual raiden rig: the room exported from the AHa-3D Blender twin, the desk top and the rail.

The room is visual only (OBJ meshes in the metric frame, placed by ``rig.json`` ``world_from_metric``). Collision:
the desk top (the TableArena table box) and the rail box, plus the floor plane at the real floor height.
"""

from __future__ import annotations

import json
import os

import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string, new_element
from scipy.spatial.transform import Rotation

from mesa.sim.envs.rigs import load_rig

_SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "raiden_lab.xml")


def world_from_metric(rig: dict) -> np.ndarray:
    return np.array(rig["world_from_metric"]["matrix"], dtype=float)


def to_world(rig: dict, p_metric) -> np.ndarray:
    W = world_from_metric(rig)
    return W[:3, :3] @ np.asarray(p_metric, float) + W[:3, 3]


def _quat_wxyz(R) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def box_in_world(rig: dict, x, y, z):
    """(size, centre) in the world of a metric axis-aligned box; the frames differ by a z rotation of 90 deg steps."""
    lo, hi = to_world(rig, [x[0], y[0], z[0]]), to_world(rig, [x[1], y[1], z[1]])
    return np.abs(hi - lo), (hi + lo) / 2


def desk_geometry(rig: dict):
    """(table_full_size, table_offset) for TableArena: the desk top box, table_offset at the top-face centre."""
    d = rig["table"]["desk_box_metric"]
    size, centre = box_in_world(rig, d["x"], d["y"], [d["top_z"] - d["thickness"], d["top_z"]])
    top = centre + np.array([0.0, 0.0, size[2] / 2])
    return tuple(float(v) for v in size), tuple(float(v) for v in top)


class RaidenBimanualArena(TableArena):
    """Desk-top table (collision, hidden) inside the exported room."""

    def __init__(self, rig_name="raiden_bimanual", table_friction=None, **kwargs):
        rig = load_rig(rig_name)
        size, offset = desk_geometry(rig)
        friction = table_friction if table_friction is not None else tuple(rig["table"].get("friction", (0.6, 0.005, 0.0001)))
        super().__init__(table_full_size=size, table_friction=friction, table_offset=offset, has_legs=False, xml=_SCENE_XML)
        self.rig = rig
        self._hide_table_visuals()
        self._place_floor()
        self._add_room()
        self._add_rail()
        self._apply_lights()

    def _hide_table_visuals(self):
        # the exported desk mesh carries the look; group 3 is not rendered
        for name in ("table_visual", "table_top_visual"):
            g = self.table_body.find(f"./geom[@name='{name}']")
            if g is not None:
                g.set("group", "3")
                g.set("rgba", "0 0 0 0")
                g.attrib.pop("material", None)

    def _place_floor(self):
        z = to_world(self.rig, [0.0, 0.0, self.rig["layout"]["room"].get("floor_z_metric", 0.0)])[2]
        self.floor.set("pos", array_to_string([0.0, 0.0, z]))
        self.floor.set("group", "3")  # collision only; the carpet mesh is the visual

    def _add_room(self):
        room_cfg = self.rig["layout"]["room"]
        room_dir = os.path.join(self.rig["dir"], os.path.dirname(room_cfg["manifest"]))
        manifest = json.load(open(os.path.join(self.rig["dir"], room_cfg["manifest"])))
        W = world_from_metric(self.rig)
        body = new_element(tag="body", name="room", pos=array_to_string(W[:3, 3]), quat=array_to_string(_quat_wxyz(W[:3, :3])))
        for g in manifest["geoms"]:
            name = f"room_{g['name']}"
            self.asset.append(new_element(tag="mesh", name=name, file=os.path.join(room_dir, g["mesh"])))
            attrs = dict(name=f"{name}_visual", type="mesh", mesh=name, contype="0", conaffinity="0", group="1")
            if "texture" in g:
                # colorspace linear = use the PNG values as they are (display values, like rgba); Blender's PNGs carry an
                # sRGB chunk, for which MuJoCo's "auto" would linearise them and render the room dark and saturated
                self.asset.append(new_element(tag="texture", name=f"tex-{name}", type="2d", colorspace="linear",
                                              file=os.path.join(room_dir, g["texture"])))
                # baked lightmaps: emission 1 minus the light the MuJoCo lights add there (layout.json room_emission)
                emission = self.rig["layout"].get("room_emission", {}).get(g["name"], g.get("emission", 0.0))
                self.asset.append(new_element(tag="material", name=f"{name}_mat", texture=f"tex-{name}", texuniform="false",
                                              reflectance="0", shininess="0", specular="0", emission=f"{emission:g}"))
                attrs["material"] = f"{name}_mat"
            else:
                spec = 0.3 if g.get("metallic", 0) > 0.5 else 0.1
                self.asset.append(new_element(tag="material", name=f"{name}_mat", rgba=array_to_string(g["rgba"]),
                                              reflectance="0", shininess=f"{max(0.0, 1 - g.get('roughness', 1.0)) * 0.5:.3f}",
                                              specular=f"{spec:g}", emission=f"{g.get('emission', 0.0):g}"))
                attrs["material"] = f"{name}_mat"
            body.append(new_element(tag="geom", **attrs))
        self.worldbody.append(body)
        bg = self.rig["layout"].get("background", {})
        if "skybox_rgb" in bg:
            sky = self.asset.find("./texture[@type='skybox']")
            sky.set("rgb1", array_to_string(bg["skybox_rgb"]))
            sky.set("rgb2", array_to_string(bg["skybox_rgb"]))

    def _add_rail(self):
        r = self.rig["rail"]["box_metric"]
        size, centre = box_in_world(self.rig, r["x"], r["y"], r["z"])
        self.worldbody.append(new_element(tag="geom", name="rail_collision", type="box", pos=array_to_string(centre),
                                          size=array_to_string(size / 2), group="3", rgba="0.6 0.6 0.6 1",
                                          friction=array_to_string(self.rig["table"].get("friction", (0.6, 0.005, 0.0001)))))

    def _apply_lights(self):
        cfg = self.rig["layout"].get("lights", {})
        for light in list(self.worldbody.findall("./light")):
            self.worldbody.remove(light)
        W = world_from_metric(self.rig)
        for s in cfg.get("spots", []):
            pos = to_world(self.rig, s["pos_metric"])
            d = W[:3, :3] @ np.asarray(s["dir_metric"], float)
            diffuse = np.asarray(s["colour"], float) * s.get("diffuse", cfg.get("spot_diffuse", 0.3))
            self.worldbody.append(new_element(
                tag="light", name=f"light_{s['name']}", pos=array_to_string(pos), dir=array_to_string(d), directional="false",
                diffuse=array_to_string(diffuse), specular=array_to_string([cfg.get("spot_specular", 0.05)] * 3),
                cutoff=str(cfg.get("spot_cutoff_deg", 80)), exponent=str(cfg.get("spot_exponent", 1)),
                castshadow=str(bool(s.get("castshadow", False))).lower()))
