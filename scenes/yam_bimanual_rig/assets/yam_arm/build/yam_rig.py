"""Pose helpers for the YAM arm asset (run inside Blender).

The arm root (an Empty tagged yam_arm_root=1) holds the joint controls as custom properties:
joint1..joint6 in radians (MJCF joint1..joint6) and gripper in metres (per-finger stroke, MJCF joint7
= joint8, 0 closed .. 0.0475 open). Link objects follow through simple-expression drivers, so no
Python auto-run is needed. Resolve links by their 'mjcf_body' property, not by object name.
"""
import bpy

ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
CONTROLS = ARM_JOINTS + ('gripper',)


def roots(scene=None):
    scene = scene or bpy.context.scene
    return [o for o in scene.objects if o.get('yam_arm_root')]


def links(root):
    """{mjcf_body: object} for one arm instance."""
    out = {}
    for obj in [root, *root.children_recursive]:
        name = obj.get('mjcf_body')
        if name is not None:
            if name in out:
                raise ValueError(f'duplicate mjcf_body {name} under {root.name}')
            out[name] = obj
    return out


def set_joints(root, q=None, gripper=None, update=True):
    """Set arm joints (sequence of 6 radians, or a dict by name) and/or the gripper stroke (m)."""
    values = {}
    if q is not None:
        values.update(q if isinstance(q, dict) else dict(zip(ARM_JOINTS, q)))
        if not isinstance(q, dict) and len(q) not in (6, 7):
            raise ValueError('q needs 6 arm joints (or 7 with the gripper)')
        if not isinstance(q, dict) and len(q) == 7:
            values['gripper'] = q[6]
    if gripper is not None:
        values['gripper'] = gripper
    for key, value in values.items():
        if key not in CONTROLS:
            raise KeyError(key)
        lo, hi = root.id_properties_ui(key).as_dict()['min'], root.id_properties_ui(key).as_dict()['max']
        if not lo - 1e-9 <= float(value) <= hi + 1e-9:
            raise ValueError(f'{key}={value} outside [{lo}, {hi}]')
        root[key] = float(value)
    if update:
        root.update_tag()
        bpy.context.view_layer.update()
    return {k: root[k] for k in CONTROLS}


def get_joints(root):
    return {k: float(root[k]) for k in CONTROLS}


# UMI-FT finger tags (ArUco DICT_4X4_50) on the rig, read from IMG_5411 and the ZED scene image: (tip_left, tip_right)
ARM_TAG_IDS = {'left_arm': (7, 6), 'right_arm': (1, 0)}


def set_finger_tags(root, tip_left, tip_right):
    """Set the ArUco id shown on each claw's tag block (one of the tag mesh's material slots, ids 0, 1, 6, 7)."""
    for obj in root.children_recursive:
        if obj.get('claw_part') == 'tag':
            tag_id = tip_left if obj['claw_finger'] == 'tip_left' else tip_right
            slots = [m.get('aruco_id') for m in obj.data.materials]
            obj.data.polygons[0].material_index = slots.index(tag_id)
            obj['aruco_id'] = tag_id


def set_claw_mount(root, roll, update=True):
    """Roll (rad) of the claw pair about the link_6 z axis (asset version 4: 'yam_claw_mount'; 0 = the MJCF gripper)."""
    if 'claw_mount_roll' not in root:
        raise KeyError('this YAM asset has no claw mount (asset version < 4)')
    root['claw_mount_roll'] = float(roll)
    if update:
        root.update_tag()
        bpy.context.view_layer.update()


def setup_arm(root, arm):
    """Per-arm claw state of the rig: ArUco tags and, on asset version 4, the fitted claw mount roll."""
    import json
    set_finger_tags(root, *ARM_TAG_IDS[arm])
    if 'claw_mount_roll_by_arm' in root:
        set_claw_mount(root, json.loads(root['claw_mount_roll_by_arm']).get(arm, 0.0))
