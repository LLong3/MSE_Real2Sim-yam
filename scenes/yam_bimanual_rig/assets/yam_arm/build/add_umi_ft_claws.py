"""Replace the stock linear_4310 claws of the YAM arm asset with the rig's UMI-FT claws (visual only).

    source kimodo_blender/env.sh
    $BLENDER_BIN -b <stock asset .blend> --python-exit-code 1 --python build/add_umi_ft_claws.py -- --out <new .blend> \
        [--claw source/umi_ft/claw_v4.json]

Input: the asset built by build_yam_arm.py (stock fingers). Parts: source/umi_ft/ (build/umi_ft_parts.py).
The MJCF bodies tip_left/tip_right, their slide joints and drivers are unchanged. Per finger body:
- the stock tip geom (MJCF mesh, still used by fk_check.py) is hidden from render and drawn as wire;
- 'carriage': the stock tip faces above z6 = -70 mm (rack slider and its end block; linear_4310, unchanged);
- the claw: UMI-FT fin-ray pad (its flat contact face in a second, black material slot), UMI-FT clamp, UMI-FT holder
  (clipped, black tag block), a 1.0 mm yellow placeholder in the CoinFT slot, 5 mm adapter to the carriage, ArUco
  tag on the tag block's top face. The holder group (holder, tag, adapter) is shifted by holder_shift_F_m along x_F
  so that the slot closes to the placeholder (claw.json schema 2: the user's claw answers, 09-30).
Claw placement (link_6 frame at raiden home, gripper 0.0475 m open), from the 09-30 wrist/claw fit (README
"Fingers"): pad contact face at y6 = -/+ open_half, finger-frame origin at z6 = pad_z6, pad mid-plane at
x6 = pad_x6 (source/umi_ft/claw.json, shared with umi_ft_mjcf.py and claw_model.py).
The right claw is the left claw mirrored in y6; tag images are not mirrored.
Schema 3 (claw_v4.json, 09-30): a per-arm claw mount: an Empty
'yam_claw_mount' under link_6 (identity at roll 0) holds both finger bodies; its rotation about the link_6 z axis
(the gripper centre line) follows the root property 'claw_mount_roll' (rad; yam_rig.set_claw_mount /
yam_rig.setup_arm with the per-arm values in root['claw_mount_roll_by_arm']). The saved default is 0.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'build'))
import build_yam_arm as B  # noqa: E402  (read_stl, make_material, srgb_to_linear)
import yam_rig  # noqa: E402

UMI = HERE / 'source' / 'umi_ft'
_ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
CLAW_PATH = Path(_ARGV[_ARGV.index('--claw') + 1]) if '--claw' in _ARGV else UMI / 'claw.json'
CLAW_PATH = CLAW_PATH if CLAW_PATH.is_absolute() else HERE / CLAW_PATH
CLAW = json.loads(CLAW_PATH.read_text())
assert CLAW.get('schema') in (2, 3), 'claw.json schema 2 or 3 expected'
MOUNT_ROLL = {arm: float(v['roll_rad']) for arm, v in CLAW.get('arm_mount', {}).items() if isinstance(v, dict)}   # schema 3
OPEN_HALF = CLAW['open_half_m']          # pad contact face to the gripper mid-plane at home
PAD_Z6 = CLAW['pad_z6_m']                # finger-frame origin (UMI finger STL z = 0) along link_6 z
PAD_X6 = CLAW['pad_x6_m']                # pad mid-plane (finger STL y = -8.6 mm) along link_6 x
HOLDER_SHIFT = CLAW['holder_shift_F_m']  # holder group along x_F (closes the CoinFT slot to the placeholder)
HOLDER_GROUP = set(CLAW['holder_group'])
CARRIAGE_CUT_Z6 = CLAW['carriage_cut_z6_m']
HOME_Q = CLAW['home_gripper_m']
PAD_THICK = CLAW['pad_thickness_m']
BOXES = {k: (tuple(b['x']), tuple(b['y']), tuple(b['z']), b['material']) for k, b in CLAW['boxes_F'].items()}
TAG_CENTRE_F = (-0.02885, -0.0231 - 0.00015, -0.0523)   # tag block top face, 1.5 mm from the outer face
TAG_SIZE = 0.0246
ARM_TAG_IDS = dict(left_arm=(7, 6), right_arm=(1, 0))   # (tip_left, tip_right); default below = left arm
R_LEFT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])  # finger frame -> link_6 axes (left claw)

# sRGB base colours by eye from IMG_5411 and the wrist images; AgX renders these saturated colours pale
# (salmon/cream), the Standard view transform keeps them (review renders use Standard).
MATS = {
    'pad': dict(name='UMI-FT | Orange TPU', srgb=(0.90, 0.24, 0.02), rough=0.38, bump=0.05, noise=900.0,
                note='fin-ray pad, TPU 95A (UMI-FT guide): satin, slightly translucent', subsurface=0.1),
    'clamp': dict(name='UMI-FT | Yellow PLA', srgb=(0.96, 0.80, 0.0), rough=0.45, bump=0.08, noise=800.0,
                  note='two-screw finger clamp'),
    'white': dict(name='UMI-FT | White PLA', srgb=(0.90, 0.90, 0.88), rough=0.5, bump=0.1, noise=700.0,
                  note='finger holder, adapter'),
    'black': dict(name='UMI-FT | Black PLA', srgb=(0.04, 0.04, 0.045), rough=0.5, bump=0.06, noise=900.0,
                  note='tag block and adapter'),
    'face': dict(name='UMI-FT | Black contact face', srgb=(0.025, 0.025, 0.028), rough=0.85, bump=0.15, noise=1500.0,
                 note='thin friction sheet on the pad contact face: appearance only, no thickness (user 09-30)'),
}


def T_l6_finger(open_half=OPEN_HALF):
    M = np.eye(4); M[:3, :3] = R_LEFT
    M[:3, 3] = [PAD_X6 + PAD_THICK / 2, -open_half, PAD_Z6]
    return M


def body_T_l6(link):
    """link_6-from-body at home, from the stored MJCF body pose and slide joint (as the drivers do)."""
    pos, quat = Vector(link['body_pos']), Quaternion(link['body_quat_wxyz'])
    d = quat.to_matrix() @ Vector(link['joint_axis'])
    M = np.array(Matrix.Translation(pos + d * HOME_Q) @ quat.to_matrix().to_4x4())
    return M


def box(b):
    (x0, x1), (y0, y1), (z0, z1) = b[:3]
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, f


def to_body(v_f, finger, T_body_l6):
    """finger-frame vertices -> body frame (right finger mirrored in link_6 y)."""
    v = np.c_[v_f, np.ones(len(v_f))] @ T_l6_finger().T
    if finger == 'tip_right':
        v[:, 1] *= -1
    return (v @ np.linalg.inv(T_body_l6).T)[:, :3]


def new_object(name, verts, faces, mat, parent, coll, props, smooth=True, uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts).tolist(), [], [tuple(int(i) for i in f) for f in faces])
    me.update()
    if smooth:
        me.shade_smooth(); me.set_sharp_from_angle(angle=math.radians(35))
    if uv is not None:
        layer = me.uv_layers.new(name='UVMap')
        for loop in me.loops:
            layer.data[loop.index].uv = uv[loop.vertex_index]
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.parent = parent
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_basis = Matrix.Identity(4)
    for k, v in props.items():
        ob[k] = v
    return ob


def tag_material(tag_id, path):
    mat = bpy.data.materials.new(f'UMI-FT | ArUco 4x4_50 id {tag_id}')
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes['Principled BSDF']
    img = bpy.data.images.load(str(path))
    img.pack()
    img.colorspace_settings.name = 'sRGB'
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img; tex.interpolation = 'Closest'; tex.location = (-400, 200)
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.45
    mat.diffuse_color = (0.3, 0.3, 0.3, 1)
    mat['aruco_dictionary'] = 'DICT_4X4_50'
    mat['aruco_id'] = tag_id
    return mat


def add_claw_mount(root, links):
    """Empty between link_6 and both finger bodies; roll about the link_6 z axis from root['claw_mount_roll'] (rad)."""
    link6 = links['link_6']
    mount = bpy.data.objects.new('yam_claw_mount', None)
    root.users_collection[0].objects.link(mount)
    mount.parent = link6
    mount.matrix_parent_inverse = Matrix.Identity(4)
    mount.matrix_basis = Matrix.Identity(4)
    mount.rotation_mode = 'QUATERNION'
    mount.empty_display_size = 0.02
    mount['claw_mount'] = ('per-arm roll of the claw pair (finger bodies with carriage and claw) about the link_6 z axis, '
                           'from the root property claw_mount_roll (claw_v4.json arm_mount; 0 = the MJCF gripper)')
    root['claw_mount_roll'] = 0.0
    root.id_properties_ui('claw_mount_roll').update(min=-0.2, max=0.2, soft_min=-0.2, soft_max=0.2, subtype='ANGLE', precision=4,
                                                     description='claw pair roll about link_6 z (yam_rig.set_claw_mount)')
    root['claw_mount_roll_by_arm'] = json.dumps(MOUNT_ROLL)
    B.add_driver(mount, 'rotation_quaternion', 0, root, 'claw_mount_roll', 'cos(q*0.5)')
    B.add_driver(mount, 'rotation_quaternion', 3, root, 'claw_mount_roll', 'sin(q*0.5)')
    for finger in ('tip_left', 'tip_right'):
        body = links[finger]
        M = body.matrix_world.copy()
        body.parent = mount
        body.matrix_parent_inverse = Matrix.Identity(4)
        bpy.context.view_layer.update()
        d = max(abs(body.matrix_world[i][j] - M[i][j]) for i in range(4) for j in range(4))
        assert d < 1e-6, (finger, d)      # roll 0: the finger bodies keep their MJCF poses
    return mount


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--claw', help='claw config (default source/umi_ft/claw.json); read at import')
    out = Path(ap.parse_args(argv).out)
    if out.exists():
        raise FileExistsError(out)
    parts = json.loads((UMI / 'parts' / 'parts.json').read_text())
    (root,) = yam_rig.roots()
    coll = root.users_collection[0]
    links = yam_rig.links(root)
    mats = {}
    for k, spec in MATS.items():
        m = B.make_material(spec)
        if spec.get('subsurface'):
            bsdf = m.node_tree.nodes['Principled BSDF']
            bsdf.inputs['Subsurface Weight'].default_value = spec['subsurface']
            bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.45, 0.15)
            bsdf.inputs['Subsurface Scale'].default_value = 0.0015
        mats[k] = m
    tag_mats = {int(i): tag_material(int(i), HERE / p) for i, p in parts['tags']['files'].items()}
    black_housing = bpy.data.materials['YAM | Black gripper housing']
    meshes_f = {'pad': B.read_stl(UMI / 'umift_finger_no_contact_mic.stl'),
                'clamp': B.read_stl(UMI / 'parts' / 'clamp.stl'),
                'holder_white': B.read_stl(UMI / 'parts' / 'holder_white.stl'),
                'holder_black': B.read_stl(UMI / 'parts' / 'holder_black.stl')}
    mat_of = dict(pad='pad', clamp='clamp', holder_white='white', holder_black='black')
    shift_F = np.array([HOLDER_SHIFT, 0.0, 0.0])
    for part in ('holder_white', 'holder_black'):
        if part in HOLDER_GROUP:
            meshes_f[part] = (meshes_f[part][0] + shift_F, meshes_f[part][1])
    pv, pf = meshes_f['pad']            # contact face: faces with normal +x_F on x_F = 0 (material slot 1)
    n = np.cross(pv[pf[:, 1]] - pv[pf[:, 0]], pv[pf[:, 2]] - pv[pf[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    pad_face = (n[:, 0] > 0.99) & (pv[pf].mean(1)[:, 0] > -0.0005)
    assert pad_face.sum() == 26, pad_face.sum()
    created = []
    for finger in ('tip_left', 'tip_right'):
        body = links[finger]
        T_bl6 = body_T_l6(body)          # link_6 from body
        # stock geom: keep for FK/MJCF, hide from render; split off the carriage
        (stock,) = [o for o in body.children if o.get('mjcf_geom_mesh') == finger]
        stock.hide_render = True
        stock.display_type = 'WIRE'
        stock['stock_claw_hidden'] = 'replaced by the UMI-FT claw in render; kept for fk_check and the MJCF'
        me = stock.data
        V = np.array([v.co[:] for v in me.vertices])
        G = np.array(stock.matrix_basis)                      # body from geom
        Vb = V @ G[:3, :3].T + G[:3, 3]
        z6 = (Vb @ T_bl6[:3, :3].T + T_bl6[:3, 3])[:, 2]
        F = [p.vertices[:] for p in me.polygons]
        keep = [f for f in F if z6[list(f)].mean() > CARRIAGE_CUT_Z6]
        used = sorted({i for f in keep for i in f}); remap = {o: n for n, o in enumerate(used)}
        created.append(new_object(f'yam_{finger}_carriage', Vb[used], [[remap[i] for i in f] for f in keep], black_housing,
                                  body, coll, dict(claw_part='carriage', claw_finger=finger, asset_part_key=f'carriage_{finger}',
                                                   note=f'stock {finger} faces above z6 = {CARRIAGE_CUT_Z6} m (linear_4310 rack slider)')))
        flip = finger == 'tip_right'
        for part, (v_f, f) in meshes_f.items():
            faces = f[:, ::-1] if flip else f
            ob = new_object(f'yam_{finger}_umi_{part}', to_body(v_f, finger, T_bl6), faces, mats[mat_of[part]], body, coll,
                            dict(claw_part=part, claw_finger=finger, asset_part_key=f'claw_{finger}_{part}'))
            if part == 'pad':
                ob.data.materials.append(mats['face'])
                for i in np.nonzero(pad_face)[0]:
                    ob.data.polygons[int(i)].material_index = 1
                ob['contact_face_material_slot'] = 1
            created.append(ob)
        for part, b in BOXES.items():
            v_f, f = box(b)
            if part in HOLDER_GROUP:
                v_f = v_f + shift_F
            faces = [t[::-1] for t in f] if flip else f
            created.append(new_object(f'yam_{finger}_umi_{part}', to_body(v_f, finger, T_bl6), faces, mats[b[3]], body, coll,
                                      dict(claw_part=part, claw_finger=finger, asset_part_key=f'claw_{finger}_{part}'), smooth=False))
        # tag: square on the tag block top face; image right = +y6, image up = -z6 (toward the fingertip)
        c = (T_l6_finger() @ np.r_[np.array(TAG_CENTRE_F) + (shift_F if 'tag' in HOLDER_GROUP else 0), 1.0])[:3]
        if flip:
            c[1] *= -1
        u, v = np.array([0, 1.0, 0]) * TAG_SIZE / 2, np.array([0, 0, -1.0]) * TAG_SIZE / 2
        corners = np.array([c - u - v, c + u - v, c + u + v, c - u + v])
        corners_b = (np.c_[corners, np.ones(4)] @ np.linalg.inv(T_bl6).T)[:, :3]
        tag_id = ARM_TAG_IDS['left_arm'][0 if finger == 'tip_left' else 1]
        ids = sorted(tag_mats)
        tag = new_object(f'yam_{finger}_umi_tag', corners_b, [(0, 1, 2, 3)], tag_mats[ids[0]], body, coll,
                         dict(claw_part='tag', claw_finger=finger, asset_part_key=f'claw_{finger}_tag', aruco_id=tag_id),
                         smooth=False, uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
        for i in ids[1:]:       # all tag materials travel with an append; the face picks one (yam_rig.set_finger_tags)
            tag.data.materials.append(tag_mats[i])
        tag.data.polygons[0].material_index = ids.index(tag_id)
        created.append(tag)
    root['yam_asset_version'] = 4 if CLAW['schema'] == 3 else 3
    if MOUNT_ROLL:
        add_claw_mount(root, links)
    root['finger_claw'] = ('UMI-FT claw (rig, 09-29; user claw answers 09-30): source/umi_ft; stock tip geoms hidden from render; '
                           'tags DICT_4X4_50, set per arm with yam_rig.set_finger_tags')
    placement = dict(schema=2, open_half_m=OPEN_HALF, pad_z6_m=PAD_Z6, pad_x6_m=PAD_X6,
                     holder_shift_F_m=HOLDER_SHIFT, home_gripper_m=HOME_Q,
                     gap_at_q0_m=round(2 * (OPEN_HALF - HOME_Q), 5), arm_tag_ids=ARM_TAG_IDS)
    if CLAW['schema'] == 3:
        placement.update(schema=3, claw_json=CLAW_PATH.name, placeholder_x_F_m=CLAW['boxes_F']['placeholder']['x'],
                         claw_mount_roll_rad=MOUNT_ROLL)
    root['claw_placement'] = json.dumps(placement)
    txt = bpy.data.texts['yam_rig.py']
    txt.from_string((HERE / 'build' / 'yam_rig.py').read_text())
    bpy.context.view_layer.update()
    # checks: every claw object is parented to its finger body; the pad contact face is where it was placed
    for ob in created:
        assert ob.parent is links[ob['claw_finger']], ob.name
    rel = {}
    for finger in ('tip_left', 'tip_right'):
        pad = bpy.data.objects[f'yam_{finger}_umi_pad']
        M = links['link_6'].matrix_world.inverted() @ pad.matrix_world
        ys = [(M @ pad.data.vertices[i].co).y for p in pad.data.polygons if p.material_index == 1 for i in p.vertices]
        rel[finger] = (min(ys), max(ys))
    print('pad contact faces y6 at home (m):', rel)
    assert abs(rel['tip_left'][1] + OPEN_HALF) < 1e-6 and abs(rel['tip_left'][0] + OPEN_HALF) < 1e-6, rel
    assert abs(rel['tip_right'][0] - OPEN_HALF) < 1e-6 and abs(rel['tip_right'][1] - OPEN_HALF) < 1e-6, rel
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    print('SAVED', out, len(created), 'claw objects')


if __name__ == '__main__':
    main()
