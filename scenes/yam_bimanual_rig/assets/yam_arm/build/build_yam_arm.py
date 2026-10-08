"""Build the poseable YAM arm + linear_4310 gripper asset from raiden's MJCF.

    source kimodo_blender/env.sh
    $BLENDER_BIN -b --factory-startup --python-exit-code 1 \
        --python scenes/yam_bimanual_rig/assets/yam_arm/build/build_yam_arm.py -- --out <new .blend>

Inputs (made by build/mjcf_reference.py and build/material_labels.py with raiden's venv):
source/mjcf_model.json (bodies, joints, XML geom poses, STL paths) and source/face_labels.npz.

Hierarchy: root Empty = MJCF world (arm base) frame; one Empty per MJCF body, parented as in the
MJCF, whose frame is the MuJoCo body frame (xpos/xquat); mesh geoms are children at their XML
pos/quat, carrying the raw STL geometry. Joints are simple-expression drivers reading the root's
joint1..joint6 (rad) and gripper (m) properties. See build/yam_rig.py.
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
PROJECT = HERE.parents[3]
sys.path.insert(0, str(PROJECT / 'src'))
from aha3d.blender.orientation import tag_orientation  # noqa: E402

ASSET_VERSION = 1
# Saved default pose = raiden FOLLOWER_HOME_POS: arm joints 0, gripper 1.0 (open) = MJCF joint7 = joint8 = 0.0475
HOME_GRIPPER = 0.0475
COLLECTION = 'YAM arm (i2rt yam + linear_4310)'
PREFIX = 'yam'
ROOT_NAME = 'yam_arm'

# Appearance from the rig video IMG_5411 and the ZED frames (not measured). sRGB base colours.
MATERIALS = {
    'shell': dict(name='YAM | Light grey shell', srgb=(0.70, 0.71, 0.71), rough=0.5, bump=0.08, noise=900.0,
                  note='link2/link3 twin-tube covers: light grey, matte to slightly satin'),
    'motor': dict(name='YAM | Black motor housing', srgb=(0.06, 0.062, 0.066), rough=0.42, bump=0.06, noise=1200.0,
                  note='joint motors, elbow/wrist housings, link1, link4, link5'),
    'base': dict(name='YAM | Black base', srgb=(0.06, 0.062, 0.065), rough=0.5, bump=0.06, noise=1200.0,
                 note='arm foot (MJCF base mesh)'),
    'gripper': dict(name='YAM | Black gripper housing', srgb=(0.07, 0.072, 0.075), rough=0.45, bump=0.06, noise=1200.0,
                    note='linear_4310 housing'),
    'finger': dict(name='YAM | White printed finger', srgb=(0.9, 0.9, 0.88), rough=0.55, bump=0.12, noise=700.0,
                   note='finger bodies (the real fingers are custom; see README)'),
}


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_material(spec):
    mat = bpy.data.materials.new(spec['name'])
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (500, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (200, 0)
    colour = [srgb_to_linear(c) for c in spec['srgb']] + [1.0]
    bsdf.inputs['Base Color'].default_value = colour
    bsdf.inputs['Roughness'].default_value = spec['rough']
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    # Object-space micro relief and a slight roughness variation (printed/moulded plastic).
    coord = nt.nodes.new('ShaderNodeTexCoord'); coord.location = (-700, 0)
    noise = nt.nodes.new('ShaderNodeTexNoise'); noise.location = (-500, 0)
    noise.inputs['Scale'].default_value = spec['noise']
    noise.inputs['Detail'].default_value = 4.0
    noise.inputs['Roughness'].default_value = 0.55
    nt.links.new(coord.outputs['Object'], noise.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.location = (-250, -200)
    ramp.inputs['To Min'].default_value = spec['rough'] - 0.04
    ramp.inputs['To Max'].default_value = spec['rough'] + 0.04
    nt.links.new(noise.outputs['Fac'], ramp.inputs['Value'])
    nt.links.new(ramp.outputs['Result'], bsdf.inputs['Roughness'])
    bump = nt.nodes.new('ShaderNodeBump'); bump.location = (-250, 150)
    bump.inputs['Strength'].default_value = spec['bump']
    bump.inputs['Distance'].default_value = 0.0002
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    mat.diffuse_color = colour  # viewport solid colour
    mat.roughness = spec['rough']
    mat['yam_material_note'] = spec['note']
    mat['yam_base_color_srgb'] = list(spec['srgb'])
    return mat


def read_stl(path):
    """Binary STL -> (unique vertices, faces) with faces in file order (face i = STL facet i)."""
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    if len(raw) != 84 + 50 * n:
        raise ValueError(f'not a binary STL: {path}')
    rec = np.frombuffer(raw, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
    tri = rec['v'].reshape(-1, 3).astype(np.float64)
    verts, inverse = np.unique(tri, axis=0, return_inverse=True)
    return verts, inverse.reshape(-1, 3)


def make_mesh(name, path, labels=None, slots=('motor',)):
    verts, faces = read_stl(path)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    if len(mesh.polygons) != len(faces):
        raise RuntimeError(f'{name}: face count changed')
    mesh.update()
    mesh.shade_smooth()
    mesh.set_sharp_from_angle(angle=math.radians(35))
    for slot in slots:
        mesh.materials.append(MATS[slot])
    if labels is not None:
        if len(labels) != len(mesh.polygons):
            raise RuntimeError(f'{name}: {len(labels)} labels for {len(mesh.polygons)} faces')
        mesh.polygons.foreach_set('material_index', labels.astype(np.int32))
        mesh.update()
    mesh['stl_source'] = str(path)
    return mesh


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def add_driver(obj, path, index, root, control, expression):
    fc = obj.driver_add(path, index)
    drv = fc.driver
    drv.type = 'SCRIPTED'
    var = drv.variables.new()
    var.name = 'q'
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = 'OBJECT'
    var.targets[0].id = root
    var.targets[0].data_path = f'["{control}"]'
    drv.expression = expression
    for mod in list(fc.modifiers):
        fc.modifiers.remove(mod)
    if not drv.is_simple_expression:
        raise RuntimeError(f'driver needs Python: {expression}')
    return fc


def num(x):
    return f'({x:.12g})'


def build(out_path):
    model = json.loads((HERE / 'source' / 'mjcf_model.json').read_text())
    labels = dict(np.load(HERE / 'source' / 'face_labels.npz'))
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    coll = bpy.data.collections.new(COLLECTION)
    scene.collection.children.link(coll)

    global MATS
    MATS = {key: make_material(spec) for key, spec in MATERIALS.items()}

    root = bpy.data.objects.new(ROOT_NAME, None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.1
    coll.objects.link(root)
    root['yam_arm_root'] = 1
    root['yam_asset_version'] = ASSET_VERSION
    root['mjcf_body'] = 'world'
    root['mjcf_source_json'] = json.dumps(model['source'], sort_keys=True)
    root['description'] = ('i2rt YAM arm + linear_4310 gripper from raiden MJCF; frame = MJCF world/base frame '
                           '(origin at the bottom of the arm foot, +Z up, +X toward the gripper at q=0)')
    joints = {j['name']: j for b in model['bodies'] for j in b['joints']}
    for jn in ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'):
        lo, hi = joints[jn]['range']
        root[jn] = 0.0
        root.id_properties_ui(jn).update(min=lo, max=hi, soft_min=lo, soft_max=hi, subtype='ANGLE', precision=4,
                                         description=f'MJCF {jn} (rad), axis {joints[jn]["axis"]} in its body frame')
    glo, ghi = joints['joint7']['range']
    root['gripper'] = HOME_GRIPPER
    root.id_properties_ui('gripper').update(min=glo, max=ghi, soft_min=glo, soft_max=ghi, subtype='DISTANCE', precision=4,
                                            description='per-finger stroke (m) = MJCF joint7 = joint8; 0 closed, 0.0475 open')
    tag_orientation(root, dict(schema_version=1, front_axis='X', up_axis='Z', symmetry='none', origin='source_root',
                               semantic_front='generic', status='authored',
                               evidence='MJCF world frame of raiden i2rt yam.xml; at q=0 the gripper points +X '
                                        '(grasp_site x=+0.248 m), base foot bottom at z=0'))

    control_of = {f'joint{i}': f'joint{i}' for i in range(1, 7)}
    control_of.update(joint7='gripper', joint8='gripper')
    # equality joint7 = joint8 (MuJoCo: q[joint1] = poly(q[joint2]), polycoef 0 1 0 0 0)
    for eq in model['equality']:
        assert eq['polycoef'] == [0.0, 1.0, 0.0, 0.0, 0.0] and (eq['joint1'], eq['joint2']) == ('joint7', 'joint8'), eq

    objects = {'world': root}
    slot_sets = {'world': ('base',), 'link1': ('motor',), 'link2': ('motor', 'shell'), 'link3': ('motor', 'shell'),
                 'link4': ('motor',), 'link5': ('motor',), 'link_6': ('gripper',),
                 'tip_left': ('finger',), 'tip_right': ('finger',)}
    for body in model['bodies']:
        name = body['name']
        if name == 'world':
            obj = root
        else:
            obj = bpy.data.objects.new(f'{PREFIX}_{name}', None)
            obj.empty_display_type = 'PLAIN_AXES'
            obj.empty_display_size = 0.03
            coll.objects.link(obj)
            obj.parent = objects[body['parent']]
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.rotation_mode = 'QUATERNION'
            pos, quat = body['pos'], body['quat']
            obj.location = pos
            obj.rotation_quaternion = quat
            obj['mjcf_body'] = name
            obj['mjcf_parent'] = body['parent']
            obj['body_pos'] = list(pos)
            obj['body_quat_wxyz'] = list(quat)
            if len(body['joints']) != 1:
                raise RuntimeError(f'{name}: expected one joint')
            (j,) = body['joints']
            if any(abs(v) > 1e-12 for v in j['pos']):
                raise RuntimeError(f'{name}: joint pos must be the body origin')
            control = control_of[j['name']]
            obj['joint_name'] = j['name']
            obj['joint_type'] = j['type']
            obj['joint_axis'] = list(j['axis'])
            obj['joint_range'] = list(j['range'])
            obj['joint_control'] = control
            axis = Vector(j['axis']).normalized()
            if j['type'] == 'hinge':
                # q_body * (cos(q/2), sin(q/2) axis): each component = C*cos(q/2) + S*sin(q/2)
                C = quat
                S = quat_mul(quat, (0.0, axis.x, axis.y, axis.z))
                for i in range(4):
                    add_driver(obj, 'rotation_quaternion', i, root, control,
                               f'{num(C[i])}*cos(q*0.5)+{num(S[i])}*sin(q*0.5)')
            elif j['type'] == 'slide':
                d = Quaternion(quat).to_matrix() @ axis
                for i in range(3):
                    add_driver(obj, 'location', i, root, control, f'{num(pos[i])}+{num(d[i])}*q')
            else:
                raise RuntimeError(j['type'])
        objects[name] = obj
        for k, g in enumerate(body['geoms']):
            mesh = make_mesh(f'{PREFIX}_{g["mesh"]}_mesh', g['file'],
                             labels=labels.get(name), slots=slot_sets[name])
            gobj = bpy.data.objects.new(f'{PREFIX}_{g["mesh"]}_geom', mesh)
            coll.objects.link(gobj)
            gobj.parent = obj
            gobj.matrix_parent_inverse = Matrix.Identity(4)
            gobj.matrix_basis = Matrix.Translation(g['pos']) @ Quaternion(g['quat']).normalized().to_matrix().to_4x4()
            gobj['mjcf_geom_mesh'] = g['mesh']
            gobj['mjcf_geom_pos'] = list(g['pos'])
            gobj['mjcf_geom_quat_wxyz'] = list(g['quat'])
            gobj['stl_md5'] = g['file_md5']
        for s in body['sites']:
            sobj = bpy.data.objects.new(f'{PREFIX}_{s["name"]}', None)
            sobj.empty_display_type = 'ARROWS'
            sobj.empty_display_size = 0.02
            coll.objects.link(sobj)
            sobj.parent = obj
            sobj.matrix_parent_inverse = Matrix.Identity(4)
            sobj.matrix_basis = Matrix.Translation(s['pos']) @ Quaternion(s['quat']).normalized().to_matrix().to_4x4()
            sobj['mjcf_site'] = s['name']

    # Keep the pose helper inside the file for convenience (not auto-run).
    text = bpy.data.texts.new('yam_rig.py')
    text.from_string((HERE / 'build' / 'yam_rig.py').read_text())
    bpy.context.view_layer.update()
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), compress=True)
    print('SAVED', out_path, len(coll.objects), 'objects')


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    build(ap.parse_args(argv).out)
