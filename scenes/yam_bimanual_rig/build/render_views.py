"""Render rig-camera views of a saved twin scene (Cycles GPU, background; the source .blend is not modified).

blender -b SCENE.blend --python-exit-code 1 --python render_views.py -- --out DIR \
    [--views scene,left_wrist,right_wrist] [--samples 96] [--no-ids] [--override-desk-white]

Per view writes DIR/<view>.exr (linear Combined, float), <view>.png (Standard view transform),
<view>_ids.exr (object-ID colours from a 1-sample emission override) and ids.json (index -> object/root/material).
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from aha3d.blender.rendering import configure_render  # noqa: E402

VIEWS = {'scene': 'ZED scene camera', 'left_wrist': 'left_wrist_camera (approx)', 'right_wrist': 'right_wrist_camera (approx)'}


def parse():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--views', default='scene')
    p.add_argument('--samples', type=int, default=96)
    p.add_argument('--no-ids', action='store_true')
    p.add_argument('--exposure', type=float, default=0.0, help='view exposure (stops) for the PNG; EXR stays linear')
    p.add_argument('--white', default='', help='comma list of material-name prefixes rendered as white diffuse (albedo calibration)')
    return p.parse_args(argv)


def root_of(o):
    while o is not None and not o.get('instance_id'):
        o = o.parent
    return o


def assign_ids(scene):
    table = {}
    k = 0
    for o in sorted(scene.objects, key=lambda o: o.name):
        if o.type != 'MESH' or o.hide_render:
            continue
        k += 1
        o.color = ((k % 256) / 255.0, (k // 256) / 255.0, 1.0, 1.0)
        r = root_of(o)
        table[k] = dict(object=o.name, root=r['instance_id'] if r else None,
                        materials=[s.material.name for s in o.material_slots if s.material])
    return table


def id_override():
    mat = bpy.data.materials.new('ID override')
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    info = nt.nodes.new('ShaderNodeObjectInfo')
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(info.outputs['Color'], em.inputs['Color'])
    nt.links.new(em.outputs[0], out.inputs['Surface'])
    return mat


def white_override(prefixes):
    """Replace the base colour of matching materials by white (1.0): renders irradiance x (1) for albedo calibration."""
    changed = []
    for mat in bpy.data.materials:
        if not any(mat.name.startswith(p) for p in prefixes) or not mat.use_nodes:
            continue
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                for l in list(n.inputs['Base Color'].links):
                    mat.node_tree.links.remove(l)
                n.inputs['Base Color'].default_value = (1, 1, 1, 1)
                changed.append(mat.name)
    return changed


OVERVIEW = {  # inspection cameras (not saved in the scene): location, target, lens mm (ortho scale for plan)
    'overview_front': ((-2.25, -0.55, 1.75), (-2.25, 1.75, 0.75), 16.0),
    'overview_left': ((-4.45, 0.05, 1.95), (-2.35, 1.75, 0.8), 16.0),
    'overview_right': ((-0.35, 0.2, 1.9), (-2.2, 1.8, 0.85), 16.0),
    'plan': ((-2.25, 1.35, 2.7), (-2.25, 1.35, 0.0), 5.2),
}


def overview_camera(view):
    loc, tgt, lens = OVERVIEW[view]
    cd = bpy.data.cameras.new(view); cam = bpy.data.objects.new(view, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    if view == 'plan':
        cd.type = 'ORTHO'; cd.ortho_scale = lens
    else:
        cd.lens = lens; cd.sensor_width = 36.0
    cd.clip_start, cd.clip_end = 0.02, 50
    return cam


def set_camera(scene, view):
    if view in OVERVIEW:
        cam = bpy.data.objects.get(view) or overview_camera(view)
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = 1280, 800
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
        return cam
    cam = bpy.data.objects[VIEWS[view]]
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = 960, 600
    ax, ay = cam.get('pixel_aspect_xy', [1.0, 1.0])
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = float(ax), float(ay)
    return cam


def check_projection(scene, cam):
    """Compare Blender's projection with the pinhole K for a few world points (pixel-centre convention)."""
    K = cam.get('intrinsics_K') or cam.get('intrinsics_K_960x600')
    K = [list(r) for r in K]
    M = cam.matrix_world
    R = M.to_3x3(); t = M.translation
    errs = []
    for d in [(0, 0, 1), (0.3, 0.2, 1), (-0.4, -0.3, 1), (0.6, -0.35, 1)]:
        pc_cv = Vector(d)  # OpenCV camera coordinates
        pw = t + R @ Vector((pc_cv.x, -pc_cv.y, -pc_cv.z)) * 1.5
        c = world_to_camera_view(scene, cam, pw)
        u_b, v_b = c.x * 960 - 0.5, (1 - c.y) * 600 - 0.5
        u_k = K[0][0] * pc_cv.x / pc_cv.z + K[0][2]
        v_k = K[1][1] * pc_cv.y / pc_cv.z + K[1][2]
        errs.append(max(abs(u_b - u_k), abs(v_b - v_k)))
    return max(errs)


def main():
    args = parse()
    scene = bpy.context.scene
    args.out.mkdir(parents=True, exist_ok=True)
    configure_render(scene, 'material', args.samples, 'cycles', cycles_device='GPU')
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = args.exposure
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.caustics_refractive = False   # no noisy caustics through the partition glass
    scene.cycles.caustics_reflective = False
    white = white_override([p for p in args.white.split(',') if p]) if args.white else []
    report = {'blend': bpy.data.filepath, 'samples': args.samples, 'white_override': white, 'views': {}}
    table = assign_ids(scene)
    (args.out / 'ids.json').write_text(json.dumps(table, indent=1))
    for view in args.views.split(','):
        cam = set_camera(scene, view)
        bpy.context.view_layer.update()
        err = check_projection(scene, cam) if view in VIEWS else None
        img = scene.render.image_settings
        # linear combined
        img.file_format = 'OPEN_EXR'; img.color_depth = '32'; img.exr_codec = 'ZIP'
        scene.render.filepath = str(args.out / f'{view}.exr')
        bpy.ops.render.render(write_still=True)
        # display PNG from the same settings (re-render avoided: save the render result as PNG)
        res = bpy.data.images['Render Result']
        img.file_format = 'PNG'; img.color_depth = '8'; img.color_mode = 'RGB'
        res.save_render(str(args.out / f'{view}.png'), scene=scene)
        entry = dict(camera=cam.name, projection_check_max_px=err,
                     matrix_world=[list(r) for r in cam.matrix_world],
                     pixel_aspect=[scene.render.pixel_aspect_x, scene.render.pixel_aspect_y])
        if not args.no_ids:
            saved = (scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.filter_width, scene.cycles.max_bounces)
            ov = id_override()
            for layer in scene.view_layers:
                layer.material_override = ov
            scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.filter_width, scene.cycles.max_bounces = 1, False, 0.01, 0
            img.file_format = 'OPEN_EXR'; img.color_depth = '32'
            scene.render.filepath = str(args.out / f'{view}_ids.exr')
            bpy.ops.render.render(write_still=True)
            for layer in scene.view_layers:
                layer.material_override = None
            scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.filter_width, scene.cycles.max_bounces = saved
            entry['ids'] = str(args.out / f'{view}_ids.exr')
        report['views'][view] = entry
        print('VIEW_OK', view, 'projection check px', None if err is None else round(err, 4))
    (args.out / 'render.json').write_text(json.dumps(report, indent=1))
    print('RENDER_OK')


main()
