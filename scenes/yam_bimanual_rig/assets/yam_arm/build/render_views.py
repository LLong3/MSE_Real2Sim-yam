"""Render review stills of the YAM asset (Cycles, short GPU renders or CPU).

    $BLENDER_BIN -b <asset.blend> --python-exit-code 1 --python build/render_views.py -- \
        --views views.json --outdir DIR [--samples 48] [--device CPU]

views.json: list of {name, width, height, q (6), gripper, and either
  {location, look_at, lens_mm} or {K (3x3 px), T_cam_arm (4x4, OpenCV camera -> arm root frame)}},
optional "mode": "material" (default) | "clay" | "mask" (white silhouette on black).
Writes DIR/<name>.png with a transparent background (material/clay) or an opaque mask.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'build'))
import yam_rig  # noqa: E402

EXPOSURE = -1.0


def setup_scene(samples, device):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.device = device
    if device == 'GPU':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'OPTIX'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == 'OPTIX'
    scene.render.film_transparent = True
    # Standard keeps the saturated claw colours (AgX renders the orange/yellow pale)
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = EXPOSURE
    world = bpy.data.worlds.new('review_world')
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.8, 0.8, 0.82, 1)
    bg.inputs['Strength'].default_value = 0.55
    scene.world = world
    # soft office-like key from above-front and a fill
    for name, loc, energy, size in (('key', (0.6, -0.4, 1.6), 260, 1.6), ('fill', (-0.8, 0.9, 1.2), 90, 2.0)):
        light = bpy.data.lights.new(name, 'AREA')
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = loc
        direction = Vector((0, 0, 0.15)) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam = bpy.data.objects.new('review_cam', bpy.data.cameras.new('review_cam'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    return scene, cam


def clay_override(scene, on):
    if on:
        mat = bpy.data.materials.new('clay')
        mat.use_nodes = True
        mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.6, 0.6, 0.6, 1)
        scene.view_layers[0].material_override = mat
    else:
        scene.view_layers[0].material_override = None


def place_camera(scene, cam, root, v):
    W, H = v['width'], v['height']
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    data = cam.data
    data.sensor_fit = 'HORIZONTAL'
    data.sensor_width = 36.0
    data.clip_start, data.clip_end = 0.01, 50
    if 'K' in v:
        K = v['K']
        data.lens = K[0][0] * data.sensor_width / W
        # OpenCV pixel centres at integers: an unshifted camera's optical centre is ((W-1)/2, (H-1)/2)
        data.shift_x = -(K[0][2] - (W - 1) / 2) / W
        data.shift_y = (K[1][2] - (H - 1) / 2) / W
        T = Matrix(v['T_cam_arm'])
        cv_to_blender = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
        cam.matrix_world = root.matrix_world @ T @ cv_to_blender
    else:
        data.lens = v['lens_mm']
        data.shift_x = data.shift_y = 0.0
        loc = root.matrix_world @ Vector(v['location'])
        target = root.matrix_world @ Vector(v['look_at'])
        cam.matrix_world = Matrix.Translation(loc) @ (target - loc).to_track_quat('-Z', 'Y').to_matrix().to_4x4()


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--device', default='CPU')  # GPU (OptiX) only when no other GPU job runs
    args = ap.parse_args(argv)
    views = json.loads(Path(args.views).read_text())
    (root,) = yam_rig.roots()
    scene, cam = setup_scene(args.samples, args.device)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    for v in views:
        yam_rig.set_joints(root, v.get('q', [0.0] * 6), gripper=v.get('gripper', 0.0))
        place_camera(scene, cam, root, v)
        mode = v.get('mode', 'material')
        clay_override(scene, mode == 'clay')
        scene.render.film_transparent = mode != 'mask'
        if mode == 'mask':
            scene.render.engine = 'BLENDER_WORKBENCH'
            scene.display.shading.light = 'FLAT'
            scene.display.shading.color_type = 'SINGLE'
            scene.display.shading.single_color = (1, 1, 1)
            scene.display.shading.background_type = 'VIEWPORT'
            scene.display.shading.background_color = (0, 0, 0)
            scene.view_settings.view_transform = 'Standard'
            scene.view_settings.exposure = 0.0
        else:
            scene.render.engine = 'CYCLES'
            scene.view_settings.view_transform = 'Standard'
            scene.view_settings.exposure = EXPOSURE
        scene.render.filepath = str(out / f'{v["name"]}.png')
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA' if mode != 'mask' else 'RGB'
        bpy.ops.render.render(write_still=True)
        print('RENDERED', scene.render.filepath)


if __name__ == '__main__':
    main()
