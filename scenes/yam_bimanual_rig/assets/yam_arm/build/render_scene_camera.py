"""Render both YAM arms at home from the ZED scene camera of the metric frame (review only).

    $BLENDER_BIN -b --factory-startup --python-exit-code 1 --python build/render_scene_camera.py -- \
        --metric ../../metric/metric_frame.json --out evidence/scene_camera [--samples 64] [--device CPU] [--asset X.blend]

Appends the asset collection of yam_arm.blend twice, places each root at arms.<arm>.world_from_base
(metric frame), sets raiden home (joints 0, gripper open) and each arm's finger tag ids, and renders from
scene_camera (K, pose.world_from_camera_blender, 960x600, no distortion) with a transparent background.
Reads only the arms and scene_camera sections of the metric frame.
Writes <out>/render_rgba.png, <out>/tris_metric.npz (world triangles per arm|part of the render-visible
meshes, for the silhouette masks) and <out>/camera.json (K, world_from_camera_opencv, arm poses used).
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'build'))
import yam_rig  # noqa: E402

COLLECTION = 'YAM arm (i2rt yam + linear_4310)'
HOME_GRIPPER = 0.0475
EXPOSURE = -1.8


def append_arm(scene, asset):
    with bpy.data.libraries.load(str(asset), link=False) as (src, dst):
        dst.collections = [COLLECTION]
    coll = dst.collections[0]
    scene.collection.children.link(coll)
    return next(o for o in coll.objects if o.get('yam_arm_root'))


def setup_render(scene, samples, device):
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
    # Standard keeps the saturated claw colours (AgX renders the orange/yellow pale); lights set for it
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = EXPOSURE
    world = bpy.data.worlds.new('review_world')
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.8, 0.8, 0.82, 1)
    bg.inputs['Strength'].default_value = 0.6
    scene.world = world
    # office ceiling light over the desk (metric frame: desk x -3.05..-1.53, y 1.32..2.08, top z 0.72)
    light = bpy.data.lights.new('ceiling', 'AREA')
    light.shape, light.size, light.size_y, light.energy = 'RECTANGLE', 1.2, 0.6, 450
    obj = bpy.data.objects.new('ceiling', light)
    scene.collection.objects.link(obj)
    obj.location = (-2.3, 1.7, 2.7)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--device', default='CPU')  # GPU (OptiX) only when no other GPU job runs
    ap.add_argument('--asset', default=str(HERE / 'yam_arm.blend'))
    a = ap.parse_args(argv)
    full = json.loads(Path(a.metric).read_text())
    metric = dict(arms=full['arms'], scene_camera=full['scene_camera'], created_utc=full.get('created_utc'))
    cam_spec = metric['scene_camera']
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    setup_render(scene, a.samples, a.device)
    tris, arms_used = {}, {}
    for arm in ('left_arm', 'right_arm'):
        root = append_arm(scene, a.asset)
        root.matrix_world = Matrix(metric['arms'][arm]['world_from_base'])
        yam_rig.set_joints(root, [0.0] * 6, gripper=HOME_GRIPPER)
        yam_rig.set_finger_tags(root, *yam_rig.ARM_TAG_IDS[arm])
        arms_used[arm] = dict(root=root.name, world_from_base=metric['arms'][arm]['world_from_base'],
                              finger_tag_ids=yam_rig.ARM_TAG_IDS[arm])
        for o in root.children_recursive:
            if o.type != 'MESH' or o.hide_render:   # stock claw geoms are hidden from render
                continue
            M = np.array(o.matrix_world)
            v = np.array([vv.co[:] for vv in o.data.vertices]) @ M[:3, :3].T + M[:3, 3]
            # triangulate (the tag is a quad)
            f = np.array([t for p in o.data.polygons for t in
                          ([p.vertices[0], p.vertices[i], p.vertices[i + 1]] for i in range(1, len(p.vertices) - 1))])
            tris[f'{arm}|{o.get("mjcf_geom_mesh") or o["asset_part_key"]}'] = v[f].astype(np.float32)

    W, H = cam_spec['image_size_wh']
    K = np.array(cam_spec['K'])
    cam = bpy.data.objects.new('zed_scene_camera', bpy.data.cameras.new('zed_scene_camera'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    d = cam.data
    d.sensor_fit, d.sensor_width = 'HORIZONTAL', 36.0
    d.lens = K[0, 0] * d.sensor_width / W
    # OpenCV pixel centres at integers: the optical centre of an unshifted camera is ((W-1)/2, (H-1)/2)
    d.shift_x = -(K[0, 2] - (W - 1) / 2) / W
    d.shift_y = (K[1, 2] - (H - 1) / 2) / W
    d.clip_start, d.clip_end = 0.01, 20.0
    cam.matrix_world = Matrix(cam_spec['pose']['world_from_camera_blender'])
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.filepath = str(out / 'render_rgba.png')
    bpy.ops.render.render(write_still=True)
    np.savez_compressed(out / 'tris_metric.npz', **tris)
    (out / 'camera.json').write_text(json.dumps(dict(
        metric_frame=str(Path(a.metric).resolve()), metric_created_utc=metric.get('created_utc'),
        image_size_wh=[W, H], K=K.tolist(), world_from_camera_opencv=cam_spec['pose']['world_from_camera_opencv'],
        blender=dict(lens_mm=d.lens, shift_x=d.shift_x, shift_y=d.shift_y), arms=arms_used,
        joints='raiden home: joint1..6 = 0, gripper 0.0475 m (open)', samples=a.samples), indent=1))
    print('RENDERED', scene.render.filepath, 'lens', round(d.lens, 4), 'shift', round(d.shift_x, 5), round(d.shift_y, 5))


if __name__ == '__main__':
    main()
