"""Render one YAM arm at raiden home from its wrist camera (review only; CPU Cycles).

    $BLENDER_BIN -b --factory-startup --python-exit-code 1 --python build/render_wrist_camera.py -- \
        --arm left_arm --out evidence/wrist_camera [--samples 32] [--asset X.blend] \
        [--calibration ~/.config/raiden/calibration_results.json] [--correction evidence/claw_fit/fit.json]

Camera: raiden's hand-eye result for <side>_wrist_camera (cam2gripper, OpenCV, gripper = MJCF grasp_site; TSAI,
7 poses, 09-29) and its HD1200 intrinsics halved for the 960x600 stream (K has fx != fy; equal to the live stream
K). The pose is approximate: --correction applies the small per-camera correction of fit_claw_wrist.py
(asset_placement; rotation about the camera centre, translation in link_6), fitted jointly with the claw. Blender pixels are square: the render uses fx on both axes at 960 x H' (H' = 600 fx/fy)
and compose_wrist_camera.py stretches it to 960x600. Writes <out>/<arm>_render_rgba.png, <out>/<arm>_tris.npz
(camera-frame triangles per part of the render-visible meshes) and <out>/<arm>_camera.json.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'build'))
import render_scene_camera as RS  # noqa: E402  (append_arm, setup_render)
import yam_rig  # noqa: E402

W, H = 960, 600
CV_TO_BLENDER = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('left_arm', 'right_arm'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--asset', default=str(HERE / 'yam_arm.blend'))
    ap.add_argument('--calibration', default=str(Path.home() / '.config/raiden/calibration_results.json'))
    ap.add_argument('--correction', default=None, help='fit.json of fit_claw_wrist.py')
    a = ap.parse_args(argv)
    side = a.arm.split('_')[0]
    cal = json.loads(Path(a.calibration).read_text())['cameras'][f'{side}_wrist_camera']
    K = np.array(cal['intrinsics']['camera_matrix']) * (W / cal['intrinsics']['image_size'][0]); K[2, 2] = 1
    he = cal['hand_eye_calibration']
    T_grasp_cam = np.eye(4); T_grasp_cam[:3, :3] = he['rotation_matrix']; T_grasp_cam[:3, 3] = he['translation_vector']
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    RS.setup_render(scene, a.samples, 'CPU')
    root = RS.append_arm(scene, a.asset)
    yam_rig.set_joints(root, [0.0] * 6, gripper=RS.HOME_GRIPPER)
    yam_rig.set_finger_tags(root, *yam_rig.ARM_TAG_IDS[a.arm])
    # the scene render's ceiling light sits in world coordinates; put it above this arm's gripper instead
    scene.objects['ceiling'].location = root.matrix_world @ Matrix.Translation((0.3, 0.0, 1.9)).to_translation()
    site = next(o for o in root.children_recursive if o.get('mjcf_site') == 'grasp_site')
    bpy.context.view_layer.update()
    T_arm_cam = np.array(root.matrix_world.inverted() @ site.matrix_world) @ T_grasp_cam
    correction = None
    if a.correction:
        correction = json.loads(Path(a.correction).read_text())['asset_placement']['camera_corrections'][side]
        T_arm_l6 = np.array(root.matrix_world.inverted() @ yam_rig.links(root)['link_6'].matrix_world)
        T = np.linalg.inv(T_arm_l6) @ T_arm_cam
        rv = np.asarray(correction['rotvec_rad']); th = np.linalg.norm(rv)
        if th > 0:
            k = rv / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            T[:3, :3] = T[:3, :3] @ (np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx)
        T[:3, 3] += correction['translation_l6_m']
        T_arm_cam = T_arm_l6 @ T

    Hs = int(round(H * K[0, 0] / K[1, 1]))            # square-pixel render height
    cy_s = K[1, 2] * Hs / H
    cam = bpy.data.objects.new('wrist_camera', bpy.data.cameras.new('wrist_camera'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    d = cam.data
    d.sensor_fit, d.sensor_width = 'HORIZONTAL', 36.0
    d.lens = K[0, 0] * d.sensor_width / W
    d.shift_x = -(K[0, 2] - (W - 1) / 2) / W
    d.shift_y = (cy_s - (Hs - 1) / 2) / W
    d.clip_start, d.clip_end = 0.005, 20.0
    cam.matrix_world = root.matrix_world @ Matrix(T_arm_cam.tolist()) @ CV_TO_BLENDER
    scene.render.resolution_x, scene.render.resolution_y = W, Hs
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.filepath = str(out / f'{a.arm}_render_rgba.png')
    bpy.ops.render.render(write_still=True)

    cam_from_arm = np.linalg.inv(T_arm_cam)
    tris = {}
    for o in root.children_recursive:
        if o.type != 'MESH' or o.hide_render:
            continue
        M = cam_from_arm @ np.array(root.matrix_world.inverted() @ o.matrix_world)
        v = np.array([vv.co[:] for vv in o.data.vertices]) @ M[:3, :3].T + M[:3, 3]
        f = np.array([t for p in o.data.polygons for t in
                      ([p.vertices[0], p.vertices[i], p.vertices[i + 1]] for i in range(1, len(p.vertices) - 1))])
        tris[o.get('mjcf_geom_mesh') or o['asset_part_key']] = v[f].astype(np.float32)
    np.savez_compressed(out / f'{a.arm}_tris.npz', **tris)
    (out / f'{a.arm}_camera.json').write_text(json.dumps(dict(
        arm=a.arm, calibration=a.calibration, calibration_timestamp=json.loads(Path(a.calibration).read_text()).get('timestamp'),
        camera=f'{side}_wrist_camera', K_960x600=K.tolist(), T_arm_from_camera_opencv=T_arm_cam.tolist(),
        T_grasp_site_from_camera_opencv=T_grasp_cam.tolist(), render_size_wh=[W, Hs], stretch_to_wh=[W, H],
        joints='raiden home: joint1..6 = 0, gripper 0.0475 m (open)', finger_tag_ids=yam_rig.ARM_TAG_IDS[a.arm],
        camera_correction=correction, samples=a.samples, asset=a.asset), indent=1))
    print('RENDERED', scene.render.filepath, 'camera in arm frame (m)', np.round(T_arm_cam[:3, 3], 4).tolist())


if __name__ == '__main__':
    main()
