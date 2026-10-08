"""Metric-frame helpers for the yam_bimanual_rig build (numpy only; also importable in Blender)."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
METRIC = SCENE / 'metric' / 'metric_frame.json'
ZED_IMAGE = Path('/home/frank/robot/bimanual_bringup/pi05/scene_camera.jpg')
LEFT_WRIST_IMAGE = Path('/home/frank/robot/bimanual_bringup/pi05/left_wrist_camera.jpg')
RIGHT_WRIST_IMAGE = Path('/home/frank/robot/bimanual_bringup/pi05/right_wrist_camera.jpg')


def metric():
    return json.loads(METRIC.read_text())


def scene_camera(m=None):
    m = m or metric()
    c = m['scene_camera']
    K = np.array(c['K'], float)
    T = np.array(c['pose']['world_from_camera_opencv'], float)
    return K, T, tuple(c['image_size_wh'])


def project(P, K, T):
    """World points (N,3) -> pixels (N,2) and depth (N,)."""
    P = np.atleast_2d(np.asarray(P, float))
    R, t = T[:3, :3], T[:3, 3]
    pc = (P - t) @ R
    uv = pc[:, :2] / pc[:, 2:3]
    uv = uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, pc[:, 2]


def ray(uv, K, T):
    uv = np.atleast_2d(np.asarray(uv, float))
    d = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    d = d @ T[:3, :3].T
    return T[:3, 3], d / np.linalg.norm(d, axis=1, keepdims=True)


def backproject_plane(uv, K, T, n, d):
    """Intersect pixel rays with the plane n.p + d = 0."""
    o, dirs = ray(uv, K, T)
    n = np.asarray(n, float)
    s = -(o @ n + d) / (dirs @ n)
    return o + dirs * s[:, None]
