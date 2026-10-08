"""Shared loaders for the metric-frame work (read-only on the reference run)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path('/home/frank/robot/aha-3d/runs/yam_bimanual_rig/20260929t202049z-reference64-a264573a')
REF = RUN / 'reference'
WORK = HERE / 'work'
MEDIA = Path('/home/frank/robot/markdowns/bimanual/1_media/alignment')
ZED_EP = Path('/home/frank/robot/bimanual_bringup/cameras/e2e_sdk55_converted/0000')
ZED_0929 = Path('/home/frank/robot/bimanual_bringup/pi05/scene_camera.jpg')
ZED_0928_PNG = Path('/home/frank/robot/bimanual_bringup/cameras/raiden_scene_camera.png')
H, W = 378, 672


def cameras():
    cam = json.load(open(REF / 'alignment/cameras.json'))
    c2w = np.array([f['c2w'] for f in cam['frames']])
    K = np.array([f['intrinsics'] for f in cam['frames']])
    fidx = np.array([f['source_frame'] for f in cam['frames']])
    return c2w, K, fidx


def points():
    """Aligned per-pixel points (64,H,W,3), measurement-valid mask, colors; cached as npy."""
    WORK.mkdir(exist_ok=True)
    fv, fm, fc = WORK / 'ref_vertices.npy', WORK / 'ref_mvalid.npy', WORK / 'ref_colors.npy'
    if not fv.exists():
        z = np.load(REF / 'mesh/layers.npz')
        np.save(fv, z['vertices'].reshape(64, H, W, 3))
        mv = z['measurement_valid'] & (z['layer'] == 0)
        np.save(fm, mv.reshape(64, H, W))
        np.save(fc, z['colors'].reshape(64, H, W, 3))
    return np.load(fv, mmap_mode='r'), np.load(fm), np.load(fc, mmap_mode='r')


def masks():
    d = np.load(REF / 'masks/masks.npz')
    return {k: d[k] for k in ('floor', 'wall', 'glass')}


def rgb():
    return np.load(REF / 'pi3x/inputs.npz')['rgb']


def zed_frame(i=0):
    import pickle
    from PIL import Image
    img = np.array(Image.open(ZED_EP / f'rgb/scene_camera/{i:010d}.png').convert('RGB'))
    depth = np.load(ZED_EP / f'depth/scene_camera/{i:010d}.npz')['depth'].astype(np.float64) / 1000.0
    low = pickle.load(open(ZED_EP / f'lowdim/{i:010d}.pkl', 'rb'))
    K = np.array(low['intrinsics']['scene_camera'], dtype=np.float64)
    return img, depth, K


def fit_plane(P, iters=300, thr=0.01, rng=0):
    """RANSAC + least squares plane. Returns (n, d) with n.x + d = 0, inlier mask."""
    r = np.random.default_rng(rng)
    best = None
    for _ in range(iters):
        s = P[r.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(P @ n - n @ s[0]) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
    inl = np.abs((P - c) @ n) < thr
    Q = P[inl]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
    return n, -n @ c, inl


def save_json(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))
