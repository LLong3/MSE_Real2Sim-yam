"""Fit a camera to one video frame from the arm silhouette (for the review render only).

    OMP_NUM_THREADS=1 .runtime/pi3x-inference/venv/bin/python build/fit_review_camera.py \
        --tris evidence/closeup/tris_home.npz --masks evidence/closeup/sam_f1140/masks.npz --instance 4 \
        --init evidence/closeup/init_pi3x_1143_approx.json --out evidence/closeup/camera_f1140.json

Status 09-29: not finished. Run 1 (no init) reached IoU 0.70 with a visibly wrong pose; runs 2 and 3
(with an init) were killed by machine crashes during the Powell stage (evidence/closeup/*.log).
tris_home.npz: build/export_tris.py on yam_arm.blend. Single-core CPU, about 5 min coarse search
plus several minutes per Powell start.

Model silhouette = all asset triangles at home except the stock finger tips and the base plate
(the real fingers differ and SAM leaves the plate out). Ignore boxes cover the real fingers and the
wrist camera. Coarse search over camera positions (look-at through the mask centroid) with a point-splat
silhouette, then Powell on a soft IoU of the exact triangle fill over (rotation, translation, focal).
Run single-threaded (OMP_NUM_THREADS=1); multithreaded OpenCV/numpy oversubscribe here. Output: K and T_cam_arm (OpenCV camera -> arm root).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

W, H = 1920, 1080
IGNORE = [(955, 585, 1140, 715), (1072, 740, 1215, 865)]  # real fingers + wrist camera (full-res px)
P0 = np.array([0.0, 0.0, 0.14])


def load_model(path):
    d = np.load(path)
    parts = []
    for k in d.files:
        t = d[k]
        if k in ('tip_left', 'tip_right'):
            continue
        if k == 'base':
            t = t[t[:, :, 2].max(1) > 0.012]  # drop the foot plate, keep the joint-1 cylinder
        parts.append(t)
    return np.concatenate(parts)


def render(tris, rvec, tvec, K, scale):
    pts, _ = cv2.projectPoints(tris.reshape(-1, 3).astype(np.float64), rvec, tvec, K, None)
    R = cv2.Rodrigues(rvec)[0]
    z = (tris.reshape(-1, 3) @ R.T + tvec.reshape(1, 3))[:, 2].reshape(-1, 3)
    pts = np.round(pts.reshape(-1, 3, 2)[(z > 0.02).all(1)] * scale * 4).astype(np.int32)
    img = np.zeros((int(H * scale), int(W * scale)), np.uint8)
    # one call per triangle: a single fillPoly over overlapping triangles XORs them
    for t in pts:
        cv2.fillConvexPoly(img, t, 255, cv2.LINE_8, 2)
    return img > 0


def surface_points(tris, n=150000, seed=0):
    a = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tris), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tris[idx]
    return np.vstack([t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]), tris.reshape(-1, 3)])


def splat(pts, rvec, tvec, K, scale):
    """Fast approximate silhouette: project dense surface points, close small gaps."""
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    pc = pts @ R.T + np.asarray(tvec, float).reshape(1, 3)
    pc = pc[pc[:, 2] > 0.02]
    uv = (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    uv = np.floor(uv * scale).astype(np.int64)
    h, w = int(H * scale), int(W * scale)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    img = np.zeros((h, w), np.uint8)
    img[uv[ok, 1], uv[ok, 0]] = 255
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def soft(m, sigma):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)


def pose_from(C, roll, uv, K):
    """Camera at C whose pixel uv looks at P0, with roll about that ray. Returns rvec, tvec (arm->cam)."""
    d = P0 - C; d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0])
    x = np.cross(d, up); x /= np.linalg.norm(x)   # OpenCV: x right, y down, z forward
    y = np.cross(d, x)
    R_look = np.stack([x, y, d], 1)  # camera -> arm
    r = np.linalg.solve(K, [uv[0], uv[1], 1.0]); r /= np.linalg.norm(r)
    axis = np.cross(r, [0, 0, 1.0]); s = np.linalg.norm(axis)
    R_align = Rotation.from_rotvec(axis / s * np.arcsin(min(1, s))).as_matrix() if s > 1e-9 else np.eye(3)
    R_c2a = R_look @ Rotation.from_rotvec([0, 0, roll]).as_matrix() @ R_align
    R_a2c = R_c2a.T
    return cv2.Rodrigues(R_a2c)[0].ravel(), -R_a2c @ C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tris', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--instance', type=int, required=True)
    ap.add_argument('--fx', type=float, default=613.3748779296875 * W / 672)  # Pi3X predicted, source frame 1143
    ap.add_argument('--init', help='JSON with T_cam_arm (OpenCV camera -> arm) to add as a refinement start')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tris = load_model(a.tris)
    real = np.load(a.masks)['masks'][a.instance]
    valid = np.ones((H, W), bool)
    for x0, y0, x1, y1 in IGNORE:
        valid[y0:y1, x0:x1] = False
    K0 = np.array([[a.fx, 0, (W - 1) / 2], [0, a.fx, (H - 1) / 2], [0, 0, 1.0]])
    ys, xs = np.nonzero(real & valid)
    uv = (xs.mean(), ys.mean())

    s1 = 0.25
    real1 = cv2.resize(real.astype(np.uint8), None, fx=s1, fy=s1, interpolation=cv2.INTER_AREA) > 0
    valid1 = cv2.resize(valid.astype(np.uint8), None, fx=s1, fy=s1, interpolation=cv2.INTER_NEAREST) > 0

    def iou(m, r, v):
        m, r = m & v, r & v
        return (m & r).sum() / max(1, (m | r).sum())

    pts = surface_points(tris, n=400000)
    cands = []
    for az in np.radians(np.arange(0, 360, 10)):
        for el in np.radians([10, 20, 30, 40, 50, 60, 70]):
            for dist in (0.45, 0.6, 0.75, 0.9, 1.1, 1.35):
                C = P0 + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                for roll in np.radians([-20, -10, 0, 10, 20]):
                    rv, tv = pose_from(C, roll, uv, K0)
                    cands.append((iou(splat(pts, rv, tv, K0, s1), real1, valid1), rv, tv))
    cands.sort(key=lambda c: -c[0])
    print('coarse best', [round(c[0], 3) for c in cands[:8]], flush=True)

    s2 = 0.5
    real2 = soft(cv2.resize(real.astype(np.uint8), None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA) > 0, 2.0)
    valid2 = cv2.resize(valid.astype(np.uint8), None, fx=s2, fy=s2, interpolation=cv2.INTER_NEAREST) > 0

    def unpack(x, rv0, tv0):
        rv = (Rotation.from_rotvec(x[:3] * 0.05) * Rotation.from_rotvec(rv0)).as_rotvec()
        tv = tv0 + x[3:6] * 0.05
        K = K0.copy(); K[0, 0] = K[1, 1] = a.fx * np.exp(x[6] * 0.1)
        return rv, tv, K

    def cost(x, rv0, tv0):
        rv, tv, K = unpack(x, rv0, tv0)
        m = soft(render(tris, rv, tv, K, s2), 2.0)
        inter = (np.minimum(m, real2) * valid2).sum(); union = (np.maximum(m, real2) * valid2).sum()
        return 1 - inter / max(union, 1e-6)

    starts = cands[:3]
    if a.init:
        T = np.array(json.loads(Path(a.init).read_text())['T_cam_arm'])
        R_a2c = T[:3, :3].T
        rv_i, tv_i = cv2.Rodrigues(R_a2c)[0].ravel(), -R_a2c @ T[:3, 3]
        starts = [(iou(splat(pts, rv_i, tv_i, K0, s1), real1, valid1), rv_i, tv_i)] + starts
        print('init start IoU', round(starts[0][0], 3), flush=True)
    best = None
    for score, rv0, tv0 in starts:
        res = minimize(cost, np.zeros(7), args=(rv0, tv0), method='Powell',
                       options=dict(xtol=1e-3, ftol=1e-5, maxfev=1200))
        rv, tv, K = unpack(res.x, rv0, tv0)
        hard = iou(render(tris, rv, tv, K, 1.0), real, valid)
        print(f'coarse {score:.3f} -> soft {1 - res.fun:.4f} hard IoU {hard:.4f} fx {K[0, 0]:.1f} nfev {res.nfev}', flush=True)
        if best is None or hard > best[0]:
            best = (hard, rv, tv, K)
    hard, rv, tv, K = best
    R = cv2.Rodrigues(rv)[0]
    T_cam_arm = np.eye(4); T_cam_arm[:3, :3] = R.T; T_cam_arm[:3, 3] = -R.T @ tv
    out = dict(frame='IMG_5411.MOV frame 1140 (38.0 s)', instance=a.instance, iou_full_res=float(hard),
               ignore_boxes_px=IGNORE, K=K.tolist(), T_cam_arm=T_cam_arm.tolist(),
               camera_centre_arm_m=T_cam_arm[:3, 3].tolist(),
               note='Silhouette fit of the home pose to the SAM 3.1 mask; for the review render only, not a calibration.')
    Path(a.out).write_text(json.dumps(out, indent=1))
    m = render(tris, rv, tv, K, 1.0)
    vis = np.zeros((H, W, 3), np.uint8); vis[real] = (0, 160, 0); vis[m] += np.array([160, 0, 160], np.uint8)
    vis[~valid] //= 3
    cv2.imwrite(str(Path(a.out).with_suffix('.jpg')), cv2.resize(vis, (960, 540)))
    print(json.dumps({k: out[k] for k in ('iou_full_res', 'camera_centre_arm_m')}), 'fx', K[0, 0])


if __name__ == '__main__':
    main()
