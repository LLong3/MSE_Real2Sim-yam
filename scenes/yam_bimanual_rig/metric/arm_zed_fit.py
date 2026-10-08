"""Metric arm-base poses from the ZED scene image alone: MJCF (q=0, fingers open) silhouettes vs the
gripper masks (colour segmentation), camera pose from zed_metric.py (desk frame, ZED depth units).

Pose per arm in the desk frame D (origin back-left desk-top corner, x along the back edge,
y toward the wall, z up): base = (x, y, z, yaw), z = base mounting face above the desk top.
Objective: squared difference of Gaussian-blurred silhouettes in a window around each gripper.
Writes work/arm_zed_fit.json.
"""
import cv2
import numpy as np
from scipy.optimize import minimize

import common

z = np.load(common.WORK / 'zed_metric.npz')
K, Rdc, BL = z['K'], z['Rdc'], z['BL']
mesh = np.load(common.WORK / 'yam_q0_open0.0475.npz')
MV, MF = mesh['vertices'], mesh['faces']
# only the distal part (link5 onward: wrist, gripper, fingers) can enter the image window;
# the silhouette is rendered by splatting a dense surface sample (~1 mm) instead of rasterising faces
MF = MF[MV[MF].mean(1)[:, 0] > 0.04]
_rng = np.random.default_rng(0)
_a, _b, _c = MV[MF[:, 0]], MV[MF[:, 1]], MV[MF[:, 2]]
_area = 0.5 * np.linalg.norm(np.cross(_b - _a, _c - _a), axis=1)
_n = 400000
_i = _rng.choice(len(MF), _n, p=_area / _area.sum())
_r1, _r2 = np.sqrt(_rng.random(_n)), _rng.random(_n)
SURF = (1 - _r1)[:, None] * _a[_i] + (_r1 * (1 - _r2))[:, None] * _b[_i] + (_r1 * _r2)[:, None] * _c[_i]
WIN = {'left_arm': (slice(430, 600), slice(200, 440)), 'right_arm': (slice(430, 600), slice(600, 860))}
SIG = 2.0


def rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project(p_desk):
    pc = p_desk @ Rdc + BL  # desk -> camera: p_cam = Rdc^T p_desk + BL
    uv = pc[:, :2] / pc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, pc[:, 2]


def silhouette(x, shape=(600, 960)):
    p = SURF @ rz(x[3]).T + x[:3]
    uv, zc = project(p)
    ui, vi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    ok = (zc > 0.05) & (ui >= 0) & (ui < shape[1]) & (vi >= 0) & (vi < shape[0])
    img = np.zeros(shape, np.uint8)
    img[vi[ok], ui[ok]] = 255
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return img.astype(np.float32) / 255


def blur(a):
    return cv2.GaussianBlur(a, (0, 0), SIG)


def fit(mask, name, x0):
    win = WIN[name]
    target = blur(mask.astype(np.float32))[win]

    def cost(x):
        s = blur(silhouette(x))[win]
        return float(np.mean((s - target) ** 2))
    best = None
    for dyaw in (0,):
        for dz in (0.0,):
            xs = np.array(x0, float) + [0, 0, dz, np.radians(dyaw)]
            r = minimize(cost, xs, method='Nelder-Mead',
                         options=dict(xatol=1e-4, fatol=1e-7, maxiter=3000, initial_simplex=None))
            if best is None or r.fun < best.fun:
                best = r
    x = best.x
    s = silhouette(x)[win] > 0.5
    m = mask[win]
    iou = float((s & m).sum() / (s | m).sum())
    return x, iou, best.fun


def main():
    out = {}
    cam = z['cam_in_desk']
    for tag in ('0928', '0929'):
        mask = np.load(common.WORK / f'gripper_mask_{tag}.npy')
        res = {}
        for name, dx in (('left_arm', -0.31), ('right_arm', 0.31)):
            x0 = [cam[0] + dx, -0.74, 0.02, np.radians(90)]
            x, iou, c = fit(mask, name, x0)
            res[name] = dict(base_desk_frame_m=x[:3], yaw_deg=float(np.degrees(x[3])), iou=iou, cost=c)
            print(tag, name, np.round(x[:3], 4), round(float(np.degrees(x[3])), 2), 'IoU', round(iou, 3), flush=True)
        L, R = res['left_arm']['base_desk_frame_m'], res['right_arm']['base_desk_frame_m']
        res['spacing_m'] = float(np.linalg.norm((R - L)[:2]))
        out[tag] = res
    common.save_json(common.WORK / 'arm_zed_fit.json', out)
    # overlay for review
    for tag in ('0928', '0929'):
        path = common.ZED_EP / 'rgb/scene_camera/0000000000.png' if tag == '0928' else common.ZED_0929
        im = cv2.imread(str(path))
        for name in ('left_arm', 'right_arm'):
            r = out[tag][name]
            x = np.r_[r['base_desk_frame_m'], np.radians(r['yaw_deg'])]
            s = (silhouette(x) > 0.5).astype(np.uint8)
            cnt, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cnt, -1, (0, 255, 0), 1)
        cv2.imwrite(str(common.WORK / f'arm_zed_fit_{tag}.jpg'), im)


if __name__ == '__main__':
    main()
