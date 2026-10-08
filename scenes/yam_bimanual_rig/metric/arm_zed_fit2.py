"""Metric arm-base check from the ZED image: MJCF (q=0) silhouettes vs the gripper masks, ZED-direct camera.

Free per arm: base x, y (desk frame, tape metres) and yaw; base z fixed at the ICP rail-top height;
finger opening scanned (the rig's fingers carry add-on pads and tags, so the finger outline is only
approximately the MJCF one). Nelder-Mead with a scaled simplex, several starts.
Writes work/arm_zed_fit2.json and work/arm_zed_fit2_<tag>.jpg.
"""
import json

import cv2
import numpy as np
from scipy.optimize import minimize

import common

z = np.load(common.WORK / 'zed_metric.npz')
zm = json.load(open(common.WORK / 'zed_metric.json'))
ZS = 1.52 / zm['desk_length_back_edge_m']          # ZED depth -> tape scale
K, Rdc, BL = z['K'], z['Rdc'], z['BL'] * ZS          # camera frame in tape units
WIN = {'left_arm': (slice(430, 600), slice(200, 440)), 'right_arm': (slice(430, 600), slice(600, 860))}
SIG = 2.0
rng = np.random.default_rng(0)


def surf(opening, n=300000):
    m = np.load(common.WORK / f'yam_q0_open{opening}.npz')
    V, F = m['vertices'], m['faces']
    F = F[V[F].mean(1)[:, 0] > 0.04]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    i = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return (1 - r1)[:, None] * a[i] + (r1 * (1 - r2))[:, None] * b[i] + (r1 * r2)[:, None] * c[i]


def rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def silhouette(S, x, zb):
    p = S @ rz(x[2]).T + np.array([x[0], x[1], zb])
    pc = p @ Rdc + BL
    uv = pc[:, :2] / pc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    ui, vi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    ok = (pc[:, 2] > 0.05) & (ui >= 0) & (ui < 960) & (vi >= 0) & (vi < 600)
    img = np.zeros((600, 960), np.uint8)
    img[vi[ok], ui[ok]] = 255
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float32) / 255


def main():
    fm = json.load(open(common.WORK / 'fit_metric.json'))
    arms = {r['arm']: r for r in fm['candidates']['A_per_axis']['arms'] if r['model'] == 'no_fingers'}
    BLw = np.array(fm['desk_box_BL_world']) if 'desk_box_BL_world' in fm else np.array([-3.0488, 2.0839, 0.72])
    out = {}
    surfs = {o: surf(o) for o in ('0.035', '0.0475', '0.055', '0.06')}
    for tag in ('0929', '0928'):
        mask = np.load(common.WORK / f'gripper_mask_{tag}.npy')
        res = {}
        for name in ('left_arm', 'right_arm'):
            win = WIN[name]
            m = mask.copy()
            m[:, :win[1].start] = False; m[:, win[1].stop:] = False
            m[:470] = False  # drop the croissant / cup above the grippers
            target = cv2.GaussianBlur(m.astype(np.float32), (0, 0), SIG)[win]
            b = np.array(arms[name]['base']) - BLw
            zb = b[2]
            best = None
            for o, S in surfs.items():
                cost = lambda x: float(np.mean((cv2.GaussianBlur(silhouette(S, x, zb), (0, 0), SIG)[win] - target) ** 2))
                for x0 in ([b[0], b[1], np.radians(arms[name]['yaw_deg'])], [b[0] - 0.025, b[1] - 0.02, np.radians(91)]):
                    x0 = np.array(x0)
                    simplex = np.array([x0, x0 + [0.01, 0, 0], x0 + [0, 0.01, 0], x0 + [0, 0, np.radians(2)]])
                    r = minimize(cost, x0, method='Nelder-Mead', options=dict(initial_simplex=simplex, xatol=2e-4, fatol=1e-8, maxiter=800))
                    if best is None or r.fun < best[0]:
                        best = (r.fun, r.x, o)
            c, x, o = best
            s = silhouette(surfs[o], x, zb)[win] > 0.5
            mm = m[win]
            iou = float((s & mm).sum() / (s | mm).sum())
            base_w = BLw + np.array([x[0], x[1], zb])
            res[name] = dict(base_world=base_w, yaw_deg=float(np.degrees(x[2])), opening=float(o), iou=iou, cost=c,
                             delta_vs_icp_A_m=base_w - np.array(arms[name]['base']))
            print(tag, name, base_w.round(4), round(float(np.degrees(x[2])), 2), 'open', o, 'IoU', round(iou, 3),
                  'delta vs ICP', (base_w - np.array(arms[name]['base'])).round(4), flush=True)
        out[tag] = res
        im = cv2.imread(str(common.ZED_0929 if tag == '0929' else common.ZED_EP / 'rgb/scene_camera/0000000000.png'))
        for name, r in res.items():
            b = r['base_world'] - BLw
            s = (silhouette(surfs[str(r['opening']) if str(r['opening']) in surfs else '0.0475'], [b[0], b[1], np.radians(r['yaw_deg'])], b[2]) > 0.5).astype(np.uint8)
            cnt, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cnt, -1, (0, 255, 0), 1)
        cv2.imwrite(str(common.WORK / f'arm_zed_fit2_{tag}.jpg'), im[400:600, 180:880])
    common.save_json(common.WORK / 'arm_zed_fit2.json', out)


if __name__ == '__main__':
    main()
