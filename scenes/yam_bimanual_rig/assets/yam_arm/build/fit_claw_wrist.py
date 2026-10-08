"""Fit the UMI-FT claw placement to the 09-29 wrist images (CPU; how the constants in add_umi_ft_claws.py were set).

    ~/robot/raiden/.venv/bin/python build/fit_claw_wrist.py --out evidence/claw_fit [--xmid 0.0]

Model: pad + strip + clamp (source/umi_ft) placed in link_6 at raiden home by (open_half c, pad_z6 d, pad_x6 xmid),
left claw mirrored for the right. Cameras: raiden hand-eye (cam2gripper to grasp_site) + HD1200 K/2 (equal to the
live 960x600 stream K). Real masks: HSV thresholds (orange pad, yellow clamp) and the dark strip next to the pad.
Loss: 1 - mean IoU (pad, clamp, strip) over both wrist images. Fit (Nelder-Mead): c, d with xmid fixed, plus a
small correction per camera (rotation about its centre, translation in link_6; prior 2 deg / 10 mm).
The hand-eye translation is weakly constrained (7 captures), so xmid trades off against it; xmid = 0 (claw centred
on the carriage like the stock claw) is chosen and the opening is checked on the ZED scene image, where the
pad-to-pad gap does not depend on the arm-pose error (metric frame: arms + scene_camera only).
Also evaluates the placement used by the asset (source/umi_ft/claw.json): camera corrections only, IoU with the raw
and the corrected cameras, and the scene gap. The corrected cameras are used for the 05 wrist review render.
Writes <out>/fit.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parents[1]
UMI = HERE / 'source' / 'umi_ft'
REAL = Path.home() / 'robot/bimanual_bringup/pi05'
CAL = Path.home() / '.config/raiden/calibration_results.json'
METRIC = HERE.parents[1] / 'metric' / 'metric_frame.json'
STRIP_T, PAD_THICK, HOME_Q = 0.0015, 0.0172, 0.0475
R_LEFT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
T_L6_GRASP = np.diag([1.0, -1.0, -1.0, 1.0]); T_L6_GRASP[2, 3] = -0.1347   # MJCF grasp_site in link_6


def fill(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(m.shape, np.uint8); cv2.drawContours(out, cs, -1, 1, -1)
    return out > 0


def real_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k = np.ones((3, 3), np.uint8)
    o = fill(cv2.morphologyEx(((h >= 3) & (h <= 17) & (s >= 140) & (v >= 110)).astype(np.uint8), cv2.MORPH_OPEN, k))
    y = fill(cv2.morphologyEx(((h >= 20) & (h <= 34) & (s >= 110) & (v >= 130)).astype(np.uint8), cv2.MORPH_OPEN, k))
    near = cv2.dilate(o.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    dark = cv2.morphologyEx(((v < 55) & near & ~o & ~y).astype(np.uint8), cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents(dark)
    touch = set(np.unique(lab[cv2.dilate(o.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0])) - {0}
    dark = np.isin(lab, list(touch)); dark[560:] = False     # bottom rows mix the strip with the tag block
    return o, y, dark


def claw(c, d, xm):
    """(label, mesh) in link_6 at home for both claws: 3 pad, 4 clamp, 5 strip."""
    fin = trimesh.load(UMI / 'umift_finger_no_contact_mic.stl')
    clamp = trimesh.load(UMI / 'parts' / 'clamp.stl')
    strip = trimesh.creation.box(bounds=[[0, -PAD_THICK, fin.bounds[0, 2]], [STRIP_T, 0, fin.bounds[1, 2]]])
    T = np.eye(4); T[:3, :3] = R_LEFT; T[:3, 3] = [xm + PAD_THICK / 2, -c - STRIP_T, d]
    out = []
    for lab, m in ((3, fin), (4, clamp), (5, strip)):
        left = m.copy(); left.apply_transform(T)
        right = left.copy(); right.apply_transform(np.diag([1, -1, 1, 1.0]))   # winding flipped by trimesh
        out += [(lab, left), (lab, right)]
    return out


def render_labels(meshes, K, T_l6_cam, W=960, H=600):
    inv = np.linalg.inv(T_l6_cam)
    polys, depth, labs = [], [], []
    for lab, m in meshes:
        p = m.vertices @ inv[:3, :3].T + inv[:3, 3]
        t = p[m.faces]; ok = (t[:, :, 2] > 0.01).all(1)
        uv = t[ok][:, :, :2] / t[ok][:, :, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        polys.append(uv); depth.append(t[ok][:, :, 2].mean(1)); labs.append(np.full(ok.sum(), lab))
    polys, depth, labs = np.concatenate(polys), np.concatenate(depth), np.concatenate(labs)
    img = np.zeros((H, W), np.uint8)
    for i in np.argsort(-depth):
        cv2.fillConvexPoly(img, np.round(polys[i] * 16).astype(np.int32), int(labs[i]), cv2.LINE_8, 4)
    return img


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


class Wrist:
    def __init__(self):
        cal = json.loads(CAL.read_text())['cameras']
        self.cams, self.real = {}, {}
        for s in ('left', 'right'):
            c = cal[f'{s}_wrist_camera']
            K = np.array(c['intrinsics']['camera_matrix']) * 0.5; K[2, 2] = 1
            T = np.eye(4); T[:3, :3] = c['hand_eye_calibration']['rotation_matrix']; T[:3, 3] = c['hand_eye_calibration']['translation_vector']
            self.cams[s] = (K, T_L6_GRASP @ T)
            self.real[s] = real_masks(cv2.imread(str(REAL / f'{s}_wrist_camera.jpg')))

    def cam(self, s, e):
        K, T = self.cams[s]; T = T.copy()
        T[:3, :3] = T[:3, :3] @ Rot.from_rotvec(e[:3]).as_matrix(); T[:3, 3] += e[3:6]
        return K, T

    def ious(self, meshes, s, e):
        K, T = self.cam(s, e)
        L = render_labels(meshes, K, T); L[560:] = np.where(L[560:] == 5, 0, L[560:])
        o, y, d = self.real[s]
        return iou(fill(L == 3), o), iou(fill(L == 4), y), iou(L == 5, d)


def scene_gap(c, d, xm):
    """Pad-to-pad gap (px) between the two orange regions of each arm in the ZED scene image, real vs model."""
    full = json.loads(METRIC.read_text())
    sc, arms = full['scene_camera'], full['arms']
    poses = json.loads((HERE / 'fk' / 'mujoco_poses.json').read_text())['configs']['home']['bodies']['link_6']
    T_base_l6 = np.eye(4); T_base_l6[:3, :3] = poses['mat']; T_base_l6[:3, 3] = poses['pos']
    K, Wc = np.array(sc['K']), np.array(sc['pose']['world_from_camera_opencv'])
    real = real_masks(cv2.imread(str(REAL / 'scene_camera.jpg')))[0]
    model = np.zeros_like(real)
    for arm in ('left_arm', 'right_arm'):
        T = np.linalg.inv(np.array(arms[arm]['world_from_base']) @ T_base_l6) @ Wc
        model |= render_labels(claw(c, d, xm), K, T) == 3

    def gaps(m, x0, x1):
        m = fill(m[:, x0:x1])
        g = {}
        for y in range(500, 571, 5):          # rows where both pads show (a gap of >= 20 px between two runs)
            xs = np.nonzero(m[y])[0]
            if len(xs) > 1 and np.diff(xs).max() >= 20:
                g[y] = int(np.diff(xs).max())
        return g
    out = {}
    for arm, x0 in (('left_arm', 230), ('right_arm', 630)):
        gr, gm = gaps(real, x0, x0 + 170), gaps(model, x0, x0 + 170)
        rows = sorted(set(gr) & set(gm))
        out[arm] = dict(rows=rows, real_px=float(np.mean([gr[y] for y in rows])) if rows else None,
                        model_px=float(np.mean([gm[y] for y in rows])) if rows else None)
    return out


def fit_cameras(wr, meshes, e0=None):
    def loss(x):
        eL, eR = x[:6], x[6:]
        tot = np.mean(wr.ious(meshes, 'left', eL)) + np.mean(wr.ious(meshes, 'right', eR))
        pen = sum(np.sum((np.degrees(e[:3]) / 2) ** 2) + np.sum((e[3:] / 0.010) ** 2) for e in (eL, eR))
        return 1 - tot / 2 + 0.002 * pen
    x0 = np.zeros(12) if e0 is None else np.asarray(e0)
    steps = ([np.radians(1.5)] * 3 + [0.005] * 3) * 2
    r = minimize(loss, x0, method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-5, maxiter=6000, initial_simplex=x0 + np.vstack([np.zeros(12), np.diag(steps)])))
    return r.x[:6], r.x[6:], float(r.fun)


def report(wr, c, d, xm, eL, eR):
    meshes = claw(c, d, xm)
    return dict(camera_corrections={s: dict(rotvec_deg=np.degrees(e[:3]).round(3).tolist(), translation_l6_mm=(np.array(e[3:]) * 1e3).round(2).tolist(),
                                            rotvec_rad=np.asarray(e[:3]).tolist(), translation_l6_m=np.asarray(e[3:]).tolist())
                                    for s, e in (('left', eL), ('right', eR))},
                iou_pad_clamp_strip=dict(corrected={s: np.round(wr.ious(meshes, s, e), 3).tolist() for s, e in (('left', eL), ('right', eR))},
                                         raw_hand_eye={s: np.round(wr.ious(meshes, s, np.zeros(6)), 3).tolist() for s in ('left', 'right')}),
                scene_gap_check=scene_gap(c, d, xm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--xmid', type=float, default=0.0)
    ap.add_argument('--x0', type=float, nargs=2, default=(0.0577, -0.1388), help='start c, d (m)')
    a = ap.parse_args()
    wr = Wrist()

    def loss(x):
        c, d = x[:2]; eL, eR = x[2:8], x[8:14]
        meshes = claw(c, d, a.xmid)
        tot = np.mean(wr.ious(meshes, 'left', eL)) + np.mean(wr.ious(meshes, 'right', eR))
        pen = sum(np.sum((np.degrees(e[:3]) / 2) ** 2) + np.sum((e[3:] / 0.010) ** 2) for e in (eL, eR))
        return 1 - tot / 2 + 0.002 * pen
    x0 = np.r_[a.x0, np.zeros(12)]
    steps = [0.004, 0.006] + ([np.radians(1.5)] * 3 + [0.005] * 3) * 2
    r = minimize(loss, x0, method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-5, maxiter=8000, initial_simplex=x0 + np.vstack([np.zeros(14), np.diag(steps)])))
    c, d = (float(v) for v in r.x[:2])
    res = dict(inputs=dict(calibration=str(CAL), wrist_images=[str(REAL / f'{s}_wrist_camera.jpg') for s in ('left', 'right')],
                           metric_frame_sections=['arms', 'scene_camera']),
               xmid_fixed_m=a.xmid, strip_m=STRIP_T,
               free_fit=dict(start_c_d_m=list(a.x0), open_half_m=c, pad_z6_m=d, loss=float(r.fun), **report(wr, c, d, a.xmid, r.x[2:8], r.x[8:14])))
    claw_cfg = json.loads((UMI / 'claw.json').read_text())
    cc, dd, xx = claw_cfg['open_half_m'], claw_cfg['pad_z6_m'], claw_cfg['pad_x6_m']
    eL, eR, f = fit_cameras(wr, claw(cc, dd, xx), np.r_[r.x[2:8], r.x[8:14]])
    res['asset_placement'] = dict(source='source/umi_ft/claw.json', open_half_m=cc, pad_z6_m=dd, pad_x6_m=xx, loss=f,
                                  **report(wr, cc, dd, xx, eL, eR))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'fit.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
