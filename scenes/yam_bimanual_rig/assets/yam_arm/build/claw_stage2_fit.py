"""Stage 2 (09-30): claw placement and the arm pose at home, with the stage-1 wrist cameras (CPU, raiden venv).

    ~/robot/raiden/.venv/bin/python -u build/claw_stage2_fit.py --stage1 RUN/stage1/cameras_stage1.json \
        --captures RUN/captures_sq0.03556/captures.json --variant b --out RUN/stage2/b [--start RUN/stage2/a/fit.json]

Model (claw_model.py, claw.json schema 2): both claws in link_6 at raiden home from C = (open_half c, pad_z6 d,
pad_x6 x), shared by both arms; the stock housing and carriage (MuJoCo) as occluders and for the scene
silhouette. World: link_6 = B FK(q) with B the arm base. Wrist camera T_wc (world, OpenCV); hand-eye X = link_6 from
camera = (B FK_l6)^-1 T_wc. The stage-1 camera has a measured world rotation (background) but no measured
translation (it was set from the metric base and raiden's hand-eye), so in the variants where the arm moves, the
camera keeps its stage-1 world rotation and rides with link_6: translation = B FK_l6 X_ref (+ offset in c).
Data (09-29 16:00 images, arms at home):
  wrist (960x600): orange pad, yellow clamp (+ placeholder), black contact face (HSV, as fit_claw_wrist.py);
  scene camera (metric K and pose): SAM 3.1 gripper silhouette per arm, orange and yellow HSV masks inside it.
Loss per arm: (1 - mean IoU wrist pad/yellow/face) + (1 - mean IoU scene silhouette/pad/yellow) + priors
(0.005 per sigma^2) + background edges of the wrist image where the camera moves (0.015 per px^2, soft-L1 2 px;
correspondences of wrist_bg_fit.py at the stage-1 pose, search 4 px).
Variants:
  a         stage-1 cameras fixed, metric bases; fit C only (the claw mount explains everything).
  a_mount   as a, plus a claw-pair rotation about link_6 (0, 0, -0.07) shared by both arms (tilted mount).
  b         per-arm base correction (yaw, pitch, roll about the base axes, xyz in the base frame; weak prior
            10 / 5 / 5 deg, 50 mm) + C; camera rotation = stage 1, translation rides with link_6 (X_ref = raiden's
            hand-eye); prior on the X rotation vs X_ref (3 deg).
  b_capX    as b with X_ref = the capture-fit hand-eye (--captures, 'joint' variant).
  b_joints  metric bases, per-arm joint offsets dq1..dq6 (prior 5 deg) + C; camera as b.
  c         as b, plus a free camera offset in the world (prior 15 mm = the X translation prior).
  b_mount   as b_capX, plus a claw-pair rotation about link_6 (0, 0, -0.07) shared by both arms: separates a
            tilted claw mount from the arm pose, with the camera mount X pinned by the calibration captures.
  c_mount   as c with X_ref = the capture-fit hand-eye, plus the shared claw-pair rotation of b_mount.
Optimisation: Nelder-Mead in normalised units, coordinate descent (arm left, arm right, shared C), restarted until
a round improves the loss by less than 1e-4 (at most --rounds rounds).
Writes OUT/fit.json (parameters, per-term losses, IoUs, centroid offsets, base / X / camera poses) and overlays.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

import claw_model as C
import wrist_bg_fit as BG
import wristfit_common as W

LAM = 0.005           # loss per prior sigma^2
LAM_BG = 0.015        # loss per background px^2
SIG = dict(x_rot=np.radians(3), x_t=0.015, b_yaw=np.radians(10), b_rp=np.radians(5), b_t=0.05, dq=np.radians(5))
MOVING = ('b', 'b_capX', 'b_joints', 'c', 'b_mount', 'c_mount')
BASE_VARIANTS = ('b', 'b_capX', 'c', 'b_mount', 'c_mount')
MOUNT_VARIANTS = ('a_mount', 'b_mount', 'c_mount')
CAM_OFFSET = ('c', 'c_mount')
MOUNT_PIVOT_L6 = np.array([0, 0, -0.070])
SAM_INST = {'left': 1, 'right': 0}
LAB = C.LABELS


def fill(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(m.shape, np.uint8); cv2.drawContours(out, cs, -1, 1, -1)
    return out > 0


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def centroid_offset(m, r):
    if m.sum() < 20 or r.sum() < 20:
        return None
    ym, xm = np.nonzero(m); yr, xr = np.nonzero(r)
    return [round(float(xm.mean() - xr.mean()), 2), round(float(ym.mean() - yr.mean()), 2)]


def wrist_real(img):
    """orange pad, yellow clamp, dark contact face next to the pad (as fit_claw_wrist.real_masks, 9x9 contact window)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k = np.ones((3, 3), np.uint8)
    o = fill(cv2.morphologyEx(((h >= 3) & (h <= 17) & (s >= 140) & (v >= 110)).astype(np.uint8), cv2.MORPH_OPEN, k))
    y = fill(cv2.morphologyEx(((h >= 20) & (h <= 34) & (s >= 110) & (v >= 130)).astype(np.uint8), cv2.MORPH_OPEN, k))
    near = cv2.dilate(o.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    dark = cv2.morphologyEx(((v < 55) & near & ~o & ~y).astype(np.uint8), cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents(dark)
    # 9x9 contact window (fit_claw_wrist.py used 5x5, which drops the right image's left-claw face: a 2-3 px lighter seam)
    touch = set(np.unique(lab[cv2.dilate(o.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0])) - {0}
    dark = np.isin(lab, list(touch)); dark[560:] = False
    return dict(pad=o, yellow=y, face=dark)


def scene_real(img, sam):
    """per arm: SAM silhouette, orange and yellow (claw_check.py thresholds) inside the dilated SAM mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    o = (H >= 3) & (H <= 18) & (S > 110) & (V > 90)
    y = (H >= 19) & (H <= 35) & (S > 110) & (V > 90)
    k = np.ones((3, 3), np.uint8)
    out = {}
    for side, i in SAM_INST.items():
        m = sam[i]
        near = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        out[side] = dict(sil=m, pad=cv2.morphologyEx((o & near).astype(np.uint8), cv2.MORPH_CLOSE, k) > 0,
                         yellow=cv2.morphologyEx((y & near).astype(np.uint8), cv2.MORPH_CLOSE, k) > 0)
    return out


def delta_base(p):
    """base-frame correction: yaw (z), pitch (y), roll (x) in rad, translation in the base frame (m)."""
    M = np.eye(4); M[:3, :3] = Rot.from_euler('ZYX', p[:3]).as_matrix(); M[:3, 3] = p[3:6]
    return M


class Problem:
    def __init__(self, a):
        self.variant = a.variant
        self.lam_bg = getattr(a, 'lam_bg', LAM_BG)
        cfg = C.load_cfg()
        self.home_gripper = cfg['home_gripper_m']
        self.pF = C.parts_F(cfg)
        self.fk = W.FK()
        self.arm_home, self.T6_home = C.arm_tris(self.fk)      # links 4-5 are not visible in any of the three views
        self.bases = W.metric_bases()
        cams = W.jload(a.stage1)
        self.T1 = {s: np.array(cams[s]) for s in W.SIDES}
        self.Kw = {s: W.wrist_K(s) for s in W.SIDES}
        self.Ks, self.S = W.scene_camera()
        f0 = self.fk(np.zeros(6))
        self.T_l6_gs = np.linalg.inv(f0['link_6']) @ f0['grasp_site']
        self.X_raiden = {s: self.T_l6_gs @ W.hand_eye(s) for s in W.SIDES}
        self.X_cap = None
        if a.captures:
            cap = W.jload(a.captures)['joint']['arms']
            self.X_cap = {s: self.T_l6_gs @ np.array(cap[s]['T_grasp_site_cam']) for s in W.SIDES}
            self.B_cap = {s: np.array(cap[s]['T_world_base']) for s in W.SIDES}
        self.X_ref = self.X_cap if self.variant in ('b_capX', 'b_mount', 'c_mount') else self.X_raiden
        self.wr = {s: wrist_real(W.real_image(s)) for s in W.SIDES}
        sam = np.load(W.HERE / 'evidence/scene_camera/sam/masks.npz')['masks']
        self.scene_img = cv2.imread(str(W.REAL / 'scene_camera.jpg'))
        self.sr = scene_real(self.scene_img, sam)
        self.bg = {}
        if self.variant in MOVING:
            segs_all = W.named_segments()
            for s in W.SIDES:
                img = W.real_image(s)
                ex = W.exclusion(s, img)
                G = W.colour_gradient(img)
                meas = BG.measure(G, self.Kw[s], self.T1[s], {k: segs_all[k] for k in BG.EDGES[s]}, ex, 4)
                rows = []
                for name, v in meas.items():
                    ok = np.isfinite(v['off'])
                    if ok.sum() < 5:
                        continue
                    rows.append((name, v['P'][ok], v['uv'][ok] + v['off'][ok][:, None] * v['nrm'], v['nrm']))
                self.bg[s] = rows
        self._claw_cache = {}
        self._arm_cache = {}

    # ---- parameters ----
    def n_arm(self):
        return dict(a=0, a_mount=0, b=6, b_capX=6, b_joints=6, c=9, b_mount=6, c_mount=9)[self.variant]

    def n_shared(self):
        return 6 if self.variant in MOUNT_VARIANTS else 3

    def steps_arm(self):
        r = np.radians(1.5)
        if self.variant == 'b_joints':
            return np.array([r] * 6)
        s = [r, r, r, 0.008, 0.008, 0.008]
        return np.array(s + [0.008] * 3 if self.variant in CAM_OFFSET else s)

    def steps_shared(self):
        s = [0.002, 0.003, 0.003]
        return np.array(s + [np.radians(1.5)] * 3 if self.variant in MOUNT_VARIANTS else s)

    # ---- geometry ----
    def claws_l6(self, cv):
        key = tuple(np.round(cv, 7))
        if key not in self._claw_cache:
            items = C.claw_tris_l6(self.pF, *cv[:3])
            if self.variant in MOUNT_VARIANTS:
                R = Rot.from_rotvec(cv[3:6]).as_matrix()
                items = [(l, (t - MOUNT_PIVOT_L6) @ R.T + MOUNT_PIVOT_L6) for l, t in items]
            if len(self._claw_cache) > 64:
                self._claw_cache.clear()
            self._claw_cache[key] = items
        return self._claw_cache[key]

    def arm_pose(self, side, av):
        """(base B, arm tris in base, T6 = link_6 in base, T_wc)."""
        B = self.bases[side]; arm, T6 = self.arm_home, self.T6_home
        Twc = self.T1[side].copy()
        if self.variant in BASE_VARIANTS:
            B = B @ delta_base(av[:6])
        if self.variant == 'b_joints':
            key = tuple(np.round(av[:6], 7))
            if key not in self._arm_cache:
                if len(self._arm_cache) > 16:
                    self._arm_cache.clear()
                self._arm_cache[key] = C.arm_tris(self.fk, q6=av[:6])
            arm, T6 = self._arm_cache[key]
        if self.variant in MOVING:        # stage-1 world rotation; translation rides with link_6
            Twc[:3, 3] = (B @ T6 @ self.X_ref[side])[:3, 3]
        if self.variant in CAM_OFFSET:
            Twc[:3, 3] += av[6:9]
        return B, arm, T6, Twc

    def render(self, side, cv, av):
        B, arm, T6, Twc = self.arm_pose(side, av)
        claws_w = C.transform(C.transform(self.claws_l6(cv), T6), B)
        arm_w = C.transform(arm, B)
        wrist_items = claws_w + [(l, t) for l, t in arm_w if l in (LAB['housing'], LAB['carriage'])]
        Lw = C.render_labels(wrist_items, self.Kw[side], Twc)
        Ls = C.render_labels(claws_w + arm_w, self.Ks, self.S)
        return Lw, Ls, (B, T6, Twc)

    # ---- loss ----
    def terms(self, side, cv, av, detail=False):
        Lw, Ls, (B, T6, Twc) = self.render(side, cv, av)
        yel = (LAB['clamp'], LAB['placeholder'])
        mw = dict(pad=fill(Lw == LAB['pad']), yellow=fill(np.isin(Lw, yel)), face=(Lw == LAB['pad_face']))
        mw['face'][560:] = False
        ms = dict(sil=Ls > 0, pad=Ls == LAB['pad'], yellow=np.isin(Ls, yel))
        rw, rs = self.wr[side], self.sr[side]
        iw = {k: iou(mw[k], rw[k]) for k in mw}
        isc = {k: iou(ms[k], rs[k]) for k in ms}
        t = dict(wrist=1 - np.mean(list(iw.values())), scene=1 - np.mean(list(isc.values())))
        X = np.linalg.inv(B @ T6) @ Twc
        pr = {}
        if self.variant not in ('a', 'a_mount'):
            dX = np.linalg.inv(self.X_ref[side]) @ X
            pr['x_rot'] = (np.linalg.norm(Rot.from_matrix(dX[:3, :3]).as_rotvec()) / SIG['x_rot']) ** 2
            if self.variant in CAM_OFFSET:
                pr['x_t'] = float(np.sum(((X[:3, 3] - self.X_ref[side][:3, 3]) / SIG['x_t']) ** 2))
        if self.variant in BASE_VARIANTS:
            pr['base'] = float((av[0] / SIG['b_yaw']) ** 2 + np.sum((av[1:3] / SIG['b_rp']) ** 2) + np.sum((av[3:6] / SIG['b_t']) ** 2))
        if self.variant == 'b_joints':
            pr['dq'] = float(np.sum((av[:6] / SIG['dq']) ** 2))
        if self.variant in MOVING:
            per = []
            for name, P, q, n in self.bg[side]:
                r = (W.project(P, self.Kw[side], Twc)[0] - q) @ n
                per.append(np.mean(2 * 4 * (np.sqrt(1 + (r / 2) ** 2) - 1)))
            t['background'] = self.lam_bg * float(np.mean(per))
        t['prior'] = LAM * float(sum(pr.values()))
        total = float(sum(t.values()))
        if not detail:
            return total
        det = dict(loss=total, terms={k: round(float(v), 5) for k, v in t.items()}, prior_sigma2={k: round(float(v), 3) for k, v in pr.items()},
                   iou_wrist={k: round(v, 4) for k, v in iw.items()}, iou_scene={k: round(v, 4) for k, v in isc.items()},
                   centroid_model_minus_real_px=dict(
                       wrist={k: centroid_offset(mw[k], rw[k]) for k in mw},
                       scene={k: centroid_offset(ms[k], rs[k]) for k in ms}),
                   T_world_base=B.tolist(), T_base_link6=T6.tolist(), T_world_cam=Twc.tolist(), T_link6_cam=X.tolist())
        return det, (Lw, Ls)

    def shared_penalty(self, cv):
        # the pads do not touch when closed (user 09-30): open_half > home stroke + 0.5 mm
        lim = self.home_gripper + 0.0005
        return 0.0 if cv[0] > lim else 1.0 + 100 * (lim - cv[0])


def nm(f, x0, steps, maxiter):
    s = np.asarray(steps, float)
    g = lambda u: f(x0 + u * s)
    n = len(x0)
    r = minimize(g, np.zeros(n), method='Nelder-Mead',
                 options=dict(xatol=0.01, fatol=2e-5, maxiter=maxiter, initial_simplex=np.vstack([np.zeros(n), np.eye(n)])))
    return x0 + r.x * s, float(r.fun), int(r.nfev)


def fit(P, cv, av, rounds=6, log=print, fix_shared=False):
    prev = np.inf
    for rd in range(rounds):
        t0 = time.time()
        if P.n_arm():
            for side in W.SIDES:
                av[side], f, n = nm(lambda x: P.terms(side, cv, x), av[side], P.steps_arm(), 1500 + 300 * P.n_arm())
                log(f'round {rd} arm {side}: loss {f:.5f} nfev {n} params {np.round(av[side], 5).tolist()}')
        if fix_shared:
            f = sum(P.terms(s, cv, av[s]) for s in W.SIDES); n = 0
        else:
            cv, f, n = nm(lambda x: sum(P.terms(s, x, av[s]) for s in W.SIDES) + P.shared_penalty(x), cv, P.steps_shared(), 2000)
        log(f'round {rd} shared: loss {f:.5f} nfev {n} C {np.round(cv, 5).tolist()} ({time.time() - t0:.0f} s)')
        if prev - f < 1e-4:
            break
        prev = f
    return cv, av


def describe_base(P, side, B):
    Bm = P.bases[side]
    d = np.linalg.inv(Bm) @ B
    ypr = np.degrees(Rot.from_matrix(d[:3, :3]).as_euler('ZYX'))
    out = dict(vs_metric=dict(yaw_pitch_roll_deg_base_axes=ypr.round(3).tolist(), t_base_mm=(d[:3, 3] * 1e3).round(1).tolist(),
                              t_world_mm=((B[:3, 3] - Bm[:3, 3]) * 1e3).round(1).tolist()),
               yaw_world_deg=round(float(np.degrees(np.arctan2(B[1, 0], B[0, 0]))), 3))
    if P.X_cap is not None:
        Bc = P.B_cap[side]
        dc = np.linalg.inv(Bc) @ B
        out['vs_capture_base'] = dict(yaw_pitch_roll_deg_base_axes=np.degrees(Rot.from_matrix(dc[:3, :3]).as_euler('ZYX')).round(3).tolist(),
                                      t_world_mm=((B[:3, 3] - Bc[:3, 3]) * 1e3).round(1).tolist())
    return out


def describe_X(P, side, X):
    out = {}
    for name, Xr in (('raiden', P.X_raiden), ('captures', P.X_cap)):
        if Xr is None:
            continue
        d = np.linalg.inv(Xr[side]) @ X
        out[f'vs_{name}'] = dict(rot_deg=round(float(np.degrees(np.linalg.norm(Rot.from_matrix(d[:3, :3]).as_rotvec()))), 3),
                                 rotvec_link6_deg=np.degrees(Rot.from_matrix(X[:3, :3] @ Xr[side][:3, :3].T).as_rotvec()).round(3).tolist(),
                                 t_link6_mm=((X[:3, 3] - Xr[side][:3, 3]) * 1e3).round(1).tolist())
    return out


def overlays(P, side, Lw, Ls, out):
    img = W.real_image(side).copy()
    rw = P.wr[side]
    for m, col in ((rw['pad'], (0, 0, 255)), (rw['yellow'], (255, 0, 255)), (rw['face'], (255, 255, 0))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, col, 1)
    for m, col in ((fill(Lw == LAB['pad']), (0, 140, 255)), (fill(np.isin(Lw, (LAB['clamp'], LAB['placeholder']))), (0, 255, 255)),
                   (Lw == LAB['pad_face'], (0, 255, 0))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, col, 1)
    cv2.imwrite(str(out / f'wrist_{side}.jpg'), img)
    sc = P.scene_img.copy()
    rs = P.sr[side]
    for m, col in ((rs['sil'], (0, 255, 0)), (rs['pad'], (0, 0, 255)), (rs['yellow'], (255, 0, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(sc, cs, -1, col, 1)
    for m, col in ((Ls > 0, (255, 255, 255)), (Ls == LAB['pad'], (0, 140, 255)), (np.isin(Ls, (LAB['clamp'], LAB['placeholder'])), (0, 255, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(sc, cs, -1, col, 1)
    x0 = 200 if side == 'left' else 600
    cv2.imwrite(str(out / f'scene_{side}_zoom4.jpg'), cv2.resize(sc[480:600, x0:x0 + 240], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', type=Path, required=True)
    ap.add_argument('--captures', type=Path)
    ap.add_argument('--variant', required=True, choices=['a', 'a_mount', 'b', 'b_capX', 'b_joints', 'c', 'b_mount', 'c_mount'])
    ap.add_argument('--start', type=Path, help='fit.json of an earlier variant: start C (and matching arm params)')
    ap.add_argument('--start-base', choices=['metric', 'implied'], default='implied',
                    help='b/b_capX/c start: metric base, or the base implied by the stage-1 camera and the X prior')
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--lam-bg', type=float, default=LAM_BG, help='background loss per px^2 (trade-off runs)')
    ap.add_argument('--C', type=float, nargs='*', help='start / fixed claw placement C (m, rad): open_half pad_z6 pad_x6 [mount rotvec]')
    ap.add_argument('--fix-shared', action='store_true', help='keep C fixed (final fit with the asset placement)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / 'fit.log', 'a')

    def log(*x):
        s = ' '.join(str(v) for v in x); print(s, flush=True); logf.write(s + '\n'); logf.flush()
    P = Problem(a)
    cv = np.array([0.052, -0.137, 0.0] + ([0.0] * 3 if a.variant in MOUNT_VARIANTS else []))
    av = {s: np.zeros(P.n_arm()) for s in W.SIDES}
    if a.start:
        st = W.jload(a.start)
        cv[:3] = st['shared']['C'][:3]
        for s in W.SIDES:
            p = np.array(st['arms'][s]['params'])
            if len(p) and st['variant'] in BASE_VARIANTS and a.variant in BASE_VARIANTS:
                n = min(len(p), len(av[s]))
                av[s][:n] = p[:n]
    elif a.variant in BASE_VARIANTS and a.start_base == 'implied':
        # base rotation that gives link_6 the world rotation implied by the stage-1 camera and X_ref (about the base origin)
        f0 = P.fk(np.zeros(6))
        for s in W.SIDES:
            Bi = P.T1[s] @ np.linalg.inv(f0['link_6'] @ P.X_ref[s])
            d = np.linalg.inv(P.bases[s]) @ Bi
            av[s][:3] = Rot.from_matrix(d[:3, :3]).as_euler('ZYX')
    elif a.variant == 'b_joints':       # joint offsets that best reproduce the link_6 rotation implied by camera + X_ref
        for s in W.SIDES:
            L = np.linalg.inv(P.bases[s]) @ P.T1[s] @ np.linalg.inv(P.X_ref[s])

            def r(q):
                T6 = P.fk(q, frames=('link_6',))['link_6']
                d = np.linalg.inv(L) @ T6
                return np.r_[Rot.from_matrix(d[:3, :3]).as_rotvec() / np.radians(0.5), q / np.radians(5)]
            from scipy.optimize import least_squares
            av[s] = least_squares(r, np.zeros(6)).x
            log(f'b_joints start {s}: dq deg {np.degrees(av[s]).round(2).tolist()}')
    t0 = time.time()
    log(f'variant {a.variant} start C {cv.round(5).tolist()} arms {[np.round(av[s], 5).tolist() for s in W.SIDES]}')
    for s in W.SIDES:
        log('start', s, json.dumps(P.terms(s, cv, av[s], True)[0]['terms']))
    if a.C:
        cv[:len(a.C)] = a.C
    cv, av = fit(P, cv, av, a.rounds, log, a.fix_shared)
    res = dict(variant=a.variant, inputs=dict(stage1=str(a.stage1), captures=str(a.captures) if a.captures else None,
                                              start=str(a.start) if a.start else None, start_base=a.start_base,
                                              C_given=a.C, fix_shared=a.fix_shared,
                                              claw_json=str(C.UMI / 'claw.json'), sigmas=SIG, lam=LAM, lam_bg=a.lam_bg),
               shared=dict(C=cv.tolist(), open_half_m=float(cv[0]), pad_z6_m=float(cv[1]), pad_x6_m=float(cv[2]),
                           gap_at_q0_mm=round(float(2 * (cv[0] - 0.0475) * 1e3), 2),
                           mount_rotvec_deg=np.degrees(cv[3:6]).round(3).tolist() if a.variant in MOUNT_VARIANTS else None),
               arms={}, seconds=round(time.time() - t0, 1))
    tot = {}
    for s in W.SIDES:
        det, (Lw, Ls) = P.terms(s, cv, av[s], True)
        B = np.array(det['T_world_base']); X = np.array(det['T_link6_cam'])
        det['params'] = av[s].tolist()
        if a.variant == 'b_joints':
            det['joint_offsets_deg'] = np.degrees(av[s]).round(3).tolist()
        det['base'] = describe_base(P, s, B)
        det['hand_eye'] = describe_X(P, s, X)
        if a.variant in CAM_OFFSET:
            det['camera_translation_vs_stage1_mm'] = (av[s][6:9] * 1e3).round(1).tolist()   # world offset from the link_6-riding position (key name kept)
        segs_all = W.named_segments()
        img = W.real_image(s)
        meas = BG.measure(W.colour_gradient(img), P.Kw[s], np.array(det['T_world_cam']), {k: segs_all[k] for k in BG.EDGES[s]},
                          W.exclusion(s, img), 12)
        det['background_edges'] = BG.summary(meas)
        res['arms'][s] = det
        for k, v in det['terms'].items():
            tot[k] = tot.get(k, 0) + v
        overlays(P, s, Lw, Ls, a.out)
        log(s, json.dumps({k: det[k] for k in ('terms', 'iou_wrist', 'iou_scene', 'centroid_model_minus_real_px')}))
        log(s, 'base', json.dumps(det['base']), 'X', json.dumps(det['hand_eye']), 'bg', det['background_edges']['all_samples_median_abs_px'])
    res['loss_terms_total'] = {k: round(v, 5) for k, v in tot.items()}
    res['loss_data'] = round(tot.get('wrist', 0) + tot.get('scene', 0), 5)
    (a.out / 'fit.json').write_text(json.dumps(res, indent=1))
    log('C', np.round(cv, 5).tolist(), 'total terms', res['loss_terms_total'], f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
