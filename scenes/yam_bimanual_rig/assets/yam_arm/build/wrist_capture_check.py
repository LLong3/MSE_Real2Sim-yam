"""Arm base and hand-eye from raiden's 09-29 calibration captures, in the metric frame (CPU, raiden venv; read only).

    ~/robot/raiden/.venv/bin/python build/wrist_capture_check.py --out RUN/captures [--square 0.03556]

The 7 captures of ~/.config/raiden/calibration_captures/20260929_171127 (17:11, 1920x1200) show the taped ChArUco
board (10x7, DICT_4X4_50) in all three cameras at 7 arm poses. The scene camera is the verified metric camera
(metric_frame.json, desk edges <= 0.7 px), so it places the board in the world. Each wrist view then gives the
wrist camera in the world at a known joint vector, and per arm the base pose B and the hand-eye X
(camera in grasp_site) are both identifiable: T_world_cam_i = B FK(q_i) X.

Detection: gripper tags share ids with the board; markers off the board plane are dropped (RANSAC homography on
the marker centres, as ~/robot/bimanual_bringup/calib/solve_captures.py). Fit: pixel residuals of all board
corners in the wrist images (+ the scene image for the board pose), soft-L1 1 px. Variants:
  fixed35   square --square (default 35 mm, the installed calibration; the user measured 35.56 mm on 09-30),
            board pose from the scene camera only (the variant keeps its historical name);
  joint     same square, board pose refined with the scene and wrist corners;
  scale     as joint, square size free (the robot motion is metric);
  offsets   as joint, plus offsets of joints 2, 3, 5 per arm (prior 2 deg).
The marker side keeps the board ratio 25.2 / 35.
Writes <out>/captures.json with B, X, residuals, the predicted home wrist camera and its difference to the
raiden chain (hand-eye + base implied by raiden's scene extrinsic and the metric scene camera).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import wristfit_common as W

SQ = 0.035
MK = 0.0252
MK_RATIO = 0.0252 / 0.035


def board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    b = cv2.aruco.CharucoBoard((10, 7), SQ, MK, d)
    return b, cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def detect(img, b, det):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mc, mid, _ = det.detectMarkers(gray)
    if mid is None or len(mid) < 4:
        return None
    centers = {int(i): np.asarray(o).reshape(4, 3)[:, :2].mean(0) for o, i in zip(b.getObjPoints(), b.getIds().ravel())}
    src = np.array([centers[int(i)] for i in mid.ravel()]); dst = np.array([c[0].mean(0) for c in mc])
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    keep = inl.ravel().astype(bool)
    n, c, ids = cv2.aruco.interpolateCornersCharuco(tuple(m for m, k in zip(mc, keep) if k), mid[keep], gray, b)
    if not n or n < 6:
        return None
    return c.reshape(-1, 2), ids.ravel(), int((~keep).sum())


def rt(x):
    return W.T_of(Rot.from_rotvec(x[:3]).as_matrix(), x[3:6])


def xr(T):
    return np.r_[Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def proj(Pw, K, Twc):
    pc = (Pw - Twc[:3, 3]) @ Twc[:3, :3]
    return pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--square', type=float, default=0.035, help='ChArUco square side (m)')
    a = ap.parse_args()
    global SQ, MK
    SQ, MK = a.square, a.square * MK_RATIO
    a.out.mkdir(parents=True, exist_ok=True)
    meta = W.jload(W.CAPTURES / 'captures.json')
    cal = W.jload(W.CAL)
    m = W.jload(W.METRIC)
    Ks = {n: np.array(v['camera_matrix'], float) for n, v in meta['intrinsics'].items()}
    Kmet, S = W.scene_camera(m)
    b, det = board()
    obj = b.getChessboardCorners()                        # (54, 3) board frame, metres at 35 mm
    fk = W.FK()
    arms = {'left': 'follower_l', 'right': 'follower_r'}
    data = {'scene': [], 'left': [], 'right': []}
    det_log = []
    for cap in meta['captures']:
        row = dict(pose=cap['pose'])
        for cam in ('scene_camera', 'left_wrist_camera', 'right_wrist_camera'):
            r = detect(cv2.imread(str(W.CAPTURES / cap['images'][cam])), b, det)
            row[cam] = None if r is None else dict(corners=len(r[1]), dropped_markers=r[2])
            if r is None:
                continue
            key = cam.split('_')[0]
            q = np.array(cap['joints'][arms[key]])[:6] if key in arms else None
            data[key].append(dict(pose=cap['pose'], uv=r[0], ids=r[1], q=q))
        det_log.append(row)

    # board pose in the world from the scene camera (metric pose, raiden's full-res K), per capture and joint
    Ksc = Ks['scene_camera']
    per = []
    for d in data['scene']:
        ok, rv, tv = cv2.solvePnP(obj[d['ids']], d['uv'], Ksc, None, flags=cv2.SOLVEPNP_ITERATIVE)
        per.append(S @ W.T_of(cv2.Rodrigues(rv)[0], tv))
    Wb0 = W.T_of(Rot.from_matrix([P[:3, :3] for P in per]).mean().as_matrix(), np.mean([P[:3, 3] for P in per], 0))
    spread = [dict(pose=d['pose'], dt_mm=np.round((P[:3, 3] - Wb0[:3, 3]) * 1e3, 2).tolist(),
                   drot_deg=round(W.pose_delta(Wb0, P)[0], 3)) for d, P in zip(data['scene'], per)]

    def board_world(Wb, s):
        return (obj * s) @ Wb[:3, :3].T + Wb[:3, 3]

    fkq = {side: [fk(d['q'])['grasp_site'] for d in data[side]] for side in W.SIDES}
    bases = W.metric_bases(m)
    X0 = {s: W.hand_eye(s, cal) for s in W.SIDES}

    def unpack(x, var):
        i = 0
        out = {}
        for s in W.SIDES:
            out[s] = dict(B=rt(x[i:i + 6]), X=rt(x[i + 6:i + 12])); i += 12
            if var == 'offsets':
                out[s]['dq'] = x[i:i + 3]; i += 3
        Wb = rt(x[i:i + 6]) if var != 'fixed35' else Wb0; i += 6 if var != 'fixed35' else 0
        sc = x[i] if var == 'scale' else 1.0
        return out, Wb, sc

    def fkd(side, j, dq):
        if dq is None:
            return fkq[side][j]
        q = data[side][j]['q'].copy(); q[[1, 2, 4]] += dq
        return fk(q)['grasp_site']

    def residuals(x, var, detail=False):
        P, Wb, sc = unpack(x, var)
        r, det_ = [], {}
        if var != 'fixed35':
            Pw = board_world(Wb, sc)
            for d in data['scene']:
                e = proj(Pw[d['ids']], Ksc, S) - d['uv']
                r.append(e.ravel()); det_.setdefault('scene', []).append(np.linalg.norm(e, axis=1))
        Pw = board_world(Wb, sc)
        for s in W.SIDES:
            K = Ks[f'{s}_wrist_camera']
            for j, d in enumerate(data[s]):
                Twc = P[s]['B'] @ fkd(s, j, P[s].get('dq')) @ P[s]['X']
                e = proj(Pw[d['ids']], K, Twc) - d['uv']
                r.append(e.ravel()); det_.setdefault(s, []).append(np.linalg.norm(e, axis=1))
            if var == 'offsets':
                r.append(np.degrees(P[s]['dq']) / 2.0)
        return (np.concatenate(r), det_) if detail else np.concatenate(r)

    results = dict(inputs=dict(captures=str(W.CAPTURES), calibration=str(W.CAL), metric=str(W.METRIC),
                               board=dict(squares=[10, 7], square_m=SQ, marker_m=MK, dictionary='DICT_4X4_50')),
                   detections=det_log, board_from_scene_camera=dict(T_world_board=Wb0.tolist(), per_capture=spread))
    # board plane vs the desk top (the board is taped on the desk)
    Pw0 = board_world(Wb0, 1.0)
    results['board_from_scene_camera']['board_z_m'] = dict(min=float(Pw0[:, 2].min()), max=float(Pw0[:, 2].max()), mean=float(Pw0[:, 2].mean()))
    results['board_from_scene_camera']['board_normal_world'] = Wb0[:3, 2].round(4).tolist()
    x_prev = None
    for var in ('fixed35', 'joint', 'scale', 'offsets'):
        x0 = []
        for s in W.SIDES:
            x0 += list(xr(bases[s])) + list(xr(X0[s]))
            if var == 'offsets':
                x0 += [0, 0, 0]
        if var != 'fixed35':
            x0 += list(xr(Wb0))
        if var == 'scale':
            x0 += [1.0]
        x0 = np.array(x0, float)
        if x_prev is not None:     # warm start the shared blocks
            P, Wb, _ = unpack(x_prev[0], x_prev[1])
            j = 0
            for s in W.SIDES:
                x0[j:j + 6] = xr(P[s]['B']); x0[j + 6:j + 12] = xr(P[s]['X']); j += 15 if var == 'offsets' else 12
        sol = least_squares(residuals, x0, args=(var,), loss='soft_l1', f_scale=1.0, max_nfev=4000)
        x_prev = (sol.x, var)
        P, Wb, sc = unpack(sol.x, var)
        _, det_ = residuals(sol.x, var, True)
        out = dict(square_m=SQ * sc, T_world_board=Wb.tolist(), arms={})
        zb = board_world(Wb, sc)[:, 2]
        out['board_above_desk_top_mm'] = dict(mean=round(float((zb.mean() - m['desk']['top_z_m']) * 1e3), 1),
                                              min=round(float((zb.min() - m['desk']['top_z_m']) * 1e3), 1),
                                              max=round(float((zb.max() - m['desk']['top_z_m']) * 1e3), 1))
        out['residual_px'] = {k: dict(median=round(float(np.median(np.concatenate(v))), 2), p90=round(float(np.percentile(np.concatenate(v), 90)), 2),
                                      per_capture_median=[round(float(np.median(e)), 2) for e in v]) for k, v in det_.items()}
        fk0 = fk(W.HOME_Q)
        for s in W.SIDES:
            B, X = P[s]['B'], P[s]['X']
            Th = B @ fk0['grasp_site'] @ X
            ang_b, dt_b = W.pose_delta(bases[s], B)
            ang_x, dt_x = W.pose_delta(X0[s], X)
            e = Rot.from_matrix(bases[s][:3, :3].T @ B[:3, :3]).as_euler('xyz', degrees=True)
            out['arms'][s] = dict(
                T_world_base=B.tolist(), T_grasp_site_cam=X.tolist(), T_world_cam_home=Th.tolist(),
                T_link6_cam=(np.linalg.inv(fk0['link_6']) @ fk0['grasp_site'] @ X).tolist(),
                base_vs_metric=dict(rot_deg=round(ang_b, 3), trans_world_mm=np.round(dt_b, 1).tolist(),
                                    rot_xyz_deg_in_metric_base=np.round(e, 3).tolist(),
                                    yaw_deg=round(float(np.degrees(np.arctan2(B[1, 0], B[0, 0]))), 3)),
                hand_eye_vs_raiden=dict(rot_deg=round(ang_x, 3), trans_grasp_site_mm=np.round(dt_x, 1).tolist(),
                                        t_mm=np.round(X[:3, 3] * 1e3, 1).tolist()),
                joint_offsets_deg=np.degrees(P[s]['dq']).round(3).tolist() if 'dq' in P[s] else None)
        results[var] = out
        print(var, 'square', round(SQ * sc * 1e3, 3), 'mm', json.dumps(out['residual_px']), 'board above desk', out['board_above_desk_top_mm'])
        for s in W.SIDES:
            o = out['arms'][s]
            print('  ', s, 'base vs metric', o['base_vs_metric'], '| X vs raiden', o['hand_eye_vs_raiden'], '| dq', o['joint_offsets_deg'])
    # raiden's own chain in the metric frame: base implied by raiden's scene extrinsic and the metric scene camera
    sc_ext = cal['cameras']['scene_camera']['extrinsics']
    B_l = S @ np.linalg.inv(W.T_of(sc_ext['rotation_matrix'], sc_ext['translation_vector']))
    B_r = B_l @ np.linalg.inv(np.array(cal['bimanual_transform']['right_base_to_left_base']))
    fk0 = fk(W.HOME_Q)
    results['raiden_chain'] = {s: dict(T_world_base=Bs.tolist(), T_world_cam_home=(Bs @ fk0['grasp_site'] @ X0[s]).tolist(),
                                       base_vs_metric=dict(rot_deg=round(W.pose_delta(bases[s], Bs)[0], 3),
                                                           trans_world_mm=np.round(W.pose_delta(bases[s], Bs)[1], 1).tolist()))
                               for s, Bs in (('left', B_l), ('right', B_r))}
    (a.out / 'captures.json').write_text(json.dumps(results, indent=1))
    print('board z', results['board_from_scene_camera']['board_z_m'], 'normal', results['board_from_scene_camera']['board_normal_world'])
    print('board per-capture spread', spread)


if __name__ == '__main__':
    main()
