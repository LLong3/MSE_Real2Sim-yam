"""Stage 1 (09-30): wrist-camera poses from the background only (CPU, raiden venv).

    ~/robot/raiden/.venv/bin/python build/wrist_bg_fit.py --out RUN/stage1

Background model: named 3D edges of the metric frame and the v2 build config (wristfit_common.named_segments);
per camera only the edges identified in its 09-29 image are used (EDGES). The grippers, claws and the croissant
are masked (hand-drawn polygons + colour masks). Per sample of a projected model edge, the real edge is the
colour-gradient peak along the edge normal (sub-pixel); the pose minimises the normal distances (soft-L1, 2 px),
re-searching with a shrinking window (moving-edges tracking, start: the v1 line-fit rotation).
Fits: (a) stage-1 camera = rotation fitted, translation kept at the raw hand-eye camera (metric base): the
background fixes the rotation but not the translation; (b) free 6-DoF; (c) 6-DoF with a 20 mm translation prior.
Reports per-edge offsets for the raw hand-eye, v1, v2 and the fits, the 6-DoF covariance at (a) (1 px per edge
sample, samples decimated to one per 8 px) and the fitted camera in world, link_6 and grasp_site (metric base).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import wristfit_common as W


def edge_search(G, uv, nrm, search, ex, step=0.5, thresh=12.0):
    """For samples uv (N,2) with normal nrm (2,), find the strongest colour-gradient peak within +-search px.
    Returns offsets (N,) (nan where none) and the peak strength."""
    gx, gy = G
    H, Wd = ex.shape
    s = np.arange(-search, search + step / 2, step)
    q = uv[:, None, :] + s[None, :, None] * nrm[None, None, :]          # N, S, 2
    xi, yi = q[..., 0], q[..., 1]
    ok = (xi >= 1) & (xi < Wd - 2) & (yi >= 1) & (yi < H - 2)
    x0 = np.clip(np.floor(xi).astype(int), 0, Wd - 2); y0 = np.clip(np.floor(yi).astype(int), 0, H - 2)
    fx, fy = xi - x0, yi - y0

    def bil(A):
        return ((A[y0, x0] * ((1 - fx) * (1 - fy))[..., None]) + (A[y0, x0 + 1] * (fx * (1 - fy))[..., None])
                + (A[y0 + 1, x0] * ((1 - fx) * fy)[..., None]) + (A[y0 + 1, x0 + 1] * (fx * fy)[..., None]))
    g = np.abs(bil(gx) * nrm[0] + bil(gy) * nrm[1]).max(-1)          # N, S
    g[~ok] = 0
    g[ex[np.clip(np.round(yi).astype(int), 0, H - 1), np.clip(np.round(xi).astype(int), 0, Wd - 1)]] = 0
    i = g.argmax(1)
    peak = g[np.arange(len(g)), i]
    off = s[i].astype(float)
    inner = (i > 0) & (i < len(s) - 1)
    a, b, c = g[np.arange(len(g)), np.clip(i - 1, 0, None)], peak, g[np.arange(len(g)), np.clip(i + 1, None, len(s) - 1)]
    den = a - 2 * b + c
    sub = np.where(inner & (den < 0), 0.5 * (a - c) / np.where(den == 0, -1, den), 0.0)
    off = off + sub * step
    off[(peak < thresh) | ~inner] = np.nan
    return off, peak


def samples(seg, K, T, ex, n=240, margin=4):
    """projected samples of a 3D segment inside the image and outside the exclusion; world points, uv, normal."""
    a, b = map(np.asarray, seg)
    P = a + np.linspace(0, 1, n)[:, None] * (b - a)
    uv, z = W.project(P, K, T)
    keep = (z > 0.05) & (uv[:, 0] > margin) & (uv[:, 0] < 960 - margin) & (uv[:, 1] > margin) & (uv[:, 1] < 600 - margin)
    if keep.sum() < 6:
        return None
    P, uv = P[keep], uv[keep]
    inside = ~ex[np.clip(np.round(uv[:, 1]).astype(int), 0, 599), np.clip(np.round(uv[:, 0]).astype(int), 0, 959)]
    if inside.sum() < 6:
        return None
    t = uv[-1] - uv[0]; t /= np.linalg.norm(t)
    # decimate to ~one sample per 3 px of image length
    P, uv = P[inside], uv[inside]
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
    pick = np.unique(np.searchsorted(L, np.arange(0, L[-1] + 1e-6, 3.0)).clip(0, len(L) - 1))
    return P[pick], uv[pick], np.array([-t[1], t[0]])


def measure(G, K, T, segs, ex, search):
    out = {}
    for name, seg in segs.items():
        s = samples(seg, K, T, ex)
        if s is None:
            continue
        P, uv, nrm = s
        off, peak = edge_search(G, uv, nrm, search, ex)
        out[name] = dict(P=P, uv=uv, nrm=nrm, off=off, peak=peak)
    return out


def summary(meas):
    res = {}
    for k, v in meas.items():
        o = v['off'][np.isfinite(v['off'])]
        if len(o) >= 5:
            res[k] = dict(n=int(len(o)), n_samples=int(len(v['off'])), median_abs_px=round(float(np.median(np.abs(o))), 2),
                          mean_signed_px=round(float(np.mean(o)), 2), p90_abs_px=round(float(np.percentile(np.abs(o), 90)), 2))
        else:
            res[k] = dict(n=int(len(o)), n_samples=int(len(v['off'])), note='edge not found (occluded, outside or no gradient)')
    allo = np.concatenate([v['off'][np.isfinite(v['off'])] for v in meas.values()]) if meas else np.zeros(0)
    per = [r['median_abs_px'] for r in res.values() if 'median_abs_px' in r]
    return dict(per_edge=res, all_samples_median_abs_px=round(float(np.median(np.abs(allo))), 2) if len(allo) else None,
                mean_of_edge_medians_px=round(float(np.mean(per)), 2) if per else None, edges_found=len(per))


def fit(G, K, T0, segs, ex, schedule=(14, 10, 7, 5, 4, 4), use=None, prior=None):
    """moving-edges 6-DoF fit. prior: optional (T_ref, sigma_rad, sigma_m) soft prior."""
    T = T0.copy()
    log = []
    for search in schedule:
        meas = measure(G, K, T, {k: v for k, v in segs.items() if use is None or k in use}, ex, search)
        rows = []
        for name, v in meas.items():
            ok = np.isfinite(v['off'])
            if ok.sum() < 5:
                continue
            q = v['uv'][ok] + v['off'][ok][:, None] * v['nrm']
            w = 1.0 / np.sqrt(ok.sum() / 20.0) if ok.sum() > 20 else 1.0     # long edges do not dominate
            rows.append((name, v['P'][ok], q, v['nrm'], w))

        def resid(x):
            Tx = W.perturb(T, x)
            r = []
            for name, P, q, n, w in rows:
                uv, _ = W.project(P, K, Tx)
                r.append(w * ((uv - q) @ n))
            if prior is not None:
                Tr, sr, st = prior
                d = np.linalg.inv(Tr) @ Tx
                r.append(Rot.from_matrix(d[:3, :3]).as_rotvec() / sr)
                r.append((Tx[:3, 3] - Tr[:3, 3]) / st)
            return np.concatenate(r)
        sol = least_squares(resid, np.zeros(6), loss='soft_l1', f_scale=2.0, x_scale=[0.01] * 3 + [0.005] * 3)
        T = W.perturb(T, sol.x)
        log.append(dict(search_px=search, edges=[r[0] for r in rows], n=int(sum(len(r[1]) for r in rows)),
                        step_deg=round(float(np.degrees(np.linalg.norm(sol.x[:3]))), 3), step_mm=round(float(np.linalg.norm(sol.x[3:]) * 1e3), 2)))
    return T, log, rows


def covariance(K, T, rows, px=1.0, per_px=8.0):
    """pose covariance from the Jacobian of the final rows; 1 px noise per sample, samples decimated to 1 per 8 px
    (neighbouring samples are correlated). Returns std of (rot xyz deg in camera frame, translation xyz mm in world)."""
    def f(x):
        Tx = W.perturb(T, x)
        return np.concatenate([((W.project(P, K, Tx)[0] - q) @ n) for _, P, q, n, _ in rows])
    x0 = np.zeros(6); f0 = f(x0); J = np.zeros((len(f0), 6))
    for i in range(6):
        h = 1e-5 if i < 3 else 1e-5
        e = np.zeros(6); e[i] = h
        J[:, i] = (f(e) - f0) / h
    J = J / np.sqrt(per_px / 3.0)      # samples are 3 px apart
    C = np.linalg.pinv(J.T @ J) * px ** 2
    sd = np.sqrt(np.diag(C))
    ev, evec = np.linalg.eigh(C[3:, 3:])
    return dict(rot_sd_deg=np.degrees(sd[:3]).round(3).tolist(), trans_sd_mm=(sd[3:] * 1e3).round(2).tolist(),
                weakest_translation_dir_world=evec[:, -1].round(3).tolist(), weakest_translation_sd_mm=round(float(np.sqrt(ev[-1]) * 1e3), 2),
                cov=C.tolist())


EDGES = {  # background edges identified in each 09-29 wrist image (checked on the v1/v2 overlays)
    'left': ['desk_back_edge', 'desk_left_edge', 'wall_end_vertical'],
    'right': ['desk_back_edge', 'desk_right_edge', 'monitor_desk_left_edge', 'privacy_panel_left_edge'],
}
SCHEDULE = (14, 10, 7, 5, 4, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = W.jload(W.METRIC); cal = W.jload(W.CAL)
    segs_all = W.named_segments(m)
    fk = W.FK()(W.HOME_Q)
    bases = W.metric_bases(m)
    builds = W.build_poses()
    res = dict(inputs=dict(metric=str(W.METRIC), build_config=str(W.BUILD_CFG), calibration=str(W.CAL),
                           images=[str(W.REAL / f'{s}_wrist_camera.jpg') for s in W.SIDES]),
               segments={k: [list(map(float, p)) for p in v] for k, v in segs_all.items()}, edges_used=EDGES,
               method=('moving-edges fit: colour-gradient peak along the projected edge normal (search 14 -> 4 px), '
                       'soft-L1 2 px; grippers/claws/croissant masked'), cameras={})
    for side in W.SIDES:
        img = W.real_image(side)
        ex = W.exclusion(side, img)
        G = W.colour_gradient(img)
        K = W.wrist_K(side, cal)
        X = W.hand_eye(side, cal)
        segs = {k: segs_all[k] for k in EDGES[side]}
        T_he = bases[side] @ fk['grasp_site'] @ X
        cands = dict(raw_hand_eye=T_he, v1_line_fit=builds['delivery_v1'][side], v2_claw_fit=builds['delivery_v2'][side])
        r = dict(K=K.tolist(), candidates={})
        for name, T in cands.items():
            r['candidates'][name] = dict(T_world_cam=T.tolist(), edges=summary(measure(G, K, T, segs, ex, 12)))
        # (a) stage-1 camera: rotation from the background, translation of the raw hand-eye camera (metric base)
        T0 = cands['v1_line_fit'].copy(); T0[:3, 3] = T_he[:3, 3]
        Ta, log_a, rows_a = fit(G, K, T0, segs, ex, SCHEDULE, prior=(T0, np.radians(30), 1e-5))
        # (b) free 6-DoF from (a); (c) 6-DoF with a 20 mm translation prior at the hand-eye position
        Tb, log_b, rows_b = fit(G, K, Ta, segs, ex, SCHEDULE)
        Tc, log_c, _ = fit(G, K, Ta, segs, ex, SCHEDULE, prior=(T0, np.radians(30), 0.02))
        cov = covariance(K, Ta, rows_a)
        out = {}
        for name, T, log in (('stage1_rotation_only', Ta, log_a), ('free_6dof', Tb, log_b), ('prior20mm_6dof', Tc, log_c)):
            comp = {}
            for cn, Tn in cands.items():
                ang, dt = W.pose_delta(Tn, T)
                Tl = W.l6_of_world(side, Tn, fk, bases)
                Tl_fit = W.l6_of_world(side, T, fk, bases)
                dl = np.linalg.inv(Tl) @ Tl_fit
                comp[cn] = dict(rot_deg=round(ang, 3), trans_world_mm=np.round(dt, 1).tolist(),
                                rot_link6_rotvec_deg=np.degrees(Rot.from_matrix(dl[:3, :3]).as_rotvec()).round(3).tolist(),
                                trans_link6_mm=np.round((Tl_fit[:3, 3] - Tl[:3, 3]) * 1e3, 1).tolist(),
                                world_rotvec_deg=np.degrees(Rot.from_matrix(T[:3, :3] @ Tn[:3, :3].T).as_rotvec()).round(3).tolist())
            out[name] = dict(T_world_cam=T.tolist(), T_link6_cam_via_metric_base=W.l6_of_world(side, T, fk, bases).tolist(),
                             T_grasp_site_cam_via_metric_base=(np.linalg.inv(bases[side] @ fk['grasp_site']) @ T).tolist(),
                             edges=summary(measure(G, K, T, segs, ex, 12)), log=log, delta_vs=comp)
        out['stage1_rotation_only']['covariance_6dof'] = cov
        r['fits'] = out
        res['cameras'][side] = r
        print(side)
        for name, v in list(r['candidates'].items()) + [(k, v) for k, v in out.items()]:
            e = v['edges']
            print(f'  {name:22s} all-sample median {e["all_samples_median_abs_px"]}  mean of edge medians {e["mean_of_edge_medians_px"]}',
                  {k: v2.get('median_abs_px') for k, v2 in e['per_edge'].items()})
        for name in out:
            print('  ', name, {k: (v['rot_deg'], v['trans_world_mm']) for k, v in out[name]['delta_vs'].items()})
        print('  cov', cov['rot_sd_deg'], cov['trans_sd_mm'], 'weakest', cov['weakest_translation_dir_world'], cov['weakest_translation_sd_mm'])
    (a.out / 'stage1_background_fit.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
