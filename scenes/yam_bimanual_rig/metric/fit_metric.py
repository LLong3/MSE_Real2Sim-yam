"""Reference -> metric transform: candidates, residuals at every anchor, MJCF arm registration.

Candidates (see metric_frame.py): A per-axis (recommended), B uniform from the desk length, C scale 1,
D blind uniform from the desk height. For each: desk box, floor, wall, arm ICP (MJCF q=0 at scale 1,
base on the rail top, yaw about Z), arm top height, rail top, camera height vs the ZED depth.
Writes work/fit_metric.json (all candidates) and work/arm_metric_<cand>.json (per-arm fits).
Run with the pi3x-mesh env.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import common
import metric_frame as mf
from arm_icp import rz, sample_surface, voxel

GAP = 0.0  # desk back edge flush with the wall (reference front-edge-to-wall 0.756)
S_A = (1.06, 1.00, 1.09)

rng = np.random.default_rng(1)


def build(a):
    cx, cy = a['desk_centre_xy']
    zf, ztop = a['floor_z_near_desk'], a['desk_top_z_at_centre']
    H = a['desk_minus_floor_per_view_median']
    def M_of(S, pin):
        S = np.asarray(S, float)
        M = np.eye(4)
        M[:3, :3] = np.diag(S)
        if pin == 'floor':      # floor -> 0 (desk top then lands at S_z * H)
            tz = -S[2] * zf
        else:                   # desk top -> 0.72
            tz = mf.TAPE['top'] - S[2] * ztop
        M[:3, 3] = [cx - S[0] * cx, cy - S[1] * cy, tz]
        return M
    return {
        'A_per_axis': dict(S=list(S_A), M=M_of(S_A, 'floor'),
                           note='sx 1.06 from the desk length (1.42-1.44 predicted), sy 1.00 (desk depth 0.756-0.77 predicted incl. ~1 cm wall gap; arm length), sz 1.09 from desk-top-minus-floor 0.661 (58 views); floor -> 0'),
        'B_uniform_1.06': dict(S=[1.06] * 3, M=M_of([1.06] * 3, 'top'), note='uniform 1.06 (desk length), desk top pinned at 0.72'),
        'C_scale_1': dict(S=[1.0] * 3, M=M_of([1.0] * 3, 'top'), note='scale 1, desk top pinned at 0.72 (vertical offset 6.9 cm)'),
        'D_uniform_1.09': dict(S=[1.09] * 3, M=M_of([1.09] * 3, 'floor'), note='blind uniform 1.09 (desk height)'),
    }


def arm_points(M, xc):
    d = np.load(common.WORK / 'clean_cloud.npz')
    P = d['points']
    box = (np.abs(P[:, 0] - xc) < 0.17) & (P[:, 1] > 1.0) & (P[:, 1] < 1.75) & (P[:, 2] > 0.675) & (P[:, 2] < 1.05)
    return mf.apply(M, voxel(P[box], 0.006))


def icp(A, model, x0, f_scale=0.01):
    tree = cKDTree(model)

    def res(p):
        q = (A - p[:3]) @ rz(p[3])
        return tree.query(q)[0]
    r = least_squares(res, np.asarray(x0, float), loss='soft_l1', f_scale=f_scale, x_scale=[0.01, 0.01, 0.01, 0.02])
    d = res(r.x)
    inl = d < 0.025
    r2 = least_squares(lambda p: res(p)[inl], r.x, loss='soft_l1', f_scale=f_scale, x_scale=[0.01, 0.01, 0.01, 0.02])
    d = res(r2.x)
    return r2.x, d


def arm_job(args):
    cand, M, name, xc, model_tag = args
    M = np.asarray(M)
    A = arm_points(M, xc)
    mesh = np.load(common.WORK / 'yam_q0_open0.0475.npz', allow_pickle=True)
    if model_tag == 'no_fingers':
        model = sample_surface(common.WORK / 'yam_q0_open0.0475_nofingers.npz', 300000)
    else:
        model = sample_surface(common.WORK / 'yam_q0_open0.0475.npz', 300000)
    c = mf.apply(M, np.array([[xc, 1.39, 0.668]]))[0]
    x, d = icp(A, model, [c[0], c[1], c[2], np.radians(91)])
    # arm top: 99th percentile height of points near link4 top (base-frame x in [-0.02, 0.10])
    q = (A - x[:3]) @ rz(x[3])
    near_top = (q[:, 0] > -0.02) & (q[:, 0] < 0.10) & (np.abs(q[:, 1]) < 0.05)
    top = float(np.percentile(A[near_top, 2], 99)) if near_top.sum() > 50 else float('nan')
    return dict(candidate=cand, arm=name, model=model_tag, base=x[:3], yaw_deg=float(np.degrees(x[3])),
                n=int(len(A)), median_m=float(np.median(d)), rms_inliers_m=float(np.sqrt(np.mean(d[d < 0.025] ** 2))),
                inlier_frac=float(np.mean(d < 0.025)), p90_m=float(np.percentile(d, 90)),
                arm_top_z_p99=top)


def main():
    a = mf.reference_anchors()
    cands = build(a)
    zm = json.load(open(common.WORK / 'zed_metric.json'))
    zed_scale = mf.TAPE['length'] / zm['desk_length_back_edge_m']  # ZED depth -> tape units
    cam_desk = np.array(zm['camera_in_desk_frame_m']) * zed_scale
    mesh = np.load(common.WORK / 'yam_q0_open0.0475.npz', allow_pickle=True)
    keep = ~np.isin(mesh['geom_body'][mesh['face_geom']], ['tip_left', 'tip_right'])
    np.savez(common.WORK / 'yam_q0_open0.0475_nofingers.npz', vertices=mesh['vertices'], faces=mesh['faces'][keep])
    jobs = []
    for k, c in cands.items():
        for name, xc in (('left_arm', -2.56), ('right_arm', -1.97)):
            for tag in (('no_fingers', 'with_fingers') if k == 'A_per_axis' else ('no_fingers',)):
                jobs.append((k, c['M'].tolist(), name, xc, tag))
    with ProcessPoolExecutor(max_workers=6) as ex:
        arm_res = list(ex.map(arm_job, jobs))
    # reference points for other anchors
    V, MV, _ = common.points()
    P = np.asarray(V)[MV]
    X, Y, Z = P.T
    rail = (X > -2.45) & (X < -2.30) & (Y > 1.37) & (Y < 1.43) & (Z > 0.64) & (Z < 0.70)
    rail2 = (X > -2.15) & (X < -2.08) & (Y > 1.37) & (Y < 1.43) & (Z > 0.64) & (Z < 0.70)
    rail_top_ref = float(np.median(np.r_[Z[rail], Z[rail2]]))
    cam_body = (X > -2.30) & (X < -2.10) & (Y > 1.40) & (Y < 1.55) & (Z > 1.44) & (Z < 1.54)
    cam_body_ref = np.median(P[cam_body], 0)
    out = dict(reference_anchors=a, zed_depth_to_tape_scale=zed_scale, camera_in_desk_frame_tape_m=cam_desk,
               rail_top_ref_z=rail_top_ref, camera_body_ref_median=cam_body_ref, candidates={})
    for k, c in cands.items():
        M = c['M']
        S = np.diag(M[:3, :3])
        L = a['desk_length'] * S[0]
        D = a['desk_depth_front_to_wall'] * S[1]
        floor = float(mf.apply(M, np.array([[-2.29, 1.70, a['floor_z_near_desk']]]))[0, 2])
        top = float(mf.apply(M, np.array([[-2.29, 1.70, a['desk_top_z_at_centre']]]))[0, 2])
        rail_top = float(mf.apply(M, np.array([[-2.29, 1.40, rail_top_ref]]))[0, 2])
        wall_top = {h: float(mf.apply(M, np.array([[-2.29, y, 0]]))[0, 1]) for h, y in a['wall_y_by_height'].items()}
        cam_ref_m = mf.apply(M, cam_body_ref[None])[0]
        arms = [r for r in arm_res if r['candidate'] == k]
        res = dict(
            S=S, M=M, note=c['note'],
            desk_length_m=L, desk_length_residual_m=L - mf.TAPE['length'],
            desk_depth_front_to_wall_m=D, desk_depth_residual_m=D - (mf.TAPE['depth'] + GAP),
            desk_top_z=top, desk_top_residual_m=top - mf.TAPE['top'],
            floor_near_desk_z=floor, floor_residual_m=floor,
            desk_height_above_floor_m=top - floor,
            rail_top_above_desk_m=rail_top - top,
            wall_y_metric_by_height=wall_top,
            camera_body_height_above_desk_m=float(cam_ref_m[2] - top),
            zed_camera_height_above_desk_m=float(cam_desk[2]),
            arms=arms,
        )
        for r in arms:
            if r['model'] == 'no_fingers':
                r['base_above_desk_top_m'] = float(r['base'][2] - top)
                r['arm_top_above_base_m'] = float(r['arm_top_z_p99'] - r['base'][2])
                r['arm_top_residual_vs_mjcf_m'] = r['arm_top_above_base_m'] - 0.2705
        out['candidates'][k] = res
        print(k, 'S', S.round(4), 'len', round(L, 4), 'depth', round(D, 4), 'floor', round(floor, 4), 'top', round(top, 4),
              'rail+', round(rail_top - top, 4), 'cam body h', round(cam_ref_m[2] - top, 4), 'zed h', round(cam_desk[2], 4))
        for r in arms:
            print('   ', r['arm'], r['model'], np.round(r['base'], 4), round(r['yaw_deg'], 2), 'med mm', round(r['median_m'] * 1000, 2),
                  'rms mm', round(r['rms_inliers_m'] * 1000, 2), 'inl', round(r['inlier_frac'], 3),
                  'top-base', round(r.get('arm_top_above_base_m', np.nan), 4))
    common.save_json(common.WORK / 'fit_metric.json', out)


if __name__ == '__main__':
    main()
