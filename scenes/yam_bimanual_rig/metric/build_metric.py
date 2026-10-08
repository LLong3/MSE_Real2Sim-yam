"""Assemble the metric world frame of the YAM bimanual rig twin -> metric_frame.json.

Inputs (all produced by the scripts in this directory from read-only sources):
  work/reference_anchors.json, work/floor_check.json  reference desk/floor anchors (metric_frame.py, floor_check.py)
  work/fit_metric.json                                 transform candidates + MJCF arm ICP per candidate (fit_metric.py)
  work/zed_metric.json/.npz                            ZED depth + edge geometry, camera in the desk frame (zed_metric.py)
  work/arm_zed_fit2.json                               MJCF gripper silhouettes vs the real ZED images (arm_zed_fit2.py)
  work/pnp_A_0929.npz                                  2D-3D PnP of the 09-29 image vs the reference (pnp_warp.py)
Final transform: p_metric = diag(1.06, 1.00, 1.09) p_ref + t; t: z from floor -> 0, x/y from a weighted
least-squares over the desk edges (sigma 1.5 cm) and the ZED-measured arm bases (sigma 1 cm).
"""
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import common
import metric_frame as mf
from arm_icp import rz, sample_surface, voxel

OUT = common.HERE / 'metric_frame.json'
S_FINAL = np.array([1.06, 1.00, 1.09])
MJCF_TOP = 0.2705  # link4 top above the base mounting face at q=0 (yam.xml + linear_4310)


def load():
    J = lambda p: json.load(open(common.WORK / p))
    return dict(a=J('reference_anchors.json'), fl=J('floor_check.json'), fm=J('fit_metric.json'),
                zm=J('zed_metric.json'), z2=J('arm_zed_fit2.json'), zn=np.load(common.WORK / 'zed_metric.npz'),
                pnp=np.load(common.WORK / 'pnp_A_0929.npz'))


def desk_box(a):
    """Metric desk box: centred on the reference desk centre in X, back edge on the reference wall line."""
    cx = a['desk_centre_xy'][0]
    yb = a['wall_y_by_height']['0.70-0.80']
    x0, x1 = cx - mf.TAPE['length'] / 2, cx + mf.TAPE['length'] / 2
    y0, y1 = yb - mf.TAPE['depth'], yb
    return dict(x=[x0, x1], y=[y0, y1], z_top=mf.TAPE['top'], back_left_top=[x0, y1, mf.TAPE['top']])


def wlsq_offset(s, xr, xm, w):
    t = np.sum(w * (xm - s * xr)) / np.sum(w)
    return t, s * xr + t - xm


def icp_fixed_z(args):
    A, x0, zb, model_path = args
    model = sample_surface(model_path, 300000)
    tree = cKDTree(model)

    def res(p):
        q = (A - [p[0], p[1], zb]) @ rz(p[2])
        return tree.query(q)[0]
    r = least_squares(res, x0, loss='soft_l1', f_scale=0.01, x_scale=[0.01, 0.01, 0.02])
    d = res(r.x)
    inl = d < 0.025
    r = least_squares(lambda p: res(p)[inl], r.x, loss='soft_l1', f_scale=0.01, x_scale=[0.01, 0.01, 0.02])
    d = res(r.x)
    return r.x, float(np.median(d)), float(np.sqrt(np.mean(d[d < 0.025] ** 2))), float(np.mean(d < 0.025))


def main():
    D = load()
    a, fm, zm, z2, zn = D['a'], D['fm'], D['zm'], D['z2'], D['zn']
    box = desk_box(a)
    BLw = np.array(box['back_left_top'])
    MA = np.array(fm['candidates']['A_per_axis']['M'])
    icpA = {r['arm'] + ':' + r['model']: r for r in fm['candidates']['A_per_axis']['arms']}

    # ---- ZED metric quantities (tape units)
    zs = mf.TAPE['length'] / zm['desk_length_back_edge_m']
    cam_desk = np.array(zm['camera_in_desk_frame_m']) * zs
    Rdc = zn['Rdc']                       # desk(=world) axes in camera coords; R_world_from_cam
    C_cam = BLw + cam_desk
    T_wc = np.eye(4)
    T_wc[:3, :3] = Rdc
    T_wc[:3, 3] = C_cam
    zed_arm = {k: np.mean([z2[t][k]['base_world'] for t in ('0928', '0929')], 0) for k in ('left_arm', 'right_arm')}
    zed_yaw = {k: float(np.mean([z2[t][k]['yaw_deg'] for t in ('0928', '0929')])) for k in ('left_arm', 'right_arm')}

    # ---- final transform: scales fixed, offsets by weighted LSQ (x, y), floor -> 0 (z)
    ref_arm = {k: (np.array(icpA[k + ':no_fingers']['base']) - MA[:3, 3]) / np.diag(MA[:3, :3]) for k in ('left_arm', 'right_arm')}
    xr = np.array([a['desk_left_x'], a['desk_right_x'], ref_arm['left_arm'][0], ref_arm['right_arm'][0]])
    xm = np.array([box['x'][0], box['x'][1], zed_arm['left_arm'][0], zed_arm['right_arm'][0]])
    yr = np.array([a['desk_front_y'], a['desk_back_y'], ref_arm['left_arm'][1], ref_arm['right_arm'][1]])
    ym = np.array([box['y'][0], box['y'][1], zed_arm['left_arm'][1], zed_arm['right_arm'][1]])
    w = np.array([1 / 1.5 ** 2, 1 / 1.5 ** 2, 1.0, 1.0])
    tx, rx = wlsq_offset(S_FINAL[0], xr, xm, w)
    ty, ry = wlsq_offset(S_FINAL[1], yr, ym, w)
    tz = -S_FINAL[2] * a['floor_z_near_desk']
    M = np.eye(4)
    M[:3, :3] = np.diag(S_FINAL)
    M[:3, 3] = [tx, ty, tz]
    dT = M[:3, 3] - MA[:3, 3]  # candidate A had the same scales; ICP results shift by dT exactly

    # ---- arms: ICP (free z) shifted into the final frame; plus a common-rail-height refit (z fixed)
    arms = {}
    d = np.load(common.WORK / 'clean_cloud.npz')
    Pc = d['points']
    jobs = []
    zfree = {k: np.array(icpA[k + ':no_fingers']['base']) + dT for k in ('left_arm', 'right_arm')}
    z_common = float(np.mean([zfree[k][2] for k in zfree]))
    for k, xc in (('left_arm', -2.56), ('right_arm', -1.97)):
        boxm = (np.abs(Pc[:, 0] - xc) < 0.17) & (Pc[:, 1] > 1.0) & (Pc[:, 1] < 1.75) & (Pc[:, 2] > 0.675) & (Pc[:, 2] < 1.05)
        A = mf.apply(M, voxel(Pc[boxm], 0.006))
        jobs.append((A, np.r_[zfree[k][:2], np.radians(icpA[k + ':no_fingers']['yaw_deg'])], z_common,
                     common.WORK / 'yam_q0_open0.0475_nofingers.npz'))
    with ProcessPoolExecutor(max_workers=2) as ex:
        fixed = list(ex.map(icp_fixed_z, jobs))
    V, MV, _ = common.points()
    P = mf.apply(M, np.asarray(V)[MV].astype(np.float64))
    X, Y, Z = P.T
    for (k, _), (x, med, rms, inl) in zip((('left_arm', 0), ('right_arm', 0)), fixed):
        rf = icpA[k + ':no_fingers']
        rw = icpA[k + ':with_fingers']
        base = np.array([x[0], x[1], z_common])
        yaw = float(np.degrees(x[2]))
        T = np.eye(4)
        T[:3, :3] = rz(np.radians(yaw))
        T[:3, 3] = base
        arms[k] = dict(
            base_position_m=base, yaw_deg=yaw, world_from_base=T,
            base_height_above_desk_top_m=base[2] - mf.TAPE['top'],
            icp=dict(model='yam.xml + linear_4310 at q=0, fingers excluded (add-on pads/tags on the rig)', points_voxel6mm=rf['n'],
                     median_m=med, rms_inliers_m=rms, inlier_frac_2p5cm=inl),
            free_z_fit=dict(base_position_m=zfree[k], yaw_deg=rf['yaw_deg'], median_m=rf['median_m'], rms_inliers_m=rf['rms_inliers_m'],
                            base_above_desk_top_m=zfree[k][2] - mf.TAPE['top'],
                            arm_top_above_base_m=rf.get('arm_top_above_base_m'), mjcf_top_above_base_m=MJCF_TOP),
            with_fingers_fit=dict(base_position_m=np.array(rw['base']) + dT, yaw_deg=rw['yaw_deg'], median_m=rw['median_m']),
            zed_silhouette_check=dict(base_position_m_mean_0928_0929=zed_arm[k], yaw_deg_mean=zed_yaw[k],
                                      per_image={t: dict(base=z2[t][k]['base_world'], yaw_deg=z2[t][k]['yaw_deg'], iou=z2[t][k]['iou'],
                                                         finger_opening_m=z2[t][k]['opening']) for t in ('0928', '0929')},
                                      delta_final_minus_zed_m=base - zed_arm[k]),
        )
    L, R_ = arms['left_arm']['base_position_m'], arms['right_arm']['base_position_m']
    spacing = dict(xy_distance_m=float(np.linalg.norm((R_ - L)[:2])), dx_m=float(R_[0] - L[0]), dy_m=float(R_[1] - L[1]),
                   zed_silhouette_xy_distance_m=float(np.linalg.norm((zed_arm['right_arm'] - zed_arm['left_arm'])[:2])),
                   midpoint_xy=((L + R_) / 2)[:2])

    # ---- rail (reference points, final frame): ~2 cm plate along the desk front edge
    top = mf.TAPE['top']
    strip = (Y > box['y'][0] - 0.01) & (Y < box['y'][0] + 0.13)
    slab = (Z > top + 0.010) & (Z < top + 0.030)
    base_mask = np.zeros(len(P), bool)
    for k in arms:
        base_mask |= (np.abs(X - arms[k]['base_position_m'][0]) < 0.11)
    pole_x = float(np.median(X[(Z > top + 0.1) & (Z < top + 0.45) & (np.abs(X + 2.28) < 0.05) & (np.abs(Y - 1.385) < 0.05)]))
    pole_y = float(np.median(Y[(Z > top + 0.1) & (Z < top + 0.45) & (np.abs(X + 2.28) < 0.05) & (np.abs(Y - 1.385) < 0.05)]))
    # height relative to the local desk surface just behind the rail (reference desk top is warped by ~1.5 cm)
    xs = np.arange(box['x'][0], box['x'][1], 0.02)
    occ, rel_h = [], []
    for x0 in xs:
        c = (X >= x0) & (X < x0 + 0.02) & (Z > top - 0.03) & (Z < top + 0.06)
        ref_s = c & (Y > box['y'][0] + 0.14) & (Y < box['y'][0] + 0.22)
        rs = c & (Y > box['y'][0] + 0.02) & (Y < box['y'][0] + 0.09)
        if ref_s.sum() < 30 or rs.sum() < 30:
            occ.append(0.0); rel_h.append(np.nan); continue
        base_z = np.median(Z[ref_s])
        h = np.percentile(Z[rs], 75) - base_z
        rel_h.append(h)
        occ.append(float(h > 0.010))
    occ, rel_h = np.array(occ), np.array(rel_h)
    # largest run of rail columns around the pole
    runs, cur = [], []
    for x0, o in zip(xs, occ):
        if o > 0:
            cur.append(x0)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if r[0] - 0.1 <= pole_x <= r[-1] + 0.1] or runs
    run = max(runs, key=len)
    rail_cols = np.array(run)
    near = strip & (np.abs(X - pole_x) > 0.05) & ~base_mask & (X > rail_cols.min()) & (X < rail_cols.max())
    base_near = (X > rail_cols.min()) & (X < rail_cols.max()) & (Y > box['y'][0] + 0.14) & (Y < box['y'][0] + 0.22) & (Z > top - 0.03) & (Z < top + 0.06)
    local_top = float(np.median(Z[base_near]))
    ybins = np.arange(box['y'][0] - 0.01, box['y'][0] + 0.13, 0.01)
    prof = []
    for y0 in ybins:
        s = near & (Y >= y0) & (Y < y0 + 0.01) & (Z > top - 0.02) & (Z < top + 0.06)
        prof.append([round(float(y0 + 0.005), 3), round(float(np.percentile(Z[s], 75) - local_top), 4) if s.sum() > 50 else None])
    on = [p for p in prof if p[1] is not None and p[1] > 0.010]
    overhang = float(arms['left_arm']['base_position_m'][0] - rail_cols.min())  # left end is well reconstructed
    rail = dict(
        x_extent_reference_m=[float(rail_cols.min()), float(rail_cols.max() + 0.02)],
        length_reference_m=float(rail_cols.max() + 0.02 - rail_cols.min()),
        x_extent_recommended_m=[float(rail_cols.min()), float(arms['right_arm']['base_position_m'][0] + overhang)],
        length_recommended_m=float(arms['right_arm']['base_position_m'][0] + overhang - rail_cols.min()),
        recommended_note='Front photo (source frame 401) shows the rail ends at equal distances outside the two arm bases '
                         '(85 px vs 90 px for a 240 px arm spacing, ~0.22 m). The left end is clear in the reference; the right end is '
                         'lost near the right arm cables, so it is mirrored from the left overhang. Tape-check the rail length.',
        y_extent_m=[on[0][0] - 0.005, on[-1][0] + 0.005] if on else None,
        top_above_desk_m=float(np.median([p[1] for p in on])) if on else None,
        top_z_m=float(top + np.median([p[1] for p in on])) if on else None,
        profile_y_vs_height_above_local_desk=prof,
        height_above_local_desk_by_x=[[round(float(x), 3), None if np.isnan(h) else round(float(h), 4)] for x, h in zip(xs, rel_h)],
        pole_xy_m=[pole_x, pole_y],
        note='Reference points (Pi3X, ~1 cm noise). Aluminium plate on the desk top, front face flush with the desk front edge; '
             'arms and the camera pole mount on it. Photo check (frames 401, 942): ends ~0.20 m outside each arm base. '
             'Ends uncertain by about 3 cm.')

    # ---- camera: depth cross-check and PnP cross-check
    pnp = D['pnp']
    Rp, tp = pnp['R'], pnp['t']
    C_pnp = -Rp.T @ tp + dT
    dR = Rdc.T @ Rp.T  # cam_from_world(final) vs cam_from_world(pnp): rotation between the two camera frames
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rdc @ Rp) - 1) / 2, -1, 1))))
    K = zn['K']
    Xp = pnp['X'] + dT
    pr = (Xp - C_cam) @ Rdc
    uvp = pr[:, :2] / pr[:, 2:3] * K[[0, 1], [0, 1]] + K[:2, 2]
    e_final = np.linalg.norm(uvp - pnp['uv'], axis=1)
    pr0 = (pnp['X'] - C_cam) @ Rdc
    e_deskanchored = np.linalg.norm(pr0[:, :2] / pr0[:, 2:3] * K[[0, 1], [0, 1]] + K[:2, 2] - pnp['uv'], axis=1)
    # desk-edge reprojection of the metric desk box with the final pose
    def proj(p):
        q = (np.asarray(p) - C_cam) @ Rdc
        return q[:, :2] / q[:, 2:3] * K[[0, 1], [0, 1]] + K[:2, 2]
    def line_dist(pts, line):
        c, dvec = np.array(line['point']), np.array(line['dir'])
        n = np.array([-dvec[1], dvec[0]])
        return (pts - c) @ n
    x0, x1 = box['x']
    y0, y1 = box['y']
    tl = np.linspace(0, 1, 50)[:, None]
    back = proj(np.c_[x0 + tl * (x1 - x0), np.full_like(tl, y1), np.full_like(tl, top)])
    left = proj(np.c_[np.full_like(tl, x0), y1 - tl * 0.5, np.full_like(tl, top)])
    right = proj(np.c_[np.full_like(tl, x1), y1 - tl * 0.5, np.full_like(tl, top)])
    edges_px = {}
    for tag in ('0928', '0929'):
        Lz = zm['edge_lines_px'][tag]
        edges_px[tag] = dict(back_edge_rms_px=float(np.sqrt(np.mean(line_dist(back[(back[:, 0] > 280) & (back[:, 0] < 760)], Lz['crease']) ** 2))),
                             left_edge_rms_px=float(np.sqrt(np.mean(line_dist(left[left[:, 1] < 598], Lz['left']) ** 2))),
                             right_edge_rms_px=float(np.sqrt(np.mean(line_dist(right[right[:, 1] < 598], Lz['right']) ** 2))))
    # ZED depth vs metric planes
    med = np.load(common.WORK / 'zed_depth_median.npy') * zs
    vv, uu = np.mgrid[0:600, 0:960]
    rays = np.dstack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)])
    rw = rays @ Rdc.T
    t_desk = (top - C_cam[2]) / rw[..., 2]
    hit = C_cam + t_desk[..., None] * rw
    on_desk = (t_desk > 0) & (hit[..., 0] > x0 + 0.02) & (hit[..., 0] < x1 - 0.02) & (hit[..., 1] > y0) & (hit[..., 1] < y1 - 0.02)
    gm = np.load(common.WORK / 'gripper_mask_0928.npy')
    gm = np.load(common.WORK / 'gripper_mask_0928.npy') | (vv > 470) & (((uu > 230) & (uu < 400)) | ((uu > 630) & (uu < 810)))
    zdesk = t_desk  # ray parameter == depth z since ray z-component is 1
    ok = on_desk & np.isfinite(med) & ~gm & ~((uu > 505) & (uu < 560) & (vv > 520))
    dres = (med - zdesk)[ok]
    t_wall = (y1 - C_cam[1]) / rw[..., 1]
    hitw = C_cam + t_wall[..., None] * rw
    on_wall = (t_wall > 0) & (hitw[..., 2] > top + 0.05) & (hitw[..., 0] > x0 + 0.1) & (hitw[..., 0] < x1 - 0.1)
    okw = on_wall & np.isfinite(med)
    wres = (med - t_wall)[okw]
    wres_hi = (med - t_wall)[okw & (hitw[..., 2] > top + 0.30)]
    camera = dict(
        device='ZED X (stereo), serial 41925345, left eye, rectified, 960x600',
        intrinsics_source=str(common.ZED_EP / 'lowdim/0000000000.pkl') + " ['intrinsics']['scene_camera'] (raiden converter, ZED SDK left_cam calibration, no flip)",
        image_size_wh=[960, 600], K=K, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        distortion=dict(model='none', coeffs=[0, 0, 0, 0, 0], note='rectified left image; raiden stores K only'),
        hfov_deg=float(np.degrees(2 * np.arctan(480 / K[0, 0]))), vfov_deg=float(np.degrees(2 * np.arctan(300 / K[1, 1]))),
        pose=dict(
            convention='world_from_camera; OpenCV camera axes (x right, y down, z forward)',
            world_from_camera_opencv=T_wc,
            world_from_camera_blender=T_wc @ np.diag([1, -1, -1, 1]),
            blender_note='Blender camera axes (x right, y up, looks along -z) = OpenCV axes with y and z negated; '
                         'matrix_world = world_from_camera_blender. Lens for sensor_width 36 mm, sensor_fit HORIZONTAL: '
                         f'focal {K[0, 0] * 36 / 960:.4f} mm; shift_x {-(K[0, 2] - 479.5) / 960:.5f}, shift_y {(K[1, 2] - 299.5) / 960:.5f} '
                         '(OpenCV pixel centres at integers; image centre (W-1)/2, (H-1)/2).',
            position_m=C_cam, optical_axis_world=Rdc[:, 2],
            pitch_down_deg=float(np.degrees(np.arcsin(-Rdc[2, 2]))),
            yaw_deg_optical_axis_from_plus_y=float(np.degrees(np.arctan2(Rdc[0, 2], Rdc[1, 2]))),
            roll_deg=float(np.degrees(np.arctan2(Rdc[2, 0], -Rdc[2, 1]))),
            height_above_floor_m=C_cam[2], height_above_desk_top_m=C_cam[2] - top,
            horizontal_distance_to_back_edge_m=y1 - C_cam[1], x_from_desk_left_end_m=C_cam[0] - x0,
            method='Direct metric pose from the ZED itself: desk-top plane from the temporal-median depth (RANSAC, rms 2.9 mm), '
                   'desk left/right edges and back crease from sub-pixel image-gradient line fits; x axis = back edge, z = desk normal; '
                   f'translation from ray-plane intersections; ZED depth scaled by {zs:.4f} so the ZED desk length ({zm["desk_length_back_edge_m"]:.4f} m) equals the tape 1.52 m.',
        ),
        checks=dict(
            vanishing_point_orientation=dict(note='desk side edges VP + back edge with K only (no depth): camera up-vector agrees with the depth plane normal to 0.3 deg; pitch 46.9 (image) vs 47.2 (depth) deg',
                                             angle_deg=0.3),
            desk_edge_reprojection_px=edges_px,
            rig_static_0928_vs_0929=dict(note='desk edge lines in the 09-28 converted frame and the 09-29 16:00 image agree to <=0.2 px; gripper silhouette fits agree to ~1 cm',
                                         left_edge_u_at_v482=[zm['edge_lines_px']['0928']['left']['point'][0], zm['edge_lines_px']['0929']['left']['point'][0]],
                                         right_edge_u_at_v482=[zm['edge_lines_px']['0928']['right']['point'][0], zm['edge_lines_px']['0929']['right']['point'][0]]),
            zed_depth_vs_metric_desk_plane=dict(pixels=int(ok.sum()), median_mm=float(np.median(dres) * 1000), rms_mm=float(np.sqrt(np.mean(dres ** 2)) * 1000),
                                                p5_p95_mm=[float(np.percentile(dres, 5) * 1000), float(np.percentile(dres, 95) * 1000)]),
            zed_depth_vs_metric_back_wall=dict(pixels=int(okw.sum()), median_mm=float(np.median(wres) * 1000), p5_p95_mm=[float(np.percentile(wres, 5) * 1000), float(np.percentile(wres, 95) * 1000)],
                                               above_0p3m_median_mm=float(np.median(wres_hi) * 1000),
                                               note='ZED NEURAL_LIGHT depth on the textureless wall is unreliable: plane fits to it vary 7-14 cm across columns and yaw 4 deg vs the back edge; not used for the pose'),
            pnp_reference=dict(
                method='2D-3D: each Pi3X source frame forward-warped (own point map) into the ZED view at the current pose, SIFT (ratio 0.8) + '
                       'fundamental RANSAC, 3D from the warped pixel; pooled over frames; PnP RANSAC (3 px) + LM; 4 iterations. '
                       'Plain SIFT against the raw source frames gave no consistent PnP (16-40 inliers, 22-117 px), see logs/match_zed.log.',
                image=str(common.ZED_0929), inliers=int(len(pnp['uv'])), frames_with_inliers=29,
                reproj_median_px=float(np.median(pnp['err'])), reproj_rms_px=float(np.sqrt(np.mean(pnp['err'] ** 2))),
                position_m=C_pnp, delta_position_vs_final_m=C_pnp - C_cam, rotation_delta_deg=ang,
                final_pose_reprojection_of_pnp_inliers=dict(median_px=float(np.median(e_final)), rms_px=float(np.sqrt(np.mean(e_final ** 2))),
                                                            desk_anchored_transform_median_px=float(np.median(e_deskanchored)),
                                                            note='reference 3D points (mostly desk-top texture) under the final camera pose: the residual is the reference '
                                                                 'shape error (1 px ~ 2.7 mm at 1 m). With the transform anchored on the desk alone (candidate A) it is '
                                                                 'about half; anchoring x/y on the ZED-measured arms moves the reference ~2 cm relative to its own desk texture.'),
                note='The PnP inliers are mostly desk-top texture (planar); the PnP pose inherits the reference shape error '
                     '(4.1 deg, 3-5 cm). The direct ZED pose is kept: it is metric, fits the desk edges to sub-pixel and its '
                     'orientation is confirmed by vanishing points.'),
        ),
    )

    # ---- arms seen by the ZED at the final camera pose: MJCF silhouettes and the reference's own gripper points
    import cv2
    def splat_mask(Pw, close=3):
        q = (Pw - C_cam) @ Rdc
        uv = q[:, :2] / q[:, 2:3] * K[[0, 1], [0, 1]] + K[:2, 2]
        ui = np.rint(uv).astype(int)
        g = (q[:, 2] > 0.05) & (ui[:, 0] >= 0) & (ui[:, 0] < 960) & (ui[:, 1] >= 0) & (ui[:, 1] < 600)
        m = np.zeros((600, 960), np.uint8)
        m[ui[g, 1], ui[g, 0]] = 1
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8)) > 0
    wins = {'left_arm': (slice(470, 600), slice(200, 440)), 'right_arm': (slice(470, 600), slice(600, 860))}
    surf06 = sample_surface(common.WORK / 'yam_q0_open0.06.npz', 250000)
    gref = (np.abs(Pc[:, 0] + 2.26) < 0.45) & (Pc[:, 1] > 1.45) & (Pc[:, 1] < 1.75) & (Pc[:, 2] > 0.72) & (Pc[:, 2] < 1.0)
    ref_grip = splat_mask(mf.apply(M, Pc[gref]), 5)
    for tag in ('0928', '0929'):
        gmask = np.load(common.WORK / f'gripper_mask_{tag}.npy')
        for k in arms:
            wv = wins[k]
            mj = splat_mask(surf06 @ rz(np.radians(arms[k]['yaw_deg'])).T + arms[k]['base_position_m'])[wv]
            rg = ref_grip[wv]
            gm_ = gmask[wv]
            ya, xa = np.nonzero(rg)
            yb, xb = np.nonzero(gm_)
            arms[k].setdefault('zed_image_check_final_pose', {})[tag] = dict(
                mjcf_open_0p06_silhouette_iou=float((mj & gm_).sum() / (mj | gm_).sum()),
                reference_gripper_points_iou=float((rg & gm_).sum() / (rg | gm_).sum()),
                reference_gripper_centroid_du_dv_px=[float(xa.mean() - xb.mean()), float(ya.mean() - yb.mean())])

    # ---- residuals of every candidate at every anchor (+ final)
    cands = {}
    for k, c in fm['candidates'].items():
        cands[k] = {kk: c[kk] for kk in ('S', 'M', 'note', 'desk_length_m', 'desk_length_residual_m', 'desk_depth_front_to_wall_m',
                                         'desk_depth_residual_m', 'desk_top_residual_m', 'floor_residual_m', 'desk_height_above_floor_m',
                                         'rail_top_above_desk_m', 'camera_body_height_above_desk_m', 'zed_camera_height_above_desk_m')}
        cands[k]['desk_depth_residual_m'] = c['desk_depth_front_to_wall_m'] - mf.TAPE['depth']  # desk flush with the wall
        cands[k]['arms_icp'] = [{kk: r[kk] for kk in ('arm', 'model', 'base', 'yaw_deg', 'median_m', 'rms_inliers_m', 'inlier_frac')}
                                | ({'arm_top_above_base_m': r['arm_top_above_base_m'], 'arm_top_residual_vs_mjcf_m': r['arm_top_residual_vs_mjcf_m']}
                                   if 'arm_top_above_base_m' in r else {}) for r in c['arms']]
    Mref = lambda p: mf.apply(M, np.atleast_2d(p))[0]
    cam_body = Mref(fm['camera_body_ref_median'])
    final_res = dict(
        desk_ends_x_m=dict(left=float(rx[0]), right=float(rx[1]), note='reference desk-end (half point density) minus the tape box, after the transform'),
        desk_length_m=float(S_FINAL[0] * a['desk_length']), desk_length_residual_m=float(S_FINAL[0] * a['desk_length'] - mf.TAPE['length']),
        desk_front_back_y_m=dict(front=float(ry[0]), back_wall=float(ry[1])),
        desk_depth_front_to_wall_m=float(S_FINAL[1] * a['desk_depth_front_to_wall']),
        desk_top_z_m=dict(plane_at_centre=float(Mref([0, 0, a['desk_top_z_at_centre']])[2]),
                          per_view_median=float(S_FINAL[2] * (a['floor_z_near_desk'] + a['desk_minus_floor_per_view_median']) + tz),
                          residual_plane_m=float(Mref([0, 0, a['desk_top_z_at_centre']])[2] - top)),
        desk_top_flatness_ref=dict(tilt_deg=a['desk_tilt_deg'], note='reference desk top rises ~1.5 cm toward the glass end (X < -2.9): reconstruction warp'),
        floor_near_desk_z_m=float(Mref([0, 0, a['floor_z_near_desk']])[2]),
        floor_away_from_desk_mm=dict(median=D['fl']['away_from_desk']['median_mm'] * S_FINAL[2], p10=D['fl']['away_from_desk']['p10_mm'] * S_FINAL[2],
                                     p90=D['fl']['away_from_desk']['p90_mm'] * S_FINAL[2]),
        back_wall=dict(y_by_height_m={h: float(S_FINAL[1] * y + ty) for h, y in a['wall_y_by_height'].items()},
                       lean_deg=1.84, note='reference wall leans back ~1.8-2.2 deg (Pi3X shape error; the pole in the same data is vertical to 0.1 deg in Y); author vertical at the desk back edge'),
        arms_xy_vs_zed_m={k: dict(dx=float(v[0]), dy=float(v[1])) for k, v in (('left_arm', (rx[2], ry[2])), ('right_arm', (rx[3], ry[3])))},
        arm_top_vs_mjcf_m={k: arms[k]['free_z_fit']['arm_top_above_base_m'] - MJCF_TOP for k in arms},
        camera_height_above_desk_m=dict(zed_depth=float(C_cam[2] - top), pnp_reference=float(C_pnp[2] - top),
                                        reference_camera_body_centre=float(cam_body[2] - top),
                                        note='reference camera-body centre should sit ~1 cm above the left-eye optical centre; residual ~+2.3 cm'),
        camera_position_pnp_minus_zed_m=C_pnp - C_cam,
    )
    out = dict(
        schema='aha3d yam_bimanual_rig metric frame v1',
        created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
        producer=dict(owner='claude-yam-metric', task='yam-metric-20260929', dir=str(common.HERE),
                      scripts=['common.py', 'clean_cloud.py', 'floor_check.py', 'zed_metric.py', 'yam_mesh.py (raiden venv)',
                               'arm_icp.py', 'metric_frame.py', 'fit_metric.py', 'arm_zed_fit2.py', 'render.py', 'pnp_warp.py',
                               'match_zed.py (plain SIFT, failed)', 'build_metric.py', 'review_images.py']),
        inputs=dict(reference_run=str(common.RUN), basis=str(common.REF / 'alignment/cameras.json'),
                    zed_episode=str(common.ZED_EP), zed_image_0929=str(common.ZED_0929), zed_image_0928=str(common.ZED_0928_PNG),
                    tape_0929=dict(desk_length_m=1.52, desk_depth_m=0.76, desk_top_above_floor_m=0.72),
                    mjcf='/home/frank/robot/raiden/third_party/i2rt/i2rt/robot_models/arm/yam/yam.xml + gripper/linear_4310/linear_4310.xml (i2rt combine_arm_and_gripper_xml), q=0, fingers open 0.0475'),
        conventions=dict(
            world='metric, metres, right-handed, Z up; floor z=0; desk top z=0.72; X along the back wall / desk length '
                  '(-X = frosted-glass end, +X = monitor-desk end); +Y toward the back wall. Origin and axes follow the aligned '
                  'reference (cameras.json) so reference coordinates change by the transform below only.',
            arm_base_frame='MJCF yam.xml world frame: origin at the base mounting face, Z up, arm folded along +x at q=0 (grippers at +x); '
                           'world_from_base = [Rz(yaw) | position]; yaw 90 deg means grippers point to +Y (the wall).',
            camera='see scene_camera.pose'),
        transform_reference_to_metric=dict(
            M=M, S=S_FINAL, t=M[:3, 3], form='p_metric = diag(sx, sy, sz) p_ref + t (no rotation; the reference is already floor/wall aligned)',
            applies_to='aligned reference coordinates (cameras.json world, layers.npz vertices, point maps); camera poses: '
                       'c2w_metric = M c2w_ref, then re-orthonormalise the rotation (non-uniform scale shears camera axes slightly)',
            derivation=dict(sx='desk length 1.52 / 1.42-1.44 predicted; arm spacing ZED 0.621-0.635 / ref 0.587 = 1.06-1.08',
                            sy='desk depth 0.76 vs 0.756-0.77 predicted; MJCF arm length fits at 1.00 along Y',
                            sz='desk top minus local floor 0.661 predicted (58 views) -> 0.72',
                            t='z: local floor near the desk (-7 mm predicted) -> 0; x, y: weighted LSQ over desk edges (1.5 cm) and ZED-measured arm bases (1 cm)'),
            uncertainty=dict(sx=0.01, sy=0.015, sz_below_desk=0.005, sz_above_desk='1.05-1.12 depending on the anchor (camera height 1.05-1.06, arm tops ~1.12)',
                             local_position_error_m='2-3 cm near the desk after the transform')),
        residuals_final=final_res,
        alternatives=cands,
        desk=dict(box_x_m=box['x'], box_y_m=box['y'], top_z_m=top, size_m=[1.52, 0.76, 0.72],
                  top_corners_m=[[box['x'][0], box['y'][0], top], [box['x'][1], box['y'][0], top], [box['x'][1], box['y'][1], top], [box['x'][0], box['y'][1], top]],
                  back_edge_to_wall_gap_m=0.0,
                  note='Rig desk (rail + arms). Right end abuts the separate monitor desk (gap ~5-13 cm seen in reference/ZED). '
                       'Height-adjustable desk frame; legs near both ends.'),
        planes=dict(convention='n . p + d = 0, n unit',
                    floor=dict(n=[0, 0, 1], d=0.0),
                    desk_top=dict(n=[0, 0, 1], d=-top),
                    back_wall=dict(n=[0, -1, 0], d=box['y'][1], note='vertical, through the desk back edge (desk flush with the wall within ~1 cm)')),
        arms=arms, arm_spacing=spacing, rail=rail, scene_camera=camera,
    )
    common.save_json(OUT, out)
    print('M', M.round(4).tolist())
    print('arms', {k: (np.round(v['base_position_m'], 4).tolist(), round(v['yaw_deg'], 2)) for k, v in arms.items()}, 'spacing', spacing['xy_distance_m'])
    print('rail', {k: rail[k] for k in ('x_extent_reference_m', 'x_extent_recommended_m', 'length_recommended_m', 'y_extent_m', 'top_above_desk_m', 'pole_xy_m')})
    print('camera C', C_cam.round(4), 'pnp', C_pnp.round(4), 'rot delta', round(ang, 2), 'edges px', edges_px)
    print('depth desk', camera['checks']['zed_depth_vs_metric_desk_plane'], 'wall', camera['checks']['zed_depth_vs_metric_back_wall'])
    print('final residuals', json.dumps(final_res, default=lambda o: np.round(o, 4).tolist() if hasattr(o, 'tolist') else o))


if __name__ == '__main__':
    main()
