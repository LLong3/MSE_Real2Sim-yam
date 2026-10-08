"""Write the wrist camera and proposed arm base records from a selected stage-2 fit (CPU, raiden venv).

    ~/robot/raiden/.venv/bin/python build/write_wrist_records.py --fit RUN/stage2/<variant>/fit.json \
        --stage1 RUN/stage1/stage1_background_fit.json --captures RUN/captures_sq0.03556/captures.json \
        --captures-alt RUN/captures_sq0.036/captures.json --out evidence/wrist_fit [--check-render DIR]

Writes OUT/wrist_cameras.json (per wrist camera: K, world / link_6 / grasp_site poses, which base the link poses
use, residuals), OUT/arm_bases_proposed.json (per arm: proposed world_from_base vs the metric base and the
capture-fit bases; identifiability notes), and the make_twin_config.py inputs OUT/cameras_world.json and
OUT/bases_proposed_4x4.json. The metric frame is read only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import wristfit_common as W


def ypr(Ba, Bb):
    d = np.linalg.inv(Ba) @ Bb
    return dict(yaw_pitch_roll_deg_base_axes=np.degrees(Rot.from_matrix(d[:3, :3]).as_euler('ZYX')).round(3).tolist(),
                t_world_mm=((Bb[:3, 3] - Ba[:3, 3]) * 1e3).round(1).tolist())


def rel(Ta, Tb):
    d = np.linalg.inv(Ta) @ Tb
    return dict(rot_deg=round(float(np.degrees(np.linalg.norm(Rot.from_matrix(d[:3, :3]).as_rotvec()))), 3),
                t_mm=((Tb[:3, 3] - Ta[:3, 3]) * 1e3).round(1).tolist())


def rail_form(fkf, Bm, L_fit, joints=(1, 2, 5)):
    """Same link_6 pose with the base on the rail: metric base height, no roll / pitch; base yaw and x, y fitted,
    plus home offsets of `joints` (0-based; default joint 2, 3, 6; joints 2 and 4 are near parallel at home, so
    the 2/4/6 form is ill-conditioned). Least squares on rotation and position."""
    from scipy.optimize import least_squares

    def pose(x):
        B = np.eye(4); B[:3, :3] = Rot.from_euler('z', x[0]).as_matrix(); B[:3, 3] = [x[1], x[2], Bm[2, 3]]
        q = np.zeros(6); q[list(joints)] = x[3:]
        return B, q, B @ fkf(q)

    def r(x):
        _, _, L = pose(x)
        return np.r_[Rot.from_matrix(L[:3, :3] @ L_fit[:3, :3].T).as_rotvec() / np.radians(0.01), (L[:3, 3] - L_fit[:3, 3]) / 1e-4]
    x0 = np.r_[np.arctan2(Bm[1, 0], Bm[0, 0]), Bm[0, 3], Bm[1, 3], np.zeros(len(joints))]
    x = least_squares(r, x0).x
    B, q, L = pose(x)
    res = r(x)
    lim = dict(joint2=(0, 3.65), joint3=(0, 3.665), joint4=(-1.571, 1.571), joint5=(-1.571, 1.571), joint6=(-2.094, 2.094))
    qd = {f'joint{j + 1}': round(float(np.degrees(q[j])), 3) for j in joints}
    return dict(world_from_base=B.tolist(), yaw_deg=round(float(np.degrees(x[0])), 3), base_position_m=B[:3, 3].round(5).tolist(),
                base_vs_metric=ypr(Bm, B), home_joint_offsets_deg=qd, home_q_rad=q.tolist(),
                in_mjcf_range={k: bool(lim[k][0] - 1e-9 <= np.radians(v) <= lim[k][1] + 1e-9) for k, v in qd.items()},
                residual=dict(rot_deg=round(float(np.linalg.norm(res[:3]) * 0.01), 4), pos_mm=round(float(np.linalg.norm(res[3:]) * 0.1), 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', type=Path, required=True)
    ap.add_argument('--stage1', type=Path, required=True)
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--captures-alt', type=Path)
    ap.add_argument('--check-render', type=Path, help='twin_check.sh dir of the final check render (claw_check.json, compare)')
    ap.add_argument('--note', default='')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fit = W.jload(a.fit); s1 = W.jload(a.stage1)
    cap = W.jload(a.captures); capa = W.jload(a.captures_alt) if a.captures_alt else None
    m = W.jload(W.METRIC)
    metric = W.metric_bases(m)
    FKm = W.FK()
    fk = FKm(W.HOME_Q)
    fkf = lambda q: FKm(q, frames=('link_6',))['link_6']
    T_l6_gs = np.linalg.inv(fk['link_6']) @ fk['grasp_site']
    builds = W.build_poses()
    cams, bases, cams_world, bases_4x4 = {}, {}, {}, {}
    for s in W.SIDES:
        ar = fit['arms'][s]
        B = np.array(ar['T_world_base']); Twc = np.array(ar['T_world_cam'])
        L = B @ fk['link_6']
        X = np.linalg.inv(L) @ Twc                               # link_6 from camera
        G = np.linalg.inv(B @ fk['grasp_site']) @ Twc            # grasp_site from camera
        Gm = np.linalg.inv(metric[s] @ fk['grasp_site']) @ Twc   # same world pose, metric base
        he = W.hand_eye(s)
        s1c = s1['cameras'][s]['fits']['stage1_rotation_only']
        cams[f'{s}_wrist'] = dict(
            camera=f'{s}_wrist_camera (ZED X One on the {s} arm gripper housing)',
            image_size_wh=[960, 600],
            K_960x600=W.wrist_K(s).tolist(),
            K_source='~/.config/raiden/calibration_results.json (09-29) ZED SDK factory HD1200 K / 2; fx != fy (render pixel aspect fy/fx)',
            T_world_cam_opencv=Twc.tolist(),
            T_link6_cam_opencv=X.tolist(),
            T_grasp_site_cam_opencv=G.tolist(),
            relative_to=dict(base=f'arm_bases_proposed.json arms.{W.ARM[s]}.world_from_base (NOT the metric base)',
                             world_from_base=B.tolist(), joints='raiden home q = 0, gripper 0.0475 m',
                             fk='source/yam_linear_4310.xml (MuJoCo), link_6 / grasp_site at q = 0'),
            if_metric_base_kept=dict(T_grasp_site_cam_opencv=Gm.tolist(),
                                     note='keeps the same world camera pose (background aligned) on the metric base; '
                                          'the claws then render off (the v2 conflict), so use the proposed base'),
            vs=dict(raiden_hand_eye=rel(T_l6_gs @ he, X), stage1_world=rel(np.array(s1c['T_world_cam']), Twc),
                    v2_world=rel(builds['delivery_v2'][s], Twc), v1_world=rel(builds['delivery_v1'][s], Twc)),
            residuals=dict(background_edges=ar['background_edges'], iou_wrist=ar['iou_wrist'], iou_scene_same_arm=ar['iou_scene'],
                           centroid_model_minus_real_px=ar['centroid_model_minus_real_px'],
                           note='numpy label render of claw_model.py (validated against Blender IDs); background: named '
                                'desk/wall edges, median |offset| px (wrist_bg_fit.py)'))
        cams_world[s] = Twc.tolist()
        bases_4x4[s] = B.tolist()
        capb = np.array(cap['joint']['arms'][s]['T_world_base'])
        bases[W.ARM[s]] = dict(
            world_from_base=B.tolist(),
            base_position_m=B[:3, 3].round(5).tolist(),
            yaw_deg=round(float(np.degrees(np.arctan2(B[1, 0], B[0, 0]))), 3),
            tilt_deg=round(float(np.degrees(np.arccos(np.clip(B[2, 2], -1, 1)))), 3),
            vs_metric=ypr(metric[s], B),
            metric_world_from_base=metric[s].tolist(),
            vs_capture_base_35p56mm=ypr(capb, B),
            capture_base_35p56mm=dict(world_from_base=capb.tolist(), vs_metric=ypr(metric[s], capb),
                                      residual_px=cap['joint']['residual_px'][s]['median']),
            capture_base_36mm=(dict(vs_metric=ypr(metric[s], np.array(capa['joint']['arms'][s]['T_world_base'])),
                                    residual_px=capa['joint']['residual_px'][s]['median']) if capa else None),
            link6_world_at_home=L.tolist(),
            rail_form=rail_form(fkf, metric[s], L),
            rail_form_alt_j2_j4_j6_ill_conditioned=rail_form(fkf, metric[s], L, (1, 3, 5)))
    src = dict(fit=str(a.fit), variant=fit['variant'], stage1=str(a.stage1), captures=str(a.captures),
               check_render=str(a.check_render) if a.check_render else None, note=a.note)
    (a.out / 'wrist_cameras.json').write_text(json.dumps(dict(
        schema=1, created='2026-09-30', task='yam-wrist-claw-fit-20260930', source=src,
        frames=('world = metric frame (scenes/yam_bimanual_rig/metric/metric_frame.json); camera axes OpenCV (x right, '
                'y down, z forward); link_6 and grasp_site = MJCF frames at raiden home (q = 0, gripper open 0.0475 m)'),
        how=('world rotation from the background (stage 1, named desk/wall edges); position rides with link_6 at the '
             'hand-eye translation of the fit variant; arm base and claw placement fitted to the wrist claws and the '
             'ZED scene camera (stage 2). The link_6 / grasp_site poses hold only with the proposed bases.'),
        claw_json=str(W.HERE / 'source/umi_ft/claw.json'), cameras=cams), indent=1))
    (a.out / 'arm_bases_proposed.json').write_text(json.dumps(dict(
        schema=1, created='2026-09-30', task='yam-wrist-claw-fit-20260930', status='proposal (metric/ not edited)',
        source=src,
        meaning=('world_from_base = equivalent arm base at raiden home: base x FK(q = 0) puts link_6 where the home '
                 'images put it (it may tilt and lift the base). rail_form = the same link_6 pose with the base on the '
                 'rail (metric height, no roll / pitch; yaw and x, y fitted) plus home joint offsets (joint 2, 3, 6): the '
                 'recommended form (no base tilt). At one pose a base yaw equals a joint-1 zero offset exactly, and a base '
                 'pitch equals a joint 2 / 3 / 4 offset up to a few mm of link_6 position, which the free base x, y absorb: '
                 'these images do not tell them apart (see the stage-2 summary).'),
        identifiable=dict(
            yes=['link_6 pose at home in the world (rotation about 0.5 deg, position about 1-2 cm along the scene-camera '
                 'rays); in rail form: base yaw (= joint-1 offset), the pitch sum joint2 - joint3 (base pitch), joint-6 '
                 'roll, base x / y'],
            no=['base tilt vs joint 2 / 3 / 4 zero offsets or sag (only link_6 is visible: links 1-5 are outside the '
                'three views)', 'the split of the pitch between joints 2, 3 and 4', 'base height vs that split',
                'a static offset vs a pose-dependent one (one pose only: the arms at home)'],
            needs=('several arm poses seen by the scene camera with the claws or wrist board in view (e.g. the '
                   'calibration captures with the arm links visible), or a direct measurement of the base on the rail')),
        arms=bases), indent=1))
    (a.out / 'cameras_world.json').write_text(json.dumps(cams_world, indent=1))
    (a.out / 'bases_proposed_4x4.json').write_text(json.dumps(bases_4x4, indent=1))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
