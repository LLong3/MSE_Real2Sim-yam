"""Summarise the stage-2 variants and test what the home images can tell apart (CPU, raiden venv).

    ~/robot/raiden/.venv/bin/python build/stage2_summary.py --run RUN [--variants a a_mount b ...] --out RUN/stage2/summary.json

Table per variant: data loss (wrist + scene, both arms), mean IoUs, scene-camera centroid offsets (model - real, px),
background-edge medians, base correction vs the metric base, hand-eye X vs raiden and vs the captures, C.
Identifiability (per arm, from a fitted variant): the link_6 pose change vs metric (rotation w, position dp) is
explained by (H0) a base rotation about the base origin, (H1) base yaw + joint 2 offset, (H2) base yaw + joint 3,
(H3) base yaw + joint 4, (H4) base yaw + joint 5 (+ joint 6 roll in H1-H4): each reproduces w exactly (least
squares); the table gives the link_6 and claw-centre position each predicts and its scene-camera pixel offset from
the fitted pose. A base yaw and a joint-1 offset are the same rotation (same axis): never separable.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import wristfit_common as W

CLAW_CENTRE_L6 = np.array([0.0, 0.0, -0.143])      # middle of the pad contact face (z6 -95 .. -192 mm)


def load(run, v):
    p = Path(run) / 'stage2' / v / 'fit.json'
    return W.jload(p) if p.exists() else None


def row(v, r):
    out = dict(variant=v, loss_data=r['loss_data'], loss_terms=r['loss_terms_total'], C_mm=[round(x * 1e3, 1) for x in r['shared']['C'][:3]],
               gap_q0_mm=r['shared']['gap_at_q0_mm'], mount_rotvec_deg=r['shared'].get('mount_rotvec_deg'), arms={})
    for s in W.SIDES:
        a = r['arms'][s]
        c = a['centroid_model_minus_real_px']
        out['arms'][s] = dict(
            iou_wrist_mean=round(float(np.mean(list(a['iou_wrist'].values()))), 3), iou_wrist=a['iou_wrist'],
            iou_scene_mean=round(float(np.mean(list(a['iou_scene'].values()))), 3), iou_scene=a['iou_scene'],
            scene_centroid_sil_px=c['scene']['sil'], scene_centroid_pad_px=c['scene']['pad'], scene_centroid_yellow_px=c['scene']['yellow'],
            wrist_centroid_pad_px=c['wrist']['pad'], wrist_centroid_yellow_px=c['wrist']['yellow'],
            background_median_px=a['background_edges']['all_samples_median_abs_px'],
            base_vs_metric=a['base']['vs_metric'], base_vs_capture=a['base'].get('vs_capture_base'), yaw_world_deg=a['base']['yaw_world_deg'],
            X_vs_raiden=a['hand_eye'].get('vs_raiden'), X_vs_captures=a['hand_eye'].get('vs_captures'),
            joint_offsets_deg=a.get('joint_offsets_deg'), camera_offset_from_link6_riding_mm=a.get('camera_translation_vs_stage1_mm'))
    return out


def identifiability(r, fk, Ks, S):
    out = {}
    metric = W.metric_bases()
    q0 = np.zeros(6)
    T6_0 = fk(q0, frames=('link_6',))['link_6']
    for s in W.SIDES:
        B = np.array(r['arms'][s]['T_world_base'])
        Bm = metric[s]
        L_fit = B @ T6_0                          # fitted link_6 in the world (base-correction variants)
        if r['arms'][s].get('joint_offsets_deg') is not None:
            L_fit = Bm @ fk(np.radians(r['arms'][s]['joint_offsets_deg']), frames=('link_6',))['link_6']
        L0 = Bm @ T6_0
        dR = L_fit[:3, :3] @ L0[:3, :3].T
        w = Rot.from_matrix(dR).as_rotvec()
        claw = lambda L: (L @ np.r_[CLAW_CENTRE_L6, 1])[:3]
        res = dict(link6_rotation_change_world_deg=np.degrees(w).round(3).tolist(),
                   link6_position_change_world_mm=((L_fit[:3, 3] - L0[:3, 3]) * 1e3).round(1).tolist(),
                   hypotheses={})

        def hyp(name, L):
            dc = claw(L) - claw(L_fit)
            uv_f = W.project(claw(L_fit)[None], Ks, S)[0][0]
            uv_h = W.project(claw(L)[None], Ks, S)[0][0]
            uv6_f = W.project(L_fit[:3, 3][None], Ks, S)[0][0]; uv6_h = W.project(L[:3, 3][None], Ks, S)[0][0]
            rr = np.degrees(np.linalg.norm(Rot.from_matrix(L[:3, :3] @ L_fit[:3, :3].T).as_rotvec()))
            res['hypotheses'][name] = dict(rotation_residual_deg=round(float(rr), 3),
                                           claw_centre_minus_fit_mm=(dc * 1e3).round(1).tolist(),
                                           scene_px_claw_centre=(uv_h - uv_f).round(2).tolist(),
                                           scene_px_link6=(uv6_h - uv6_f).round(2).tolist())
        # H0: base rotation about the base origin, base position = metric
        Bh = Bm.copy(); Bh[:3, :3] = dR @ Bm[:3, :3]
        hyp('H0_base_rotation', Bh @ T6_0)
        for name, js in (('H1_yaw_j2_j6', (0, 1, 5)), ('H2_yaw_j3_j6', (0, 2, 5)), ('H3_yaw_j4_j6', (0, 3, 5)),
                         ('H4_yaw_j5_j6_j2', (0, 4, 5, 1))):
            def rr(x):
                q = np.zeros(6); q[list(js)] = x
                L = Bm @ fk(q, frames=('link_6',))['link_6']
                return Rot.from_matrix(L[:3, :3] @ L_fit[:3, :3].T).as_rotvec() / np.radians(0.01)
            x = least_squares(rr, np.zeros(len(js))).x
            q = np.zeros(6); q[list(js)] = x
            hyp(name, Bm @ fk(q, frames=('link_6',))['link_6'])
            res['hypotheses'][name]['joint_offsets_deg'] = {f'j{j + 1}': round(float(np.degrees(v)), 3) for j, v in zip(js, x)}
        out[s] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--variants', nargs='*', default=['a', 'a_mount', 'b', 'b_capX', 'b_joints', 'c', 'b_mount', 'c_mount'])
    ap.add_argument('--ident', nargs='*', default=['b', 'c', 'b_capX', 'b_mount', 'c_mount'], help='variants for the identifiability test')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    fk = W.FK()
    Ks, S = W.scene_camera()
    table, ident = [], {}
    for v in a.variants:
        r = load(a.run, v)
        if r is None:
            continue
        table.append(row(v, r))
        if v in a.ident:
            ident[v] = identifiability(r, fk, Ks, S)
    res = dict(table=table, identifiability=ident)
    a.out.write_text(json.dumps(res, indent=1))
    for t in table:
        print(f"{t['variant']:9s} data {t['loss_data']:.3f} C {t['C_mm']} gap {t['gap_q0_mm']} mount {t['mount_rotvec_deg']}")
        for s in W.SIDES:
            x = t['arms'][s]
            print(f"   {s:5s} wristIoU {x['iou_wrist_mean']} sceneIoU {x['iou_scene_mean']} scene sil/pad/yel {x['scene_centroid_sil_px']} "
                  f"{x['scene_centroid_pad_px']} {x['scene_centroid_yellow_px']} bg {x['background_median_px']} base {x['base_vs_metric']} "
                  f"X-raiden {x['X_vs_raiden'] and x['X_vs_raiden']['rot_deg']} X-cap {x['X_vs_captures'] and x['X_vs_captures']['rot_deg']}")
    for v, d in ident.items():
        for s, r in d.items():
            print(v, s, 'w', r['link6_rotation_change_world_deg'], 'dp', r['link6_position_change_world_mm'])
            for h, x in r['hypotheses'].items():
                print('    ', h, x)


if __name__ == '__main__':
    main()
