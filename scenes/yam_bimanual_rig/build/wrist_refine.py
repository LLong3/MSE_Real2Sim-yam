"""Refine the approximate wrist-camera rotation against the real wrist image (hand-eye translation kept).

python wrist_refine.py --render RUN/renders/vNN --config configs/vNN.json --lines wrist_lines.json --out configs/vMM.json

Start: arm FK grasp_site x raiden hand-eye T_grasp_site_cam (camera matrix read back from render.json).
Real line segments read off the wrist images (wrist_lines.json) are matched to named model edges; the camera
rotation (3 small angles about its optical centre) minimises the perpendicular distance of the projected model
edge samples to the real lines (samples within the real segment's extent). The refined T_grasp_site_cam goes
into config['wrist_refinement'] and is used by build_scene.py. Still approximate: finger geometry, the bracket
and the translation are not fitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rigcam import metric

CALIB = Path('/home/frank/.config/raiden/calibration_results.json')
KEYS = {'left_wrist': 'left_wrist_camera', 'right_wrist': 'right_wrist_camera'}


def named_segments(m, cfg):
    d = m['desk']; x0, x1 = d['box_x_m']; y0, y1 = d['box_y_m']; zt = d['top_z_m']
    wy = -m['planes']['back_wall']['d'] * m['planes']['back_wall']['n'][1]
    we = cfg.get('wall_end_x_override', {}).get('x', m['planes']['back_wall']['wall_end_x_m'])
    g = cfg['glass']; md = cfg['monitor_desk']
    return {
        'desk_back_edge': ((x0, y1, zt), (x1, y1, zt)),
        'desk_left_edge': ((x0, y0, zt), (x0, y1, zt)),
        'desk_right_edge': ((x1, y0, zt), (x1, y1, zt)),
        'wall_end_vertical': ((we, wy, zt + 0.01), (we, wy, 2.4)),
        'glass_frosted_bottom': ((g['post_x'], g['plane_y'], g['frosted_z'][0]), (we, g['plane_y'], g['frosted_z'][0])),
        'monitor_desk_left_edge': ((md['x'][0], md['y'][0], md['top_z']), (md['x'][0], md['y'][1], md['top_z'])),
        'privacy_panel_left_edge': ((md['panel_x'][0], md['panel_y'][0], md['panel_z'][0]), (md['panel_x'][0], md['panel_y'][0], md['panel_z'][1] - 0.06)),
    }


def segments(m, cfg):
    return list(named_segments(m, cfg).values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--render', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--lines', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()); m = metric()
    rj = json.loads((a.render / 'render.json').read_text())
    calib = json.loads(CALIB.read_text())
    lines = json.loads(a.lines.read_text())
    segs = named_segments(m, cfg)
    out = cfg.setdefault('wrist_refinement', {})
    log = {}
    for view, key in KEYS.items():
        K = np.array(cfg['camera_model'][view]['K'])
        T0 = np.array(rj['views'][view]['matrix_world']) @ np.diag([1, -1, -1, 1])   # world_from_camera, OpenCV axes
        he = calib['cameras'][key]['hand_eye_calibration']
        Tgc = np.eye(4); Tgc[:3, :3] = he['rotation_matrix']; Tgc[:3, 3] = he['translation_vector']
        if view in cfg.get('wrist_refinement', {}):
            Tgc = np.array(cfg['wrist_refinement'][view]['T_grasp_site_cam_opencv'])
        Twg = T0 @ np.linalg.inv(Tgc)
        pairs = []
        for item in lines[view]:
            s0, s1 = segs[item['model']]
            q0, q1 = np.array(item['real'], float)
            t = (q1 - q0) / np.linalg.norm(q1 - q0); n = np.array([-t[1], t[0]])
            pairs.append((np.linspace(s0, s1, 400), q0, t, n, np.linalg.norm(q1 - q0), item['model']))
        def proj(w, P):
            R = T0[:3, :3] @ Rotation.from_rotvec(w).as_matrix()
            pc = (P - T0[:3, 3]) @ R
            return pc[:, :2] / np.clip(pc[:, 2:3], 1e-3, None) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        def res(w, per=False):
            r, pl = [], {}
            for P, q0, t, n, L, name in pairs:
                uv = proj(w, P)
                s = (uv - q0) @ t
                keep = (s >= -5) & (s <= L + 5)
                d = (uv[keep] - q0) @ n
                # fixed-size residual per line: 20 quantiles of the in-extent distances (penalise empty overlap)
                v = np.quantile(d, np.linspace(0, 1, 20)) if keep.sum() >= 5 else np.full(20, 50.0)
                r.append(v); pl[name] = float(np.median(np.abs(v)))
            return pl if per else np.concatenate(r)
        before = res(np.zeros(3), True)
        w = least_squares(res, np.zeros(3), loss='soft_l1', f_scale=3.0).x
        after = res(w, True)
        T1 = T0.copy(); T1[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(w).as_matrix()
        Tgc1 = np.linalg.inv(Twg) @ T1
        log[view] = dict(rotation_correction_deg=np.degrees(w).round(3).tolist(), angle_deg=round(float(np.degrees(np.linalg.norm(w))), 3),
                         line_offset_px_before=before, line_offset_px_after=after)
        out[view] = dict(T_grasp_site_cam_opencv=Tgc1.tolist(), rotation_correction_deg=log[view]['rotation_correction_deg'],
                         lines=str(a.lines), method='rotation about the optical centre fitted to real-image line segments; hand-eye translation kept')
    cfg.setdefault('calibration_log', []).append(dict(step='wrist_refine', render=str(a.render), **log))
    a.out.write_text(json.dumps(cfg, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
