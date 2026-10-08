"""Write a build config (+ optional metric-frame copy) for a check render of the twin with other wrist cameras.

    ~/robot/raiden/.venv/bin/python build/make_twin_config.py --cameras CAMERAS.json --out DIR [--bases BASES.json]

Input: the v2 delivery config (read only). CAMERAS.json: {"left": T_world_cam (OpenCV 4x4), "right": ...}. The
cameras go into the config as 'wrist_refinement' T_grasp_site_cam_opencv entries relative to the arm's grasp_site
at raiden home, which build_scene.py already reads (no build code change). BASES.json (optional):
{"left": world_from_base, "right": ...}: writes DIR/metric_frame.json, a copy of the metric frame with only
arms.*.world_from_base replaced (for build_scene.py --metric); the original metric frame is not touched.
Writes DIR/config.json (and DIR/metric_frame.json).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import wristfit_common as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bases', type=Path)
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = W.jload(W.BUILD_CFG)
    m = W.jload(W.METRIC)
    cams = W.jload(a.cameras)
    bases = W.metric_bases(m)
    if a.bases:
        nb = W.jload(a.bases)
        for s in W.SIDES:
            if s in nb:
                bases[s] = np.array(nb[s])
                m['arms'][W.ARM[s]]['world_from_base'] = bases[s].tolist()
                m['arms'][W.ARM[s]]['override_note'] = f'check-render copy: base from {a.bases} (09-30 wrist/claw fit); original in metric/metric_frame.json'
        (a.out / 'metric_frame.json').write_text(json.dumps(m, indent=1))
    gs = W.FK()(W.HOME_Q)['grasp_site']
    cfg['wrist_camera_source'] = 'wrist_refinement'
    cfg['wrist_refinement'] = {}
    for s in W.SIDES:
        T = np.array(cams[s])
        Tg = np.linalg.inv(bases[s] @ gs) @ T
        cfg['wrist_refinement'][f'{s}_wrist'] = dict(T_grasp_site_cam_opencv=Tg.tolist(), method=f'09-30 wrist/claw fit ({a.cameras}); {a.note}')
    cfg['note_wrist_fit_0930'] = f'check-render config written by assets/yam_arm/build/make_twin_config.py from {W.BUILD_CFG.name}; {a.note}'
    (a.out / 'config.json').write_text(json.dumps(cfg, indent=1))
    print('wrote', a.out / 'config.json')


if __name__ == '__main__':
    main()
