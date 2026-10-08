"""Validate claw_model.py (numpy claw geometry + painter's label rasterizer) against Blender (CPU, raiden venv).

    ~/robot/raiden/.venv/bin/python build/claw_model_check.py --render-dir RENDERS --claw-json CLAW.json \
        --open-half-pad 0.0525 --out OUT [--cameras render|CAMERAS.json] [--bases BASES.json]

Reference: the object-ID pass of a Blender render of the twin (render_views.py: <view>_ids.exr + ids.json; first
hit per pixel). Model: claw_model parts from CLAW.json, placed with the pad contact face at y6 = -/+ open-half-pad
(v2: 0.051 + 1.5 mm strip = 0.0525), plus the stock housing and carriage as occluders, rendered with the same
wrist cameras (render.json matrix_world) and K (wrist_K, fx != fy). Per view and part: IoU of model label vs
Blender object ID (arm of that wrist only; scene: both arms, with link4/5 as occluders). Writes OUT/check.json and OUT/<view>_labels.png (Blender | model | diff).
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import claw_model as C  # noqa: E402
import wristfit_common as W  # noqa: E402

PART_OF_OBJECT = {'carriage': 'carriage', 'gripper_geom': 'housing'}


def blender_labels(rd, view, arms):
    a = cv2.imread(str(Path(rd) / f'{view}_ids.exr'), cv2.IMREAD_UNCHANGED)
    k = np.round(a[:, :, 2] * 255).astype(int) + 256 * np.round(a[:, :, 1] * 255).astype(int)
    k[np.abs(a[:, :, 0] - 1.0) > 1e-3] = 0
    table = json.loads((Path(rd) / 'ids.json').read_text())
    lut = np.zeros(k.max() + 1, np.uint8)
    for i, v in table.items():
        i = int(i)
        if i > k.max() or v['root'] not in arms:
            continue
        name = v['object'].split('.')[0]
        part = None
        if name.startswith('yam_tip_') and '_umi_' in name:
            part = name.split('_umi_')[1]
        elif name.endswith('_carriage'):
            part = 'carriage'
        elif name == 'yam_gripper_geom':
            part = 'housing'
        if part in C.LABELS:
            lut[i] = C.LABELS[part]
    return lut[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-dir', type=Path, required=True)
    ap.add_argument('--claw-json', type=Path, required=True)
    ap.add_argument('--open-half-pad', type=float, required=True, help='pad contact face |y6| at home (m)')
    ap.add_argument('--pad-z6', type=float)
    ap.add_argument('--pad-x6', type=float)
    ap.add_argument('--skip-parts', nargs='*', default=['pad_face'], help='model parts hidden (v2: pad face behind the strip)')
    ap.add_argument('--merge-face', action='store_true', help='score the model pad face as pad (schema 2: one Blender object)')
    ap.add_argument('--bases', type=Path, help='world_from_base per side (default: metric frame)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = C.load_cfg(a.claw_json)
    zc = a.pad_z6 if a.pad_z6 is not None else cfg['pad_z6_m']
    xc = a.pad_x6 if a.pad_x6 is not None else cfg['pad_x6_m']
    pF = {k: v for k, v in C.parts_F(cfg).items() if k not in a.skip_parts}
    fk = W.FK()
    arm, T6 = C.arm_tris(fk)
    claws = C.transform(C.claw_tris_l6(pF, a.open_half_pad, zc, xc), T6)
    bases = W.metric_bases()
    if a.bases:
        bases.update({s: np.array(v) for s, v in W.jload(a.bases).items() if s in W.SIDES})
    rj = W.jload(a.render_dir / 'render.json')
    res = dict(render_dir=str(a.render_dir), claw_json=str(a.claw_json), open_half_pad_m=a.open_half_pad, pad_z6_m=zc,
               pad_x6_m=xc, skipped_parts=a.skip_parts, views={})
    names = {v: k for k, v in C.LABELS.items()}
    Ks, _ = W.scene_camera()
    arm5, _ = C.arm_tris(fk, links=('link_6', 'link5', 'link4'))
    for side in W.SIDES + ('scene',):
        view = f'{side}_wrist' if side != 'scene' else 'scene'
        if view not in rj['views']:
            continue
        T = np.array(rj['views'][view]['matrix_world']) @ W.CV2BL
        if side == 'scene':      # both arms; link4/link5 as occluders; housing and carriage are compared too
            K = Ks
            items = C.transform(arm5 + claws, bases['left']) + C.transform(arm5 + claws, bases['right'])
            Lb = blender_labels(a.render_dir, view, ('left_arm', 'right_arm'))
        else:
            K = W.wrist_K(side)
            items = C.transform(arm + claws, bases[side])
            Lb = blender_labels(a.render_dir, view, (W.ARM[side],))
        Lm = C.render_labels(items, K, T)
        Lm[Lm == C.LABELS['arm']] = 0
        if a.merge_face:
            Lm[Lm == C.LABELS['pad_face']] = C.LABELS['pad']
        r = {}
        for lab in sorted(set(np.unique(Lb)) | set(np.unique(Lm))):
            if lab == 0:
                continue
            m, b = Lm == lab, Lb == lab
            r[names[lab]] = dict(iou=round(float((m & b).sum() / max((m | b).sum(), 1)), 4), model_px=int(m.sum()), blender_px=int(b.sum()))
        allm, allb = Lm > 0, Lb > 0
        r['_any_part'] = dict(iou=round(float((allm & allb).sum() / max((allm | allb).sum(), 1)), 4))
        r['_label_agreement_in_union'] = round(float((Lm == Lb)[allm | allb].mean()), 4)
        res['views'][view] = r
        pal = np.random.default_rng(3).integers(40, 255, (32, 3)).astype(np.uint8); pal[0] = 0
        diff = np.where(((Lm != Lb) & (allm | allb))[..., None], np.array([255, 255, 255], np.uint8), 0).astype(np.uint8)
        cv2.imwrite(str(a.out / f'{view}_labels.png'), np.hstack([pal[Lb], pal[Lm], diff]))
        print(view, json.dumps(r))
    (a.out / 'check.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
