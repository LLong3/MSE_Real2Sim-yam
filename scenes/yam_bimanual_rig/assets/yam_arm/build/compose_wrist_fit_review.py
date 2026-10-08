"""Review JPEGs of the 09-30 wrist/claw fit (numpy/OpenCV, raiden venv; CPU).

    ~/robot/raiden/.venv/bin/python build/compose_wrist_fit_review.py stage1 --run RUN --out MEDIA/01_....jpg
    ~/robot/raiden/.venv/bin/python build/compose_wrist_fit_review.py claws --render DIR --cameras CAMS.json [--bases B.json] --out ...
    ~/robot/raiden/.venv/bin/python build/compose_wrist_fit_review.py scene --render DIR --out ...
    ~/robot/raiden/.venv/bin/python build/compose_wrist_fit_review.py variants --run RUN --out ...

stage1:   per wrist: real image with the named background edges projected from the v2 camera (red) and the stage-1
          camera (green) | v2 render | stage-1 render (renders with the camera model; cyan dots = real edge points
          found along the named edges).
claws:    per wrist: real | final render; outlines of the real pad (red), clamp (magenta), contact face (cyan) on both.
scene:    ZED scene camera, 4x zoom on each gripper: real | v2 render | final render (SAM outline green).
variants: stage-2 fit overlays per variant (wrist and scene zoom), with the key numbers.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import claw_stage2_fit as F
import wristfit_common as W

V2 = W.V2_RUN / 'renders' / 'delivery_v2'


def label(img, text, scale=0.7):
    img = img.copy()
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def seg_lines(img, K, T, segs, col, th=2):
    for a, b in segs.values():
        P = np.linspace(np.asarray(a), np.asarray(b), 200)
        uv, z = W.project(P, K, T)
        ok = (z > 0.05) & (np.abs(uv) < 5000).all(1)
        pts = np.round(uv[ok]).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
    return img


def real_edge_points(side, T):
    """real-image edge points found along the named background edges (wrist_bg_fit.measure, search 12 px at T)."""
    import wrist_bg_fit as BG
    img = W.real_image(side)
    segs = W.named_segments()
    meas = BG.measure(W.colour_gradient(img), W.wrist_K(side), T, {k: segs[k] for k in BG.EDGES[side]}, W.exclusion(side, img), 12)
    pts = [v['uv'][np.isfinite(v['off'])] + v['off'][np.isfinite(v['off'])][:, None] * v['nrm'] for v in meas.values()]
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def over_edges(img, pts, col=(255, 255, 0)):
    img = img.copy()
    for u, v in np.round(pts).astype(int):
        cv2.circle(img, (u, v), 2, col, -1, cv2.LINE_AA)
    return img


def stage1(a):
    import wrist_bg_fit as BG
    run = Path(a.run)
    cams1 = W.jload(run / 'stage1' / 'cameras_stage1.json')
    builds = W.build_poses()
    segs = W.named_segments()
    rows = []
    for side in W.SIDES:
        K = W.wrist_K(side)
        s = {k: segs[k] for k in BG.EDGES[side]}
        real = W.real_image(side).copy()
        seg_lines(real, K, builds['delivery_v2'][side], s, (0, 0, 255))
        seg_lines(real, K, np.array(cams1[side]), s, (0, 255, 0))
        e = real_edge_points(side, np.array(cams1[side]))
        v2 = over_edges(cv2.imread(str(V2 / f'{side}_wrist_cam.png')), e)
        s1 = over_edges(cv2.imread(str(run / 'check_stage1' / 'renders' / f'{side}_wrist_cam.png')), e)
        fit = W.jload(run / 'stage1' / 'stage1_background_fit.json')['cameras'][side]
        m2 = fit['candidates']['v2_claw_fit']['edges']['all_samples_median_abs_px']
        m1 = fit['fits']['stage1_rotation_only']['edges']['all_samples_median_abs_px']
        rows.append(np.hstack([label(real, f'{side} wrist real: model edges v2 red, stage 1 green'),
                               label(v2, f'v2 render ({m2} px); cyan = real edge points'),
                               label(s1, f'stage-1 camera render ({m1} px)')]))
    out = np.vstack(rows)
    cv2.imwrite(str(a.out), cv2.resize(out, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])


def outline(img, m, col, th=1):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA)
    return img


def claws(a):
    rd = Path(a.render)
    rows = []
    for side in W.SIDES:
        r = F.wrist_real(W.real_image(side))
        panels = []
        for name, img in (('real', W.real_image(side)), ('v2', cv2.imread(str(V2 / f'{side}_wrist_cam.png'))),
                          ('final', cv2.imread(str(rd / f'{side}_wrist_cam.png')))):
            img = img.copy()
            for m, col in ((r['pad'], (0, 0, 255)), (r['yellow'], (255, 0, 255)), (r['face'], (255, 255, 0))):
                outline(img, m, col, 2)
            panels.append(label(img, f'{side} wrist {name}; real outlines: pad red, clamp magenta, face cyan'))
        rows.append(np.hstack(panels))
    cv2.imwrite(str(a.out), cv2.resize(np.vstack(rows), None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def scene(a):
    rd = Path(a.render)
    sam = np.load(W.HERE / 'evidence/scene_camera/sam/masks.npz')['masks']
    sr = F.scene_real(cv2.imread(str(W.REAL / 'scene_camera.jpg')), sam)
    imgs = [('real 09-29', cv2.imread(str(W.REAL / 'scene_camera.jpg'))), ('v2', cv2.imread(str(V2 / 'scene_cam.png'))),
            ('new (v3 inputs)', cv2.imread(str(rd / 'scene_cam.png')))]
    rows = []
    for name, img in imgs:
        img = img.copy()
        for side in W.SIDES:
            outline(img, sr[side]['sil'], (0, 255, 0), 1)
        crops = [cv2.resize(img[490:600, x0:x0 + 220], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) for x0 in (200, 610)]
        rows.append(label(np.hstack(crops), f'{name}; green = SAM outline of the real grippers'))
    cv2.imwrite(str(a.out), cv2.resize(np.vstack(rows), None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def variants(a):
    run = Path(a.run)
    rows = []
    for v in a.variants:
        d = run / 'stage2' / v
        res = W.jload(d / 'fit.json')
        w = [cv2.resize(cv2.imread(str(d / f'wrist_{s}.jpg')), (480, 300), interpolation=cv2.INTER_AREA) for s in W.SIDES]
        sc = [cv2.resize(cv2.imread(str(d / f'scene_{s}_zoom4.jpg')), (480, 240), interpolation=cv2.INTER_AREA) for s in W.SIDES]
        txt = []
        for s in W.SIDES:
            ar = res['arms'][s]
            c = ar['centroid_model_minus_real_px']['scene']['sil']
            txt.append(f"{s}: wrist IoU {np.mean(list(ar['iou_wrist'].values())):.2f}, scene IoU {np.mean(list(ar['iou_scene'].values())):.2f}, "
                       f"scene dv {c[1] if c else 'na'} px")
        top = np.hstack([label(w[0], f'{v}: {txt[0]}', 0.45), label(w[1], txt[1], 0.45)])
        bot = np.hstack([cv2.resize(sc[0], (480, 240)), cv2.resize(sc[1], (480, 240))])
        rows.append(np.vstack([top, bot]))
    cv2.imwrite(str(a.out), cv2.resize(np.vstack(rows), None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 86])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['stage1', 'claws', 'scene', 'variants'])
    ap.add_argument('--run')
    ap.add_argument('--render')
    ap.add_argument('--variants', nargs='*', default=['a', 'a_mount', 'b', 'b_joints', 'c'])
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dict(stage1=stage1, claws=claws, scene=scene, variants=variants)[a.what](a)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
