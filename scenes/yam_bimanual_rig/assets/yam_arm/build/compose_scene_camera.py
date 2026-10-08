"""Scene-camera review: real ZED image vs the asset render, silhouette agreement per arm.

    .runtime/pi3x-inference/venv/bin/python build/compose_scene_camera.py \
        --real ~/robot/bimanual_bringup/pi05/scene_camera.jpg --dir evidence/scene_camera \
        --sam-instances left_arm=1 right_arm=0 --media ~/robot/markdowns/bimanual/1_media/yam_asset

<dir> holds the outputs of render_scene_camera.py (render_rgba.png, tris_metric.npz, camera.json) and
of sam_arm_mask.py (sam/masks.npz). Model silhouettes are exact triangle fills projected with the
metric K and pose (checked against the Cycles alpha). Per arm: IoU with the SAM 3.1 mask, precision
(rendered pixels inside the real outline) and recall (real pixels covered by the model), for the whole
asset and for the asset without the fingers (claws and carriage). Writes <dir>/metrics.json and two JPEGs.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MAGENTA, GREEN = (255, 0, 255), (0, 230, 0)


def project_mask(tris, K, cam_from_world, W, H, ss=4):
    """Pixel coverage >= 0.5, from an ss x supersampled fill (OpenCV pixel centres at integers)."""
    p = tris.reshape(-1, 3).astype(np.float64) @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
    z = p[:, 2].reshape(-1, 3)
    uv = (p[:, :2] / p[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    uv_hi = (uv.reshape(-1, 3, 2)[(z > 0.02).all(1)] + 0.5) * ss - 0.5
    pts = np.round(uv_hi * 16).astype(np.int32)
    img = np.zeros((H * ss, W * ss), np.uint8)
    for t in pts:  # one call per triangle (a single fillPoly XORs overlaps)
        cv2.fillConvexPoly(img, t, 255, cv2.LINE_8, 4)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) >= 128


def stats(model, real):
    inter, union = (model & real).sum(), (model | real).sum()
    return dict(iou=float(inter / max(union, 1)), precision=float(inter / max(model.sum(), 1)),
                recall=float(inter / max(real.sum(), 1)), model_px=int(model.sum()), real_px=int(real.sum()))


def label(im, text, scale=0.55):
    cv2.rectangle(im, (0, 0), (im.shape[1], 26), (25, 25, 25), -1)
    cv2.putText(im, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def outline(im, mask, colour, k=2):
    e = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((k, k), np.uint8)) > 0
    im[e] = colour
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--sam-instances', nargs='+', required=True, help='arm=index into sam/masks.npz')
    ap.add_argument('--media', type=Path, required=True)
    a = ap.parse_args()
    cam = json.loads((a.dir / 'camera.json').read_text())
    W, H = cam['image_size_wh']
    K = np.array(cam['K'])
    cam_from_world = np.linalg.inv(np.array(cam['world_from_camera_opencv']))
    tris = dict(np.load(a.dir / 'tris_metric.npz'))
    sam = np.load(a.dir / 'sam' / 'masks.npz')
    real = cv2.imread(a.real)
    rgba = cv2.imread(str(a.dir / 'render_rgba.png'), cv2.IMREAD_UNCHANGED)
    alpha = rgba[:, :, 3] > 127
    inst = dict(s.split('=') for s in a.sam_instances)

    res = dict(real_image=a.real, camera=str(a.dir / 'camera.json'), sam_masks=str(a.dir / 'sam' / 'masks.npz'),
               sam_instances={k: int(v) for k, v in inst.items()}, arms={})
    all_model = np.zeros((H, W), bool); all_real = np.zeros((H, W), bool)
    masks = {}
    for arm, i in inst.items():
        keys = [k for k in tris if k.startswith(arm + '|')]
        full = project_mask(np.concatenate([tris[k] for k in keys]), K, cam_from_world, W, H)
        fingers = ('tip_left', 'tip_right', '|claw_', '|carriage_')
        body = project_mask(np.concatenate([tris[k] for k in keys if not any(f in k for f in fingers)]),
                            K, cam_from_world, W, H)
        r = sam['masks'][int(i)]
        res['arms'][arm] = dict(sam_score=float(sam['scores'][int(i)]), whole_asset=stats(full, r),
                                without_fingers=stats(body, r))
        all_model |= full; all_real |= r
        masks[arm] = (full, r)
    res['both_arms'] = stats(all_model, all_real)
    res['projection_vs_cycles_alpha_iou'] = stats(all_model, alpha)['iou']
    (a.dir / 'metrics.json').write_text(json.dumps(res, indent=1))

    a.media.mkdir(parents=True, exist_ok=True)
    grey = np.full_like(real, 128)
    al = rgba[:, :, 3:4].astype(np.float32) / 255
    on_grey = (rgba[:, :, :3] * al + grey * (1 - al)).astype(np.uint8)
    over = (rgba[:, :, :3] * al * 0.6 + real * (1 - al * 0.6)).astype(np.uint8)
    over = outline(outline(over, all_real, GREEN), all_model, MAGENTA)
    s = 0.7
    panels = [cv2.resize(p, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for p in (real, on_grey, over)]
    label(panels[0], 'real: ZED scene camera 09-29, arms at home')
    label(panels[1], 'asset x2 at home, metric K + pose (no fitting)')
    label(panels[2], f'overlay: render 60%, asset outline magenta, SAM 3.1 green')
    sheet = np.hstack(panels)
    cv2.imwrite(str(a.media / '01_scene_camera_side_by_side.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    rows = []
    for arm in ('left_arm', 'right_arm'):
        ys, xs = np.nonzero(masks[arm][0] | masks[arm][1])
        x0 = max(0, (xs.min() + xs.max()) // 2 - 110); y0 = max(0, ys.min() - 20)
        x1, y1 = min(W, x0 + 220), H
        m = res['arms'][arm]
        zoom = [cv2.resize(p[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) for p in (real, on_grey, over)]
        label(zoom[0], f'{arm}: real')
        label(zoom[1], f'{arm}: asset (UMI-FT claws)')
        label(zoom[2], f'IoU {m["whole_asset"]["iou"]:.2f}, asset inside real {m["whole_asset"]["precision"]:.2f}, '
                       f'real covered {m["whole_asset"]["recall"]:.2f}', 0.5)
        rows.append(np.hstack(zoom))
    cv2.imwrite(str(a.media / '02_scene_camera_grippers_zoom.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps({k: res[k] for k in ('arms', 'both_arms', 'projection_vs_cycles_alpha_iou')}, indent=1))


if __name__ == '__main__':
    main()
