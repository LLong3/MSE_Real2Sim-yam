"""Wrist-camera review: real wrist image vs the asset render, pad and clamp silhouettes per arm.

    .runtime/pi3x-inference/venv/bin/python build/compose_wrist_camera.py --dir evidence/wrist_camera \
        --real-dir ~/robot/bimanual_bringup/pi05 --media ~/robot/markdowns/bimanual/1_media/yam_asset

<dir> holds render_wrist_camera.py outputs for left_arm and right_arm. Real masks: HSV colour thresholds for the
orange pads and yellow clamps (holes filled). Model masks: painter's-algorithm labels of the projected claw
triangles (pad, clamp; other visible parts occlude). IoU per arm with the camera the render used (raw hand-eye,
or with the fitted correction; see <arm>_camera.json and evidence/claw_fit/fit.json for both).
Writes <dir>/metrics.json and <media>/05_wrist_view_vs_real.jpg.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ORANGE, YELLOW = (0, 140, 255), (0, 230, 255)
REAL_O, REAL_Y = (0, 0, 255), (255, 0, 255)


def fill(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(m.shape, np.uint8); cv2.drawContours(out, cs, -1, 1, -1)
    return out > 0


def real_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k = np.ones((3, 3), np.uint8)
    o = cv2.morphologyEx(((h >= 3) & (h <= 17) & (s >= 140) & (v >= 110)).astype(np.uint8), cv2.MORPH_OPEN, k)
    y = cv2.morphologyEx(((h >= 20) & (h <= 34) & (s >= 110) & (v >= 130)).astype(np.uint8), cv2.MORPH_OPEN, k)
    return fill(o), fill(y)


def label_image(tris, K, W, H, ss=2):
    """Painter's algorithm, far to near. Labels: 1 pad, 2 clamp, 3 other visible parts."""
    polys, depth, lab = [], [], []
    for key, t in tris.items():
        l = 1 if key.endswith('_pad') else 2 if key.endswith('_clamp') else 3
        z = t[:, :, 2]
        ok = (z > 0.005).all(1)
        uv = t[ok][:, :, :2] / z[ok][:, :, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        polys.append(uv); depth.append(z[ok].mean(1)); lab.append(np.full(ok.sum(), l))
    polys, depth, lab = np.concatenate(polys), np.concatenate(depth), np.concatenate(lab)
    img = np.zeros((H * ss, W * ss), np.uint8)
    for i in np.argsort(-depth):
        cv2.fillConvexPoly(img, np.round(((polys[i] + 0.5) * ss - 0.5) * 16).astype(np.int32), int(lab[i]), cv2.LINE_8, 4)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


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
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--real-dir', type=Path, required=True)
    ap.add_argument('--media', type=Path, required=True)
    a = ap.parse_args()
    res, rows = {}, []
    for arm in ('left_arm', 'right_arm'):
        cam = json.loads((a.dir / f'{arm}_camera.json').read_text())
        W, H = cam['stretch_to_wh']
        K = np.array(cam['K_960x600'])
        rgba = cv2.imread(str(a.dir / f'{arm}_render_rgba.png'), cv2.IMREAD_UNCHANGED)
        rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_LINEAR)       # square-pixel render -> fx != fy
        real = cv2.imread(str(a.real_dir / f'{arm.split("_")[0]}_wrist_camera.jpg'))
        ro, ry = real_masks(real)
        L = label_image(dict(np.load(a.dir / f'{arm}_tris.npz')), K, W, H)
        mo, my = fill(L == 1), fill(L == 2)
        res[arm] = dict(camera_correction=cam.get('camera_correction'), pad_iou=iou(mo, ro), clamp_iou=iou(my, ry),
                        pad_precision=float((mo & ro).sum() / max(mo.sum(), 1)), pad_recall=float((mo & ro).sum() / max(ro.sum(), 1)),
                        real_pad_px=int(ro.sum()), model_pad_px=int(mo.sum()))
        al = rgba[:, :, 3:4].astype(np.float32) / 255
        on_grey = (rgba[:, :, :3] * al + 128 * (1 - al)).astype(np.uint8)
        over = (rgba[:, :, :3] * al * 0.55 + real * (1 - al * 0.55)).astype(np.uint8)
        over = outline(outline(over, ro, REAL_O), ry, REAL_Y)
        over = outline(outline(over, mo, ORANGE), my, YELLOW)
        s = 0.62
        p = [cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for x in (real, on_grey, over)]
        side = arm.split('_')[0]
        label(p[0], f'real: {side} wrist camera 09-29, arm at home')
        how = 'hand-eye + fitted correction' if cam.get('camera_correction') else 'raw hand-eye'
        label(p[1], f'asset at home, {how} camera (approx.), tags {cam["finger_tag_ids"]}', 0.45)
        label(p[2], f'IoU pad {res[arm]["pad_iou"]:.2f}, clamp {res[arm]["clamp_iou"]:.2f} (real red/magenta, asset orange/yellow)', 0.45)
        rows.append(np.hstack(p))
    (a.dir / 'metrics.json').write_text(json.dumps(dict(
        real_dir=str(a.real_dir), note='raiden hand-eye camera and HD1200 K/2 (approximate); correction as recorded per arm', arms=res), indent=1))
    a.media.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.media / '05_wrist_view_vs_real.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
