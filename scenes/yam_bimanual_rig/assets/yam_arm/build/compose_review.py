"""Side-by-side review sheet: real frame | asset render (same fitted camera) | overlay, plus gripper zooms.

    .runtime/pi3x-inference/venv/bin/python build/compose_review.py --real F.jpg --render R.png \
        --camera evidence/closeup/camera_f1140.json --out OUT.jpg
"""
import argparse
import json

import cv2
import numpy as np


def label(im, text, scale=0.8):
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(im, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--render', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--crop', type=int, nargs=4, default=(280, 540, 1260, 1080))
    ap.add_argument('--grip-crop', type=int, nargs=4, default=(930, 560, 1250, 880))
    a = ap.parse_args()
    cam = json.loads(open(a.camera).read())
    real = cv2.imread(a.real)
    rgba = cv2.imread(a.render, cv2.IMREAD_UNCHANGED)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
    grey = np.full_like(real, 128)
    on_grey = (rgba[:, :, :3] * alpha + grey * (1 - alpha)).astype(np.uint8)
    overlay = (rgba[:, :, :3] * alpha * 0.55 + real * (1 - alpha * 0.55)).astype(np.uint8)
    edges = cv2.Canny((alpha[:, :, 0] * 255).astype(np.uint8), 50, 150)
    overlay[cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0] = (0, 255, 255)
    x0, y0, x1, y1 = a.crop
    s = 0.9
    panels = [cv2.resize(p[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for p in (real, on_grey, overlay)]
    label(panels[0], 'real: IMG_5411 frame 1140 (38.0 s), arm at home')
    label(panels[1], 'asset at home (q=0, gripper open), fitted camera')
    label(panels[2], f'overlay (render 55%, outline); silhouette IoU {cam["iou_full_res"]:.2f}')
    top = np.hstack(panels)
    gx0, gy0, gx1, gy1 = a.grip_crop
    g = [cv2.resize(p[gy0:gy1, gx0:gx1], None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC) for p in (real, on_grey, overlay)]
    label(g[0], 'real gripper: custom fin-ray fingers, ZED X One mount', 0.6)
    label(g[1], 'MJCF linear_4310: stock fingers, no camera', 0.6)
    label(g[2], 'overlay', 0.6)
    bottom = np.hstack(g)
    pad = np.full((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), 40, np.uint8)
    sheet = np.vstack([top, np.hstack([bottom, pad]) if pad.shape[1] > 0 else bottom[:, :top.shape[1]]])
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(a.out, sheet.shape)


if __name__ == '__main__':
    main()
