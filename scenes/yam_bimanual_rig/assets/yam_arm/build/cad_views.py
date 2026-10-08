"""Flat-shaded CPU views of raiden's gripper/camera-mount STLs (for the finger/wrist-camera assessment).

    .runtime/pi3x-inference/venv/bin/python build/cad_views.py --out evidence/cad_parts
Painter's-algorithm triangle fill with Lambert shading; three orthographic views + one isometric per part,
with the bounding box in mm. No Blender, no GPU.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path('/home/frank/robot/raiden/docs/assets')
PARTS = ['finray_short', 'finray_long', 'finray_adapter', 'zed_mounter_left', 'zed_mounter_right']


def read_stl(path):
    raw = path.read_bytes()
    n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    rec = np.frombuffer(raw, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
    return rec['v'].astype(np.float64)


def view(tris, R, size=300):
    v = tris.reshape(-1, 3) - tris.reshape(-1, 3).mean(0)
    v = (v @ R.T).reshape(-1, 3, 3)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = np.clip(np.abs(n @ np.array([0.3, -0.4, 0.87])), 0, 1) * 0.75 + 0.2
    ext = np.abs(v[:, :, :2]).max() * 1.08
    uv = ((v[:, :, :2] / ext) * [1, -1] * size / 2 + size / 2) * 16
    order = np.argsort(v[:, :, 2].mean(1))  # far (-z) first; camera looks along -z
    img = np.full((size, size, 3), 235, np.uint8)
    for i in order:
        c = int(shade[i] * 255)
        cv2.fillConvexPoly(img, np.round(uv[i]).astype(np.int32), (c, c, c), cv2.LINE_AA, 4)
    return img


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    views = {'top (-Z)': np.eye(3), 'front (-Y)': rot('x', -90), 'side (+X)': rot('y', -90) @ rot('x', -90),
             'iso': rot('x', -55) @ rot('z', -35)}
    rows = []
    for p in PARTS:
        t = read_stl(ASSETS / f'{p}.STL')
        size = t.reshape(-1, 3).max(0) - t.reshape(-1, 3).min(0)
        tiles = []
        for name, R in views.items():
            im = view(t, R)
            cv2.putText(im, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
            tiles.append(im)
        head = np.full((300, 230, 3), 255, np.uint8)
        for k, line in enumerate([p, f'{len(t)} tris', 'bbox mm:', f'{size[0]:.1f} x {size[1]:.1f}', f'x {size[2]:.1f}']):
            cv2.putText(head, line, (8, 40 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack([head] + tiles))
    sheet = np.vstack(rows)
    cv2.imwrite(str(a.out / 'raiden_cad_parts.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.out / 'raiden_cad_parts.jpg', sheet.shape)


if __name__ == '__main__':
    main()
