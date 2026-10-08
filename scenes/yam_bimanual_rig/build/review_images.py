"""Compose the user review images (JPEG, modest size) from a delivery render + compare outputs."""
import json, sys
from pathlib import Path
import cv2, numpy as np
import camera_model as cm

run = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
tag = sys.argv[3] if len(sys.argv) > 3 else 'delivery_v1'; pre = sys.argv[4] if len(sys.argv) > 4 else ''
ver = tag.replace('delivery_', '')
d = run / f'renders/{tag}'; ov = run / f'renders/{tag}_overview'; pl = ov if (ov / 'plan.png').exists() else run / f'renders/{tag}_plan'
Q = [cv2.IMWRITE_JPEG_QUALITY, 85]
def lab(img, text, pos=(8, 22), s=0.6):
    img = img.copy(); cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA); return img
def rd(p): return cv2.imread(str(p))
# 01 scene camera: real | render
real = rd(cm.REAL['scene']); ren = rd(d / 'scene_cam.png')
cv2.imwrite(str(out / (pre + '01_scene_camera_real_vs_render.jpg')), np.concatenate([lab(real, 'real ZED scene camera 09-29'), lab(ren, f'render {ver} (metric K + pose, camera model)')], 1), Q)
# 02 overlay + difference
c = d / 'compare_scene'
cv2.imwrite(str(out / (pre + '02_scene_camera_overlay_and_difference.jpg')), np.concatenate([lab(rd(c / 'overlay.jpg'), '50/50 overlay'), lab(rd(c / 'difference.jpg'), '|real-render| x4 (sRGB)')], 1), Q)
# 03 edges
cv2.imwrite(str(out / (pre + '03_scene_camera_edges.jpg')), lab(rd(c / 'edges.jpg'), 'Canny edges: red real, green render, yellow both'), Q)
# 04 zooms: glass corner and monitor-desk corner (2x)
def zoom(img, x0, y0, x1, y1, f=2): return cv2.resize(img[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
zl = np.concatenate([lab(zoom(real, 0, 280, 300, 480), 'real'), lab(zoom(ren, 0, 280, 300, 480), 'render')], 1)
zr = np.concatenate([lab(zoom(real, 760, 170, 960, 600, 1), 'real'), lab(zoom(ren, 760, 170, 960, 600, 1), 'render')], 1)
zr = cv2.resize(zr, (int(zr.shape[1] * zl.shape[0] / zr.shape[0]), zl.shape[0]))
cv2.imwrite(str(out / (pre + '04_scene_camera_zoom_glass_and_monitor_corner.jpg')), np.concatenate([zl, zr], 1), Q)
# 05 wrist cameras
rows = []
for v in ('left_wrist', 'right_wrist'):
    rows.append(np.concatenate([lab(rd(cm.REAL[v]), f'real {v} 09-29'), lab(rd(d / f'{v}_cam.png'), f'render {ver} {v} (approx. pose)')], 1))
cv2.imwrite(str(out / (pre + '05_wrist_cameras_real_vs_render.jpg')), cv2.resize(np.concatenate(rows, 0), None, fx=0.75, fy=0.75), Q)
# 06 overview renders
ims = [lab(cv2.resize(rd(ov / f'{v}.png'), (800, 500)), v) for v in ('overview_front', 'overview_left', 'overview_right')]
ims.append(lab(cv2.resize(rd(pl / 'plan.png'), (800, 500)), 'plan (orthographic)'))
cv2.imwrite(str(out / (pre + '06_overview_views.jpg')), np.concatenate([np.concatenate(ims[:2], 1), np.concatenate(ims[2:], 1)], 0), Q)
print('\n'.join(sorted(str(p) for p in out.glob('*.jpg'))))
# 07 (v2 only): grippers, v1 placeholder vs v2 final claws, scene camera zoom and wrist crops
v1 = run / 'renders/delivery_v1'
if tag != 'delivery_v1' and (v1 / 'scene_cam.png').exists():
    r1 = rd(v1 / 'scene_cam.png'); r2 = ren
    zs = [lab(zoom(im, 220, 490, 820, 600, 2), t) for im, t in ((real, 'real'), (r1, 'v1 placeholder fingers'), (r2, f'{ver} UMI-FT claws'))]
    top = np.concatenate(zs, 0)
    wr = []
    for v in ('left_wrist', 'right_wrist'):
        ims = [cv2.resize(x, (400, 250)) for x in (rd(cm.REAL[v]), rd(v1 / f'{v}_cam.png'), rd(d / f'{v}_cam.png'))]
        wr.append(np.concatenate([lab(ims[0], f'real {v}', s=0.45), lab(ims[1], 'v1', s=0.45), lab(ims[2], ver, s=0.45)], 1))
    bot = np.concatenate(wr, 0)
    bot = cv2.resize(bot, (top.shape[1], int(bot.shape[0] * top.shape[1] / bot.shape[1])))
    cv2.imwrite(str(out / (pre + '07_grippers_v1_vs_v2.jpg')), np.concatenate([top, bot], 0), Q)
print('\n'.join(sorted(str(p) for p in out.glob(pre + '*.jpg'))))
