import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2, camera_model as cm
from rigcam import metric
from wrist_refine import segments
rd = Path(sys.argv[1]); cfg = json.load(open(sys.argv[2])); m = metric()
rj = json.load(open(rd / 'render.json'))
rows = []
for view in ('left_wrist', 'right_wrist'):
    K = np.array(cfg['camera_model'][view]['K']); T0 = np.array(rj['views'][view]['matrix_world']) @ np.diag([1, -1, -1, 1])
    img = cm.real_u8(view).copy()
    for i in range(0, 960, 50): cv2.line(img, (i, 0), (i, 599), (255, 255, 255), 1) if i % 100 == 0 else None
    for j in range(0, 600, 100): cv2.line(img, (0, j), (959, j), (255, 255, 255), 1)
    cols = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,255,255),(255,255,0),(255,128,0),(128,0,255)]
    for c, (a, b) in zip(cols, segments(m, cfg)):
        P = np.linspace(a, b, 300); pc = (P - T0[:3, 3]) @ T0[:3, :3]; ok = pc[:, 2] > 0.05
        uv = pc[ok, :2] / pc[ok, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        cv2.polylines(img, [np.round(uv).astype(np.int32)], False, c, 2)
    rows.append(img)
cv2.imwrite(sys.argv[3], np.concatenate(rows, 0)[:, :, ::-1])
