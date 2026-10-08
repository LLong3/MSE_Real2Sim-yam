"""Quick side-by-side of real and rendered rig-camera views (PNG from Blender, no camera model)."""
import sys, cv2, numpy as np
from pathlib import Path
d = Path(sys.argv[1]); out = Path(sys.argv[2])
real = {'scene': '/home/frank/robot/bimanual_bringup/pi05/scene_camera.jpg',
        'left_wrist': '/home/frank/robot/bimanual_bringup/pi05/left_wrist_camera.jpg',
        'right_wrist': '/home/frank/robot/bimanual_bringup/pi05/right_wrist_camera.jpg'}
rows = []
for v in sys.argv[3].split(','):
    a = cv2.imread(real[v]); b = cv2.imread(str(d / (v + '.png')))
    if len(sys.argv) > 4 and (d / (v + '_cam.png')).exists(): b = cv2.imread(str(d / (v + '_cam.png')))
    rows.append(np.concatenate([a, b, cv2.addWeighted(a, 0.5, b, 0.5, 0)], 1))
cv2.imwrite(str(out), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
