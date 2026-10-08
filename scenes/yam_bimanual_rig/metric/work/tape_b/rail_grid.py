"""Full-res crops around the projected rail ends with an absolute pixel grid (every 10 px), for picking the rail-end pixels."""
import sys
import cv2
import numpy as np
import common
sys.path.insert(0, str(common.WORK / 'tape_b'))
from rail_overlay import fr, proj, y_front, ztop

for f in map(int, sys.argv[1:]):
    img = cv2.imread(str(common.WORK / f'frames_full/{f:06d}.jpg'))
    for side, xe in (('left', -2.75), ('right', -1.80)):
        (u, v), = np.rint(proj(fr[f], [xe, y_front, ztop])[0]).astype(int)
        u0, v0 = max(u - 60, 0), max(v - 45, 0)
        c = img[v0:v0 + 90, u0:u0 + 120]
        c = cv2.resize(c, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        for x in range(0, 120, 10):
            cv2.line(c, (x * 5, 0), (x * 5, c.shape[0]), (0, 0, 255), 1)
            cv2.putText(c, str(u0 + x), (x * 5 + 2, 12), 0, 0.4, (0, 0, 255), 1)
        for y in range(0, 90, 10):
            cv2.line(c, (0, y * 5), (c.shape[1], y * 5), (255, 0, 0), 1)
            cv2.putText(c, str(v0 + y), (2, y * 5 - 2), 0, 0.4, (255, 0, 0), 1)
        cv2.imwrite(str(common.WORK / f'tape_b/grid_{f}_{side}.jpg'), c)
