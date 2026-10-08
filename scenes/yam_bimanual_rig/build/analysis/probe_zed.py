import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from rigcam import *
m = metric(); K, T, (W, H) = scene_camera(m)
img = cv2.imread(str(ZED_IMAGE))
x0, x1 = m['desk']['box_x_m']; y0, y1 = m['desk']['box_y_m']; zt = 0.72
wall_y = y1 + 0.035
def line(a, b, col, n=50):
    P = np.linspace(a, b, n); uv, z = project(P, K, T)
    uv = uv[z > 0]
    cv2.polylines(img, [np.round(uv).astype(np.int32)], False, col, 1, cv2.LINE_AA)
# desk top
for a, b in [((x0,y0,zt),(x1,y0,zt)),((x1,y0,zt),(x1,y1,zt)),((x1,y1,zt),(x0,y1,zt)),((x0,y1,zt),(x0,y0,zt))]:
    line(a, b, (0,255,0))
# wall base at desk height and floor
line((-4, wall_y, zt), (-1, wall_y, zt), (0,0,255))
line((-4, wall_y, 0), (-1, wall_y, 0), (0,0,255))
we = m['planes']['back_wall']['wall_end_x_m']
line((we, wall_y, 0), (we, wall_y, 2.5), (255,0,255))
# floor grid
for x in np.arange(-4.0, -0.9, 0.25):
    line((x, 0.5, 0), (x, 2.5, 0), (255,255,0))
for y in np.arange(0.5, 2.6, 0.25):
    line((-4.0, y, 0), (-1.0, y, 0), (255,255,0))
cv2.imwrite(sys.argv[1], img)
for name, uv in [('wall_end_top', (0,30)), ('wall_end_bot', (270,360))]:
    pass
# back-project some pixels onto floor
for uv in [(160,445),(0,330),(40,325),(240,328),(100,600),(250,600)]:
    p = backproject_plane(uv, K, T, (0,0,1), 0)[0]
    print('floor', uv, np.round(p,3))
for uv in [(0,30),(135,195),(270,360)]:
    p = backproject_plane(uv, K, T, (1,0,0), -we)[0]
    print('wall-end plane x=we', uv, np.round(p,3))
