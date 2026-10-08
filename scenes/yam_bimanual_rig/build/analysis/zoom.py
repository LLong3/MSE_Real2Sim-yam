import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2, camera_model as cm
d = Path(sys.argv[1]); view = sys.argv[2]; x0, y0, x1, y1 = map(int, sys.argv[3].split(',')); f = int(sys.argv[4]); out = sys.argv[5]
real = cm.real_u8(view)
rend = cv2.imread(str(d / f'{view}.png'))[:, :, ::-1]
if (d / f'{view}_cam.png').exists(): rend = cv2.imread(str(d / f'{view}_cam.png'))[:, :, ::-1]
a = cv2.resize(real[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
b = cv2.resize(rend[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
e1 = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), (3,3), 0), 30, 80)[y0:y1, x0:x1]
e2 = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rend, cv2.COLOR_RGB2GRAY), (3,3), 0), 30, 80)[y0:y1, x0:x1]
ov = (0.5 * a).astype(np.uint8)
E1 = cv2.resize(e1, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) > 0; E2 = cv2.resize(e2, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) > 0
ov[E1] = (255, 60, 60); ov[E2] = (60, 255, 60); ov[E1 & E2] = (255, 255, 0)
cv2.imwrite(out, np.concatenate([a, b, ov], 1)[:, :, ::-1])
