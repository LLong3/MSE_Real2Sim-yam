import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2, camera_model as cm
d = Path(sys.argv[1]); view = sys.argv[2] if len(sys.argv) > 2 else 'scene'
reg, k = cm.region_masks(d, view)
ex = cm.exclusion(view, reg)
rend = cm.to_u8(cm.render_lin(d, view)); real = cm.real_u8(view)
cols = {'desk': (255,0,0), 'wall': (0,255,0), 'glass_frosted': (0,0,255), 'glass_lower': (255,0,255), 'carpet': (255,255,0), 'glass_frame': (0,255,255), 'panel': (255,128,0), 'monitor_desk': (128,0,255)}
def outl(img):
    img = img.copy()
    for n, c in cols.items():
        cs, _ = cv2.findContours((reg[n] & ~ex).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, c, 1)
    return img
out = np.concatenate([outl(real), outl(rend)], 1)
cv2.imwrite(sys.argv[3] if len(sys.argv) > 3 else str(d / f'{view}_regions.jpg'), out[:, :, ::-1])
