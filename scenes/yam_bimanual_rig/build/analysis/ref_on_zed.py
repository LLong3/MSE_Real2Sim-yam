import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from rigcam import *
m = metric(); K, T, (W, H) = scene_camera(m)
d = np.load(Path(__file__).with_name('plan_slices.npz')); P = d['P'].astype(float); C = d['C'].astype(float)
uv, z = project(P, K, T)
ok = (z > 0.05) & (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
uv, z, C, P = uv[ok], z[ok], C[ok], P[ok]
order = np.argsort(-z)
img = np.zeros((H, W, 3), np.float32); zb = np.full((H, W), np.inf); xyz = np.zeros((H, W, 3), np.float32)
r = 2
for i in order:
    u, v = int(uv[i,0]), int(uv[i,1])
    img[max(0,v-r):v+r+1, max(0,u-r):u+r+1] = C[i]
    xyz[max(0,v-r):v+r+1, max(0,u-r):u+r+1] = P[i]
real = cv2.imread(str(ZED_IMAGE)).astype(np.float32)/255.
out = np.concatenate([real, img[:,:,::-1]], 1)
cv2.imwrite(sys.argv[1], (out*255).astype(np.uint8))
np.save(Path(sys.argv[1]).with_suffix('.xyz.npy'), xyz)
