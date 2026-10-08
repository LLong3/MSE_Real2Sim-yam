"""Back-project real-image edge points found near projected model edges onto the edge's plane."""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from rigcam import *
m = metric(); K, T, _ = scene_camera(m)
g = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(str(ZED_IMAGE)), cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float64)
gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
def edge_pts(a, b, search=14, n=60):
    uv, z = project(np.linspace(a, b, n), K, T); ok = (uv[:,0]>search)&(uv[:,0]<960-search)&(uv[:,1]>search)&(uv[:,1]<600-search); uv = uv[ok]
    t = uv[-1]-uv[0]; t /= np.linalg.norm(t); nr = np.array([-t[1], t[0]]); G = np.abs(gx*nr[0]+gy*nr[1]).astype(np.float32)
    out = []
    for p in uv:
        s = np.arange(-search, search+1); q = p[None] + s[:,None]*nr[None]
        v = cv2.remap(G, q[:,0].astype(np.float32)[None], q[:,1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        if v.max() > 20: out.append(p + s[np.argmax(v)]*nr)
    return np.array(out)
cases = json.loads(sys.argv[1])
for name, c in cases.items():
    P = edge_pts(c['a'], c['b'])
    X = backproject_plane(P, K, T, c['n'], c['d'])
    print(name, len(X), 'median', np.round(np.median(X, 0), 4), 'p10', np.round(np.percentile(X, 10, 0), 4), 'p90', np.round(np.percentile(X, 90, 0), 4))
