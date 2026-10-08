import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rigcam import *
m = metric(); K, T, _ = scene_camera(m)
for Y in np.arange(1.95, 2.13, 0.02):
    a = backproject_plane([(810,342)], K, T, (0,1,0), -Y)[0]; b = backproject_plane([(900,318)], K, T, (0,1,0), -Y)[0]
    # top: point on vertical line through a with pixel (955,192): solve z by projecting candidates
    zs = np.linspace(0.8, 1.6, 801); P = np.c_[np.full_like(zs, a[0]), np.full_like(zs, Y), zs]
    uv, _ = project(P, K, T); i = np.argmin(np.linalg.norm(uv - [955,192], axis=1))
    print(f'Y {Y:.2f} bottom-left {np.round(a,3)} bottom@900 z {b[2]:.3f} top z {zs[i]:.3f} (res {np.linalg.norm(uv[i]-[955,192]):.1f}px)')
# vertical post (bracket) through (830,390) bottom; check top (915,318)
for Y in [1.95, 1.97, 2.0]:
    a = backproject_plane([(830,390)], K, T, (0,0,1), -0.71)[0]
    zs = np.linspace(0.7, 1.3, 601); P = np.c_[np.full_like(zs, a[0]), np.full_like(zs, a[1]), zs]
    uv, _ = project(P, K, T); d = np.abs((uv[:,0]-830)*(318-390) - (uv[:,1]-390)*(915-830)) / np.hypot(85,72)
    print('bracket base', np.round(a,3), 'line residual px at z 1.0:', round(float(d[np.argmin(abs(zs-1.0))]),1))
uv,_ = project([[-1.44,1.975,0.71],[-1.44,1.975,1.0],[-3.066,1.872,0.62],[-3.066,1.872,0.99],[-3.066,1.52,0.99],[-3.89,2.155,0],[-3.89,2.155,1.0],[-3.85,2.155,0.0],[-3.85,2.12,0]],K,T); print(np.round(uv,1))
