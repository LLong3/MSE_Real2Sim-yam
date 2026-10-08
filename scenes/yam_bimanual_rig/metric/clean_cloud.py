"""Multi-view consistent subset of the aligned Pi3X point maps (read-only on the reference).

A pixel point of view i is kept when at least MIN_SUPPORT other views see a point within
TOL of the same depth along their rays and few views see through it (free-space check).
Output: work/clean_cloud.npz (points, colors, slot, pixel yx, support count).
"""
import numpy as np

import common

STEP = 2          # every 2nd pixel in u and v
TOL_REL = 0.02    # depth agreement tolerance, relative (2 %)
TOL_ABS = 0.015   # plus 1.5 cm
MIN_SUPPORT = 3
MAX_VIOLATION_FRAC = 0.25  # other views seeing through the point (free-space violation)

V, MV, C = common.points()
c2w, K, fidx = common.cameras()
w2c = np.linalg.inv(c2w)
Hh, Ww = common.H, common.W
# per-view depth maps (camera z) for the consistency test
Z = np.full((64, Hh, Ww), np.nan, np.float32)
for s in range(64):
    Pc = np.asarray(V[s]).reshape(-1, 3) @ w2c[s, :3, :3].T + w2c[s, :3, 3]
    z = Pc[:, 2].reshape(Hh, Ww)
    Z[s] = np.where(MV[s], z, np.nan)

sel = MV.copy()
grid = np.zeros_like(sel)
grid[:, ::STEP, ::STEP] = True
sel &= grid
slot, vy, ux = np.nonzero(sel)
P = np.asarray(V)[sel].astype(np.float64)
support = np.zeros(len(P), np.int16)
violate = np.zeros(len(P), np.int16)
for s in range(64):
    Pc = P @ w2c[s, :3, :3].T + w2c[s, :3, 3]
    z = Pc[:, 2]
    ok = (z > 0.05) & (slot != s)
    u = K[s, 0, 0] * Pc[:, 0] / np.where(ok, z, 1) + K[s, 0, 2]
    v = K[s, 1, 1] * Pc[:, 1] / np.where(ok, z, 1) + K[s, 1, 2]
    ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
    ok &= (ui >= 0) & (ui < Ww) & (vi >= 0) & (vi < Hh)
    zs = np.full(len(P), np.nan)
    zs[ok] = Z[s, vi[ok], ui[ok]]
    agree = ok & np.isfinite(zs) & (np.abs(zs - z) < TOL_REL * z + TOL_ABS)
    support += agree
    violate += ok & np.isfinite(zs) & (zs > z * (1 + TOL_REL) + TOL_ABS)
    print('view', s, 'visible', ok.sum(), 'agree', agree.sum(), flush=True)
keep = (support >= MIN_SUPPORT) & (violate <= MAX_VIOLATION_FRAC * (support + violate))
col = np.asarray(C)[sel]
np.savez(common.WORK / 'clean_cloud.npz', points=P[keep], colors=col[keep], slot=slot[keep],
         yx=np.stack([vy, ux], 1)[keep], support=support[keep], violate=violate[keep])
print('kept', keep.sum(), 'of', len(P))
