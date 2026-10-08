"""Point-splat renderer of the aligned Pi3X reference (z-buffer), and ZED plane-based initial pose."""
import numpy as np

import common


def reference_cloud(step=1, min_conf=None):
    V, MV, C = common.points()
    sel = MV.copy()
    if step > 1:
        grid = np.zeros_like(sel)
        grid[:, ::step, ::step] = True
        sel &= grid
    P = np.asarray(V)[sel].astype(np.float64)
    col = np.asarray(C)[sel]
    slot = np.nonzero(sel)[0]
    return P, col, slot


def render(P, col, R, t, K, size=(960, 600), radius=1):
    """R,t: world->camera (OpenCV). Returns rgb (H,W,3), depth (H,W) nan-filled, index map (H,W) into P (-1)."""
    Wd, Ht = size
    Xc = P @ R.T + t
    z = Xc[:, 2]
    ok = z > 0.05
    u = K[0, 0] * Xc[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * Xc[ok, 1] / z[ok] + K[1, 2]
    idx = np.nonzero(ok)[0]
    zz = z[ok]
    order = np.argsort(-zz)  # far first; later writes (nearer) win
    u, v, zz, idx = u[order], v[order], zz[order], idx[order]
    ui0, vi0 = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    depth = np.full(Ht * Wd, np.inf)
    owner = np.full(Ht * Wd, -1, np.int64)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            ui, vi = ui0 + du, vi0 + dv
            m = (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < Ht)
            lin = vi[m] * Wd + ui[m]
            depth_tmp = np.full(Ht * Wd, np.inf)
            own_tmp = np.full(Ht * Wd, -1, np.int64)
            depth_tmp[lin] = zz[m]
            own_tmp[lin] = idx[m]
            better = depth_tmp < depth
            depth[better] = depth_tmp[better]
            owner[better] = own_tmp[better]
    depth = depth.reshape(Ht, Wd)
    owner = owner.reshape(Ht, Wd)
    rgb = np.zeros((Ht, Wd, 3), np.uint8)
    rgb[owner >= 0] = col[owner[owner >= 0]]
    depth[~np.isfinite(depth)] = np.nan
    return rgb, depth, owner
