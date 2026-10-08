"""Register the YAM MJCF (q=0) to each reconstructed arm in the aligned reference frame.

Model: yam_mesh.py export (base frame, Z up, arm along +X at q=0). Pose: base on the rail top,
yaw about Z only: world = S * Rz(yaw) * p + (x, y, z). Robust (soft-L1) point-to-surface fit of the
reference arm points (multi-view consistent cloud) to a dense MJCF surface sample.
Fits with scale fixed to 1 and, for diagnosis, with a free isotropic scale and free per-axis scale.
Output: work/arm_icp.json.
"""
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import common

rng = np.random.default_rng(0)


def sample_surface(npz, n=300000):
    d = np.load(npz)
    v, f = d['vertices'], d['faces']
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]


def voxel(P, s):
    k = np.floor(P / s).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return P[np.sort(idx)]


def rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def to_base(P, x):
    """world -> base-frame (inverse of world = diag(S) Rz p + T), params x = [tx,ty,tz,yaw,sx,sy,sz]."""
    T, th, S = x[:3], x[3], x[4:7]
    q = (P - T) @ rz(th)  # = Rz^T (P - T)... row form
    return q / S


def fit(P, tree, x0, free, f_scale=0.01):
    x0 = np.asarray(x0, float)

    def res(p):
        x = x0.copy(); x[free] = p
        d, _ = tree.query(to_base(P, x))
        return d * np.mean(x[4:7])  # distances back in world units
    r = least_squares(res, x0[free], loss='soft_l1', f_scale=f_scale, x_scale=0.01)
    x = x0.copy(); x[free] = r.x
    d = res(r.x)
    return x, d


def main(arms=('left_arm', 'right_arm'), out_name='arm_icp.json'):
    d = np.load(common.WORK / 'clean_cloud.npz')
    P, col = d['points'], d['colors']
    out = {}
    models = {o: sample_surface(common.WORK / f'yam_q0_open{o}.npz') for o in ('0', '0.0475')}
    trees = {o: cKDTree(m) for o, m in models.items()}
    for name, xc in (('left_arm', -2.56), ('right_arm', -1.97)):
        if name not in arms:
            continue
        box = (np.abs(P[:, 0] - xc) < 0.17) & (P[:, 1] > 1.0) & (P[:, 1] < 1.75) & (P[:, 2] > 0.675) & (P[:, 2] < 1.05)
        A = voxel(P[box], 0.006)
        res = dict(n_points_box=int(box.sum()), n_points_voxel6mm=int(len(A)))
        for o in ('0', '0.0475'):
            # coarse yaw/translation search, then refine
            best = None
            for th in np.radians((90.0,)):
                for dy in (0.0,):
                    x0 = [xc, 1.39 + dy, 0.67, th, 1, 1, 1]
                    x, dd = fit(A, trees[o], x0, [0, 1, 2, 3])
                    cost = np.mean(np.minimum(dd, 0.03) ** 2)
                    if best is None or cost < best[0]:
                        best = (cost, x, dd)
            _, x1, d1 = best
            # iterate with inlier trimming (drop cables/rail/mount plate beyond 2.5 cm)
            inl = d1 < 0.025
            x1, d1i = fit(A[inl], trees[o], x1, [0, 1, 2, 3])
            d1 = trees[o].query(to_base(A, x1))[0]
            # isotropic scale: tie the three scales
            def iso_fit(x0):
                x0 = np.asarray(x0, float)
                def r_(p):
                    x = np.r_[p[:4], p[4], p[4], p[4]]
                    return trees[o].query(to_base(A[inl], x))[0] * p[4]
                r = least_squares(r_, np.r_[x0[:4], 1.0], loss='soft_l1', f_scale=0.01, x_scale=0.01)
                return np.r_[r.x[:4], r.x[4], r.x[4], r.x[4]], r_(r.x)
            xi, di = iso_fit(x1)
            xa, da = fit(A[inl], trees[o], np.r_[x1[:4], 1.0, 1.0, 1.0], [0, 1, 2, 3, 4, 5, 6])
            res[f'opening_{o}'] = dict(
                scale1=dict(x=x1[:3], yaw_deg=np.degrees(x1[3]), inlier_frac_2p5cm=float(inl.mean()),
                            rms_inliers_m=float(np.sqrt(np.mean(d1i ** 2))), median_all_m=float(np.median(d1)),
                            p90_all_m=float(np.percentile(d1, 90))),
                iso_scale=dict(x=xi[:3], yaw_deg=np.degrees(xi[3]), scale=xi[4], rms_inliers_m=float(np.sqrt(np.mean(di ** 2)))),
                axis_scale=dict(x=xa[:3], yaw_deg=np.degrees(xa[3]), scale_base_xyz=xa[4:7],
                                note='scales along the arm base axes: x = arm length (world ~Y), y = width (world ~X), z = height',
                                rms_inliers_m=float(np.sqrt(np.mean(da ** 2)))))
            print(name, o, json.dumps(res[f'opening_{o}'], default=lambda a: np.round(a, 4).tolist()), flush=True)
        out[name] = res
    common.save_json(common.WORK / out_name, out)


if __name__ == '__main__':
    import sys
    arms = tuple(sys.argv[1:]) or ('left_arm', 'right_arm')
    main(arms, 'arm_icp.json' if len(arms) == 2 else f'arm_icp_{arms[0]}.json')
