"""Classical stereo on the rectified ZED pair (no neural depth): where is the wall's textured left edge?

Inputs: work/wallcheck/zed_stereo_pair.npz (zed_stereo_pair.py), work/zed_metric.npz (desk frame from the crease).
- rectification check: vertical offset of SIFT matches;
- desk-top plane from SGBM disparity on the wood texture (independent of NEURAL_LIGHT);
- sub-pixel edge disparity, row by row, for (a) the vertical glass/wall junction (left end of the painted wall),
  (b) controls: the desk's left and right side edges near the back corners (must lie on the desk plane, y ~ 0 there).
Points go to the crease desk frame (x along the crease, y toward the wall, z up; y = 0 is the desk back edge).
Writes work/stereo_edges.json.
"""
import json

import cv2
import numpy as np

import common

S = np.load(common.WORK / 'wallcheck/zed_stereo_pair.npz')
zn = np.load(common.WORK / 'zed_metric.npz')
K = S['K']
fB = float(K[0, 0] * S['baseline_m'])
gl = cv2.cvtColor(S['left'].astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)   # pyzed BGRA -> first 3 = BGR
gr = cv2.cvtColor(S['right'].astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
out = dict(fB_px_m=fB, baseline_m=float(S['baseline_m']), frames_averaged=int(S['n']))

# ---- rectification check
sift = cv2.SIFT_create(4000)
k1, d1 = sift.detectAndCompute(np.clip(gl, 0, 255).astype(np.uint8), None)
k2, d2 = sift.detectAndCompute(np.clip(gr, 0, 255).astype(np.uint8), None)
mm = cv2.BFMatcher().knnMatch(d1, d2, k=2)
good = [m for m, n in mm if m.distance < 0.7 * n.distance]
p1 = np.array([k1[m.queryIdx].pt for m in good]); p2 = np.array([k2[m.trainIdx].pt for m in good])
dy = p1[:, 1] - p2[:, 1]
okm = (np.abs(dy) < 3) & (p1[:, 0] - p2[:, 0] > 0)
out['rectification_dy_px'] = dict(matches=int(okm.sum()), median=float(np.median(dy[okm])), mad=float(np.median(np.abs(dy[okm] - np.median(dy[okm])))))

# ---- SGBM disparity -> desk plane (camera coords)
sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=7, P1=8 * 49, P2=32 * 49, uniquenessRatio=10,
                           speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_HH)
disp = sg.compute(np.clip(gl, 0, 255).astype(np.uint8), np.clip(gr, 0, 255).astype(np.uint8)).astype(np.float64) / 16
v, u = np.mgrid[0:600, 0:960]
Zs = np.where(disp > 1, fB / np.maximum(disp, 1e-6), np.nan)
Ps = np.dstack([(u - K[0, 2]) / K[0, 0] * Zs, (v - K[1, 2]) / K[1, 1] * Zs, Zs])


def side(u0, v0, u1, v1):
    return (u1 - u0) * (v - v0) - (v1 - v0) * (u - u0)


desk = (v > 375) & (v < 595) & (side(243, 365, 120, 600) < 0) & (side(775, 360, 920, 600) > 0)
desk &= ~(((u > 230) & (u < 400) & (v > 490)) | ((u > 630) & (u < 810) & (v > 490)) | ((u > 505) & (u < 560) & (v > 520)))
desk &= np.isfinite(Zs)
ns, ds, inl = common.fit_plane(Ps[desk], thr=0.006)
if ds < 0:
    ns, ds = -ns, -ds
nd, dd = zn['nd'], zn['dd']
out['sgbm_desk_plane'] = dict(pixels=int(desk.sum()), inlier_frac=float(inl.mean()),
                              camera_height_m=float(ds), neural_camera_height_m=float(dd),
                              normal_angle_vs_neural_deg=float(np.degrees(np.arccos(np.clip(ns @ nd, -1, 1)))),
                              rms_mm=float(np.sqrt(np.mean((Ps[desk][inl] @ ns + ds) ** 2)) * 1000))


# ---- desk frame from the SGBM plane and the crease/side-edge corners (same image lines as zed_metric)
def ray_plane(uv, n, d):
    r = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    return (-d / (n @ r)) * r


def frame(n, d):
    BL, BR = ray_plane(zn['bl'], n, d), ray_plane(zn['br'], n, d)
    ez = n
    ex = BR - BL; ex -= (ex @ ez) * ez; ex /= np.linalg.norm(ex)
    return BL, np.stack([ex, np.cross(ez, ex), ez]), float(np.linalg.norm(BR - BL))


frames = {'neural_plane': (zn['BL'], zn['Rdc']), 'sgbm_plane': frame(ns, ds)[:2]}
out['sgbm_desk_length_back_edge_m'] = frame(ns, ds)[2]


# ---- sub-pixel edges
def peak(seg):
    k = int(np.argmax(seg))
    if 1 <= k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        den = y0 - 2 * y1 + y2
        return k + (0.5 * (y0 - y2) / den if den != 0 else 0.0), seg[k]
    return None, None


gxl = cv2.Sobel(cv2.GaussianBlur(gl, (5, 5), 1.0), cv2.CV_64F, 1, 0, ksize=3)
gxr = cv2.Sobel(cv2.GaussianBlur(gr, (5, 5), 1.0), cv2.CV_64F, 1, 0, ksize=3)


def edge_track(rows, u_of_v, sign, half=14, dmin=20, dmax=60):
    pts = []
    for vv in rows:
        ue = u_of_v(vv)
        a = int(round(ue - half))
        if a < 2 or a + 2 * half > 957:
            continue
        ul, sl = peak(sign * gxl[vv, a:a + 2 * half])
        if ul is None or sl < 15:
            continue
        ul += a
        b0, b1 = int(ul - dmax), int(ul - dmin)
        if b0 < 2:
            continue
        ur, sr = peak(sign * gxr[vv, b0:b1])
        if ur is None or sr < 0.5 * sl:
            continue
        ur += b0
        pts.append((vv, ul, ul - ur, sl, sr))
    return np.array(pts)


def to_frames(T):
    Z = fB / T[:, 2]
    Pc = np.c_[(T[:, 1] - K[0, 2]) / K[0, 0] * Z, (T[:, 0] - K[1, 2]) / K[1, 1] * Z, Z]
    return {k: (Pc - BL) @ R.T for k, (BL, R) in frames.items()}, Z


res = {}
# (a) glass/wall junction: dark-blue/blue on the left, beige on the right -> intensity rises left->right (sign +)
jn = edge_track(range(40, 356, 2), lambda vv: 275 + (vv - 362) * (48 / 62), +1)
# (b) desk left edge: carpet (dark) left, desk (bright) right -> sign +; line from zed_metric edge fits
Ll = json.load(open(common.WORK / 'zed_metric.json'))['edge_lines_px']['0928']


def line_u(name):
    c, d = np.array(Ll[name]['point']), np.array(Ll[name]['dir'])
    return lambda vv: c[0] + (vv - c[1]) / d[1] * d[0]


le = edge_track(range(368, 470, 2), line_u('left'), +1)
# (c) desk right edge: desk (bright) left, dark gap/floor right -> sign -
re_ = edge_track(range(358, 470, 2), line_u('right'), -1)
for name, T in (('glass_wall_junction', jn), ('desk_left_edge', le), ('desk_right_edge', re_)):
    F, Z = to_frames(T)
    r = dict(rows=int(len(T)), disparity_px_median=float(np.median(T[:, 2])), depth_m_range=[float(Z.min()), float(Z.max())])
    for k, Pd in F.items():
        lo = Pd[:, 2] < 0.15 if name == 'glass_wall_junction' else np.ones(len(Pd), bool)
        r[k] = dict(x_m_median=float(np.median(Pd[:, 0])),
                    y_mm_median=float(np.median(Pd[:, 1]) * 1000), z_mm_median=float(np.median(Pd[:, 2]) * 1000),
                    y_mm_low_part=float(np.median(Pd[lo, 1]) * 1000) if lo.any() else None,
                    lean_deg=float(np.degrees(np.arctan(np.polyfit(Pd[:, 2], Pd[:, 1], 1)[0]))) if name == 'glass_wall_junction' else None,
                    by_height=[[round(float(z), 3), round(float(y) * 1000, 1)] for z, y in zip(Pd[::8, 2], Pd[::8, 1])] if name == 'glass_wall_junction' else None)
    # per-row disparity precision: residual about a line fit of disparity vs row
    c = np.polyfit(T[:, 0], T[:, 2], 2)
    r['disparity_fit_rms_px'] = float(np.sqrt(np.mean((np.polyval(c, T[:, 0]) - T[:, 2]) ** 2)))
    res[name] = r
out['edges'] = res
# expected disparity of a point on the wall plane (y=0 in the crease frame) at the junction rows, and at y=+85 mm
BL, R = frames['neural_plane']
exp = {}
for yoff in (0.0, 0.04, 0.085):
    ds_ = []
    for vv, ul, dm, *_ in jn:
        r_ = np.linalg.inv(K) @ np.array([ul, vv, 1.0])
        # intersect the ray with the plane y = yoff in the desk frame: R[1] . (t r - BL) = yoff
        t = (yoff + R[1] @ BL) / (R[1] @ r_)
        ds_.append(fB / (t * r_[2]) - dm)
    exp[f'y{int(yoff * 1000)}mm'] = dict(median_expected_minus_measured_px=float(np.median(ds_)))
out['junction_disparity_vs_plane_hypotheses'] = exp
common.save_json(common.WORK / 'stereo_edges.json', out)
print(json.dumps(out, indent=1, default=float))


# ---- classical-only test (no neural depth, no crease frame): desk-top disparity plane d = a u + b v + c from
# the desk side edges + SIFT matches on the desk texture (RANSAC), compared with the junction disparity where the
# junction meets the crease. Flush wall: equal at the crease; a gap g: junction disparity lower.
def side2(u0, v0, u1, v1, uu, vv):
    return (u1 - u0) * (vv - v0) - (v1 - v0) * (uu - u0)


dm = (p1[:, 1] > 372) & (p1[:, 1] < 596) & (side2(243, 365, 120, 600, p1[:, 0], p1[:, 1]) < 0) & (side2(775, 360, 920, 600, p1[:, 0], p1[:, 1]) > 0) & okm
dm &= ~(((p1[:, 0] > 230) & (p1[:, 0] < 400) & (p1[:, 1] > 470)) | ((p1[:, 0] > 630) & (p1[:, 0] < 810) & (p1[:, 1] > 470)))
pts = np.r_[np.c_[p1[dm], p1[dm, 0] - p2[dm, 0]], np.c_[le[:, 1], le[:, 0], le[:, 2]], np.c_[re_[:, 1], re_[:, 0], re_[:, 2]]]
src = np.r_[np.zeros(dm.sum()), np.ones(len(le)), 2 * np.ones(len(re_))]
rng = np.random.default_rng(0)
best = None
A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
for _ in range(2000):
    i = rng.choice(len(pts), 3, replace=False)
    try:
        c = np.linalg.solve(A[i], pts[i, 2])
    except np.linalg.LinAlgError:
        continue
    inl = np.abs(A @ c - pts[:, 2]) < 0.35
    if best is None or inl.sum() > best.sum():
        best = inl
c, *_ = np.linalg.lstsq(A[best], pts[best, 2], rcond=None)
rres = A[best] @ c - pts[best, 2]
jb = jn[jn[:, 0] >= 340]  # junction rows just above the crease (v_crease ~362 at u ~275)
d_plane_at_j = c[0] * jb[:, 1] + c[1] * jb[:, 0] + c[2]
# crease pixel under the junction: where the junction line meets the crease
jl = np.polyfit(jn[:, 0], jn[:, 1], 1)
vc = 359.9975 + (np.polyval(jl, 360.0) - 518.5) / 0.99978 * (-0.02101)
uc = float(np.polyval(jl, vc))
d_plane_crease = float(c[0] * uc + c[1] * vc + c[2])
dj = np.polyfit(jn[:, 0], jn[:, 2], 2)
d_junction_crease = float(np.polyval(dj, vc))
# convert the disparity difference at the crease pixel to a Y offset along the ray (desk frame, neural plane only for direction)
r_ = np.linalg.inv(K) @ np.array([uc, vc, 1.0])
BLn, Rn = frames['neural_plane']
Zc_plane, Zc_j = fB / d_plane_crease, fB / d_junction_crease
dy_mm = float((Rn[1] @ (r_ * Zc_j) - Rn[1] @ (r_ * Zc_plane)) * 1000)
out['classical_only_crease_test'] = dict(
    note='desk-top disparity plane (affine in u,v) from classical matches only; compared with the junction edge disparity '
         'extrapolated to the crease pixel under it. y offset = distance along the ray between the two, projected on the desk-frame y.',
    plane_points=dict(sift_desk=int(dm.sum()), left_edge=int(len(le)), right_edge=int(len(re_)), inliers=int(best.sum()),
                      inliers_by_source=[int((best & (src == k)).sum()) for k in range(3)], rms_px=float(np.sqrt(np.mean(rres ** 2)))),
    crease_pixel_under_junction=[uc, float(vc)], d_desk_plane_px=d_plane_crease, d_junction_px=d_junction_crease,
    d_diff_px=d_plane_crease - d_junction_crease, junction_behind_desk_back_edge_mm=dy_mm,
    junction_rows_340_356_d_plane_minus_d_junction_px=float(np.median(d_plane_at_j - jb[:, 2])))
common.save_json(common.WORK / 'stereo_edges.json', out)
print(json.dumps(out['classical_only_crease_test'], indent=1, default=float))
