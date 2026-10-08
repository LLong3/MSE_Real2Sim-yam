"""Metric scene-camera geometry from the ZED X itself (converted episode 0000, 09-28).

- Depth: temporal median of the episode's scene-camera depth (uint16 mm, NEURAL_LIGHT, left eye).
- Desk-top plane from depth (RANSAC + LSQ); upper back-wall plane (0.3-0.78 m above the desk).
- Desk edges from image gradients (left edge, right edge, back crease), sub-pixel, line fits;
  back corners = line intersections, lifted to 3D on the desk plane (ray-plane).
- Desk frame D: origin = back-left desk-top corner, x along the back edge (left->right),
  z = desk normal (up), y = z cross x (toward the wall).
Writes work/zed_metric.json and work/zed_metric.npz.
"""
import glob

import cv2
import numpy as np

import common

img, _, K = common.zed_frame(0)
Ki = np.linalg.inv(K)
fs = sorted(glob.glob(str(common.ZED_EP / 'depth/scene_camera/*.npz')))
D = np.stack([np.load(f)['depth'] for f in fs]).astype(np.float64) / 1000.0
D[D == 0] = np.nan
med = np.nanmedian(D, 0)
sd = np.nanstd(D, 0)
np.save(common.WORK / 'zed_depth_median.npy', med)
v, u = np.mgrid[0:600, 0:960]
P = np.dstack([(u - K[0, 2]) / K[0, 0] * med, (v - K[1, 2]) / K[1, 1] * med, med])


def side(u0, v0, u1, v1):
    return (u1 - u0) * (v - v0) - (v1 - v0) * (u - u0)


# ---- desk plane from depth
desk = (v > 375) & (v < 595) & (side(243, 365, 120, 600) < 0) & (side(775, 360, 920, 600) > 0)
desk &= ~(((u > 230) & (u < 400) & (v > 490)) | ((u > 630) & (u < 810) & (v > 490)) | ((u > 505) & (u < 560) & (v > 520)))
desk &= np.isfinite(med)
nd, dd, inl = common.fit_plane(P[desk], thr=0.006)
if dd < 0:
    nd, dd = -nd, -dd  # nd points from the desk toward the camera (up)
desk_res = P[desk][inl] @ nd + dd
# ---- upper wall plane from depth (diagnostic only)
wall = (v > 60) & (v < 250) & (u > 320) & (u < 760) & np.isfinite(med)
nw, dw, inw = common.fit_plane(P[wall], thr=0.01)
if dw < 0:
    nw, dw = -nw, -dw


# ---- desk edges from image gradients
def edges(image):
    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float), (5, 5), 1.2)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)

    def peak(seg):
        k = int(np.argmax(seg))
        if 1 <= k < len(seg) - 1:
            y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
            den = y0 - 2 * y1 + y2
            return k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
        return None
    L, Rr, Cr = [], [], []
    for vv in range(372, 596, 2):
        ue = 243 - (vv - 365) * (123 / 235)
        a = int(ue - 25); k = peak(gx[vv, a:a + 50])
        if k is not None:
            L.append((a + k, vv))
        ue = 777 + (vv - 357) * (141 / 243)
        a = int(ue - 25); k = peak(-gx[vv, a:a + 50])
        if k is not None:
            Rr.append((a + k, vv))
    for uu in range(280, 760, 3):
        a = 340; k = peak(np.abs(gy[a:a + 32, uu]))
        if k is not None:
            Cr.append((uu, a + k))
    return np.array(L), np.array(Rr), np.array(Cr)


def fit_line(pts, thr=1.0, rng=np.random.default_rng(0)):
    best = None
    for _ in range(500):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        inl = np.abs((pts - pts[i]) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = pts[best]
    c = Q.mean(0)
    dvec = np.linalg.svd(Q - c)[2][0]
    n = np.array([-dvec[1], dvec[0]])
    return c, dvec, float(best.mean()), float(np.sqrt(np.mean(((Q - c) @ n) ** 2)))


def inter(l1, l2):
    A = np.array([l1[1], -l2[1]]).T
    s = np.linalg.solve(A, l2[0] - l1[0])
    return l1[0] + s[0] * l1[1]


def ray_plane(uv):
    r = Ki @ np.array([uv[0], uv[1], 1.0])
    return (-dd / (nd @ r)) * r


lines = {}
for name, path in (('0928', common.ZED_EP / 'rgb/scene_camera/0000000000.png'), ('0929', common.ZED_0929)):
    im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    L, Rr, Cr = edges(im)
    lines[name] = dict(left=fit_line(L), right=fit_line(Rr), crease=fit_line(Cr))
ln = lines['0928']
bl, br = inter(ln['left'], ln['crease']), inter(ln['right'], ln['crease'])


def at_v(line, vv):
    c, d = line[0], line[1]
    return c + (vv - c[1]) / d[1] * d


BL, BR = ray_plane(bl), ray_plane(br)
FL, FR = ray_plane(at_v(ln['left'], 599)), ray_plane(at_v(ln['right'], 599))
ez = nd
ex = BR - BL
ex -= (ex @ ez) * ez
ex /= np.linalg.norm(ex)
ey = np.cross(ez, ex)
Rdc = np.stack([ex, ey, ez])           # camera -> desk-frame rotation (rows = desk axes in camera coords)
cam_in_desk = Rdc @ (np.zeros(3) - BL)  # camera centre in the desk frame (origin BL)


def to_desk(p):
    return Rdc @ (p - BL)


dl = (FL - BL) / np.linalg.norm(FL - BL)
dr = (FR - BR) / np.linalg.norm(FR - BR)
# wall (upper plane fit) relative to the desk frame
nw_d = Rdc @ nw
wall_yaw_vs_crease = np.degrees(np.arctan2(nw_d[0], -nw_d[1]))
wall_tilt_from_vertical = np.degrees(np.arcsin(abs(nw_d[2])))
wall_y_at = {k: float(-(dw + nw @ (BL + Rdc.T @ np.array([xx, 0, 0.5]))) / (nw @ (Rdc.T @ np.array([0, 1, 0]))))
             for k, xx in (('x0.0', 0.0), ('x0.75', 0.75), ('x1.5', 1.5))}
# depth-noise and crease rounding diagnostics: wall horizontal coordinate vs height, per column
prof = {}
Pd = (P.reshape(-1, 3) - BL) @ Rdc.T
Pd = Pd.reshape(600, 960, 3)
for uu in (350, 500, 650):
    rows = []
    for vv in range(40, 356, 20):
        s = np.s_[vv:vv + 20, uu:uu + 40]
        rows.append([round(float(np.nanmedian(Pd[s][..., 2])), 3), round(float(np.nanmedian(Pd[s][..., 1])), 4)])
    prof[f'u{uu}'] = rows

res = dict(
    source=dict(episode=str(common.ZED_EP), frames=len(fs), depth='uint16 mm, ZED SDK NEURAL_LIGHT, MEASURE.DEPTH (z along the optical axis), left rectified eye',
                image_size_wh=[960, 600], K=K, distortion='none (rectified image; raiden saves the rectified left-eye K only)',
                depth_valid_fraction=float(np.isfinite(med).mean()), depth_temporal_sd_median_mm=float(np.nanmedian(sd) * 1000)),
    desk_plane=dict(normal_cam=nd, offset_m=dd, camera_height_above_desk_m=dd, inlier_frac=float(inl.mean()),
                    rms_mm=float(np.sqrt(np.mean(desk_res ** 2)) * 1000)),
    edge_lines_px={k: {n: dict(point=l[0], dir=l[1], inlier_frac=l[2], rms_px=l[3]) for n, l in v_.items()} for k, v_ in lines.items()},
    image_corners_px=dict(back_left=bl, back_right=br),
    corners_cam_m=dict(back_left=BL, back_right=BR, left_edge_at_image_bottom=FL, right_edge_at_image_bottom=FR),
    desk_length_back_edge_m=float(np.linalg.norm(BR - BL)),
    desk_width_at_image_bottom_m=float(np.linalg.norm(FR - FL)),
    angles_deg=dict(left_vs_back=float(np.degrees(np.arccos(dl @ ex))), right_vs_back=float(np.degrees(np.arccos(dr @ ex))),
                    left_vs_right=float(np.degrees(np.arccos(np.clip(dl @ dr, -1, 1))))),
    camera_in_desk_frame_m=cam_in_desk,
    R_desk_from_cam=Rdc,
    upper_wall_plane=dict(normal_cam=nw, offset_m=dw, inlier_frac=float(inw.mean()),
                          yaw_vs_back_edge_deg=float(wall_yaw_vs_crease), tilt_from_vertical_deg=float(wall_tilt_from_vertical),
                          y_in_desk_frame_at=wall_y_at,
                          angle_to_desk_plane_deg=float(np.degrees(np.arccos(abs(nd @ nw)))),
                          note='textureless wall; diagnostic only (see profile)'),
    wall_profile_desk_frame={k: dict(columns='[z above desk, y toward wall]', rows=r) for k, r in prof.items()},
)
common.save_json(common.WORK / 'zed_metric.json', res)
np.savez(common.WORK / 'zed_metric.npz', K=K, nd=nd, dd=dd, nw=nw, dw=dw, BL=BL, BR=BR, FL=FL, FR=FR, Rdc=Rdc,
         cam_in_desk=cam_in_desk, bl=bl, br=br)
print('camera height above desk', round(dd, 4), 'desk length', round(res['desk_length_back_edge_m'], 4),
      'camera in desk frame', cam_in_desk.round(4), 'angles', res['angles_deg'])
print('upper wall yaw vs back edge', round(wall_yaw_vs_crease, 2), 'tilt', round(wall_tilt_from_vertical, 2), 'y at', wall_y_at)
