"""Back-wall offset check: is the ~85 mm ZED wall depth beyond the desk back edge a real gap or depth bias?

Everything here is ZED-internal (converted episode 0000, 238 frames, NEURAL_LIGHT depth, left eye) in the
ZED desk frame of zed_metric.py: origin = back-left desk-top corner on the image crease, x along the crease,
z = desk normal, y toward the wall. A wall flush with the desk back edge has y = 0.
- wall y map (temporal-median depth) and temporal SD map (confidence proxy; the episode stores no confidence);
- textured, stereo-friendly features: the vertical frosted-glass / wall junction (left) and the crease band;
- per-column plane fits to the wall (planarity, yaw vs the crease).
Writes work/wall_check.json and work/wallcheck/zed_wall_y.jpg.
"""
import glob

import cv2
import numpy as np

import common

zn = np.load(common.WORK / 'zed_metric.npz')
K, BL, Rdc, nd, dd = zn['K'], zn['BL'], zn['Rdc'], zn['nd'], zn['dd']
fs = sorted(glob.glob(str(common.ZED_EP / 'depth/scene_camera/*.npz')))
D = np.stack([np.load(f)['depth'] for f in fs]).astype(np.float64) / 1000.0
D[D == 0] = np.nan
med = np.nanmedian(D, 0)
sd = np.nanstd(D, 0)
v, u = np.mgrid[0:600, 0:960]
P = np.dstack([(u - K[0, 2]) / K[0, 0] * med, (v - K[1, 2]) / K[1, 1] * med, med])
Pd = (P.reshape(-1, 3) - BL) @ Rdc.T
Pd = Pd.reshape(600, 960, 3)
X, Y, Z = Pd[..., 0], Pd[..., 1], Pd[..., 2]

# crease line in the image (0928 frame fit) -> v_crease(u)
crease_c, crease_d = np.array([518.5, 359.9975]), np.array([0.99978, -0.02101])
v_cr = crease_c[1] + (u - crease_c[0]) / crease_d[0] * crease_d[1]
img = cv2.cvtColor(cv2.imread(str(common.ZED_EP / 'rgb/scene_camera/0000000000.png')), cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
beige = (hsv[..., 1] < 60) & (hsv[..., 2] > 120)            # wall paint
wall = beige & (v < v_cr - 3) & (u > 240) & (u < 800) & np.isfinite(med)

out = {}
# ---- 1. wall offset by height band and by column
bands = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.90)]
cols = [(250, 350), (350, 450), (450, 550), (550, 650), (650, 750)]
tab = {}
for c0, c1 in cols:
    row = {}
    for z0, z1 in bands:
        s = wall & (u >= c0) & (u < c1) & (Z >= z0) & (Z < z1)
        row[f'z{z0:.2f}-{z1:.2f}'] = [round(float(np.median(Y[s])) * 1000, 1), round(float(np.median(sd[s])) * 1000, 1), int(s.sum())] if s.sum() > 30 else None
    tab[f'u{c0}-{c1}'] = row
out['wall_y_mm_by_column_and_height'] = dict(cells='[median y behind the crease plane mm, median temporal SD mm, pixels]', table=tab)

# ---- 2. temporal SD (confidence proxy): wall vs desk vs textured regions
desk_px = (v > v_cr + 15) & (v < 470) & (u > 300) & (u < 700) & np.isfinite(med)
out['temporal_sd_mm'] = dict(desk_top=float(np.nanmedian(sd[desk_px]) * 1000), wall=float(np.nanmedian(sd[wall]) * 1000),
                            wall_p90=float(np.nanpercentile(sd[wall], 90) * 1000), desk_top_p90=float(np.nanpercentile(sd[desk_px], 90) * 1000))

# ---- 3. per-column wall planes: tilt from vertical and offset at the crease
fits = {}
for c0, c1 in cols:
    s = wall & (u >= c0) & (u < c1) & (Z > 0.05) & (Z < 0.8)
    A = np.c_[Z[s], np.ones(s.sum())]
    (k, b), *_ = np.linalg.lstsq(A, Y[s], rcond=None)
    fits[f'u{c0}-{c1}'] = dict(y_at_crease_mm=round(float(b) * 1000, 1), lean_deg=round(float(np.degrees(np.arctan(k))), 2),
                               x_mid_m=round(float(np.median(X[s])), 3))
out['per_column_linear_fit_y_vs_height'] = fits
s = wall & (Z > 0.05) & (Z < 0.8)
A = np.c_[X[s], Z[s], np.ones(s.sum())]
(kx, kz, b), *_ = np.linalg.lstsq(A, Y[s], rcond=None)
out['wall_plane_fit'] = dict(yaw_vs_crease_deg=float(np.degrees(np.arctan(kx))), lean_deg=float(np.degrees(np.arctan(kz))),
                             y_at_x0_z0_mm=float(b * 1000), rms_mm=float(np.sqrt(np.mean((A @ [kx, kz, b] - Y[s]) ** 2)) * 1000))

# ---- 4. glass / wall junction (vertical, high-contrast edge: the best stereo feature near the wall)
g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float), (3, 3), 0.8)
gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
junc = []
for vv in range(40, 356, 4):
    # junction runs from about (227, 300) to (275, 362): search +-40 px around the extrapolated line
    ue = 275 + (vv - 362) * (48 / 62)
    a, b_ = int(max(ue - 40, 1)), int(min(ue + 40, 958))
    if b_ - a < 10:
        continue
    k = int(np.argmax(np.abs(gx[vv, a:b_])))
    ux = a + k
    if abs(gx[vv, ux]) < 40:
        continue
    beige_side = med[vv, ux + 4:ux + 9]
    glass_side = med[vv, ux - 8:ux - 3]
    if not (np.isfinite(beige_side).all() and np.isfinite(glass_side).all()):
        continue
    yb = np.median(Y[vv, ux + 4:ux + 9]); zb = np.median(Z[vv, ux + 4:ux + 9]); xb = np.median(X[vv, ux + 4:ux + 9])
    yg = np.median(Y[vv, ux - 8:ux - 3])
    yb_far = np.median(Y[vv, ux + 30:ux + 60]) if ux + 60 < 960 else np.nan
    junc.append([vv, ux, xb, zb, yb, yg, yb_far, float(np.median(sd[vv, ux + 4:ux + 9]))])
J = np.array(junc)
ok = J[:, 3] > 0.03
out['glass_wall_junction'] = dict(
    note='vertical beige/glass edge (left end of the painted wall), found per row by the horizontal image gradient; '
         'y sampled 4-8 px on the wall side, 3-8 px on the glass side, and 30-60 px into the plain wall',
    rows=int(ok.sum()), x_m_median=float(np.median(J[ok, 2])), z_range_m=[float(J[ok, 3].min()), float(J[ok, 3].max())],
    y_wall_side_mm_median=float(np.median(J[ok, 4]) * 1000), y_wall_side_mm_p10_p90=[float(np.percentile(J[ok, 4], 10) * 1000), float(np.percentile(J[ok, 4], 90) * 1000)],
    y_glass_side_mm_median=float(np.median(J[ok, 5]) * 1000),
    y_plain_wall_30_60px_right_mm_median=float(np.nanmedian(J[ok, 6]) * 1000),
    temporal_sd_mm_median=float(np.median(J[ok, 7]) * 1000),
    by_height=[[round(float(r[3]), 2), round(float(r[4]) * 1000, 1), round(float(r[6]) * 1000, 1)] for r in J[ok][::6]])

# ---- 5. crease band: depth immediately above the crease (z < 3 cm) and immediately below (desk)
above = wall & (Z >= 0.0) & (Z < 0.03)
below = (v > v_cr + 2) & (v < v_cr + 12) & (u > 300) & (u < 740) & np.isfinite(med)
out['crease_band'] = dict(wall_z0_3cm_y_mm_median=float(np.median(Y[above]) * 1000), wall_z0_3cm_pixels=int(above.sum()),
                          desk_rows_below_crease_y_mm_median=float(np.median(Y[below]) * 1000),
                          desk_rows_below_crease_z_mm_median=float(np.median(Z[below]) * 1000))
common.save_json(common.WORK / 'wall_check.json', out)

# ---- review image: y behind the crease plane on the wall, over the ZED image
vis = img.copy()
m = wall | ((v < v_cr - 3) & (u >= 150) & (u < 900) & np.isfinite(med) & (Y > -0.05))
yy = np.clip(Y * 1000, 0, 150)
cm = cv2.applyColorMap((yy / 150 * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
vis[m] = (0.35 * vis[m] + 0.65 * cm[m]).astype(np.uint8)
for r in J[ok]:
    cv2.circle(vis, (int(r[1]), int(r[0])), 2, (255, 60, 60), -1)
bar = np.zeros((600, 60, 3), np.uint8) + 255
for i in range(560):
    bar[20 + i, 10:30] = cv2.applyColorMap(np.array([[int((1 - i / 559) * 255)]], np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0, ::-1]
for val in (0, 50, 100, 150):
    yv = int(20 + (1 - val / 150) * 559)
    cv2.putText(bar, str(val), (32, yv + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
vis = np.concatenate([vis, bar], 1)
for i, t in enumerate(('ZED depth (temporal median): wall distance behind the crease plane, mm',
                       'red dots: glass/wall junction (vertical textured edge)')):
    cv2.putText(vis, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
(common.WORK / 'wallcheck').mkdir(exist_ok=True)
cv2.imwrite(str(common.WORK / 'wallcheck/zed_wall_y.jpg'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
import json
print(json.dumps({k: out[k] for k in out if k != 'glass_wall_junction'}, indent=1))
print(json.dumps(out['glass_wall_junction'], indent=1))
