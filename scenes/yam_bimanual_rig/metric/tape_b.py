"""User tape 09-29 (b) -> metric_frame.json: 35 mm desk-back-edge-to-wall gap, 1.00 m rail. Run after back_wall.py.

The tape corrects the flush verdict of back_wall.py. The desk box, transform, arms and camera pose do not change:
the camera is anchored on the ZED desk back edge, and that edge is the real wood edge (checked here). Changes:
back-wall plane = desk back edge + 0.035 (vertical); desk gap; back_wall_check verdict and notes; ZED wall-depth check
against the new plane; rail extent at the tape length. Evidence (CPU, read-only inputs), in work/tape_b/evidence.json:
  zed_edge        ZED images: wood/wall hue boundary vs the fitted desk back edge; the darker band above it
  zed_depth_step  ZED neural depth across the edge (temporal median, desk frame)
  ref_desk_back   reference desk-top-height points near the back boundary: wood vs wall-paint colour
  ref_wall        reference wall position vs height
  video_band      source-video dark band at the junction vs the 35 mm slot width (from video_slot_check.json)
  sy              y-scale reconciliation
  rail            rail-end pixels picked in source frames (work/tape_b/grid_*.jpg) -> each frame's own Pi3X point map
metric_frame.json is replaced atomically (temp file + rename). The pre-change copy is work/metric_frame.before_tape_b.json.
"""
import json
import os
import shutil
from datetime import datetime, timezone

import cv2
import numpy as np

import common
import metric_frame as mf

GAP = 0.035          # desk back edge to back wall, user tape 09-29
RAIL_LEN = 1.00      # rail (metal bar) length, user tape 09-29
OUT = common.HERE / 'metric_frame.json'
BEFORE = common.WORK / 'metric_frame.before_tape_b.json'
TB = common.WORK / 'tape_b'
# rail-end pixels (full-res source frames), picked by eye where the rail's silver top ends (grid crops work/tape_b/grid_*.jpg from
# work/tape_b/rail_grid.py, projected ticks from rail_overlay.py);
# C-clamps sit at both ends and hide part of the end face, so each pick is good to about +-5 px
RAIL_PICKS = [(341, 'left', 438, 805), (341, 'right', 1234, 782), (381, 'left', 553, 800), (381, 'right', 1300, 742),
              (401, 'left', 573, 805), (401, 'right', 1338, 733), (421, 'left', 596, 840), (421, 'right', 1328, 745),
              (942, 'left', 203, 1050)]

TB.mkdir(exist_ok=True)
if not BEFORE.exists():
    shutil.copy2(OUT, BEFORE)
J = json.load(open(OUT))
J0 = json.load(open(BEFORE))
W = lambda p: json.load(open(common.WORK / p))
zm, zn = W('zed_metric.json'), np.load(common.WORK / 'zed_metric.npz')
se, wc, gr, vs = W('stereo_edges.json'), W('wall_check.json'), W('wall_gap_reference.json'), W('video_slot_check.json')
M = np.array(J['transform_reference_to_metric']['M'])
x0, x1 = J['desk']['box_x_m']
y0, y1 = J['desk']['box_y_m']
TOP = J['desk']['top_z_m']
YW = y1 + GAP
cam = J['scene_camera']
K = np.array(cam['K'])
Twc = np.array(cam['pose']['world_from_camera_opencv'])
Rwc, C = Twc[:3, :3], Twc[:3, 3]
zs = mf.TAPE['length'] / zm['desk_length_back_edge_m']
ev = {}


def proj(P):
    q = (np.atleast_2d(P) - C) @ Rwc
    return q[:, :2] / q[:, 2:3] * K[[0, 1], [0, 1]] + K[:2, 2]


# ---- 1. ZED image: is the fitted desk back edge the wood edge? what is above it?
band_px = [float(proj([[x, y1, TOP]])[0, 1] - proj([[x, YW, TOP]])[0, 1]) for x in (x0 + 0.3, (x0 + x1) / 2, x1 - 0.3)]
ang = float(np.degrees(np.arctan2(C[2] - TOP, y1 - C[1])))
zed_edge = dict(camera_ray_below_horizontal_at_edge_deg=ang, slot_depth_seen_below_top_mm=GAP * np.tan(np.radians(ang)) * 1000,
                slot_band_px_35mm=band_px, images={})
for tag, path in (('0928', common.ZED_EP / 'rgb/scene_camera/0000000000.png'), ('0929', common.ZED_0929)):
    im = cv2.imread(str(path))
    S_ = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].astype(float)
    L_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)
    ln = zm['edge_lines_px'][tag]['crease']
    c, d = np.array(ln['point']), np.array(ln['dir'])
    dv_hue, prof = [], []
    for u in range(300, 741, 4):
        vc = c[1] + (u - c[0]) / d[0] * d[1]
        vv = np.arange(int(vc) - 25, int(vc) + 9)
        s = np.median(S_[vv[0]:vv[-1] + 1, u - 1:u + 2], 1)
        lum = np.median(L_[vv[0]:vv[-1] + 1, u - 1:u + 2], 1)
        s_wall, s_wood = np.median(s[(vv < vc - 8) & (vv > vc - 14)]), np.median(s[(vv > vc + 3)])
        half = 0.5 * (s_wall + s_wood)
        k = np.nonzero((s[:-1] < half) & (s[1:] >= half) & (vv[:-1] > vc - 6))[0]
        if len(k):
            k = k[0]
            dv_hue.append(float(vv[k] + (half - s[k]) / (s[k + 1] - s[k]) - vc))
        prof.append(np.interp(np.arange(-25, 7), vv - vc, lum))
    prof = np.median(prof, 0)
    far, lo = float(np.median(prof[:8])), float(prof[15:26].min())       # wall 18-25 px above the edge; minimum just above it
    dark = np.arange(-25, 7)[(prof < far - 0.5 * (far - lo)) & (np.arange(-25, 7) < 0)]
    onset = np.arange(-25, 7)[(prof < far - 0.1 * (far - lo)) & (np.arange(-25, 7) < 0)]
    zed_edge['images'][tag] = dict(
        hue_boundary_minus_fitted_edge_px=dict(median=float(np.median(dv_hue)), p90_abs=float(np.percentile(np.abs(dv_hue), 90)), columns=len(dv_hue)),
        luminance_above_edge=dict(profile_dv_minus25_to_plus6=np.round(prof, 1), wall_far=far, minimum=lo,
                                  half_depth_band_px=float(len(dark)), darkening_onset_px_above_edge=float(-onset.min()) if len(onset) else None))
zed_edge['finding'] = ('the fitted desk back edge is the wood/wall hue boundary (sub-pixel; 1 px ~ 3.5 mm in y), so the camera '
                       'pose is anchored on the real desk edge. Above it the wall darkens over ~10 px: the wall face seen down the '
                       f'gap (the ray past the edge drops {ang:.0f} deg and meets the wall {GAP * np.tan(np.radians(ang)) * 1000:.0f} mm below the desk top). '
                       'It is beige wall in shadow, continuous with the wall above; there is no black slot.')
ev['zed_edge'] = zed_edge

# ---- 2. ZED neural depth across the edge (desk frame of zed_metric, depth scaled to the tape desk length)
med = np.load(common.WORK / 'zed_depth_median.npy')
v_, u_ = np.mgrid[0:600, 0:960]
Pc = np.dstack([(u_ - zn['K'][0, 2]) / zn['K'][0, 0] * med, (v_ - zn['K'][1, 2]) / zn['K'][1, 1] * med, med])
Pd = ((Pc.reshape(-1, 3) - zn['BL']) @ zn['Rdc'].T).reshape(600, 960, 3) * zs
ln = zm['edge_lines_px']['0928']['crease']
rows = []
for dv in range(-30, 9):
    ys, zz = [], []
    for u in range(300, 741, 4):
        vc = ln['point'][1] + (u - ln['point'][0]) / ln['dir'][0] * ln['dir'][1]
        p = Pd[int(round(vc + dv)), u]
        if np.isfinite(p).all():
            ys.append(p[1]); zz.append(p[2])
    rows.append([dv, round(float(np.median(ys)) * 1000, 1), round(float(np.median(zz)) * 1000, 1)])
R_ = np.array(rows)
up = R_[R_[:, 0] < 0]
ev['zed_depth_step'] = dict(
    rows='[image rows from the fitted edge (negative = above it), median y behind the desk back edge mm, median z above the desk top mm], columns u 300-740',
    table=rows, y_mm_where_z_crosses_0=float(np.interp(0, up[::-1, 2], up[::-1, 1])),
    y_mm_at_z_40_95mm=[float(up[(up[:, 2] > 40) & (up[:, 2] < 95), 1].min()), float(up[(up[:, 2] > 40) & (up[:, 2] < 95), 1].max())],
    neural_bands_0_10cm_mm=[min(v['z0.00-0.05'][0] for v in wc['wall_y_mm_by_column_and_height']['table'].values()),
                            max(v['z0.05-0.10'][0] for v in wc['wall_y_mm_by_column_and_height']['table'].values())],
    finding='the depth leaves the desk plane at the fitted edge and recedes at desk-top level to y 8-21 mm within 6 rows, then the '
            'wall rises at y 22-37 mm up to 9 cm above the desk: a wall ~35 mm behind the edge, blurred by the neural depth over the '
            '~10 px the gap spans. Higher up (0.2-0.9 m) the textureless wall reads 60-180 mm (bias).')

# ---- 3. reference: desk-top-height points near the back boundary, and the wall vs height
V, MV, Cc = common.points()
P = mf.apply(M, np.asarray(V)[MV].astype(np.float64))
col = np.asarray(Cc)[MV]
X, Y, Z = P.T
sel = (X > -2.95) & (X < -1.75) & (Y > y1 - 0.12) & (Y < y1 + 0.08) & (Z > 0.60) & (Z < 0.90)
hsv = cv2.cvtColor(col[sel].reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
wood, paint = (hsv[:, 1] > 70) & (hsv[:, 0] < 25), hsv[:, 1] < 45
Xs, Ys, Zs = X[sel], Y[sel], Z[sel]
bands = [r for r in gr['per_x_band'] if -2.95 < r['x'] < -1.75]
p99 = float(np.median([r['desk_back_y_p99'] for r in bands]))
acc = {}
for r in bands:
    b = (np.abs(Xs - r['x']) < 0.05) & (np.abs(Zs - r['desk_top_z']) < 0.008)
    for dy in np.arange(-0.06, 0.015, 0.005):
        s = b & (Ys - r['desk_back_y_p99'] >= dy) & (Ys - r['desk_back_y_p99'] < dy + 0.005)
        k = round(float(dy + 0.0025) * 1000, 1)
        a = acc.setdefault(k, [0, 0.0, 0.0])
        a[0] += int(s.sum()); a[1] += float(wood[s].sum()); a[2] += float(paint[s].sum())
cb = [[k, n, w / n, p / n] for k, (n, w, p) in sorted(acc.items()) if n > 20]
cbA = np.array(cb)
i = int(np.nonzero(cbA[:, 2] < 0.5)[0][0])
wood_edge_off = float(np.interp(0.5, [cbA[i, 2], cbA[i - 1, 2]], [cbA[i, 0], cbA[i - 1, 0]]))  # mm from the p99 boundary
ev['ref_desk_back'] = dict(
    bins='[y - p99 back boundary (mm, 5 mm bins), points, wood fraction, wall-paint fraction]; desk-top height (|z - local top| < 8 mm), x -2.95..-1.75',
    table=np.round(cbA, 3), p99_boundary_y=p99, wood_edge_minus_p99_mm=wood_edge_off, wood_edge_y=p99 + wood_edge_off / 1000,
    wood_edge_minus_desk_back_edge_mm=(p99 + wood_edge_off / 1000 - y1) * 1000,
    slot_points_below_top_per_band=[r['slot_points_below_top'] for r in bands], wall_points_per_band=[r['wall_points'] for r in bands],
    finding=f'the wood ends {-wood_edge_off:.0f} mm before the reference back boundary; the last ~2 cm of "desk top" are wall-paint pixels '
            'flattened to desk height. Pi3X merged the wall seen down the gap into a desk-height strip and pulled the lower wall onto '
            'it; the slot interior has almost no points (100-430 per 10 cm band vs ~40k wall points).')
w = (X > -2.9) & (X < -1.7) & (Y > 1.95) & (Y < 2.35)
wall_rows = []
for z0 in np.arange(0.72, 2.02, 0.10):
    s = w & (Z >= z0) & (Z < z0 + 0.10)
    if s.sum() >= 5000:
        yy = float(np.median(Y[s]))
        wall_rows.append([round(float(z0 + 0.05 - TOP), 2), round((yy - y1) * 1000, 1), round((yy - p99) * 1000, 1), int(s.sum())])
ev['ref_wall'] = dict(rows='[height above the desk top m, wall y - desk back edge mm (metric), wall y - reference p99 boundary mm, points]',
                      table=wall_rows,
                      finding='the reference wall leans back from the desk-height strip: 0-9 mm behind its boundary up to 0.55 m above '
                              'the desk, 30 mm at 1.0-1.2 m. The lean is the merge, not a tilted wall.')

# ---- 4. source video: the dark band at the junction vs the 35 mm slot
vb = {}
for f, s in vs['summary'].items():
    vb[f] = dict(dark_band_px=s['dark_band_px_median'], slot_px_35mm=round(3.5 * s['slot_px_per_cm_median'], 1),
                 dark_band_cm_if_slot=round(s['implied_gap_if_band_is_slot_cm'], 2),
                 thin_line_cm_if_slot=round(vs['narrow_dip'][f]['dip_width_over_slot_px_per_cm_median'], 2))
ev['video_band'] = dict(per_frame=vb, finding='in all 6 close views the darker band above the junction (24-38 px) is at least as wide as a 35 mm slot '
                                              '(19-24 px). The old check called it lighting and read only the thin line inside it (0.6-0.8 cm), '
                                              'which is the deepest shadow at the occluding edge, not the slot width.')

# ---- 5. y scale
fr_ref = y0 + J['residuals_final']['desk_front_back_y_m']['front']       # reference front edge, metric frame
icp = {k: W(f'arm_icp_{k}.json')[k] for k in ('left_arm', 'right_arm')}
arm_sy = [1 / icp[k][f'opening_{o}']['axis_scale']['scale_base_xyz'][0] for k in icp for o in ('0', '0.0475')]
up_wall = np.median([r[1] for r in wall_rows if 0.95 <= r[0] <= 1.25]) / 1000 + y1
wood_y = p99 + wood_edge_off / 1000
merged = J['residuals_final']['back_wall']['y_by_height_m']['0.70-0.80']
ev['sy'] = dict(
    reference_front_edge_y=fr_ref, reference_wood_edge_y=wood_y, reference_merged_boundary_y=merged, reference_upper_wall_y_1p0_1p2m=up_wall,
    depths_at_sy_1=dict(front_to_wood=wood_y - fr_ref, front_to_merged_boundary=merged - fr_ref, front_to_p99=p99 - fr_ref, front_to_upper_wall=up_wall - fr_ref),
    implied_sy=dict(desk_wood_0p76=mf.TAPE['depth'] / (wood_y - fr_ref), upper_wall_0p795=(mf.TAPE['depth'] + GAP) / (up_wall - fr_ref),
                    merged_as_wall_0p795=(mf.TAPE['depth'] + GAP) / (merged - fr_ref), merged_as_edge_0p76=mf.TAPE['depth'] / (merged - fr_ref),
                    arm_length_icp=[min(arm_sy), max(arm_sy)]),
    finding='the reference desk-height measure (0.757) was neither desk (0.76) nor desk + gap (0.795): it ends at the merged strip. '
            'The wood depth (0.737) gives sy 1.03, the upper wall 1.01, the MJCF arm length 1.00-1.04. Desk + gap would need sy 1.05, '
            'at or past every other bound. Keep sy 1.00: with the desk anchored on the merged boundary, the box, arms and rail '
            'move by <= 3 mm between (sy 1.00, merged anchor) and (sy 1.03, wood anchor).')

# ---- 6. rail ends from the source frames' own point maps
c2w_all, _, fidx = common.cameras()
cams = json.load(open(common.REF / 'alignment/cameras.json'))
sc = 1920 / cams['processed_size_wh'][0]
picks = []
for f, side, u, v in RAIL_PICKS:
    k = int(np.nonzero(fidx == f)[0][0])
    xs = []
    for du in (-8, 0, 8):
        up_, vp_ = int(round((u + du) / sc)), int(round(v / sc))
        blk = mf.apply(M, np.asarray(V[k, vp_ - 1:vp_ + 2, up_ - 1:up_ + 2]).reshape(-1, 3).astype(float))
        xs.append(np.median(blk, 0))
    picks.append(dict(frame=f, side=side, px=[u, v], x=float(xs[1][0]), y=float(xs[1][1]), z=float(xs[1][2]),
                      x_per_5px=float(abs(xs[2][0] - xs[0][0]) / 16 * 5)))
arms = J['arms']
bl, br = arms['left_arm']['base_position_m'][0], arms['right_arm']['base_position_m'][0]
mid = 0.5 * (bl + br)
both = [f for f in {p['frame'] for p in picks} if {q['side'] for q in picks if q['frame'] == f} == {'left', 'right'}]
end = lambda f, s: next(p['x'] for p in picks if p['frame'] == f and p['side'] == s)
per_frame = {str(f): dict(left=end(f, 'left'), right=end(f, 'right'), length=end(f, 'right') - end(f, 'left'),
                          centre=0.5 * (end(f, 'left') + end(f, 'right')), overhang_left=bl - end(f, 'left'),
                          overhang_right=end(f, 'right') - br) for f in sorted(both)}
ctr = float(np.mean([v['centre'] for v in per_frame.values()]))
xr = [mid - RAIL_LEN / 2, mid + RAIL_LEN / 2]
ev['rail'] = dict(picks=picks, per_frame=per_frame, point_map_centre=ctr, point_map_centre_sd=float(np.std([v['centre'] for v in per_frame.values()])),
                  arm_base_midpoint_x=mid, pole_x=J['rail']['pole_xy_m'][0], x_extent=xr,
                  finding=f'the source-frame point maps put the rail ends at {np.mean([v["left"] for v in per_frame.values()]):.3f} / '
                          f'{np.mean([v["right"] for v in per_frame.values()]):.3f} (length {np.mean([v["length"] for v in per_frame.values()]):.3f}, '
                          f'2-3 cm short of the tape at each end) with the centre at {ctr:.3f}, {abs(ctr - mid) * 100:.1f} cm from the arm-base '
                          'midpoint. The tape rail is centred on the arm bases.')
common.save_json(TB / 'evidence.json', ev)

# ---- 7. ZED depth vs the new back-wall plane (same pixels and rules as build_metric.py)
medz = med * zs
rays = np.dstack([(u_ - K[0, 2]) / K[0, 0], (v_ - K[1, 2]) / K[1, 1], np.ones_like(u_, float)])
rw = rays @ Rwc.T
t_wall = (YW - C[1]) / rw[..., 1]
hitw = C + t_wall[..., None] * rw
on_wall = (t_wall > 0) & (hitw[..., 0] > x0 + 0.1) & (hitw[..., 0] < x1 - 0.1) & np.isfinite(medz)
wres = lambda m: (medz - t_wall)[m] * 1000
m_all, m_hi = on_wall & (hitw[..., 2] > TOP + 0.05), on_wall & (hitw[..., 2] > TOP + 0.30)
m_near = on_wall & (hitw[..., 2] > TOP) & (hitw[..., 2] < TOP + 0.10)
chk_old = J0['scene_camera']['checks']['zed_depth_vs_metric_back_wall']
wall_chk = dict(
    pixels=int(m_all.sum()), median_mm=float(np.median(wres(m_all))), p5_p95_mm=[float(np.percentile(wres(m_all), 5)), float(np.percentile(wres(m_all), 95))],
    above_0p3m_median_mm=float(np.median(wres(m_hi))),
    near_edge_0_10cm=dict(pixels=int(m_near.sum()), median_mm=float(np.median(wres(m_near))),
                          p10_p90_mm=[float(np.percentile(wres(m_near), 10)), float(np.percentile(wres(m_near), 90))]),
    plane_y_m=YW,
    note='against the tape plane (desk back edge + 0.035). Next to the desk (0-10 cm) the ZED depth agrees with the plane; '
         'higher up the textureless wall reads a median 5 cm (p95 20 cm) behind it (NEURAL_LIGHT bias). Not used for the pose.',
    verdict='tape_gap_35mm; zed near-edge depth agrees, upper-wall depth biased',
    before_tape_b=dict(plane_y_m=y1, median_mm=chk_old['median_mm'], above_0p3m_median_mm=chk_old['above_0p3m_median_mm'], verdict=chk_old['verdict']))

# ---- 8. merge
now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
J['inputs']['tape_0929_b'] = dict(desk_back_edge_to_wall_gap_m=GAP, rail_length_m=RAIL_LEN,
                                  source='user tape measure, 09-29 (second set): gap between the rig desk top back edge and the back wall; '
                                         'length of the rail (metal bar)')
d = J['desk']
d['back_edge_to_wall_gap_m'] = GAP
d['back_edge_to_wall_gap_bounds_m'] = [0.030, 0.040]
d['back_edge_to_wall_gap_note'] = ('tape 0.035 m (tape_0929_b). ZED near the desk agrees: neural depth 22-37 mm at 0-9 cm above the desk top, '
                                   'classical stereo on the painted wall end 41 mm. The earlier "flush" (back_wall.py) was wrong; see back_wall_check.')
bw = J['planes']['back_wall']
bw['d'] = YW
bw['note'] = ('vertical, desk back edge + 0.035 m (tape_0929_b); the desk back edge (y 2.0839) is the real wood edge and the camera is '
              'anchored on it. Pi3X puts the wall 17-54 mm in front of this plane (it merged the wall seen down the gap into the desk); '
              'the ZED neural depth higher up reads 4-15 cm behind it (bias). Build the wall vertical here.')
bw['wall_end_note'] = ('painted wall ends ~0.045-0.06 m right of the desk left end. Its end edge lies on the wall plane: ZED classical stereo '
                       'puts it 41 mm behind the desk back edge at desk height (6 mm behind the plane), 48 mm median over 0-0.55 m. '
                       'Left of it, the frosted glass is recessed: ~0.07 m behind the desk back edge in the reference (the reference wall '
                       'reads 4-5 cm short there, so up to ~0.12 m).')
R0 = J['residuals_final']['back_wall']
R0['note'] = ('reference wall (Pi3X, metric frame). It leans back ~1.8 deg because Pi3X merged the wall seen down the 35 mm gap into '
              'a desk-height strip and pulled the lower wall onto it (not a tilted wall; the pole is vertical to 0.1 deg). Author the '
              'wall vertical at desk back edge + 0.035 (planes.back_wall).')
R0['reference_desk_back_edge_minus_plane_m'] = (p99 - YW)
R0['reference_desk_back_boundary_minus_desk_back_edge_m'] = (p99 - y1)
R0['reference_wood_edge_minus_desk_back_edge_m'] = (wood_y - y1)
R0['reference_wall_minus_plane_m_by_height'] = {f'{r[0]:.2f}': round(r[1] / 1000 - GAP, 4) for r in wall_rows}
T = J['transform_reference_to_metric']
T['derivation']['sy'] = ('1.00 kept. Reference desk-height measures at sy 1: front to wood edge 0.737 (-> 1.03), front to the merged '
                         'desk/wall boundary 0.757, front to the upper wall (1.0-1.2 m above the desk) 0.787 (-> 1.01); MJCF arm length '
                         '1.00-1.04. The merged boundary is where Pi3X flattened the wall seen down the 35 mm gap; it is neither the desk '
                         'edge nor the wall. Desk + gap (0.795) on it would need 1.05. sy 1.00 with the desk anchored on the merged '
                         'boundary and sy 1.03 with it anchored on the wood edge place the box, arms and rail within 3 mm.')
T['uncertainty']['sy'] = 0.03
T['uncertainty']['sy_note'] = 'range 1.00-1.04 (tape_0929_b review); far from the desk, reference y distances may read up to 4 % short'
J['alternatives']['A_per_axis']['note'] = ('sx 1.06 from the desk length (1.42-1.44 predicted), sy 1.00 (see derivation.sy: the desk-height '
                                           'boundary is a Pi3X desk/wall merge; wood depth and arm length give 1.00-1.04), sz 1.09 from '
                                           'desk-top-minus-floor 0.661 (58 views); floor -> 0')
Pz = cam['pose']
Pz['horizontal_distance_to_back_wall_m'] = YW - C[1]
cam['checks']['zed_depth_vs_metric_back_wall'] = wall_chk
cam['checks']['desk_back_edge_is_wood_edge'] = dict(
    hue_boundary_minus_fitted_edge_px={t: zed_edge['images'][t]['hue_boundary_minus_fitted_edge_px'] for t in zed_edge['images']},
    note='ZED wood/wall hue boundary vs the fitted desk back edge the pose is anchored on (1 px ~ 3.5 mm in y): the pose does not depend '
         'on the wall; it is unchanged by the 35 mm gap.')

rl = J['rail']
rl['x_extent_recommended_m'] = xr
rl['length_recommended_m'] = RAIL_LEN
rl['length_tape_m'] = RAIL_LEN
rl['ends_outside_arm_bases_m'] = [bl - xr[0], xr[1] - br]
rl['end_evidence'] = dict(point_map_ends_per_frame=per_frame, point_map_centre_x=ctr, left_end_frame_942=end(942, 'left'),
                          reference_profile_ends_x=rl['x_extent_reference_m'], file='work/tape_b/evidence.json (rail), work/tape_b/grid_*.jpg')
rl['recommended_note'] = (f'tape 1.00 m (tape_0929_b), centred on the arm-base midpoint (x {mid:.4f}): each end {bl - xr[0]:.4f} m outside its '
                          f'arm base. Evidence: the rail ends picked in source frames 341/381/401/421 read {ctr:.3f} for the centre through '
                          f'each frame\'s own Pi3X point map ({abs(ctr - mid) * 100:.1f} cm from the arm midpoint); front photo 401 shows '
                          'equal overhangs. The same picks read the rail 0.94-0.98 m long: Pi3X is 2-3 cm short at each end.')
rl['note'] = ('Aluminium plate on the desk top, front face flush with the desk front edge; the arms and the camera pole mount on it. '
              'C-clamps at both ends hold it to the desk front edge. x extent: tape length, centred on the arm bases (ends +-1 cm). '
              'y extent, top height and profiles: reference points (Pi3X, ~1 cm noise).')

bc = J['back_wall_check']
bc['answer'] = ('(a) a real gap: 35 mm by tape (tape_0929_b). The earlier answer "(b) ZED NEURAL_LIGHT bias; flush to ~1 cm" was wrong. '
                'Near the desk the ZED depth agrees with the gap; only higher up is the 85 mm median NEURAL_LIGHT bias.')
bc['why_flush_was_wrong'] = [
    'Pi3X: the wall pixels seen down the gap were flattened into a ~2 cm desk-height strip behind the wood, and the lower wall was pulled '
    'onto it. "Desk-top points reach the wall" was that strip, not the desk (work/tape_b/evidence.json ref_desk_back).',
    f'ZED: the ray past the edge drops {ang:.0f} deg; a 35 mm gap spans ~10 px and shows beige wall in shadow, continuous with the wall '
    'above. No black slot is expected. The "blur-limited 5 px line" was measured against a wall baseline 8-20 px above the edge, '
    'which itself lies in the gap band (10 px) and its shadow (onset 15 px).',
    'Video: the darker band above the junction (24-38 px) is at least a 35 mm slot wide (19-24 px) in all 6 views, consistent with the '
    'gap; it was dismissed as lighting and only the thin line inside it (0.6-0.8 cm) was read.',
    'The two metric ZED readings next to the desk were explained away: neural depth 20-43 mm at 0-10 cm and classical stereo 41 mm at '
    'the painted wall end (read as a reveal). Both were the gap.']
bc['not_a_gap'] = 'WITHDRAWN (tape_0929_b): see why_flush_was_wrong. Previous text: ' + J0['back_wall_check']['not_a_gap']
bc['not_transform'] = ('still true: the check is ZED-internal and the desk back edge is the ZED wood edge. With the tape plane the ZED near-edge '
                       'depth matches (near_edge_0_10cm in scene_camera.checks.zed_depth_vs_metric_back_wall).')
bc['wall_end_detail'] = ('the painted wall end next to the glass is on the wall plane (ZED classical stereo 41 mm behind the desk back edge at '
                         'desk height, 6 mm behind the tape plane). It is not a reveal; the frosted glass left of it is recessed further.')
opt = bc['options']
opt['A_flush_adopted']['status'] = 'rejected by tape_0929_b (gap 0.035 m)'
opt['B_small_gap']['status'] = 'rejected by tape_0929_b (gap 0.035 m)'
opt['C_tape_gap_adopted'] = dict(
    back_edge_to_wall_gap_m=GAP, back_wall_plane_y_m=YW, status='adopted (tape_0929_b)',
    residuals=dict(zed_neural_near_edge_0_10cm_median_mm=wall_chk['near_edge_0_10cm']['median_mm'],
                   zed_neural_wall_median_mm=wall_chk['median_mm'],
                   zed_stereo_wall_end_at_desk_height_mm=se['classical_only_crease_test']['junction_behind_desk_back_edge_mm'] - GAP * 1000,
                   reference_wall_minus_plane_mm_at_0p05_and_1p05m=[wall_rows[0][1] - GAP * 1000, next(r[1] for r in wall_rows if r[0] == 1.05) - GAP * 1000]))
bc['settle_by'] = 'settled by tape_0929_b: gap 0.035 m (user tape, 09-29).'
E = bc['evidence']
E['zed_neural_depth_not_a_wall']['finding'] = ('corrected: near the desk (0-10 cm) the ZED neural depth sees the wall 20-43 mm behind the edge, '
                                               'which is the gap; higher up the textureless wall surface is non-planar and biased')
E['reference_local_gap']['finding'] = ('corrected: the "desk-top points reaching the wall" are wall-paint pixels that Pi3X flattened to desk '
                                       'height; the wood ends ~2 cm earlier (tape_b ref_desk_back)')
E['video_and_zed_dark_line']['finding'] = ('corrected: the thin line is the deepest shadow at the edge; the broader darker band around it '
                                           'is the wall seen down the 35 mm gap (tape_b video_band, zed_edge)')
E['zed_classical_stereo']['finding'] = ('corrected: the painted wall end 41 mm behind the desk back edge at desk height is the wall itself '
                                        '(tape 35 mm), not a reveal')
E['zed_edge_is_wood_edge'] = dict(finding=zed_edge['finding'],
                                  hue_boundary_minus_fitted_edge_px={t: zed_edge['images'][t]['hue_boundary_minus_fitted_edge_px']['median'] for t in zed_edge['images']},
                                  darker_band_px={t: zed_edge['images'][t]['luminance_above_edge']['darkening_onset_px_above_edge'] for t in zed_edge['images']},
                                  slot_band_px_35mm=float(np.median(band_px)), file='work/tape_b/evidence.json (zed_edge)')
E['zed_depth_step'] = dict(finding=ev['zed_depth_step']['finding'], y_mm_where_z_crosses_0=ev['zed_depth_step']['y_mm_where_z_crosses_0'],
                           y_mm_at_z_40_95mm=ev['zed_depth_step']['y_mm_at_z_40_95mm'], file='work/tape_b/evidence.json (zed_depth_step)')
E['reference_desk_top_colour'] = dict(finding=ev['ref_desk_back']['finding'], wood_edge_minus_p99_mm=wood_edge_off,
                                      wood_edge_minus_desk_back_edge_mm=ev['ref_desk_back']['wood_edge_minus_desk_back_edge_mm'],
                                      file='work/tape_b/evidence.json (ref_desk_back)')
E['video_band_vs_35mm'] = dict(finding=ev['video_band']['finding'], per_frame=vb, file='work/tape_b/evidence.json (video_band)')
bc['revision'] = 'tape_0929_b: verdict corrected from flush to a 35 mm gap; old option/evidence texts kept with corrected findings'

pr = J['producer']
pr['scripts'] = [s for s in pr['scripts'] if s != 'tape_b.py'] + ['tape_b.py']
pr['updated_utc'] = now
pr['updated_by'] = ('claude-yam-metric-tape (task yam-metric-tape-20260929): tape_0929_b - back wall at desk back edge + 0.035, '
                    'rail 1.00 m; transform, desk box, arms and camera pose unchanged')
J['revision'] = dict(
    id='tape_0929_b', utc=now, task='yam-metric-tape-20260929', owner='claude-yam-metric-tape',
    inputs='inputs.tape_0929_b: desk back edge to wall 0.035 m, rail 1.00 m (user tape, 09-29)',
    changed=['planes.back_wall.d: 2.0839 -> desk back edge + 0.035 = %.4f' % YW, 'desk.back_edge_to_wall_gap_m: 0 -> 0.035 (bounds 0.03-0.04)',
             'back_wall_check: flush verdict withdrawn (why_flush_was_wrong)',
             'scene_camera.checks.zed_depth_vs_metric_back_wall: recomputed against the new plane',
             'rail.x_extent_recommended_m: [-2.7888, -1.7406] (1.048 m) -> [%.4f, %.4f] (1.00 m, centred on the arm bases)' % tuple(xr),
             'notes: planes.back_wall, residuals_final.back_wall, transform derivation.sy (+ uncertainty 0.03), alternatives.A note, rail'],
    unchanged=['transform (S, t, M)', 'desk box (x -3.0488..-1.5288, y 1.3239..2.0839, top 0.72)', 'arm bases and yaws',
               'scene_camera K and pose (anchored on the ZED desk back edge = the wood edge)'],
    previous_file='work/metric_frame.before_tape_b.json', evidence='work/tape_b/evidence.json')

DEST = TB / 'metric_frame.dry.json' if os.environ.get('TAPE_B_DRY') else OUT   # TAPE_B_DRY=1: write a copy for review only
tmp = DEST.with_name(DEST.name + '.tmp')
common.save_json(tmp, J)
json.load(open(tmp))
os.replace(tmp, DEST)
print(json.dumps(dict(plane_y=YW, zed_edge={t: zed_edge['images'][t]['hue_boundary_minus_fitted_edge_px'] for t in zed_edge['images']},
                      band_px=band_px, dark_band={t: zed_edge['images'][t]['luminance_above_edge']['half_depth_band_px'] for t in zed_edge['images']},
                      onset={t: zed_edge['images'][t]['luminance_above_edge']['darkening_onset_px_above_edge'] for t in zed_edge['images']},
                      depth_step=dict(y0=ev['zed_depth_step']['y_mm_where_z_crosses_0'], y40_95=ev['zed_depth_step']['y_mm_at_z_40_95mm']),
                      wood_edge_off=wood_edge_off, wood_minus_edge_mm=ev['ref_desk_back']['wood_edge_minus_desk_back_edge_mm'],
                      sy=ev['sy'], rail=dict(x=xr, per_frame=per_frame, centre=ctr, mid=mid, e942=end(942, 'left')),
                      wall_chk=wall_chk, ref_wall=wall_rows), indent=1, default=float))
