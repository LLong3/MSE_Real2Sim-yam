"""Desk back edge vs back wall in the Pi3X reference (metric frame of metric_frame.json), per X band.

- desk back boundary: high percentiles of Y of desk-top points (|z - local desk top| < 8 mm);
- wall just above the desk: median Y of points 3-15 cm above the desk top, behind the desk;
- slot evidence: points 2-15 cm BELOW the desk top with Y beyond the desk back boundary (a real gap shows the
  wall face down inside the slot to views from above);
- the painted wall's left edge (glass junction) region reported separately.
Writes work/wall_gap_reference.json.
"""
import json

import numpy as np

import common
import metric_frame as mf

J = json.load(open(common.HERE / 'metric_frame.json'))
M = np.array(J['transform_reference_to_metric']['M'])
x0, x1 = J['desk']['box_x_m']
y1 = J['desk']['box_y_m'][1]
top = J['desk']['top_z_m']
V, MV, _ = common.points()
P = mf.apply(M, np.asarray(V)[MV].astype(np.float64))
view = np.nonzero(MV)[0]
X, Y, Z = P.T
near = (Y > y1 - 0.30) & (Y < y1 + 0.20) & (X > x0 - 0.3) & (X < x1 + 0.3) & (Z > top - 0.25) & (Z < top + 0.40)
X, Y, Z, view = X[near], Y[near], Z[near], view[near]
rows = []
for xa in np.arange(x0, x1 - 1e-6, 0.10):
    b = (X >= xa) & (X < xa + 0.10)
    # local desk top from points 10-25 cm in front of the back edge
    lt = b & (Y > y1 - 0.25) & (Y < y1 - 0.10) & (np.abs(Z - top) < 0.04)
    if lt.sum() < 50:
        continue
    zt = float(np.median(Z[lt]))
    dk = b & (np.abs(Z - zt) < 0.008) & (Y > y1 - 0.25)
    wl = b & (Z > zt + 0.03) & (Z < zt + 0.15) & (Y > y1 - 0.06)
    wl_hi = b & (Z > zt + 0.15) & (Z < zt + 0.35) & (Y > y1 - 0.06)
    back = float(np.percentile(Y[dk], 99)) if dk.sum() > 50 else np.nan
    wy = float(np.median(Y[wl])) if wl.sum() > 30 else np.nan
    slot = b & (Z < zt - 0.02) & (Z > zt - 0.15) & (Y > back - 0.005) & (Y < wy + 0.03)
    rows.append(dict(x=round(float(xa + 0.05), 2), desk_top_z=round(zt, 4), desk_back_y_p99=round(back, 4),
                     desk_back_y_p995=round(float(np.percentile(Y[dk], 99.5)), 4) if dk.sum() > 50 else None,
                     wall_y_3_15cm=round(wy, 4), wall_y_15_35cm=round(float(np.median(Y[wl_hi])), 4) if wl_hi.sum() > 30 else None,
                     gap_wall_minus_desk_back_mm=round((wy - back) * 1000, 1),
                     slot_points_below_top=int(slot.sum()), desk_points=int(dk.sum()), wall_points=int(wl.sum()),
                     views_seeing_wall=int(len(np.unique(view[wl])))))
g = np.array([r['gap_wall_minus_desk_back_mm'] for r in rows])
out = dict(note='metric frame of metric_frame.json (sy = 1.00); gap = median wall Y (3-15 cm above the desk) minus the p99 Y of desk-top points',
           per_x_band=rows, gap_mm_median=float(np.nanmedian(g)), gap_mm_range=[float(np.nanmin(g)), float(np.nanmax(g))],
           desk_back_minus_plane_mm_median=float(np.nanmedian([r['desk_back_y_p99'] for r in rows]) * 1000 - y1 * 1000),
           wall_minus_plane_mm_median=float(np.nanmedian([r['wall_y_3_15cm'] for r in rows]) * 1000 - y1 * 1000))
# painted wall's left end (next to the frosted glass): wall Y in 2 cm slices, 5-45 cm above the desk top.
# Beige (paint-coloured) points; left of the paint the glass plane shows up ~7 cm behind the wall.
V2, MV2, C2 = common.points()
P2 = mf.apply(M, np.asarray(V2)[MV2].astype(np.float64))
col = np.asarray(C2)[MV2].astype(float)
s2 = (P2[:, 0] > x0 - 0.20) & (P2[:, 0] < x0 + 0.40) & (P2[:, 1] > y1 - 0.15) & (P2[:, 1] < y1 + 0.30) & (P2[:, 2] > top + 0.05) & (P2[:, 2] < top + 0.45)
P2, col = P2[s2], col[s2]
beige = (np.abs(col[:, 0] - col[:, 2]) < 60) & (col.mean(1) > 110)
edge_rows = []
for xa in np.arange(-0.10, 0.36, 0.02):
    b = (P2[:, 0] - x0 >= xa) & (P2[:, 0] - x0 < xa + 0.02) & beige
    if b.sum() > 20:
        dy = (P2[b, 1] - y1) * 1000
        edge_rows.append([round(float(xa + 0.01), 2), int(b.sum()), round(float(np.median(dy)), 1),
                          round(float(np.percentile(dy, 10)), 1), round(float(np.percentile(dy, 90)), 1)])
out['wall_left_end_slices'] = dict(columns='[x - desk left end (m), beige points, median y - plane (mm), p10, p90]', rows=edge_rows,
                                   note='glass/recess ~+71 mm behind the plane for x < x0 + 0.045; mixed x0 + 0.05..0.09; '
                                        'main wall face from x0 + 0.11 (median -11..-14 mm incl. the Pi3X lean and the 15 mm ty residual)')
common.save_json(common.WORK / 'wall_gap_reference.json', out)
for r in rows:
    print(r)
print({k: v for k, v in out.items() if k not in ('per_x_band', 'wall_left_end_slices')})
for r in edge_rows:
    print(r)
