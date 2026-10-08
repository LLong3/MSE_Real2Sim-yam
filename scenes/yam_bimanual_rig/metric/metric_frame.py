"""Reference -> metric transform candidates and the reference desk anchors (shared by the fit scripts).

Reference anchors are measured from the aligned Pi3X point maps (measurement-valid pixels), predicted m:
  - desk top plane (RANSAC) and the floor near the desk (floor_check.py),
  - desk ends and front/back edges: half-plateau of the desk-top point density across each edge.
Metric: floor z=0, desk top 0.72, desk 1.52 (X) x 0.76 (Y) (tape, 09-29).
Transform form: p_metric = S (p_ref - c_ref) + c_metric, S = diag(sx, sy, sz), no rotation
(the reference is already floor/wall aligned), c = desk centre on the floor.
"""
import json

import numpy as np

import common

TAPE = dict(length=1.52, depth=0.76, top=0.72)


def density_edge(values, outside, inside, interior, step=0.005, frac=0.3):
    """Edge = first bin, scanning from outside toward inside, whose point density reaches `frac` of the
    interior density (median over the `interior` interval)."""
    lo, hi = min(outside, inside), max(outside, inside)
    bins = np.arange(lo, hi + step, step)
    h, e = np.histogram(values, bins)
    c = 0.5 * (e[1:] + e[:-1])
    hs = np.convolve(h, np.ones(3) / 3, mode='same')
    ilo, ihi = min(interior), max(interior)
    plateau = np.median(np.histogram(values[(values > ilo) & (values < ihi)], np.arange(ilo, ihi + step, step))[0])
    order = np.arange(len(c)) if outside < inside else np.arange(len(c))[::-1]
    for k in order:
        if hs[k] >= frac * plateau:
            return float(c[k]), float(plateau)
    return float('nan'), float(plateau)


def reference_anchors(cache=True):
    f = common.WORK / 'reference_anchors.json'
    if cache and f.exists():
        return json.load(open(f))
    V, MV, _ = common.points()
    P = np.asarray(V)[MV]
    X, Y, Z = P.T
    fl = json.load(open(common.WORK / 'floor_check.json'))
    top_slab = (Z > 0.62) & (Z < 0.70)
    left, right, front = [], [], []
    for y0 in (1.40, 1.50, 1.60, 1.70, 1.80, 1.90):
        s = top_slab & (Y > y0) & (Y < y0 + 0.1)
        left.append(density_edge(X[s & (X > -3.12) & (X < -2.75)], -3.12, -2.90, (-2.90, -2.80))[0])
        right.append(density_edge(X[s & (X > -1.80) & (X < -1.52)], -1.52, -1.70, (-1.80, -1.70))[0])
    for x0 in (-2.95, -2.75, -2.45, -2.10, -1.85):
        s = top_slab & (X > x0) & (X < x0 + 0.15)
        front.append(density_edge(Y[s & (Y > 1.22) & (Y < 1.60)], 1.22, 1.40, (1.45, 1.60))[0])
    # back wall: plane of wall points at desk height x-range, 0.75-1.0 m (just above the desk top)
    w = (X > -2.9) & (X < -1.7) & (Y > 1.95) & (Y < 2.3)
    wall_y = {f'{lo:.2f}-{hi:.2f}': float(np.median(Y[w & (Z > lo) & (Z < hi)])) for lo, hi in
              ((0.70, 0.80), (0.80, 1.00), (1.00, 1.30), (1.30, 1.60), (1.60, 1.90))}
    desk = (X > -2.9) & (X < -1.7) & (Y > 1.45) & (Y < 2.0) & (Z > 0.6) & (Z < 0.7)
    n, d, _ = common.fit_plane(P[desk], thr=0.008)
    if n[2] < 0:
        n, d = -n, -d
    a = dict(
        desk_left_x=float(np.median(left)), desk_left_x_per_band=left,
        desk_right_x=float(np.median(right)), desk_right_x_per_band=right,
        desk_front_y=float(np.median(front)), desk_front_y_per_band=front,
        wall_y_by_height=wall_y, desk_back_y=wall_y['0.70-0.80'],
        desk_plane_normal=n, desk_plane_offset=float(d),
        desk_top_z_at_centre=float((-d - n[0] * -2.295 - n[1] * 1.705) / n[2]),
        desk_tilt_deg=float(np.degrees(np.arccos(n[2]))),
        floor_z_near_desk=fl['near_desk']['local_plane_z_at_desk_centre_mm'] / 1000.0,
        floor_z_under_desk=fl['under_desk']['local_plane_z_at_desk_centre_mm'] / 1000.0,
        desk_minus_floor_per_view_median=fl['per_view_desk_minus_floor']['desk_minus_floor_median_m'],
    )
    a['desk_length'] = a['desk_right_x'] - a['desk_left_x']
    a['desk_depth_front_to_wall'] = a['desk_back_y'] - a['desk_front_y']
    a['desk_height_above_local_floor'] = a['desk_top_z_at_centre'] - a['floor_z_near_desk']
    a['desk_centre_xy'] = [0.5 * (a['desk_left_x'] + a['desk_right_x']), 0.5 * (a['desk_front_y'] + a['desk_back_y'])]
    common.save_json(f, a)
    return a


def candidates(a):
    """All candidates map the desk centre (floor level) to the same metric point and the desk top to 0.72."""
    Lr, Dr, Hr = a['desk_length'], a['desk_depth_front_to_wall'], a['desk_height_above_local_floor']
    cx, cy = a['desk_centre_xy']
    c_ref = np.array([cx, cy, a['floor_z_near_desk']])
    out = {
        'A_per_axis': dict(S=[TAPE['length'] / Lr, 1.0, TAPE['top'] / Hr],
                           note='sx from desk length, sy = 1 (depth axis of the first pass; desk depth 0.77 = 0.76 + ~1 cm wall gap), sz from desk height over the local floor'),
        'B_uniform_length': dict(S=[TAPE['length'] / Lr] * 3,
                                 note='uniform scale from desk length; desk top pinned at 0.72, floor left where it lands'),
        'C_uniform_1': dict(S=[1.0, 1.0, 1.0], note='scale 1; desk top pinned at 0.72 (vertical offset only)'),
        'D_uniform_height': dict(S=[TAPE['top'] / Hr] * 3, note='blind uniform scale from desk height'),
    }
    for k, v in out.items():
        S = np.array(v['S'])
        # desk top (ref) -> 0.72 exactly for every candidate; XY desk centre fixed
        top_ref = a['desk_top_z_at_centre']
        t = np.array([cx - S[0] * cx, cy - S[1] * cy, TAPE['top'] - S[2] * top_ref])
        # metric desk centre: keep the reference desk-centre XY (metric frame origin = reference origin)
        M = np.eye(4)
        M[:3, :3] = np.diag(S)
        M[:3, 3] = t
        v['M'] = M
    return out


def apply(M, P):
    return P @ M[:3, :3].T + M[:3, 3]
