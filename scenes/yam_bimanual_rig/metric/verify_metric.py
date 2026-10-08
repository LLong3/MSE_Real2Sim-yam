"""Self-consistency check of metric_frame.json (read-only; CPU; seconds).

- completeness: no null/NaN outside the documented profile gaps (rail profiles use null for empty bins);
- transform: M = [diag(S) | t]; the stated residuals recomputed from work/reference_anchors.json + M;
- camera: rotation orthonormal, det +1; blender = opencv @ diag(1,-1,-1); position/optical axis/pitch derived
  fields agree; K, FOV and Blender lens/shift agree; desk-box back corners reproject onto the ZED crease corners;
- arms: world_from_base = [Rz(yaw) | base]; spacing agrees with the bases;
- desk/planes agree with the box;
- tape_0929_b: back wall vertical at desk back edge + the taped gap; the ZED wood/wall hue boundary lies on the projected
  desk back edge and the 35 mm gap spans ~10 px above it; the ZED wall-depth check reproduces against the plane;
  rail length = tape, rail inside the desk, ends outside the arm bases as stated.
Prints PASS/FAIL per check; exit 1 on any FAIL. Writes nothing.
"""
import json
import math
import sys

import cv2
import numpy as np

import common
import metric_frame as mf

J = json.load(open(common.HERE / 'metric_frame.json'))
a = json.load(open(common.WORK / 'reference_anchors.json'))
zn = np.load(common.WORK / 'zed_metric.npz')
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
    if not ok:
        fails.append(name)


def walk(o, path=''):
    bad = []
    if isinstance(o, dict):
        for k, v in o.items():
            bad += walk(v, f'{path}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += walk(v, f'{path}[{i}]')
    elif o is None or (isinstance(o, float) and not math.isfinite(o)):
        bad.append(path)
    return bad


bad = [p for p in walk(J) if not p.startswith(('.rail.profile_y_vs_height_above_local_desk', '.rail.height_above_local_desk_by_x'))]
check('completeness: no null/NaN outside rail profiles', not bad, ', '.join(bad[:5]))
for sec in ('inputs', 'conventions', 'transform_reference_to_metric', 'residuals_final', 'alternatives', 'desk', 'planes',
            'arms', 'arm_spacing', 'rail', 'scene_camera'):
    check(f'section {sec} present and non-empty', bool(J.get(sec)))

T = J['transform_reference_to_metric']
M, S, t = np.array(T['M']), np.array(T['S']), np.array(T['t'])
check('M = [diag(S) | t]', np.allclose(M[:3, :3], np.diag(S)) and np.allclose(M[:3, 3], t) and np.allclose(M[3], [0, 0, 0, 1]))
R = J['residuals_final']
box = J['desk']
x0, x1 = box['box_x_m']
y0, y1 = box['box_y_m']
top = box['top_z_m']
ap = lambda p: M[:3, :3] @ np.asarray(p, float) + M[:3, 3]
calc = {
    'desk_ends_x_m.left': S[0] * a['desk_left_x'] + t[0] - x0,
    'desk_ends_x_m.right': S[0] * a['desk_right_x'] + t[0] - x1,
    'desk_length_m': S[0] * a['desk_length'],
    'desk_length_residual_m': S[0] * a['desk_length'] - 1.52,
    'desk_front_back_y_m.front': S[1] * a['desk_front_y'] + t[1] - y0,
    'desk_front_back_y_m.back_wall': S[1] * a['desk_back_y'] + t[1] - y1,
    'desk_depth_front_to_wall_m': S[1] * a['desk_depth_front_to_wall'],
    'desk_top_z_m.plane_at_centre': ap([0, 0, a['desk_top_z_at_centre']])[2],
    'desk_top_z_m.per_view_median': S[2] * (a['floor_z_near_desk'] + a['desk_minus_floor_per_view_median']) + t[2],
    'floor_near_desk_z_m': ap([0, 0, a['floor_z_near_desk']])[2],
}
for h, y in a['wall_y_by_height'].items():
    calc[f'back_wall.y_by_height_m.{h}'] = S[1] * y + t[1]


def get(d, dotted):
    for k in dotted.split('.', 1)[0:1]:
        pass
    cur = d
    parts = dotted.split('.')
    i = 0
    while i < len(parts):
        # keys may contain dots (height bands like '0.70-0.80')
        for j in range(len(parts), i, -1):
            k = '.'.join(parts[i:j])
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
                i = j
                break
        else:
            raise KeyError(dotted)
    return cur


worst = max(abs(get(R, k) - v) for k, v in calc.items())
check('residuals_final reproduced from anchors + M', worst < 1e-6, f'max |diff| {worst:.2e} m over {len(calc)} values')

cam = J['scene_camera']
P = cam['pose']
Tc = np.array(P['world_from_camera_opencv'])
Rc, C = Tc[:3, :3], Tc[:3, 3]
check('camera rotation orthonormal, det +1', np.allclose(Rc.T @ Rc, np.eye(3), atol=1e-9) and abs(np.linalg.det(Rc) - 1) < 1e-9)
check('blender = opencv @ diag(1,-1,-1,1)', np.allclose(np.array(P['world_from_camera_blender']), Tc @ np.diag([1, -1, -1, 1])))
check('OpenCV x axis ~ +X (camera looks at the wall, +Y)', Rc[0, 0] > 0.99 and Rc[1, 2] > 0.5, f'x_cam={Rc[:, 0].round(3)}, z_cam={Rc[:, 2].round(3)}')
check('position / optical axis / heights consistent', np.allclose(P['position_m'], C) and np.allclose(P['optical_axis_world'], Rc[:, 2])
      and abs(P['height_above_desk_top_m'] - (C[2] - top)) < 1e-9 and abs(P['horizontal_distance_to_back_edge_m'] - (y1 - C[1])) < 1e-9)
check('pitch_down_deg = asin(-z_cam.z)', abs(P['pitch_down_deg'] - math.degrees(math.asin(-Rc[2, 2]))) < 1e-9)
K = np.array(cam['K'])
check('K fields and FOV consistent', K[0, 0] == cam['fx'] and K[1, 2] == cam['cy'] and abs(cam['hfov_deg'] - math.degrees(2 * math.atan(480 / K[0, 0]))) < 1e-9)
check('K equals the ZED SDK left-eye K (zed_metric)', np.allclose(K, zn['K']))
lens = K[0, 0] * 36 / 960
check('Blender lens in note', f'focal {lens:.4f} mm' in P['blender_note'], f'{lens:.4f} mm')
# desk box back corners -> ZED image crease corners (0928 line intersections)
for name, Pw, uv in (('back-left', [x0, y1, top], zn['bl']), ('back-right', [x1, y1, top], zn['br'])):
    q = Rc.T @ (np.array(Pw) - C)
    px = K[:2, :2] @ (q[:2] / q[2]) + K[:2, 2]
    e = float(np.linalg.norm(px - uv))
    check(f'desk {name} corner reprojects onto the ZED image corner', e < 0.5, f'{e:.2f} px')

for k, arm in J['arms'].items():
    W = np.array(arm['world_from_base'])
    c, s = math.cos(math.radians(arm['yaw_deg'])), math.sin(math.radians(arm['yaw_deg']))
    check(f'{k} world_from_base = [Rz(yaw) | base]', np.allclose(W[:3, :3], [[c, -s, 0], [s, c, 0], [0, 0, 1]]) and np.allclose(W[:3, 3], arm['base_position_m']))
L, Rr = np.array(J['arms']['left_arm']['base_position_m']), np.array(J['arms']['right_arm']['base_position_m'])
check('arm_spacing consistent', abs(J['arm_spacing']['xy_distance_m'] - np.linalg.norm((Rr - L)[:2])) < 1e-9)
check('desk corners / size / planes consistent',
      np.allclose(box['top_corners_m'], [[x0, y0, top], [x1, y0, top], [x1, y1, top], [x0, y1, top]]) and abs(x1 - x0 - 1.52) < 1e-9
      and abs(y1 - y0 - 0.76) < 1e-9 and abs(J['planes']['desk_top']['d'] + top) < 1e-12
      and abs(J['planes']['back_wall']['d'] - (y1 + box['back_edge_to_wall_gap_m'])) < 1e-9)

# ---- tape_0929_b: back wall with the taped gap
tb = J['inputs']['tape_0929_b']
gap, bwp = box['back_edge_to_wall_gap_m'], J['planes']['back_wall']
check('back wall vertical, normal -Y', np.allclose(bwp['n'], [0, -1, 0]))
check('desk gap = tape_0929_b gap, within its bounds', abs(gap - tb['desk_back_edge_to_wall_gap_m']) < 1e-12
      and box['back_edge_to_wall_gap_bounds_m'][0] <= gap <= box['back_edge_to_wall_gap_bounds_m'][1], f'{gap * 1000:.1f} mm')
yw = bwp['d']
check('camera horizontal distance to the back wall consistent', abs(P['horizontal_distance_to_back_wall_m'] - (yw - C[1])) < 1e-9)


def proj(Pw):
    q = (np.atleast_2d(Pw) - C) @ Rc
    return q[:, :2] / q[:, 2:3] @ K[:2, :2].T + K[:2, 2]


# the ZED wood/wall hue boundary must sit on the projected desk back edge (the pose is anchored on the real edge, not on the wall)
for tag, path in (('0928', common.ZED_EP / 'rgb/scene_camera/0000000000.png'), ('0929', common.ZED_0929)):
    Sat = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2HSV)[..., 1].astype(float)
    dv = []
    for xx in np.linspace(x0 + 0.25, x1 - 0.25, 60):
        (u, ve), = proj([xx, y1, top])
        ui = int(round(u))
        vv = np.arange(int(ve) - 14, int(ve) + 8)
        sc = np.median(Sat[vv[0]:vv[-1] + 1, ui - 1:ui + 2], 1)
        half = 0.5 * (np.median(sc[vv < ve - 8]) + np.median(sc[vv > ve + 3]))
        k = np.nonzero((sc[:-1] < half) & (sc[1:] >= half) & (vv[:-1] > ve - 6))[0]
        if len(k):
            dv.append(vv[k[0]] + (half - sc[k[0]]) / (sc[k[0] + 1] - sc[k[0]]) - ve)
    check(f'ZED {tag}: wood/wall hue boundary on the projected desk back edge', len(dv) > 40 and abs(np.median(dv)) < 1.0,
          f'median {np.median(dv):+.2f} px, {len(dv)} columns (1 px ~ 3.5 mm)')
band = [proj([xx, y1, top])[0, 1] - proj([xx, yw, top])[0, 1] for xx in (x0 + 0.3, x1 - 0.3)]
check('35 mm gap spans 8-12 px above the edge in the ZED image', all(8 < b < 12 for b in band), ', '.join(f'{b:.1f} px' for b in band))
# ZED wall-depth check reproduced against the plane (same rules as tape_b.py)
zm = json.load(open(common.WORK / 'zed_metric.json'))
medz = np.load(common.WORK / 'zed_depth_median.npy') * (mf.TAPE['length'] / zm['desk_length_back_edge_m'])
vv_, uu_ = np.mgrid[0:600, 0:960]
rw = np.dstack([(uu_ - K[0, 2]) / K[0, 0], (vv_ - K[1, 2]) / K[1, 1], np.ones_like(uu_, float)]) @ Rc.T
tw = (yw - C[1]) / rw[..., 1]
hw = C + tw[..., None] * rw
ow = (tw > 0) & (hw[..., 0] > x0 + 0.1) & (hw[..., 0] < x1 - 0.1) & np.isfinite(medz)
wchk = cam['checks']['zed_depth_vs_metric_back_wall']
m_all, m_near = ow & (hw[..., 2] > top + 0.05), ow & (hw[..., 2] > top) & (hw[..., 2] < top + 0.10)
d_all, d_near = float(np.median((medz - tw)[m_all]) * 1000), float(np.median((medz - tw)[m_near]) * 1000)
check('ZED wall-depth check reproduced against planes.back_wall', abs(wchk['plane_y_m'] - yw) < 1e-12 and abs(wchk['median_mm'] - d_all) < 0.01
      and abs(wchk['near_edge_0_10cm']['median_mm'] - d_near) < 0.01, f'median {d_all:.1f} mm, 0-10 cm above the desk {d_near:.1f} mm')
check('ZED depth next to the desk (0-10 cm) on the tape plane within 10 mm', abs(d_near) < 10, f'{d_near:+.1f} mm')

# ---- tape_0929_b: rail
rl = J['rail']
r0, r1 = rl['x_extent_recommended_m']
check('rail length = tape_0929_b', abs(r1 - r0 - tb['rail_length_m']) < 1e-9 and abs(rl['length_recommended_m'] - tb['rail_length_m']) < 1e-12
      and abs(rl['length_tape_m'] - tb['rail_length_m']) < 1e-12, f'{r1 - r0:.4f} m')
bl_, br_ = L[0], Rr[0]
check('rail ends outside the arm bases as stated (0.1-0.3 m each)', np.allclose(rl['ends_outside_arm_bases_m'], [bl_ - r0, r1 - br_])
      and all(0.1 < o < 0.3 for o in rl['ends_outside_arm_bases_m']), ', '.join(f'{o:.4f}' for o in rl['ends_outside_arm_bases_m']))
check('rail inside the desk; front within 1.5 cm of the desk front edge', x0 < r0 and r1 < x1 and abs(rl['y_extent_m'][0] - y0) < 0.015
      and rl['y_extent_m'][1] < y1 and rl['top_z_m'] > top, f'x {r0:.4f}..{r1:.4f}, y {rl["y_extent_m"][0]:.3f}..{rl["y_extent_m"][1]:.3f}')
print('RESULT', 'FAIL: ' + ', '.join(fails) if fails else 'all checks pass')
sys.exit(1 if fails else 0)
