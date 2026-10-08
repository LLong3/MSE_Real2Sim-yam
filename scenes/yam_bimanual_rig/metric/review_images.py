"""Review images for the metric frame -> ~/robot/markdowns/bimanual/1_media/alignment/.

01_zed_overlay.jpg     real 09-29 scene-camera image with the metric desk box, back-wall grid, rail and MJCF arms
                       (top), and blended with the metric reference render (bottom); final camera pose.
02_side_top_view.jpg   side (Y-Z) and top (X-Y) views of the metric frame: floor, desk, wall, rail, arms, pole, camera.
04_scale_evidence.jpg  implied reference->metric scale per axis and anchor.
05_back_wall_check.jpg back wall with the 35 mm tape gap (tape_b.py evidence): ZED edge close-up and luminance across the edge,
                       wall position vs height (ZED neural depth, ZED classical stereo, reference), the gap band in a close source
                       view, the gap read per view, and the reference desk-top colour near its back boundary.
Usage: review_images.py [01 02 04 05]  (default: all)
(03_floor_near_desk.jpg is written by floor_check.py.)
"""
import json
import sys

import cv2
import matplotlib
import numpy as np

import common
import metric_frame as mf
import render
from arm_icp import rz, sample_surface

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

J = json.load(open(common.HERE / 'metric_frame.json'))
M = np.array(J['transform_reference_to_metric']['M'])
cam = J['scene_camera']
K = np.array(cam['K'])
T_wc = np.array(cam['pose']['world_from_camera_opencv'])
R_wc, C = T_wc[:3, :3], T_wc[:3, 3]
box = J['desk']
x0, x1 = box['box_x_m']
y0, y1 = box['box_y_m']
TOP = box['top_z_m']
YW = J['planes']['back_wall']['d']   # back wall plane y (desk back edge + gap)
GAP = box['back_edge_to_wall_gap_m']
rail = J['rail']
rx0, rx1 = rail['x_extent_recommended_m']
ry0, ry1 = rail['y_extent_m']
rtop = rail['top_z_m']
arms = J['arms']
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#0b0b0b', '#52514e'
rng = np.random.default_rng(0)
arm_surf = sample_surface(common.WORK / 'yam_q0_open0.0475.npz', 250000)


def arm_world(k):
    a = arms[k]
    return arm_surf @ rz(np.radians(a['yaw_deg'])).T + np.array(a['base_position_m'])


def project(P):
    q = (P - C) @ R_wc
    ok = q[:, 2] > 0.05
    uv = q[:, :2] / np.where(ok, q[:, 2], 1)[:, None] * K[[0, 1], [0, 1]] + K[:2, 2]
    return uv, ok


def draw_poly3d(img, P, color, thick=2, closed=False, n=60):
    pts = []
    segs = list(zip(P[:-1], P[1:])) + ([(P[-1], P[0])] if closed else [])
    for a, b in segs:
        t = np.linspace(0, 1, n)[:, None]
        uv, ok = project(a + t * (b - a))
        uv = uv[ok]
        for p, q in zip(uv[:-1], uv[1:]):
            cv2.line(img, tuple(np.round(p * 16).astype(int)), tuple(np.round(q * 16).astype(int)), color, thick, cv2.LINE_AA, shift=4)


def hex2bgr(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def overlay():
    real = cv2.imread(str(common.ZED_0929))
    a = real.copy()
    # back wall grid (vertical plane, desk back edge + gap); bottom line = the wall at desk-top height
    for x in np.arange(x0, x1 + 1e-6, 0.25):
        draw_poly3d(a, np.array([[x, YW, TOP], [x, YW, 2.0]]), hex2bgr(AQUA), 1)
    for z in (TOP, 0.9, 1.2, 1.5, 1.8):
        draw_poly3d(a, np.array([[x0 - 0.3 * (z > TOP), YW, z], [x1 + 0.3 * (z > TOP), YW, z]]), hex2bgr(AQUA), 1)
    # desk top rectangle and desk front face
    draw_poly3d(a, np.array([[x0, y0, TOP], [x1, y0, TOP], [x1, y1, TOP], [x0, y1, TOP]]), hex2bgr(YELLOW), 2, closed=True)
    # rail top outline
    draw_poly3d(a, np.array([[rx0, ry0, rtop], [rx1, ry0, rtop], [rx1, ry1, rtop], [rx0, ry1, rtop]]), hex2bgr(BLUE), 1, closed=True)
    # MJCF arms: silhouette contours
    for k in arms:
        uv, ok = project(arm_world(k))
        m = np.zeros((600, 960), np.uint8)
        ui = np.rint(uv[ok]).astype(int)
        g = (ui[:, 0] >= 0) & (ui[:, 0] < 960) & (ui[:, 1] >= 0) & (ui[:, 1] < 600)
        m[ui[g, 1], ui[g, 0]] = 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(a, cnt, -1, hex2bgr(MAGENTA), 2, cv2.LINE_AA)
    for i, (txt, col) in enumerate((('desk top 1.52 x 0.76 at z 0.72', YELLOW),
                                    (f'back wall plane (desk back edge + {GAP * 1000:.0f} mm), 0.25 m grid', AQUA),
                                    ('rail top (tape 1.00 m; outside this view)', BLUE), ('MJCF arms q=0 at registered bases', MAGENTA))):
        cv2.putText(a, txt, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(a, txt, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hex2bgr(col), 1, cv2.LINE_AA)
    # blended metric reference render
    d = np.load(common.WORK / 'clean_cloud.npz')
    P = mf.apply(M, d['points'])
    far = np.linalg.norm(P - C, axis=1) > 0.3
    rgb, _, _ = render.render(P[far], d['colors'][far], R_wc.T, -R_wc.T @ C, K, radius=1)
    b = (0.5 * real + 0.5 * cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)).astype(np.uint8)
    for txt in ('real 09-29 image blended 50/50 with the metric-transformed Pi3X reference',):
        cv2.putText(b, txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(b, txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.concatenate([a, b], 0)
    cv2.imwrite(str(common.MEDIA / '01_zed_overlay.jpg'), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


def side_top():
    V, MV, Cc = common.points()
    P = mf.apply(M, np.asarray(V)[MV][::3].astype(np.float64))
    col = np.asarray(Cc)[MV][::3] / 255.0
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2), dpi=90, gridspec_kw=dict(width_ratios=[1, 1.25]))
    # --- side view: slice through the left arm
    xl = arms['left_arm']['base_position_m'][0]
    s = (np.abs(P[:, 0] - xl) < 0.10) & (P[:, 1] > 0.6) & (P[:, 1] < 2.3) & (P[:, 2] < 2.1)
    ax[0].scatter(P[s, 1], P[s, 2], s=0.3, c=col[s], alpha=0.35, rasterized=True)
    ax[0].axhline(0, color=INK, lw=1.2)
    ax[0].add_patch(plt.Rectangle((y0, TOP - 0.03), y1 - y0, 0.03, fc=YELLOW, ec=INK, lw=1, alpha=0.8, label='desk top (tape)'))
    ax[0].plot([YW, YW], [0, 2.1], color=AQUA, lw=2, label=f'back wall (vertical, desk back edge + {GAP * 1000:.0f} mm, tape)')
    ax[0].annotate(f'{GAP * 1000:.0f} mm gap', xy=(0.5 * (y1 + YW), TOP), xytext=(1.93, TOP + 0.14), fontsize=8, color=INK,
                   arrowprops=dict(arrowstyle='->', color=INK, lw=0.8))
    ax[0].add_patch(plt.Rectangle((ry0, TOP), ry1 - ry0, rtop - TOP, fc=BLUE, ec=BLUE, lw=1, label='rail'))
    aw = arm_world('left_arm')
    ax[0].scatter(aw[::20, 1], aw[::20, 2], s=0.3, c=MAGENTA, alpha=0.5, rasterized=True)
    ax[0].plot([], [], 'o', color=MAGENTA, ms=5, label='MJCF left arm (q=0)')
    px, py = rail['pole_xy_m']
    ax[0].plot([py, py], [rtop, C[2] + 0.12], color=INK2, lw=3, alpha=0.7, label='pole (reference)')
    ax[0].plot(C[1], C[2], 'o', color=ORANGE, ms=8, label='ZED left eye')
    fwd = R_wc[:, 2]
    upc = -R_wc[:, 1]
    vf = np.radians(cam['vfov_deg'] / 2)
    for sgn in (-1, 1):
        d = np.cos(vf) * fwd + sgn * np.sin(vf) * upc
        tt = (TOP - C[2]) / d[2] if d[2] < 0 else 1.0
        e = C + min(tt, 1.6) * d
        ax[0].plot([C[1], e[1]], [C[2], e[2]], color=ORANGE, lw=1, ls='--')
    ax[0].set_xlim(0.9, 2.25); ax[0].set_ylim(-0.05, 1.95); ax[0].set_aspect('equal')
    ax[0].set_xlabel('Y (m, toward the back wall)'); ax[0].set_ylabel('Z (m)')
    ax[0].set_title(f'Side view, slice |X - left arm| < 0.10 m; reference points in metric frame', fontsize=10)
    ax[0].legend(fontsize=7.5, loc='upper left', frameon=False)
    ax[0].grid(alpha=0.25)
    # --- top view
    s = (P[:, 2] > 0.2) & (P[:, 2] < 1.2) & (P[:, 0] > -3.5) & (P[:, 0] < -1.0) & (P[:, 1] > 0.9) & (P[:, 1] < 2.3)
    ax[1].scatter(P[s, 0], P[s, 1], s=0.2, c=col[s], alpha=0.25, rasterized=True)
    ax[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=YELLOW, lw=2, label='desk 1.52 x 0.76'))
    ax[1].plot([x0 - 0.4, x1 + 0.4], [YW, YW], color=AQUA, lw=2, label=f'back wall (desk back edge + {GAP * 1000:.0f} mm)')
    ax[1].add_patch(plt.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fc=BLUE, ec=BLUE, alpha=0.35, label=f'rail {rx1 - rx0:.2f} m (tape)'))
    for k in arms:
        aw = arm_world(k)
        ax[1].scatter(aw[::25, 0], aw[::25, 1], s=0.3, c=MAGENTA, alpha=0.5, rasterized=True)
        b = arms[k]['base_position_m']
        ax[1].plot(b[0], b[1], '+', color=INK, ms=10)
        ax[1].annotate(f"{k.split('_')[0]}: ({b[0]:.3f}, {b[1]:.3f})\nyaw {arms[k]['yaw_deg']:.1f} deg", (b[0], b[1]),
                       xytext=(b[0] - 0.13, 1.05), fontsize=8, color=INK)
    ax[1].plot(px, py, 's', color=INK2, ms=6, label='pole')
    ax[1].plot(C[0], C[1], 'o', color=ORANGE, ms=8, label='ZED left eye')
    hf = np.radians(cam['hfov_deg'] / 2)
    f2 = fwd[:2] / np.linalg.norm(fwd[:2])
    r2 = R_wc[:2, 0] / np.linalg.norm(R_wc[:2, 0])
    for sgn in (-1, 1):
        d = np.cos(hf) * f2 + sgn * np.sin(hf) * r2
        ax[1].plot([C[0], C[0] + 0.7 * d[0]], [C[1], C[1] + 0.7 * d[1]], color=ORANGE, lw=1, ls='--')
    ax[1].set_xlim(-3.3, -1.25); ax[1].set_ylim(0.95, 2.25); ax[1].set_aspect('equal')
    ax[1].set_xlabel('X (m, along the back wall)'); ax[1].set_ylabel('Y (m)')
    ax[1].set_title('Top view, 0.2 < Z < 1.2 m; metric frame', fontsize=10)
    ax[1].legend(fontsize=7.5, loc='upper right', frameon=False, ncol=2)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(common.MEDIA / '02_side_top_view.jpg', dpi=90, pil_kwargs=dict(quality=85))


def scale_evidence():
    fm = json.load(open(common.WORK / 'fit_metric.json'))
    a = fm['reference_anchors']
    z2 = json.load(open(common.WORK / 'arm_zed_fit2.json'))
    MA = np.array(fm['candidates']['A_per_axis']['M'])
    ref = {r['arm']: (np.array(r['base']) - MA[:3, 3]) / np.diag(MA[:3, :3]) for r in fm['candidates']['A_per_axis']['arms'] if r['model'] == 'no_fingers'}
    icp_raw = {k: json.load(open(common.WORK / f'arm_icp_{k}.json'))[k] for k in ('left_arm', 'right_arm')}
    ref_sp = abs(ref['right_arm'][0] - ref['left_arm'][0])
    zsp = [abs(z2[t]['right_arm']['base_world'][0] - z2[t]['left_arm']['base_world'][0]) for t in ('0928', '0929')]
    top_ref = a['desk_top_z_at_centre']
    zc = J['residuals_final']['camera_height_above_desk_m']
    cam_ref_body = (fm['camera_body_ref_median'][2] - top_ref) - 0.01 / 1.09
    ev = [
        ('X', 'desk length 1.52 / 1.42-1.44', [1.52 / 1.44, 1.52 / a['desk_length']]),
        ('X', 'arm spacing, ZED / reference', [z / ref_sp for z in zsp]),
        ('Y', 'desk depth 0.76 / 0.756-0.77', [0.76 / 0.77, 0.76 / a['desk_depth_front_to_wall']]),
        ('Y', 'MJCF arm length (ICP per-axis)', [1 / icp_raw[k][f'opening_{o}']['axis_scale']['scale_base_xyz'][0] for k in icp_raw for o in ('0', '0.0475')]),
        ('Z', 'desk top - floor 0.72 / 0.661 (58 views)', [0.72 / a['desk_minus_floor_per_view_median']]),
        ('Z', 'camera height: ZED / reference camera body', [zc['zed_depth'] / cam_ref_body]),
        ('Z', 'camera height: ZED / reference PnP', [zc['zed_depth'] / ((zc['pnp_reference']) / 1.09)]),
        ('Z', 'MJCF arm top / reference arm top', [0.2705 / (J['arms'][k]['free_z_fit']['arm_top_above_base_m'] / 1.09) for k in ('left_arm', 'right_arm')]),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=90)
    colors = {'X': BLUE, 'Y': ORANGE, 'Z': AQUA}
    for i, (axis, lab, vals) in enumerate(ev):
        yv = len(ev) - 1 - i
        ax.plot([min(vals), max(vals)], [yv, yv], color=colors[axis], lw=2)
        ax.plot(vals, [yv] * len(vals), 'o', color=colors[axis], ms=7)
    for v, lab in ((1.0, 'scale 1'), (1.10, 'blind 1.10')):
        ax.axvline(v, color=INK2, lw=1, ls=':')
        ax.text(v, len(ev) - 0.35, lab, ha='center', fontsize=8, color=INK2)
    for axis, v in zip('XYZ', J['transform_reference_to_metric']['S']):
        ax.plot([v, v], [-0.5, len(ev) - 0.6], color=colors[axis], lw=1, alpha=0.5)
    ax.text(1.062, -0.85, 'chosen sx 1.06, sy 1.00, sz 1.09 (thin lines)', fontsize=8, color=INK2, ha='center')
    ax.set_xlim(0.97, 1.15); ax.set_ylim(-1.1, len(ev) - 0.1)
    ax.set_yticks([len(ev) - 1 - i for i in range(len(ev))], [f'{e[0]}: {e[1]}' for e in ev], fontsize=9)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('implied metric / reference scale')
    ax.set_title('Reference-to-metric scale by axis and anchor: anisotropic, not a floor-plane error', fontsize=10)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(common.MEDIA / '04_scale_evidence.jpg', dpi=90, pil_kwargs=dict(quality=88))


def back_wall():
    ev = json.load(open(common.WORK / 'tape_b/evidence.json'))
    wc = json.load(open(common.WORK / 'wall_check.json'))
    se = json.load(open(common.WORK / 'stereo_edges.json'))
    vs = json.load(open(common.WORK / 'video_slot_check.json'))
    fig, ax = plt.subplots(2, 3, figsize=(18, 10.2), dpi=90)

    def clean(a):
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        a.grid(alpha=0.25)

    # (a) ZED close-up of the desk back edge
    img = cv2.cvtColor(cv2.imread(str(common.ZED_0929)), cv2.COLOR_BGR2RGB)
    u0, u1, v0, v1 = 380, 640, 332, 378
    a = ax[0, 0]
    a.imshow(img[v0:v1, u0:u1], extent=(u0 - 0.5, u1 - 0.5, v1 - 0.5, v0 - 0.5), interpolation='bicubic')
    xs = np.linspace(x0, x1, 80)
    e = project(np.c_[xs, np.full_like(xs, y1), np.full_like(xs, TOP)])[0]
    w = project(np.c_[xs, np.full_like(xs, YW), np.full_like(xs, TOP)])[0]
    a.plot(e[:, 0], e[:, 1], color=YELLOW, lw=1.5, label='desk back edge (metric box, final pose)')
    a.plot(w[:, 0], w[:, 1], color=AQUA, lw=1.5, ls='--', label=f'wall at desk-top height, {GAP * 1000:.0f} mm behind')
    a.set_xlim(u0 - 0.5, u1 - 0.5); a.set_ylim(v1 - 0.5, v0 - 0.5); a.set_aspect('auto')
    a.legend(fontsize=7.5, loc='lower left', framealpha=0.85)
    ze = ev['zed_edge']
    a.set_title(f"ZED 09-29 (px): the wood ends on the desk back edge (hue boundary "
                f"{ze['images']['0929']['hue_boundary_minus_fitted_edge_px']['median']:+.1f} px).\n"
                f"Between the lines: the wall seen down the gap, in shadow ({np.median(ze['slot_band_px_35mm']):.0f} px)", fontsize=9.5, color=INK)
    # (b) ZED luminance across the edge
    a = ax[0, 1]
    dv = np.arange(-25, 7)
    for tag, colr in (('0928', ORANGE), ('0929', BLUE)):
        pr = np.array(ze['images'][tag]['luminance_above_edge']['profile_dv_minus25_to_plus6'])
        a.plot(-dv, pr, '-o', color=colr, ms=3, lw=1.5, label=f'ZED {tag} (median over columns u 300-740)')
    a.axvspan(0, np.median(ze['slot_band_px_35mm']), color=AQUA, alpha=0.15, lw=0, label=f'{GAP * 1000:.0f} mm gap band')
    a.axvline(0, color=INK, lw=1)
    a.text(-0.7, a.get_ylim()[1] - 2, 'desk (wood)', ha='left', va='top', fontsize=8, color=INK2)
    a.text(0.7, a.get_ylim()[1] - 2, 'wall', ha='right', va='top', fontsize=8, color=INK2)
    a.set_xlabel('image rows above the desk back edge (px)'); a.set_ylabel('luminance')
    a.set_title(f"Luminance across the edge: the wall darkens toward the edge over the gap band.\n"
                f"Ray past the edge drops {ze['camera_ray_below_horizontal_at_edge_deg']:.0f} deg: the gap shows wall down to "
                f"{ze['slot_depth_seen_below_top_mm']:.0f} mm below the desk top, no black slot", fontsize=9.5, color=INK)
    a.invert_xaxis()
    a.legend(fontsize=7.5, loc='lower left', frameon=False)
    clean(a)
    # (c) wall position vs height
    a = ax[0, 2]
    st = np.array(ev['zed_depth_step']['table'])
    st = st[st[:, 0] <= 0]
    a.plot(st[:, 1], st[:, 2] / 1000, '-', color=ORANGE, lw=2.2, label='ZED neural depth, rows just above the edge')
    tab = wc['wall_y_mm_by_column_and_height']['table']
    bands = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.90)]
    zc = np.array([(z0_ + z1_) / 2 for z0_, z1_ in bands])
    vals = np.median([[tab[c][f'z{z0_:.2f}-{z1_:.2f}'][0] for z0_, z1_ in bands] for c in tab], 0)
    a.plot(vals, zc, 'o--', color=ORANGE, lw=1, ms=4, alpha=0.7, label='ZED neural depth on the wall (median by band)')
    eh = np.array(se['edges']['glass_wall_junction']['neural_plane']['by_height'])
    a.plot(eh[:, 1], eh[:, 0], '-s', color=BLUE, lw=1.5, ms=4, label='ZED classical stereo: painted wall end')
    rw_ = np.array(ev['ref_wall']['table'])
    a.plot(rw_[:, 1], rw_[:, 0], '-D', color=MAGENTA, lw=1.5, ms=4, label='Pi3X reference wall (metric frame)')
    a.axvline(GAP * 1000, color=AQUA, lw=3, alpha=0.8, label=f'back wall plane, tape {GAP * 1000:.0f} mm')
    a.axvline(0, color=INK2, lw=1, ls=':')
    a.text(-1, 1.13, 'old "flush" plane', fontsize=8, color=INK2, ha='right')
    a.add_patch(plt.Rectangle((-60, -0.03), 60, 0.03, fc=YELLOW, ec=INK, lw=0.8, alpha=0.8))
    a.text(-30, -0.05, 'desk top', ha='center', va='top', fontsize=8, color=INK)
    a.set_xlim(-60, 175); a.set_ylim(-0.08, 1.2)
    a.set_xlabel('distance behind the desk back edge (mm)'); a.set_ylabel('height above the desk top (m)')
    a.set_title('Wall behind the desk back edge. ZED depth and stereo sit on the 35 mm plane\n'
                'next to the desk; Pi3X pulls the lower wall onto the desk edge', fontsize=9.5, color=INK)
    a.legend(fontsize=7.2, loc='upper right', frameon=False)
    clean(a)
    # (d) close source view with the 35 mm band
    f = '1083'
    rws = vs['per_frame'][f]
    im2 = cv2.cvtColor(cv2.imread(str(common.WORK / f'frames_full/{int(f):06d}.jpg')), cv2.COLOR_BGR2RGB)
    J0 = np.array([r['junction_px'] for r in rws])
    c0 = J0[len(J0) // 2]
    tl = np.polyfit(J0[:, 0], J0[:, 1], 1)
    dd = np.array([1.0, tl[0]]); dd /= np.linalg.norm(dd)
    n = np.array([-dd[1], dd[0]])
    if n[1] > 0:
        n = -n
    xa, ya = int(c0[0] - 150), int(c0[1] - 100)
    crop = im2[max(ya, 0):ya + 200, max(xa, 0):xa + 300]
    a = ax[1, 0]
    a.imshow(crop, extent=(xa, xa + crop.shape[1], ya + crop.shape[0], ya))
    off = 3.5 * np.median([r['slot_band_px_for_gap']['10mm'] for r in rws])
    for o, colr, lab in ((0, YELLOW, 'junction (wood edge)'), (off, AQUA, f'wall at desk-top height, {GAP * 1000:.0f} mm gap')):
        p0, q0 = c0 + o * n - 400 * dd, c0 + o * n + 400 * dd
        a.plot([p0[0], q0[0]], [p0[1], q0[1]], color=colr, lw=1.3, ls='-' if o == 0 else '--', label=lab)
    a.set_xlim(xa, xa + crop.shape[1]); a.set_ylim(ya + crop.shape[0], ya)
    a.set_axis_off()
    a.legend(fontsize=7.5, loc='lower left', framealpha=0.85)
    a.set_title(f'Source frame {f} (close, from above): the darker band between the\n'
                'lines is the wall seen down the gap; it was read as lighting', fontsize=9.5, color=INK)
    # (e) gap read per view
    a = ax[1, 1]
    vb = ev['video_band']['per_frame']
    keys = list(vb)
    thin = [vb[k]['thin_line_cm_if_slot'] for k in keys]
    broad = [vb[k]['dark_band_cm_if_slot'] for k in keys]
    yy = np.arange(len(keys))[::-1]
    a.barh(yy + 0.18, broad, height=0.34, color=AQUA, label='darker band at the junction (half depth)')
    a.barh(yy - 0.18, thin, height=0.34, color=BLUE, label='thin line inside it (the old reading)')
    a.axvline(GAP * 100, color=INK, lw=1.5)
    a.text(GAP * 100 + 0.08, yy[0] + 0.55, f'tape {GAP * 100:.1f} cm', fontsize=8.5, color=INK)
    a.set_yticks(yy, [f'video {k}' for k in keys], fontsize=8.5)
    a.tick_params(axis='y', length=0)
    a.set_xlim(0, 7.5); a.set_xlabel('gap implied if the feature were the slot (cm)')
    a.legend(fontsize=7.5, loc='lower right', frameon=False)
    a.set_title('Gap read per close view: the darker band contains the 35 mm slot;\n'
                'the thin line is only the deepest shadow at the edge', fontsize=9.5, color=INK)
    clean(a)
    # (f) reference desk-top colour near its back boundary
    a = ax[1, 2]
    rb = ev['ref_desk_back']
    t = np.array(rb['table'])
    t = t[t[:, 1] >= 200]
    dy = (t[:, 0] + (rb['p99_boundary_y'] - y1) * 1000)
    a.plot(dy, t[:, 2], '-o', color=YELLOW, ms=4, lw=2, label='wood colour')
    a.plot(dy, t[:, 3], '-o', color=INK2, ms=4, lw=2, label='wall-paint colour')
    a.axvline(rb['wood_edge_minus_desk_back_edge_mm'], color=ORANGE, lw=1.2, ls='--')
    a.text(rb['wood_edge_minus_desk_back_edge_mm'] - 1, 0.5, 'reference wood edge', rotation=90, ha='right', va='center', fontsize=8, color=ORANGE)
    a.axvline((rb['p99_boundary_y'] - y1) * 1000, color=INK2, lw=1, ls=':')
    a.text((rb['p99_boundary_y'] - y1) * 1000 + 1, 0.5, 'reference "desk back"\n(= its wall line)', ha='left', va='center', fontsize=8, color=INK2)
    a.set_xlabel('y - desk back edge (mm); reference points at desk-top height')
    a.set_ylabel('fraction of points'); a.set_ylim(-0.03, 1.05)
    a.legend(fontsize=7.5, loc='upper left', frameon=False)
    a.set_title(f"Pi3X: the last {-rb['wood_edge_minus_p99_mm']:.0f} mm of its 'desk top' are wall paint: the wall seen\n"
                'down the gap, flattened to desk height ("desk-top points reach the wall")', fontsize=9.5, color=INK)
    clean(a)
    fig.suptitle(f'Back wall: {GAP * 1000:.0f} mm gap (tape 09-29). The earlier "flush" reading was wrong: from above the gap looks like '
                 'shadowed wall, and Pi3X flattened it into the desk', fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(common.MEDIA / '05_back_wall_check.jpg', dpi=90, pil_kwargs=dict(quality=88))


if __name__ == '__main__':
    common.MEDIA.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ['01', '02', '04', '05']
    for k, fn in (('01', overlay), ('02', side_top), ('04', scale_evidence), ('05', back_wall)):
        if k in todo:
            fn()
    print('written', sorted(p.name for p in common.MEDIA.iterdir()))
