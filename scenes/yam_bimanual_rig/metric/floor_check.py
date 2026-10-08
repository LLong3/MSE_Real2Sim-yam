"""Floor-near-desk diagnosis in the aligned reference (predicted m): is the global floor plane wrong at the desk?

- Floor = SAM 3.1 floor mask & measurement-valid pixels, |z| < 0.15 (aligned frame, global floor z=0).
- 10 cm height map of the floor, local plane fits near the desk and far from it.
- Per view: desk-top height minus floor height near the desk, both from that view's own point map.
Writes work/floor_check.json and the review plot 1_media/alignment/03_floor_near_desk.jpg.
"""
import matplotlib
import numpy as np

import common

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

V, MV, _ = common.points()
M = common.masks()
fl = M['floor'] & MV
P = np.asarray(V)[fl]
slot = np.nonzero(fl)[0]
sel = np.abs(P[:, 2]) < 0.15
P, slot = P[sel], slot[sel]
X, Y, Z = P.T

# desk footprint in the reference (from the multi-view consistent cloud, desk_edges in fit_metric.py)
DX, DY = (-2.965, -1.556), (1.322, 2.088)
near = (X > -3.15) & (X < -1.40) & (Y > 1.0) & (Y < 2.1)
under = (X > DX[0] + 0.05) & (X < DX[1] - 0.05) & (Y > DY[0] + 0.05) & (Y < DY[1] - 0.02)
far = ~((X > -3.6) & (X < -1.0) & (Y > 0.6))
out = dict(n_floor_points=int(len(P)))


def stats(m):
    z = Z[m]
    n, d, inl = common.fit_plane(P[m], thr=0.01)
    if n[2] < 0:
        n, d = -n, -d
    return dict(n=int(m.sum()), median_mm=float(np.median(z) * 1000), p10_mm=float(np.percentile(z, 10) * 1000),
                p90_mm=float(np.percentile(z, 90) * 1000),
                local_plane_tilt_deg=float(np.degrees(np.arccos(n[2]))),
                local_plane_z_at_desk_centre_mm=float((-d - n[0] * -2.26 - n[1] * 1.70) / n[2] * 1000))


out['near_desk'] = stats(near)
out['under_desk'] = stats(under)
out['away_from_desk'] = stats(far)

# per-view desk-top minus local floor, both in that view's own points
desk_n = np.array([0.008, 0.0027, 1.0]); desk_n /= np.linalg.norm(desk_n)
per_view = []
for s in range(64):
    Ps = np.asarray(V[s])[MV[s]]
    Xs, Ys, Zs = Ps.T
    dsel = (Xs > -2.9) & (Xs < -1.65) & (Ys > 1.45) & (Ys < 2.0) & (Zs > 0.6) & (Zs < 0.7)
    fsel = (M['floor'][s] & MV[s])[MV[s]] & (Xs > -3.15) & (Xs < -1.4) & (Ys > 1.0) & (Ys < 2.1) & (np.abs(Zs) < 0.15)
    if dsel.sum() > 500 and fsel.sum() > 200:
        per_view.append([s, float(np.median(Zs[dsel])), float(np.median(Zs[fsel])), int(dsel.sum()), int(fsel.sum())])
pv = np.array(per_view)
out['per_view_desk_minus_floor'] = dict(
    views=int(len(pv)), columns='slot, desk_top_z, floor_z_near_desk, n_desk, n_floor',
    desk_minus_floor_median_m=float(np.median(pv[:, 1] - pv[:, 2])),
    desk_minus_floor_min_m=float((pv[:, 1] - pv[:, 2]).min()), desk_minus_floor_max_m=float((pv[:, 1] - pv[:, 2]).max()),
    rows=pv.round(4).tolist())
common.save_json(common.WORK / 'floor_check.json', out)
print({k: v for k, v in out.items() if k != 'per_view_desk_minus_floor'})
print('per view desk-floor', out['per_view_desk_minus_floor']['desk_minus_floor_median_m'],
      out['per_view_desk_minus_floor']['desk_minus_floor_min_m'], out['per_view_desk_minus_floor']['desk_minus_floor_max_m'],
      'views', len(pv))

# ---------------- plot
step = 0.1
xs = np.arange(-4.8, 0.61, step)
ys = np.arange(-0.7, 2.31, step)
Hm = np.full((len(ys), len(xs)), np.nan)
ix = np.floor((X - xs[0]) / step).astype(int)
iy = np.floor((Y - ys[0]) / step).astype(int)
ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
lin = iy[ok] * len(xs) + ix[ok]
order = np.argsort(lin)
lin_s, z_s = lin[order], Z[ok][order]
u, st, cnt = np.unique(lin_s, return_index=True, return_counts=True)
for k, s0, c in zip(u, st, cnt):
    if c >= 30:
        Hm.flat[k] = np.median(z_s[s0:s0 + c])

fig = plt.figure(figsize=(12, 10.5), dpi=90)
gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], width_ratios=[1.7, 1], hspace=0.28, wspace=0.22)
ax = fig.add_subplot(gs[0, :])
im = ax.imshow(Hm * 1000, origin='lower', extent=[xs[0], xs[-1] + step, ys[0], ys[-1] + step], cmap='RdBu_r',
               vmin=-60, vmax=60, interpolation='nearest')
ax.add_patch(plt.Rectangle((DX[0], DY[0]), DX[1] - DX[0], DY[1] - DY[0], fill=False, ec='#0b0b0b', lw=1.5))
ax.text(DX[0] + 0.02, DY[1] + 0.04, 'rig desk footprint', fontsize=9, color='#0b0b0b')
ax.add_patch(plt.Rectangle((-3.15, 1.0), 1.75, 1.1, fill=False, ec='#52514e', lw=1, ls='--'))
ax.text(-3.13, 0.88, 'near-desk window', fontsize=8, color='#52514e')
ax.set_xlabel('X (reference, predicted m)'); ax.set_ylabel('Y (m)')
ax.set_title('Floor height vs the global floor plane (median per 10 cm cell, mm); SAM 3.1 floor pixels', fontsize=11)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label('mm')
ax.set_aspect('equal')

ax2 = fig.add_subplot(gs[1, 0])
bins = np.arange(-3.6, -0.9, 0.1)
for m, lab, c in ((near, 'Y 1.0-2.1 (under / in front of desk)', '#2a78d6'), ((Y > 0.2) & (Y < 1.0) & (X > -3.6), 'Y 0.2-1.0 (room side)', '#eb6834')):
    med = [np.median(Z[m & (X >= b) & (X < b + 0.1)]) * 1000 if (m & (X >= b) & (X < b + 0.1)).sum() > 30 else np.nan for b in bins]
    ax2.plot(bins + 0.05, med, '-o', color=c, ms=4, lw=2, label=lab)
ax2.axhline(0, color='#52514e', lw=1)
ax2.axhline(-66, color='#e34948', lw=1.5, ls='--')
ax2.text(-3.55, -61, 'floor needed for a 0.72 m desk at scale 1 (-66 mm)', color='#0b0b0b', fontsize=8)
ax2.axvspan(DX[0], DX[1], color='#e8e8e4', zorder=0)
ax2.text(DX[0] + 0.02, 22, 'desk X span', fontsize=8, color='#52514e')
ax2.set_ylim(-80, 35); ax2.set_xlabel('X (m)'); ax2.set_ylabel('floor height (mm)')
ax2.set_title('Floor height along the desk (10 cm bins)', fontsize=10)
ax2.legend(fontsize=8, loc='lower right', frameon=False)

ax3 = fig.add_subplot(gs[1, 1])
dmf = (pv[:, 1] - pv[:, 2]) * 1000
ax3.plot(pv[:, 0], dmf, 'o', color='#2a78d6', ms=5)
ax3.axhline(720, color='#e34948', lw=1.5, ls='--')
ax3.text(1, 712, 'tape 0.72 m', fontsize=8, color='#0b0b0b', va='top')
ax3.set_ylim(620, 740)
ax3.set_xlabel('source view (slot)'); ax3.set_ylabel('desk top minus floor (mm)')
n_out = int((dmf < 620).sum())
ax3.set_title(f'Per view, own point map: median {np.median(dmf):.0f} mm' + (f' ({n_out} view < 620 mm not shown)' if n_out else ''), fontsize=10)
fig.savefig(common.MEDIA / '03_floor_near_desk.jpg', dpi=90, pil_kwargs=dict(quality=88))
print('saved')
