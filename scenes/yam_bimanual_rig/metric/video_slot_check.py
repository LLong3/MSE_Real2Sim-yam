"""Slot check in the source video: how wide would a desk-wall gap look, and how wide is the dark band?

For source frames that look down on the desk back edge, project (reference cameras.json, reference coordinates)
the desk back edge and the wall line at desk-top height behind it for gaps of 1/2/4/8.5 cm; a gap exposes the wall
face inside the slot between these two lines. Then measure the brightness profile across the actual junction
(full-res frame, strongest desk->wall gradient near the projected edge) and the width of its dark band.
Writes work/video_slot_check.json and work/wallcheck/video_slot_<frame>.jpg.
"""
import json

import cv2
import numpy as np

import common

cam = json.load(open(common.REF / 'alignment/cameras.json'))
fr = {f['source_frame']: f for f in cam['frames']}
gr = json.load(open(common.WORK / 'wall_gap_reference.json'))
J = json.load(open(common.HERE / 'metric_frame.json'))
M = np.array(J['transform_reference_to_metric']['M'])
Sx, Sy, Sz = np.diag(M)[:3]
tx, ty, tz = M[:3, 3]
# reference-frame desk back edge: p99 back boundary and local desk top per band (metric -> reference)
bands = [(r['x'], r['desk_back_y_p99'], r['desk_top_z']) for r in gr['per_x_band'] if -2.95 < r['x'] < -1.65]
s = 1920 / cam['processed_size_wh'][0]
out = {}
for f in (942, 962, 982, 1063, 1083, 1103):
    F = fr[f]
    c2w = np.array(F['c2w'])
    K = np.array(F['intrinsics']) * s
    K[2, 2] = 1
    w2c = np.linalg.inv(c2w)
    img = cv2.imread(str(common.WORK / f'frames_full/{f:06d}.jpg'))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g = cv2.GaussianBlur(g, (3, 3), 0.8)

    def pr(Pm):
        Pr = np.c_[(Pm[:, 0] - tx) / Sx, (Pm[:, 1] - ty) / Sy, (Pm[:, 2] - tz) / Sz]
        q = (w2c[:3, :3] @ Pr.T).T + w2c[:3, 3]
        ok = q[:, 2] > 0.1
        return (q[:, :2] / q[:, 2:3]) @ K[:2, :2].T + K[:2, 2], ok
    rows = []
    for x, yb, zt in bands:
        e, ok = pr(np.array([[x, yb, zt]]))
        if not ok[0] or not (60 < e[0, 0] < 1860 and 60 < e[0, 1] < 1020):
            continue
        e2, _ = pr(np.array([[x + 0.05, yb, zt]]))
        t = (e2 - e)[0]; t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])
        up, _ = pr(np.array([[x, yb, zt + 0.10]]))           # n must point from the desk to the wall (up the wall)
        if (up - e)[0] @ n < 0:
            n = -n
        band_px = {}
        for gap in (0.01, 0.02, 0.04, 0.085):
            w, _ = pr(np.array([[x, yb + gap, zt]]))
            band_px[f'{int(gap * 1000)}mm'] = round(float((w - e)[0] @ n), 1)
        # actual junction: strongest desk(bright/textured)->wall transition along n within +-40 px
        ts = np.arange(-40, 41, 0.5)
        pts = e[0] + ts[:, None] * n
        prof = np.array([cv2.getRectSubPix(g.astype(np.float32), (1, 1), (float(p[0]), float(p[1])))[0, 0] for p in pts])
        dprof = np.gradient(prof)
        k = int(np.argmax(np.abs(dprof[10:-10]))) + 10
        j0 = e[0] + ts[k] * n
        tt = np.arange(-20, 80.5, 1.0)                           # along n: negative = desk, positive = wall
        pp = np.array([cv2.getRectSubPix(g.astype(np.float32), (1, 1), (float(p[0]), float(p[1])))[0, 0] for p in j0 + tt[:, None] * n])
        wall_far = float(np.median(pp[(tt > 50)]))
        mn_i = int(np.argmin(pp[(tt >= -2) & (tt <= 40)])) + int(np.nonzero(tt >= -2)[0][0])
        mn = float(pp[mn_i])
        half = (wall_far + mn) / 2
        # width of the dark band: contiguous run below `half` around the minimum
        a = mn_i
        while a > 0 and pp[a - 1] < half:
            a -= 1
        b = mn_i
        while b < len(pp) - 1 and pp[b + 1] < half:
            b += 1
        rows.append(dict(x=x, junction_px=[round(float(j0[0]), 1), round(float(j0[1]), 1)], junction_offset_from_projection_px=float(ts[k]),
                         slot_band_px_for_gap=band_px, dark_band_width_px=float(tt[b] - tt[a] + 1), dark_band_min_rel=round(mn / wall_far, 2),
                         dark_band_from_to_px=[float(tt[a]), float(tt[b])]))
    if not rows:
        continue
    out[str(f)] = rows
    # crop of the most central sample for viewing
    r = rows[len(rows) // 2]
    cx, cy = map(int, r['junction_px'])
    crop = img[max(cy - 120, 0):cy + 120, max(cx - 160, 0):cx + 160].copy()
    cv2.imwrite(str(common.WORK / f'wallcheck/video_slot_{f}.jpg'), crop)
summ = {}
for f, rows in out.items():
    summ[f] = dict(n=len(rows), dark_band_px_median=float(np.median([r['dark_band_width_px'] for r in rows])),
                   slot_px_per_cm_median=float(np.median([r['slot_band_px_for_gap']['10mm'] for r in rows])),
                   implied_gap_if_band_is_slot_cm=float(np.median([r['dark_band_width_px'] / r['slot_band_px_for_gap']['10mm'] for r in rows])),
                   band_for_85mm_px=float(np.median([r['slot_band_px_for_gap']['85mm'] for r in rows])))
common.save_json(common.WORK / 'video_slot_check.json', dict(per_frame=out, summary=summ))
print(json.dumps(summ, indent=1))


# ---- narrow dip at the junction (separate from the broad soft shadow): depression below a local linear baseline
# fitted on the wall 8-20 px from the junction; width = run below half the max depression. Same measure on the ZED.
def dip(tt, pp):
    base = np.polyfit(tt[(tt >= 8) & (tt <= 20)], pp[(tt >= 8) & (tt <= 20)], 1)
    dep = np.polyval(base, tt) - pp
    w = (tt >= -1) & (tt <= 8)
    k = np.argmax(np.where(w, dep, -1e9))
    a = b = k
    while a > 0 and dep[a - 1] > dep[k] / 2:
        a -= 1
    while b < len(dep) - 1 and dep[b + 1] > dep[k] / 2:
        b += 1
    return float(tt[b] - tt[a] + (tt[1] - tt[0])), float(dep[k] / np.polyval(base, tt[k]))


dips = {}
for f, rows in out.items():
    img = cv2.cvtColor(cv2.imread(str(common.WORK / f'frames_full/{int(f):06d}.jpg')), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ws, ds = [], []
    for r in rows:
        # direction along the junction from neighbouring samples; n toward the wall
        j = np.array(r['junction_px'])
        others = [np.array(q['junction_px']) for q in rows if q is not r]
        o = min(others, key=lambda q: np.linalg.norm(q - j))
        t = (o - j) / np.linalg.norm(o - j)
        n = np.array([-t[1], t[0]])
        tt = np.arange(-6, 25.01, 0.5)
        pp = np.array([cv2.getRectSubPix(img, (1, 1), (float(p[0]), float(p[1])))[0, 0] for p in j + tt[:, None] * n])
        if np.median(pp[tt > 10]) < np.median(pp[tt < -3]) - 60:   # wrong side (n pointed into the desk)
            pp = np.array([cv2.getRectSubPix(img, (1, 1), (float(p[0]), float(p[1])))[0, 0] for p in j - tt[:, None] * n])
        w_, d_ = dip(tt, pp)
        ws.append(w_ / r['slot_band_px_for_gap']['10mm']); ds.append(d_)
    dips[f] = dict(dip_width_over_slot_px_per_cm_median=float(np.median(ws)), dip_depth_rel_median=float(np.median(ds)))
Sp = np.load(common.WORK / 'wallcheck/zed_stereo_pair.npz')
gz = cv2.cvtColor(Sp['left'].astype(np.float32), cv2.COLOR_BGR2GRAY)
zm = np.load(common.WORK / 'zed_metric.npz')


def zproj(pd):  # ZED desk frame -> pixel
    pc = zm['Rdc'].T @ pd + zm['BL']
    q = zm['K'] @ pc
    return q[:2] / q[2]


zed_px_per_cm = float(np.median([zproj(np.array([xx, 0, 0]))[1] - zproj(np.array([xx, 0.01, 0]))[1] for xx in (0.3, 0.75, 1.2)]))
zed_band_85 = float(np.median([zproj(np.array([xx, 0, 0]))[1] - zproj(np.array([xx, 0.085, 0]))[1] for xx in (0.3, 0.75, 1.2)]))
zw = []
for u0 in range(300, 741, 20):
    vc = 359.9975 + (u0 - 518.5) / 0.99978 * (-0.02101)
    tt = np.arange(-6, 25.01, 0.5)
    pp = np.array([cv2.getRectSubPix(gz, (1, 1), (float(u0), float(vc - t_)))[0, 0] for t_ in tt])
    zw.append(dip(tt, pp))
dips['zed_scene_camera'] = dict(dip_width_px_median=float(np.median([w for w, _ in zw])), slot_px_per_cm=zed_px_per_cm, slot_band_px_for_85mm=zed_band_85,
                                dip_width_over_slot_px_per_cm_median=float(np.median([w for w, _ in zw]) / zed_px_per_cm),
                                dip_depth_rel_median=float(np.median([d for _, d in zw])))
common.save_json(common.WORK / 'video_slot_check.json', dict(per_frame=out, summary=summ, narrow_dip=dips,
                 note='dip_width_over_slot_px_per_cm = the gap in cm IF the narrow dark line were the slot interior; '
                      'a contact line gives a camera-blur-limited width in px, independent of the px/cm scale'))
print(json.dumps(dips, indent=1))
