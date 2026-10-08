"""Compare a rendered rig-camera view with the real image: images + metrics.

python compare.py --render DIR --config configs/vNN.json --view scene --out DIR/compare_scene

Writes <out>/{side_by_side,overlay,difference,edges,ratio}.jpg, <view>_cam.png (render through the camera
model) and metrics.json:
  edges    per named structural edge: the 3D edge is projected with the metric camera; along its normal the
           real-image gradient peak is searched (+-12 px); offset stats in px (median |d|, p90 |d|, n)
  iou      scene view: gripper silhouette IoU (render arm IDs vs SAM 3.1 masks, read only)
  regions  per region (eroded render-ID masks minus exclusions): mean sRGB real/render, median linear ratio,
           luminance ratio, CIE76 dE of the mean Lab, per-pixel mean |dL|
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import camera_model as cm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigcam import metric, project  # noqa: E402


def named_edges(m, cfg):
    d = m['desk']; x0, x1 = d['box_x_m']; y0, y1 = d['box_y_m']; zt = d['top_z_m']
    wy = y1 + cfg['measurements']['wall_gap_m']
    we = cfg.get('wall_end_x_override', {}).get('x', m['planes']['back_wall']['wall_end_x_m'])
    g = cfg['glass']; gy = g['plane_y']
    px = g['post_x'] + g['post_size'][0] / 2
    py0 = gy - g['post_size'][1] / 2
    md = cfg['monitor_desk']
    return {
        'desk_back_edge': ((x0, y1, zt), (x1, y1, zt)),
        'desk_left_edge': ((x0, y0, zt), (x0, y1, zt)),
        'desk_right_edge': ((x1, y0, zt), (x1, y1, zt)),
        'wall_end_vertical': ((we, wy, zt + 0.02), (we, wy, 2.6)),
        'glass_frosted_bottom': ((g['post_x'] + 0.05, gy, g['frosted_z'][0]), (we - 0.03, gy, g['frosted_z'][0])),
        'glass_mullion_post_edge': ((px, py0, 0.03), (px, py0, 1.2)),
        'monitor_desk_left_edge': ((md['x'][0], md['y'][0], md['top_z']), (md['x'][0], md['y'][1], md['top_z'])),
        'privacy_panel_left_edge': ((md['panel_x'][0], md['panel_y'][0], md['panel_z'][0]), (md['panel_x'][0], md['panel_y'][0], md['panel_z'][1] - md['panel_corner_r'])),
    }


def edge_offsets(gray, K, T, a, b, search=12, n=80, occl=None):
    P = np.linspace(a, b, n)
    uv, z = project(P, K, T)
    ok = (z > 0) & (uv[:, 0] > search) & (uv[:, 0] < 960 - search) & (uv[:, 1] > search) & (uv[:, 1] < 600 - search)
    uv = uv[ok]
    if len(uv) < 5:
        return None
    t = uv[-1] - uv[0]; t /= np.linalg.norm(t); nrm = np.array([-t[1], t[0]])
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g = np.abs(gx * nrm[0] + gy * nrm[1])
    offs = []
    for p in uv:
        if occl is not None and occl[int(round(p[1])), int(round(p[0]))]:
            continue
        s = np.arange(-search, search + 1)
        q = p[None] + s[:, None] * nrm[None]
        vals = cv2.remap(g.astype(np.float32), q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        if vals.max() < 20:
            continue
        i = int(np.argmax(vals))
        if 0 < i < len(s) - 1:  # parabolic sub-pixel peak
            den = vals[i - 1] - 2 * vals[i] + vals[i + 1]
            sub = 0.5 * (vals[i - 1] - vals[i + 1]) / den if den != 0 else 0.0
        else:
            sub = 0.0
        offs.append(s[i] + sub)
    if len(offs) < 5:
        return None
    o = np.abs(np.array(offs))
    return dict(n=len(o), median_abs_px=round(float(np.median(o)), 2), p90_abs_px=round(float(np.percentile(o, 90)), 2),
                mean_signed_px=round(float(np.mean(offs)), 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--render', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--view', default='scene')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    view = a.view
    camp = cfg['camera_model'][view]
    real8 = cm.real_u8(view); reall = cm.real_lin(view)
    rl = cm.apply_camera(cm.render_lin(a.render, view), camp, camp['K'])
    rend8 = cm.to_u8(rl)
    cv2.imwrite(str(a.render / f'{view}_cam.png'), rend8[:, :, ::-1])
    reg, k = cm.region_masks(a.render, view)
    ex = cm.exclusion(view, reg)
    metrics = {'render': str(a.render), 'config': str(a.config), 'view': view, 'camera_model': camp}

    # images
    sbs = np.concatenate([real8, rend8], 1)
    cv2.imwrite(str(a.out / 'side_by_side.jpg'), sbs[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(a.out / 'overlay.jpg'), cv2.addWeighted(real8, 0.5, rend8, 0.5, 0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    diff = np.abs(real8.astype(int) - rend8.astype(int)).mean(2)
    heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(a.out / 'difference.jpg'), heat, [cv2.IMWRITE_JPEG_QUALITY, 90])
    Y = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
    ratio = np.log2(np.clip(Y(reall), 1e-4, None) / np.clip(Y(rl), 1e-4, None))
    rimg = cv2.applyColorMap(np.clip((ratio + 1) * 127.5, 0, 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED)
    rimg[ex] = (rimg[ex] * 0.3).astype(np.uint8)
    cv2.imwrite(str(a.out / 'ratio_log2_pm1.jpg'), rimg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    g1 = cv2.GaussianBlur(cv2.cvtColor(real8, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(rend8, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    e1 = cv2.Canny(g1, 30, 80) > 0; e2 = cv2.Canny(g2, 30, 80) > 0
    ev = (0.45 * real8).astype(np.uint8); ev[e1] = (255, 60, 60); ev[e2] = (60, 255, 60); ev[e1 & e2] = (255, 255, 0)
    cv2.imwrite(str(a.out / 'edges.jpg'), ev[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    # global edge chamfer (render edges -> nearest real edge), outside exclusions, capped at 20 px
    dt = cv2.distanceTransform((~e1).astype(np.uint8), cv2.DIST_L2, 5)
    sel = e2 & ~ex
    dd = np.minimum(dt[sel], 20)
    metrics['edge_chamfer_render_to_real_px'] = dict(median=round(float(np.median(dd)), 2), mean_capped20=round(float(dd.mean()), 2),
                                                     frac_within_2px=round(float((dd <= 2).mean()), 3), n=int(sel.sum()))

    # named structural edges (scene camera)
    if view == 'scene':
        m = metric(); K, T = np.array(m['scene_camera']['K']), np.array(m['scene_camera']['pose']['world_from_camera_opencv'])
        occl = ex | cv2.dilate((reg['left_arm'] | reg['right_arm']).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        edges = {}
        for name, (p0, p1) in named_edges(m, cfg).items():
            edges[name] = dict(real=edge_offsets(g1.astype(np.float64), K, T, p0, p1, occl=occl),
                               render=edge_offsets(g2.astype(np.float64), K, T, p0, p1, occl=occl))
        metrics['edges'] = edges
        # gripper IoU
        sam = np.load(cm.SAM)['masks']
        iou = {}
        for arm, i in cm.SAM_INSTANCES.items():
            r = reg[arm]; s = sam[i]
            inter, uni = (r & s).sum(), (r | s).sum()
            iou[arm] = dict(iou=round(float(inter / max(uni, 1)), 4), render_px=int(r.sum()), sam_px=int(s.sum()),
                            precision=round(float(inter / max(r.sum(), 1)), 4), recall=round(float(inter / max(s.sum(), 1)), 4))
        r = reg['left_arm'] | reg['right_arm']; s = sam[0] | sam[1]
        iou['both'] = dict(iou=round(float((r & s).sum() / max((r | s).sum(), 1)), 4))
        metrics['gripper_iou'] = iou

    # regions
    L1, L2 = cm.lab(real8), cm.lab(rend8)
    regions = {}
    for n in cm.REGIONS:
        msk = reg[n] & ~ex
        if msk.sum() < 100:
            continue
        la, lb = L1[msk].mean(0), L2[msk].mean(0)
        regions[n] = dict(pixels=int(msk.sum()),
                          real_srgb_mean=np.round(real8[msk].mean(0), 1).tolist(), render_srgb_mean=np.round(rend8[msk].mean(0), 1).tolist(),
                          linear_ratio_real_over_render=np.round(np.median(reall[msk], 0) / np.clip(np.median(rl[msk], 0), 1e-6, None), 3).tolist(),
                          luminance_ratio=round(float(np.median(Y(reall[msk])) / max(np.median(Y(rl[msk])), 1e-6)), 3),
                          dE76_mean_lab=round(float(np.linalg.norm(la - lb)), 2),
                          mean_abs_dL=round(float(np.abs(L1[msk][:, 0] - L2[msk][:, 0]).mean()), 2),
                          srgb_mae=round(float(np.abs(real8[msk].astype(int) - rend8[msk].astype(int)).mean()), 2))
    metrics['regions'] = regions
    allm = ~ex
    metrics['whole_image_srgb_mae_excl'] = round(float(np.abs(real8[allm].astype(int) - rend8[allm].astype(int)).mean()), 2)
    (a.out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: metrics[k] for k in metrics if k not in ('camera_model',)}, indent=1))


if __name__ == '__main__':
    main()
