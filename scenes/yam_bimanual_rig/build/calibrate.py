"""Fit flat albedos / emission / light power per region and the post-render camera model to the real images.

python calibrate.py --render DIR --config configs/vNN.json --out configs/vMM.json [--steps albedo,vignette,wrist]

albedo:   per-region ratio mean(real_lin) / mean(camera(render_lin)) per channel, applied to the region's
          material parameter (scene camera only; the wrist cameras never change materials).
vignette: fit k1, k2 of V(r) for the scene camera on wall + desk + carpet luminance, with a free scale per region.
wrist:    per wrist camera: RGB gain (+ optional vignette) on desk + wall, materials fixed.
Every step writes its measurements into the output config under 'calibration_log'.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import camera_model as cm

TARGET = {  # region -> (material key, parameter)
    'desk': ('desk_top', 'albedo'),
    'wall': ('wall', 'albedo'),
    'carpet': ('carpet', 'albedo'),
    'glass_frosted': ('frosted', 'emission'),
    'glass_frame': ('frame_alu', 'albedo'),
    'panel': ('panel', 'tint'),
    'monitor_desk': ('monitor_desk_top', 'albedo'),
    'brackets': ('bracket', 'albedo'),
}


def stats(real, rend, mask):
    # medians are robust to small misalignment at region borders (bright frame, dark gaps)
    return np.median(real[mask], 0), np.median(rend[mask], 0), int(mask.sum())


def cam_K(cfg, view):
    return cfg['camera_model'][view]['K']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--render', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', default='albedo')
    p.add_argument('--damping', type=float, default=1.0, help='exponent on the ratio (1 = full step)')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    new = copy.deepcopy(cfg)
    log = {'render': str(a.render), 'from': str(a.config), 'steps': a.steps.split(','), 'regions': {}}
    steps = a.steps.split(',')
    view = 'scene'
    real = cm.real_lin(view)
    rend0 = cm.render_lin(a.render, view)
    reg, _ = cm.region_masks(a.render, view)
    ex = cm.exclusion(view, reg)
    camp = cfg['camera_model'][view]
    K = camp['K']
    rend = cm.apply_camera(rend0, camp, K)

    if 'vignette' in steps:
        names = [n for n in ('wall', 'desk', 'carpet') if (reg[n] & ~ex).sum() > 500]
        h, w = real.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        r2 = ((u - K[0][2]) / K[0][0]) ** 2 + ((v - K[1][2]) / K[1][1]) ** 2
        Y = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
        sel = [(reg[n] & ~ex) for n in names]
        samp = [np.flatnonzero(s)[::7] for s in sel]
        yr = [np.log(np.clip(Y(real.reshape(-1, 3)[i]), 1e-4, None)) for i in samp]
        yb = [np.log(np.clip(Y(rend0.reshape(-1, 3)[i]), 1e-4, None)) for i in samp]
        rr = [r2.reshape(-1)[i] for i in samp]
        def res(x):
            k1, k2 = x[:2]; out = []
            for j in range(len(names)):
                V = np.clip(1 + k1 * rr[j] + k2 * rr[j] ** 2, 1e-3, None)
                out.append(yr[j] - (yb[j] + np.log(V) + x[2 + j]))
            return np.concatenate(out)
        fit = least_squares(res, np.r_[-0.3, 0.05, np.zeros(len(names))])
        k1, k2 = fit.x[:2]
        new['camera_model'][view]['k1'], new['camera_model'][view]['k2'] = float(k1), float(k2)
        # global exposure: the wall keeps its albedo; the camera gain absorbs the vignette's mean darkening
        gw = float(np.exp(fit.x[2 + names.index('wall')]))
        new['camera_model'][view]['gain'] = [round(gw, 4)] * 3
        log['vignette'] = dict(regions=names, k1=float(k1), k2=float(k2), rms_log=float(np.sqrt(np.mean(fit.fun ** 2))),
                               V_at_r=[[float(r), float(1 + k1 * r * r + k2 * r ** 4)] for r in (0.25, 0.5, 0.75, 1.0, 1.25)])
        rend = cm.apply_camera(rend0, new['camera_model'][view], K)

    if 'albedo' in steps:
        for region, (mkey, param) in TARGET.items():
            m = reg[region] & ~ex
            if m.sum() < 200:
                continue
            mr, mb, n = stats(real, rend, m)
            ratio = np.clip(mr / np.clip(mb, 1e-6, None), 0.2, 5.0) ** a.damping
            mat = new['materials'].setdefault(mkey, {})
            old = np.asarray(mat.get(param, [1, 1, 1]), np.float64)
            val = old * ratio
            if param == 'albedo':
                val = np.clip(val, 0.002, 0.95)
            mat[param] = [round(float(x), 5) for x in val]
            log['regions'][region] = dict(pixels=n, real_lin=mr.round(5).tolist(), render_lin=mb.round(5).tolist(),
                                          ratio=ratio.round(4).tolist(), param=f'{mkey}.{param}', old=old.round(5).tolist(), new=mat[param])
        # daylight behind the glass: the carpet seen through the clear lower glass
        m = reg['glass_lower'] & ~ex
        if m.sum() > 200:
            mr, mb, n = stats(real, rend, m)
            # remove the part explained by the (already scaled) carpet albedo change: scale light by the residual
            carpet_ratio = np.asarray(log['regions'].get('carpet', {}).get('ratio', [1, 1, 1]))
            ratio = np.clip(mr / np.clip(mb * carpet_ratio, 1e-6, None), 0.2, 5.0) ** a.damping
            d = new['lighting']['daylight']
            scale = float(ratio.mean())
            col = np.asarray(d['color']) * ratio / scale
            d['power'] = round(d['power'] * scale, 2)
            d['color'] = [round(float(c), 4) for c in col / col.max()]
            log['regions']['glass_lower'] = dict(pixels=n, real_lin=mr.round(5).tolist(), render_lin=mb.round(5).tolist(),
                                                 ratio=ratio.round(4).tolist(), param='lighting.daylight power/color')

    if 'wrist' in steps:
        for wv in ('left_wrist', 'right_wrist'):
            if not (a.render / f'{wv}.exr').exists():
                continue
            rl = cm.real_lin(wv); rb = cm.render_lin(a.render, wv)
            rg, _ = cm.region_masks(a.render, wv)
            exw = cm.exclusion(wv, rg)
            m = (rg['desk'] | rg['wall']) & ~exw
            c = new['camera_model'][wv]
            if 'wrist_vignette' in steps:
                Kw = c['K']; h, w = rl.shape[:2]; u, v = np.meshgrid(np.arange(w), np.arange(h))
                r2 = (((u - Kw[0][2]) / Kw[0][0]) ** 2 + ((v - Kw[1][2]) / Kw[1][1]) ** 2).reshape(-1)
                Yf = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
                idx = [np.flatnonzero((rg[n] & ~exw).reshape(-1))[::5] for n in ('desk', 'wall')]
                def res(x):
                    return np.concatenate([np.log(np.clip(Yf(rl.reshape(-1, 3)[i]), 1e-4, None)) - np.log(np.clip(Yf(rb.reshape(-1, 3)[i]), 1e-4, None))
                                           - np.log(np.clip(1 + x[0] * r2[i] + x[1] * r2[i] ** 2, 1e-3, None)) - x[2] for i in idx])
                fit = least_squares(res, np.r_[-0.3, 0.05, 0.0])
                c['k1'], c['k2'] = float(fit.x[0]), float(fit.x[1])
            rbv = rb * cm.vignette(rb.shape, c['K'], c.get('k1', 0), c.get('k2', 0))[..., None]
            gain = rl[m].mean(0) / np.clip(rbv[m].mean(0), 1e-6, None)
            c['gain'] = [round(float(x), 4) for x in gain]
            log[wv] = dict(pixels=int(m.sum()), gain=c['gain'], k1=c.get('k1', 0), k2=c.get('k2', 0))
    new.setdefault('calibration_log', []).append(log)
    a.out.write_text(json.dumps(new, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
