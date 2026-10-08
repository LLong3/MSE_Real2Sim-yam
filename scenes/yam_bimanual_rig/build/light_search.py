"""Grid search of the ceiling-panel row position against the scene-camera image (sequential, one GPU job at a time).

python light_search.py --config configs/vNN.json --out RUN/light_search_vNN
For each (dx, y) the panel row is shifted by dx and moved to y; the scene is built and rendered (scene view,
16 samples); the vignette + per-region scales are refitted and the rms log-luminance residual on wall + desk +
monitor desk is recorded. Writes results.json sorted by score. Materials and other lights are unchanged.
"""
import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import camera_model as cm

HERE = Path(__file__).resolve().parent


def score(render_dir, K):
    real = cm.real_lin('scene'); rend = cm.render_lin(render_dir, 'scene')
    reg, _ = cm.region_masks(render_dir, 'scene'); ex = cm.exclusion('scene', reg)
    Y = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
    h, w = real.shape[:2]; u, v = np.meshgrid(np.arange(w), np.arange(h))
    r2 = (((u - K[0][2]) / K[0][0]) ** 2 + ((v - K[1][2]) / K[1][1]) ** 2).reshape(-1)
    names = ['wall', 'desk']
    idx = [np.flatnonzero((reg[n] & ~ex).reshape(-1))[::5] for n in names]
    yr = [np.log(np.clip(Y(real.reshape(-1, 3)[i]), 1e-4, None)) for i in idx]
    yb = [np.log(np.clip(Y(rend.reshape(-1, 3)[i]), 1e-4, None)) for i in idx]
    def res(x):
        return np.concatenate([yr[j] - yb[j] - np.log(np.clip(1 + x[0] * r2[i] + x[1] * r2[i] ** 2, 1e-3, None)) - x[2 + j]
                               for j, i in enumerate(idx)])
    f = least_squares(res, np.r_[-0.4, 0.05, np.zeros(len(names))])
    per = {}
    o = 0
    for j, n in enumerate(names):
        k = len(idx[j]); per[n] = float(np.sqrt(np.mean(f.fun[o:o + k] ** 2))); o += k
    return float(np.sqrt(np.mean(f.fun ** 2))), per, f.x.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dx', default='-0.6,-0.3,0,0.3,0.6')
    p.add_argument('--y', default='0.3,0.8,1.3,1.8')
    p.add_argument('--world', default='1', help='multipliers of the world (ambient) strength')
    a = p.parse_args()
    base = json.loads(a.config.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    env = 'source ../../../kimodo_blender/env.sh >/dev/null 2>&1; '
    results = []
    import itertools
    for dx, y, wm in itertools.product(map(float, a.dx.split(',')), map(float, a.y.split(',')), map(float, a.world.split(','))):
            tag = f'dx{dx:+.2f}_y{y:.2f}_w{wm:.1f}'
            cfg = copy.deepcopy(base)
            cfg['lighting']['world']['strength'] *= wm
            for pnl in cfg['lighting']['ceiling_panels']:
                pnl['center'][0] += dx; pnl['center'][1] = y
            cp = a.out / f'{tag}.json'; cp.write_text(json.dumps(cfg, indent=1))
            blend = a.out / f'{tag}.blend'; rdir = a.out / tag
            cmd = (env + f'"$BLENDER_BIN" -b --factory-startup --python-exit-code 1 --python build_scene.py -- --config {cp} --out {blend} > {a.out}/{tag}_build.log 2>&1 && '
                   f'"$BLENDER_BIN" -b {blend} --python-exit-code 1 --python render_views.py -- --out {rdir} --views scene --samples 16 > {a.out}/{tag}_render.log 2>&1')
            subprocess.run(['bash', '-c', cmd], cwd=HERE, check=True)
            s, per, x = score(rdir, base['camera_model']['scene']['K'])
            results.append(dict(tag=tag, dx=dx, y=y, world_mult=wm, rms_log=s, per_region=per, fit=x))
            print(tag, round(s, 4), {k: round(v, 3) for k, v in per.items()}, flush=True)
            blend.unlink()
    results.sort(key=lambda r: r['rms_log'])
    (a.out / 'results.json').write_text(json.dumps(results, indent=1))
    print('BEST', results[0])


if __name__ == '__main__':
    main()
