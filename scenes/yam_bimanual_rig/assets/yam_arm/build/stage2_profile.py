"""Loss profile of one arm parameter of a stage-2 variant (CPU, raiden venv): how well the home images fix it.

    ~/robot/raiden/.venv/bin/python -u build/stage2_profile.py --fit RUN/stage2/c/fit.json --side right --param 1 \
        --grid-deg -2 0 2 4 6 8 10 12 14 --out RUN/stage2/profile_c_right_pitch.json

For each grid value the parameter is fixed and the other arm parameters are re-fitted (Nelder-Mead, C fixed at the
variant's value), started from the variant's solution and from the best neighbour. param: index into the arm
vector (base variants: 0 yaw, 1 pitch, 2 roll [rad], 3-5 translation [m]; c: 6-8 camera offset [m]).
--shared k profiles the shared claw placement C[k] instead (0 open_half, 1 pad_z6, 2 pad_x6; grid in mm) with both
arms re-fitted; in variant c the camera offset starts moved with the claws (pad_z6 / pad_x6 along link_6 z / x),
so the wrist view starts unchanged and the scene and background decide.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import claw_stage2_fit as F
import wristfit_common as W


def refit_arm(P, side, cv, x0):
    y, fv, _ = F.nm(lambda x: P.terms(side, cv, x), x0, P.steps_arm(), 1500)
    for _ in range(2):
        y2, fv2, _ = F.nm(lambda x: P.terms(side, cv, x), y, P.steps_arm(), 1500)
        done = fv - fv2 < 1e-4
        if fv2 < fv:
            y, fv = y2, fv2
        if done:
            break
    return y, fv


def profile_shared(a, fit, P, cv0):
    k = a.shared
    rows = []
    t0 = time.time()
    for g in np.array(a.grid_mm) * 1e-3:
        cv = cv0.copy(); cv[k] = g
        tot, arms = 0.0, {}
        for side in W.SIDES:
            x0 = np.array(fit['arms'][side]['params'])
            if P.variant in F.CAM_OFFSET and k in (1, 2):   # move the camera with the claws (link_6 z6 / x6)
                B, arm, T6, Twc = P.arm_pose(side, x0)
                ax = np.array([0, 0, 1.0]) if k == 1 else np.array([1.0, 0, 0])
                x0 = x0.copy(); x0[6:9] += (B @ T6)[:3, :3] @ ax * (g - cv0[k])
            y, fv = refit_arm(P, side, cv, x0)
            det, _ = P.terms(side, cv, y, True)
            arms[side] = dict(loss=fv, terms=det['terms'], iou_wrist=det['iou_wrist'], iou_scene=det['iou_scene'], params=y.tolist(),
                              scene_centroid_sil_px=det['centroid_model_minus_real_px']['scene']['sil'])
            tot += fv
        rows.append(dict(value_mm=float(g * 1e3), loss=tot, arms=arms))
        print(f"{g * 1e3:8.2f} loss {tot:.4f} " + ' '.join(f"{s}:{arms[s]['terms']}" for s in W.SIDES) + f' ({time.time() - t0:.0f} s)', flush=True)
    a.out.write_text(json.dumps(dict(fit=str(a.fit), shared=k, C=cv0.tolist(), rows=rows), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', type=Path, required=True)
    ap.add_argument('--side', choices=W.SIDES, required=True)
    ap.add_argument('--param', type=int, default=1)
    ap.add_argument('--grid-deg', type=float, nargs='*')
    ap.add_argument('--grid-mm', type=float, nargs='*')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shared', type=int, help='profile C[k] (mm grid) with both arms re-fitted')
    a = ap.parse_args()
    fit = W.jload(a.fit)

    class Args:
        pass
    pa = Args(); pa.variant = fit['variant']; pa.stage1 = fit['inputs']['stage1']; pa.captures = fit['inputs']['captures']
    pa.lam_bg = fit['inputs'].get('lam_bg', F.LAM_BG)
    P = F.Problem(pa)
    cv = np.array(fit['shared']['C'])
    if a.shared is not None:
        return profile_shared(a, fit, P, cv)
    av0 = np.array(fit['arms'][a.side]['params'])
    grid = np.radians(a.grid_deg) if a.grid_deg else np.array(a.grid_mm) * 1e-3
    free = [i for i in range(len(av0)) if i != a.param]
    steps = P.steps_arm()[free]
    rows, prev = [], None
    t0 = time.time()
    for g in grid:
        best = None
        for start in [av0] + ([prev] if prev is not None else []):
            x0 = start.copy(); x0[a.param] = g

            def f(y):
                x = x0.copy(); x[free] = y
                return P.terms(a.side, cv, x)
            y, fv, n = F.nm(f, x0[free], steps, 1200)
            for _ in range(2):         # restarts
                y2, fv2, n2 = F.nm(f, y, steps, 1200)
                if fv - fv2 < 1e-4:
                    y, fv = (y2, fv2) if fv2 < fv else (y, fv)
                    break
                y, fv = y2, fv2
            x = x0.copy(); x[free] = y
            if best is None or fv < best[0]:
                best = (fv, x)
        prev = best[1]
        det, _ = P.terms(a.side, cv, best[1], True)
        rows.append(dict(value=float(g), value_deg_or_mm=float(np.degrees(g)) if a.grid_deg else float(g * 1e3), loss=det['loss'],
                         terms=det['terms'], iou_wrist=det['iou_wrist'], iou_scene=det['iou_scene'], params=best[1].tolist(),
                         scene_centroid_sil_px=det['centroid_model_minus_real_px']['scene']['sil']))
        print(f"{rows[-1]['value_deg_or_mm']:7.2f} loss {det['loss']:.4f} {det['terms']} ({time.time() - t0:.0f} s)", flush=True)
    a.out.write_text(json.dumps(dict(fit=str(a.fit), side=a.side, param=a.param, C=cv.tolist(), rows=rows), indent=1))


if __name__ == '__main__':
    main()
