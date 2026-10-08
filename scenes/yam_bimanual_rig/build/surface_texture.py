"""Image-derived albedo for a planar surface from the ZED scene image (desk top or back wall).

python surface_texture.py --white-render RUN/renders/vNN_white --config configs/vNN.json --surface desk --out textures/desk_albedo_vNN.exr

albedo_obs = real_lin / (camera(render with this surface's base colour = 1)), i.e. the real radiance divided by the
modelled irradiance, vignette and exposure. It is rectified onto the surface plane (desk: 1 mm texels over the
desk box; wall: 5 mm texels over the rect in the config), occluded or excluded texels (arms, grippers, croissant,
vial, other objects) are inpainted, and texels outside the ZED view blend to the surface median.
The wall map is low-pass filtered (sigma in the config) so that it keeps only the shading residual.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--white-render', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--surface', choices=['desk', 'wall'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--erode', type=int, default=2, help='px eroded from the surface region (object borders)')
    p.add_argument('--arm-dilate', type=int, default=6, help='px grown around arms/grippers (render IDs and SAM)')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()); m = metric()
    camp = cfg['camera_model']['scene']; K = np.array(camp['K']); T = np.array(m['scene_camera']['pose']['world_from_camera_opencv'])
    real = cm.real_lin('scene')
    L1 = cm.apply_camera(cm.render_lin(a.white_render, 'scene'), camp, camp['K'])
    reg, k = cm.region_masks(a.white_render, 'scene', erode=0)
    ex = cm.exclusion('scene', reg, dilate_arms=a.arm_dilate)
    if a.arm_dilate > 8:  # grow the SAM gripper masks too: real fingers are larger than the placeholder asset
        sam = np.load(cm.SAM)['masks']
        ex |= cv2.dilate((sam[0] | sam[1]).astype(np.uint8), np.ones((2 * a.arm_dilate + 1,) * 2, np.uint8)).astype(bool)
    ex[:3] = ex[-3:] = True; ex[:, :3] = ex[:, -3:] = True
    if a.surface == 'desk':
        region = reg['desk']
        x0, x1 = m['desk']['box_x_m']; y0, y1 = m['desk']['box_y_m']; zt = m['desk']['top_z_m']
        res = 0.001
        W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        xs = x0 + (np.arange(W) + 0.5) * res; ys = y0 + (np.arange(H) + 0.5) * res
        X, Yg = np.meshgrid(xs, ys)
        P = np.c_[X.ravel(), Yg.ravel(), np.full(X.size, zt)]
        sigma_lp = cfg['materials']['desk_top'].get('texture_lowpass_m', 0.0)
    else:
        region = reg['wall']
        t = cfg['materials']['wall']['texture_rect']; res = 0.005
        wy = -m['planes']['back_wall']['d'] * m['planes']['back_wall']['n'][1]
        W, H = int(round((t[1] - t[0]) / res)), int(round((t[3] - t[2]) / res))
        xs = t[0] + (np.arange(W) + 0.5) * res; zs = t[2] + (np.arange(H) + 0.5) * res
        X, Z = np.meshgrid(xs, zs)
        P = np.c_[X.ravel(), np.full(X.size, wy), Z.ravel()]
        sigma_lp = cfg['materials']['wall'].get('texture_lowpass_m', 0.2)
    uv, z = project(P, K, T)
    mu = uv[:, 0].reshape(H, W).astype(np.float32); mv = uv[:, 1].reshape(H, W).astype(np.float32)
    obs = real / np.clip(L1, 1e-5, None)
    valid_img = (region & ~ex).astype(np.float32)
    valid_img = cv2.erode(valid_img, np.ones((2 * a.erode + 1,) * 2, np.uint8))
    tex = np.stack([cv2.remap(obs[:, :, c].astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderValue=0) for c in range(3)], -1)
    val = cv2.remap(valid_img, mu, mv, cv2.INTER_NEAREST, borderValue=0) > 0.5
    val &= (z.reshape(H, W) > 0)
    med = np.median(tex[val], 0)
    # fill: normalised convolution at increasing scales (keeps the observed texels, smooth inside holes)
    fill = np.zeros_like(tex); wsum = np.zeros(val.shape, np.float32)
    acc = tex * val[..., None]; wv = val.astype(np.float32)
    out = tex.copy()
    missing = ~val
    for s in (0.01, 0.03, 0.08, 0.2):
        sp = s / res
        num = cv2.GaussianBlur(acc, (0, 0), sp); den = cv2.GaussianBlur(wv, (0, 0), sp)
        est = num / np.clip(den[..., None], 1e-6, None)
        ok = missing & (den > 0.05)
        out[ok] = est[ok]; missing &= ~ok
    out[missing] = med
    # far from any observation: blend to the median over ~0.15 m
    dist = cv2.distanceTransform((~val).astype(np.uint8), cv2.DIST_L2, 5) * res
    wgt = np.clip(1 - dist / 0.15, 0, 1)[..., None]
    out = wgt * out + (1 - wgt) * med
    if sigma_lp > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigma_lp / res)
    out = np.clip(out, 0.005, 0.95).astype(np.float32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # EXR rows: image row 0 = top of the texture; Blender UV v=0 is the bottom -> flip so v follows +y / +z
    cv2.imwrite(str(a.out), out[::-1, :, ::-1])
    prev = cm.to_u8(out[::-1] * 0.5)
    cv2.imwrite(str(a.out.with_suffix('.preview.jpg')), prev[:, :, ::-1])
    rep = dict(erode_px=a.erode, arm_dilate_px=a.arm_dilate, surface=a.surface, out=str(a.out), size=[W, H], texel_m=res, observed_fraction=float(val.mean()),
               median_albedo=med.round(4).tolist(), lowpass_m=sigma_lp, white_render=str(a.white_render), config=str(a.config))
    a.out.with_suffix('.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
