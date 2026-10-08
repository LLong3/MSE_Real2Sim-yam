"""Shared numpy helpers: images, ID regions, exclusions and the post-render camera model.

Camera model (applied to the linear Cycles render): out_lin = gain_rgb * V(r) * render_lin, then the sRGB
OETF (Blender 'Standard'). V(r) = 1 + k1 r^2 + k2 r^4 with r the distance to the principal point divided by fx.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
REAL = {'scene': Path('/home/frank/robot/bimanual_bringup/pi05/scene_camera.jpg'),
        'left_wrist': Path('/home/frank/robot/bimanual_bringup/pi05/left_wrist_camera.jpg'),
        'right_wrist': Path('/home/frank/robot/bimanual_bringup/pi05/right_wrist_camera.jpg')}
SAM = SCENE / 'assets/yam_arm/evidence/scene_camera/sam/masks.npz'   # read only
SAM_INSTANCES = {'left_arm': 1, 'right_arm': 0}

REGIONS = {
    'desk': ['Rig desk | top'],
    'wall': ['Back wall | painted'],
    'glass_frosted': ['Glass A | frosted band', 'Glass B | frosted band'],
    'glass_lower': ['Glass A | clear lower', 'Glass B | clear lower'],
    'carpet': ['Floor | carpet'],
    'glass_frame': ['Glass frame | mullion post', 'Glass frame | wall-end jamb', 'Glass frame | floor channel'],
    'panel': ['Monitors | privacy panel'],
    'monitor_desk': ['Monitor desk | top'],
    'brackets': ['Rig desk | panel bracket 0', 'Monitors | monitor arm segment 0'],
    'claw_pad': ['yam_tip_left_umi_pad', 'yam_tip_right_umi_pad'],
    'claw_clamp': ['yam_tip_left_umi_clamp', 'yam_tip_right_umi_clamp'],
}
# task objects on the real table (not modelled by request) and fingers in the wrist images
EXCLUDE_POLYS = {
    'scene': [[(630, 442), (684, 442), (684, 502), (630, 502)], [(528, 506), (568, 506), (568, 562), (528, 562)]],
    'left_wrist': [[(0, 600), (0, 470), (170, 390), (230, 340), (355, 312), (300, 600)],
                   [(575, 325), (650, 325), (760, 355), (960, 500), (960, 600), (690, 600)]],
    'right_wrist': [[(0, 600), (0, 500), (160, 380), (230, 320), (350, 295), (300, 600)],
                    [(570, 305), (700, 325), (780, 370), (960, 520), (960, 600), (690, 600)],
                    [(425, 330), (515, 330), (515, 440), (425, 440)]],
}


def srgb_to_lin(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0, None)
    return np.clip(np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055), 0, 1)


def real_lin(view):
    return srgb_to_lin(cv2.imread(str(REAL[view]))[:, :, ::-1] / 255.0)


def real_u8(view):
    return cv2.imread(str(REAL[view]))[:, :, ::-1]


def render_lin(d, view):
    return cv2.imread(str(Path(d) / f'{view}.exr'), cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float64)


def ids(d, view):
    a = cv2.imread(str(Path(d) / f'{view}_ids.exr'), cv2.IMREAD_UNCHANGED)
    k = np.round(a[:, :, 2] * 255).astype(int) + 256 * np.round(a[:, :, 1] * 255).astype(int)
    k[np.abs(a[:, :, 0] - 1.0) > 1e-3] = 0
    return k


def table(d):
    return {int(k): v for k, v in json.loads((Path(d) / 'ids.json').read_text()).items()}


def region_masks(d, view, erode=3):
    k, t = ids(d, view), table(d)
    out = {}
    for name, prefixes in REGIONS.items():
        sel = [i for i, v in t.items() if any(v['object'] == p or v['object'].startswith(p + '.') for p in prefixes)]
        out[name] = np.isin(k, sel)
    for arm in ('left_arm', 'right_arm'):
        out[arm] = np.isin(k, [i for i, v in t.items() if v['root'] == arm])
    if erode:
        ker = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
        out = {n: cv2.erode(m.astype(np.uint8), ker).astype(bool) if n not in ('left_arm', 'right_arm') else m for n, m in out.items()}
    return out, k


def exclusion(view, region=None, d=None, dilate_arms=8):
    ex = np.zeros((600, 960), np.uint8)
    for poly in EXCLUDE_POLYS.get(view, []):
        cv2.fillPoly(ex, [np.array(poly, np.int32)], 1)
    if view == 'scene':
        sam = np.load(SAM)['masks']
        ex |= cv2.dilate((sam[0] | sam[1]).astype(np.uint8), np.ones((15, 15), np.uint8))
    if region is not None:
        arms = (region['left_arm'] | region['right_arm']).astype(np.uint8)
        ex |= cv2.dilate(arms, np.ones((2 * dilate_arms + 1, 2 * dilate_arms + 1), np.uint8))
    return ex.astype(bool)


def vignette(shape, K, k1, k2):
    h, w = shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    r2 = ((u - K[0][2]) / K[0][0]) ** 2 + ((v - K[1][2]) / K[1][1]) ** 2
    return 1 + k1 * r2 + k2 * r2 ** 2


def apply_camera(lin, cam, K):
    V = vignette(lin.shape, K, cam.get('k1', 0.0), cam.get('k2', 0.0))
    g = np.asarray(cam.get('gain', [1, 1, 1]), np.float64)
    return lin * V[..., None] * g


def to_u8(lin):
    return (lin_to_srgb(lin) * 255 + 0.5).astype(np.uint8)


def lab(u8):
    return cv2.cvtColor(np.ascontiguousarray(u8.astype(np.uint8)), cv2.COLOR_RGB2LAB).astype(np.float64) * [100 / 255, 1, 1] - [0, 128, 128]
