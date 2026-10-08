"""Shared helpers for the 09-30 wrist-camera and claw fits (numpy/OpenCV/MuJoCo; raiden venv, CPU only).

Frames: world = metric frame (scenes/yam_bimanual_rig/metric/metric_frame.json, read only); camera poses are
world_from_camera with OpenCV axes (x right, y down, z forward). Wrist K = raiden's ZED SDK HD1200 K / 2 (960x600,
fx != fy, as the live stream). FK: MuJoCo on source/yam_linear_4310.xml (raiden's i2rt model).
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parents[1]                 # assets/yam_arm
SCENE = HERE.parents[1]                                    # scenes/yam_bimanual_rig
ROOT = SCENE.parents[1]                                    # aha-3d
METRIC = SCENE / 'metric' / 'metric_frame.json'
BUILD_CFG = SCENE / 'build' / 'configs' / 'delivery_v2.json'
V2_RUN = ROOT / 'runs/yam_bimanual_rig/20260929t225835z-build-1de8ab34'
REAL = Path.home() / 'robot/bimanual_bringup/pi05'
CAL = Path.home() / '.config/raiden/calibration_results.json'
CAPTURES = Path.home() / '.config/raiden/calibration_captures/20260929_171127'
XML = HERE / 'source' / 'yam_linear_4310.xml'
CV2BL = np.diag([1.0, -1.0, -1.0, 1.0])
SIDES = ('left', 'right')
ARM = {'left': 'left_arm', 'right': 'right_arm'}
HOME_Q = np.zeros(6)


def jload(p):
    return json.loads(Path(p).read_text())


def T_of(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t)
    return M


def pose_delta(Ta, Tb):
    """Tb relative to Ta: rotation angle (deg) and translation difference (mm, in the frame of the parent of both)."""
    d = np.linalg.inv(Ta) @ Tb
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(d[:3, :3]).as_rotvec()))), (Tb[:3, 3] - Ta[:3, 3]) * 1e3


def perturb(T, x):
    """x = (rotvec about the camera centre in the camera frame, translation in the parent frame) -> new pose."""
    M = T.copy()
    M[:3, :3] = T[:3, :3] @ Rot.from_rotvec(x[:3]).as_matrix()
    M[:3, 3] = T[:3, 3] + x[3:6]
    return M


def project(P, K, T):
    """world points (N,3) -> pixels (N,2), depth (N,) for camera pose T (world_from_camera, OpenCV)."""
    P = np.atleast_2d(np.asarray(P, float))
    pc = (P - T[:3, 3]) @ T[:3, :3]
    uv = pc[:, :2] / np.clip(pc[:, 2:3], 1e-6, None) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, pc[:, 2]


class FK:
    """MuJoCo FK of the combined i2rt YAM + linear_4310 model (base frame)."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)

    def __call__(self, q6, gripper=0.0475, frames=('link_6', 'grasp_site')):
        q = np.zeros(self.m.nq); q[:6] = q6
        if self.m.nq >= 8:
            q[6] = q[7] = gripper
        self.d.qpos[:] = q
        self.mj.mj_kinematics(self.m, self.d)
        out = {}
        for f in frames:
            bid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, f)
            if bid >= 0:
                out[f] = T_of(self.d.xmat[bid].reshape(3, 3), self.d.xpos[bid])
                continue
            sid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_SITE, f)
            out[f] = T_of(self.d.site_xmat[sid].reshape(3, 3), self.d.site_xpos[sid])
        return out


def wrist_K(side, cal=None):
    cal = cal or jload(CAL)
    c = cal['cameras'][f'{side}_wrist_camera']['intrinsics']
    s = 960 / c['image_size'][0]
    K = np.array(c['camera_matrix'], float) * s; K[2, 2] = 1
    return K


def hand_eye(side, cal=None):
    """raiden 09-29 hand-eye: grasp_site_from_camera (OpenCV)."""
    cal = cal or jload(CAL)
    he = cal['cameras'][f'{side}_wrist_camera']['hand_eye_calibration']
    return T_of(he['rotation_matrix'], he['translation_vector'])


def metric_bases(m=None):
    m = m or jload(METRIC)
    return {s: np.array(m['arms'][ARM[s]]['world_from_base']) for s in SIDES}


def scene_camera(m=None):
    m = m or jload(METRIC)
    sc = m['scene_camera']
    return np.array(sc['K'], float), np.array(sc['pose']['world_from_camera_opencv'], float)


def build_poses():
    """v1 (line fit) and v2 (claw fit) wrist cameras as rendered in the 09-29 build run (world_from_camera, OpenCV)."""
    out = {}
    for v in ('delivery_v1', 'delivery_v2'):
        r = jload(V2_RUN / 'renders' / v / 'render.json')
        out[v] = {s: np.array(r['views'][f'{s}_wrist']['matrix_world']) @ CV2BL for s in SIDES}
    return out


def l6_of_world(side, T_w_cam, fk_home=None, bases=None):
    """camera pose in link_6 at raiden home via the metric base."""
    bases = bases or metric_bases()
    fk_home = fk_home or FK()(HOME_Q)
    return np.linalg.inv(bases[side] @ fk_home['link_6']) @ T_w_cam


# ---------- background model: named 3D segments (metric frame + build config v2, read only) ----------

def named_segments(m=None, cfg=None):
    m = m or jload(METRIC); cfg = cfg or jload(BUILD_CFG)
    d = m['desk']; x0, x1 = d['box_x_m']; y0, y1 = d['box_y_m']; zt = d['top_z_m']
    wy = y1 + cfg['measurements']['wall_gap_m']
    we = cfg['wall_end_x_override']['x']
    g = cfg['glass']; gy = g['plane_y']
    md = cfg['monitor_desk']; mz = md['top_z']
    pu = md['power_unit']
    px0, (py0, _), (pz0, pz1) = md['panel_x'][0], md['panel_y'], md['panel_z']
    seg = {
        # rig desk top (laminate top edge): wood vs wall / floor
        'desk_back_edge': ((x0, y1, zt), (x1, y1, zt)),
        'desk_left_edge': ((x0, y0, zt), (x0, y1, zt)),
        'desk_right_edge': ((x1, y0, zt), (x1, y1, zt)),
        # painted wall end (wall vs frosted glass), vertical on the wall plane
        'wall_end_vertical': ((we, wy, zt + 0.03), (we, wy, 2.4)),
        'glass_frosted_bottom': ((g['post_x'] + 0.05, gy, g['frosted_z'][0]), (we - 0.03, gy, g['frosted_z'][0])),
        # monitor desk (right of the rig desk)
        'monitor_desk_left_edge': ((md['x'][0], md['y'][0], mz), (md['x'][0], md['y'][1], mz)),
        'privacy_panel_left_edge': ((px0, py0, pz0), (px0, py0, pz1 - md['panel_corner_r'])),
        'privacy_panel_bottom_edge': ((px0 + 0.01, py0, pz0), (px0 + 0.6, py0, pz0)),
        'power_unit_front_top': ((pu['x'][0], pu['y'][0], pu['z'][1]), (pu['x'][1], pu['y'][0], pu['z'][1])),
        'power_unit_front_bottom': ((pu['x'][0], pu['y'][0], pu['z'][0]), (pu['x'][1], pu['y'][0], pu['z'][0])),
        'power_unit_left': ((pu['x'][0], pu['y'][0], pu['z'][0]), (pu['x'][0], pu['y'][0], pu['z'][1])),
    }
    return seg


# ---------- images and exclusion masks ----------

def real_image(side):
    return cv2.imread(str(REAL / f'{side}_wrist_camera.jpg'))


def claw_colour_masks(img):
    """orange pad, yellow clamp (HSV), as fit_claw_wrist.py."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    o = (h >= 3) & (h <= 17) & (s >= 140) & (v >= 110)
    y = (h >= 20) & (h <= 34) & (s >= 110) & (v >= 130)
    return o, y


# the real grippers, claws and the croissant in the 09-29 wrist images (hand-drawn, generous; 960x600)
GRIPPER_POLYS = {
    'left': [[(0, 600), (0, 455), (150, 380), (215, 335), (345, 315), (360, 330), (300, 600)],
             [(575, 322), (650, 325), (760, 355), (960, 490), (960, 600), (690, 600)]],
    'right': [[(0, 600), (0, 480), (150, 370), (215, 318), (340, 295), (355, 310), (300, 600)],
              [(570, 305), (700, 325), (780, 370), (960, 500), (960, 600), (690, 600)],
              [(425, 330), (515, 330), (515, 440), (425, 440)]],
}


def exclusion(side, img=None, grow=6):
    ex = np.zeros((600, 960), np.uint8)
    for poly in GRIPPER_POLYS[side]:
        cv2.fillPoly(ex, [np.array(poly, np.int32)], 1)
    if img is not None:
        o, y = claw_colour_masks(img)
        ex |= (o | y).astype(np.uint8)
    if grow:
        ex = cv2.dilate(ex, np.ones((2 * grow + 1,) * 2, np.uint8))
    return ex.astype(bool)


def colour_gradient(img):
    """per-pixel gradient (gx, gy) of the channel with the largest gradient magnitude (Lab, blurred)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] *= 1.5
    gx = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)], -1)
    gy = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)], -1)
    return gx, gy
