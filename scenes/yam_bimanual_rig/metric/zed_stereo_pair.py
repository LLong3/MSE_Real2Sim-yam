"""Rectified left/right scene-camera images from the raw SVO2 (read-only), no SDK depth (DEPTH_MODE.NONE).

Run with the raiden venv (pyzed 5.5): ~/robot/raiden/.venv/bin/python zed_stereo_pair.py
Writes work/wallcheck/zed_stereo_pair.npz: mean of the first N frames per eye (rig static), K, baseline.
"""
import sys
from pathlib import Path

import numpy as np
import pyzed.sl as sl

SVO = '/home/frank/robot/bimanual_bringup/cameras/e2e_sdk55/0000/cameras/scene_camera.svo2'
OUT = Path(__file__).resolve().parent / 'work/wallcheck/zed_stereo_pair.npz'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

init = sl.InitParameters()
init.set_from_svo_file(SVO)
init.depth_mode = sl.DEPTH_MODE.NONE
init.coordinate_units = sl.UNIT.METER
init.svo_real_time_mode = False
cam = sl.Camera()
err = cam.open(init)
if err != sl.ERROR_CODE.SUCCESS:
    raise SystemExit(f'open failed: {err}')
info = cam.get_camera_information()
cal = info.camera_configuration.calibration_parameters
res = info.camera_configuration.resolution
print('serial', info.serial_number, 'resolution', res.width, res.height, 'frames', cam.get_svo_number_of_frames())
L, R = [], []
ml, mr = sl.Mat(), sl.Mat()
rt = sl.RuntimeParameters()
while len(L) < N:
    e = cam.grab(rt)
    if e != sl.ERROR_CODE.SUCCESS:
        print('grab stopped:', e)
        break
    cam.retrieve_image(ml, sl.VIEW.LEFT)
    cam.retrieve_image(mr, sl.VIEW.RIGHT)
    L.append(ml.get_data()[..., :3].astype(np.float32))
    R.append(mr.get_data()[..., :3].astype(np.float32))
lc, rc = cal.left_cam, cal.right_cam
K = np.array([[lc.fx, 0, lc.cx], [0, lc.fy, lc.cy], [0, 0, 1]])
Kr = np.array([[rc.fx, 0, rc.cx], [0, rc.fy, rc.cy], [0, 0, 1]])
baseline = float(cal.get_camera_baseline())
np.savez_compressed(OUT, left=np.mean(L, 0).astype(np.float32), right=np.mean(R, 0).astype(np.float32),
                    left0=L[0].astype(np.uint8), right0=R[0].astype(np.uint8), K=K, K_right=Kr, baseline_m=baseline, n=len(L))
print('frames used', len(L), 'K', K.round(3).tolist(), 'K_right', Kr.round(3).tolist(), 'baseline m', baseline)
cam.close()
