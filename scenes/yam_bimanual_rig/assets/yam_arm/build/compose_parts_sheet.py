"""Real rig fingers / wrist camera next to raiden's fin-ray and ZED-mount CAD (for the user decision).

    .runtime/pi3x-inference/venv/bin/python build/compose_parts_sheet.py --out <media>/04_fingers_wrist_camera_vs_raiden_cad.jpg
Inputs: evidence/video_frames/f1140_t38.0.jpg, evidence/video_frames/crop_gripper_1.jpg,
~/robot/bimanual_bringup/pi05/right_wrist_camera.jpg and evidence/cad_parts/raiden_cad_parts.jpg (cad_views.py).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]


def tile(im, h, text):
    im = cv2.resize(im, (round(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (im.shape[1], 24), (25, 25, 25), -1)
    cv2.putText(im, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    f = cv2.imread(str(HERE / 'evidence/video_frames/f1140_t38.0.jpg'))[560:900, 900:1260]
    top = np.hstack([tile(f, 380, 'rig fingers, IMG_5411 frame 1140'),
                     tile(cv2.imread(str(HERE / 'evidence/video_frames/crop_gripper_1.jpg')), 380,
                          'rig wrist camera: ZED X One on grey bracket'),
                     tile(cv2.imread('/home/frank/robot/bimanual_bringup/pi05/right_wrist_camera.jpg'), 380,
                          'right wrist camera view, 09-29')])
    cad = cv2.imread(str(HERE / 'evidence/cad_parts/raiden_cad_parts.jpg'))
    cad = tile(cad, round(cad.shape[0] * top.shape[1] / cad.shape[1]),
               'raiden docs/assets CAD (mm): finray_short, finray_long, finray_adapter, zed_mounter_left/right (ZED Mini)')
    sheet = np.vstack([top, cad[:, :top.shape[1]]])
    s = 1400 / sheet.shape[1]
    cv2.imwrite(a.out, cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(a.out)


if __name__ == '__main__':
    main()
