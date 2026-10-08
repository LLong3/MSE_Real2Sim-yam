"""Rail-end evidence in the source frames: project metric geometry (arm base axes, pole, desk front edge, rail front-top
edge with 1 cm ticks near both ends) into full-res source frames through the reference cameras (metric -> reference by
the inverse transform). Writes work/tape_b/rail_<frame>_{left,right,full}.jpg."""
import json
import sys

import cv2
import numpy as np

import common

cam = json.load(open(common.REF / 'alignment/cameras.json'))
fr = {f['source_frame']: f for f in cam['frames']}
J = json.load(open(common.HERE / 'metric_frame.json'))
M = np.array(J['transform_reference_to_metric']['M'])
S, t = np.diag(M)[:3], M[:3, 3]
s = 1920 / cam['processed_size_wh'][0]
arms = {k: np.array(v['base_position_m']) for k, v in J['arms'].items()}
pole = J['rail']['pole_xy_m']
y_front, ztop = 1.314, J['rail']['top_z_m']
x0, x1 = J['desk']['box_x_m']
y0 = J['desk']['box_y_m'][0]


def proj(F, Pm):
    Pm = np.atleast_2d(Pm)
    Pr = (Pm - t) / S
    w2c = np.linalg.inv(np.array(F['c2w']))
    K = np.array(F['intrinsics']) * s
    K[2, 2] = 1
    q = Pr @ w2c[:3, :3].T + w2c[:3, 3]
    return q[:, :2] / q[:, 2:3] @ K[:2, :2].T + K[:2, 2], q[:, 2]


def draw(frame, out_prefix):
    F = fr[frame]
    img = cv2.imread(str(common.WORK / f'frames_full/{frame:06d}.jpg'))
    ov = img.copy()
    P = lambda p: tuple(int(round(c)) for c in proj(F, p)[0][0])
    cv2.line(ov, P([x0, y0, 0.72]), P([x1, y0, 0.72]), (255, 255, 0), 1)                    # desk front edge
    for k, b in arms.items():
        cv2.line(ov, P(b), P(b + [0, 0, 0.12]), (0, 0, 255), 2)                          # base yaw axis
        cv2.circle(ov, P(b), 4, (0, 0, 255), -1)
    cv2.line(ov, P([pole[0], pole[1], 0.74]), P([pole[0], pole[1], 1.0]), (255, 0, 255), 2)
    ticks = {}
    for xe in np.arange(-2.83, -2.70, 0.01).tolist() + np.arange(-1.84, -1.71, 0.01).tolist():
        a, b = P([xe, y_front, ztop]), P([xe, y_front, ztop + (0.02 if round(xe * 100) % 5 == 0 else 0.01)])
        cv2.line(ov, a, b, (0, 255, 0) if round(xe * 100) % 5 else (0, 255, 255), 1)
        if round(xe * 100) % 5 == 0:
            cv2.putText(ov, f'{xe:.2f}', (b[0] - 14, b[1] - 4), 0, 0.35, (0, 255, 255), 1)
        ticks[round(xe, 2)] = a
    cv2.line(ov, P([-2.83, y_front, ztop]), P([-1.71, y_front, ztop]), (0, 200, 0), 1)
    cv2.imwrite(str(common.WORK / f'tape_b/{out_prefix}_full.jpg'), cv2.resize(ov, None, fx=0.5, fy=0.5))
    for side, xe in (('left', -2.765), ('right', -1.765)):
        u, v = P([xe, y_front, ztop])
        crop = ov[max(v - 110, 0):v + 110, max(u - 180, 0):u + 180]
        cv2.imwrite(str(common.WORK / f'tape_b/{out_prefix}_{side}.jpg'), cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    # local px per cm along the rail at both ends
    return {side: float(np.linalg.norm(np.subtract(ticks[round(a + 0.01, 2)], ticks[round(a, 2)])))
            for side, a in (('left', -2.78), ('right', -1.77))}


if __name__ == '__main__':
    for f in map(int, sys.argv[1:]):
        print(f, 'px per cm', draw(f, f'rail_{f}'))
