"""2D-3D correspondences: real ZED scene-camera image <-> 64 Pi3X source frames (SIFT, F-RANSAC).

Writes work/zed_matches.npz: zed_uv (N,2), ref_xyz (N,3, aligned reference frame), frame slot, source uv.
Run with the pi3x-mesh env (cv2 + numpy).
"""
import sys

import cv2
import numpy as np

import common

ZED_IMG = sys.argv[1] if len(sys.argv) > 1 else str(common.ZED_EP / 'rgb/scene_camera/0000000000.png')
OUT = sys.argv[2] if len(sys.argv) > 2 else str(common.WORK / 'zed_matches.npz')
RATIO = 0.8

V, MV, _ = common.points()
c2w, Kref, fidx = common.cameras()
zed = cv2.imread(ZED_IMG, cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
kz, dz = sift.detectAndCompute(clahe.apply(zed), None)
print('zed keypoints', len(kz), flush=True)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))

rows = []
for slot, f in enumerate(fidx):
    src = cv2.imread(str(common.WORK / f'frames_full/{f:06d}.jpg'), cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, (1280, 720), interpolation=cv2.INTER_AREA)
    ks, ds = sift.detectAndCompute(clahe.apply(src), None)
    mm = flann.knnMatch(dz, ds, k=2)
    good = [m for m, n in mm if m.distance < RATIO * n.distance]
    if len(good) < 12:
        print(slot, f, 'good', len(good)); continue
    pz = np.float32([kz[m.queryIdx].pt for m in good])
    ps = np.float32([ks[m.trainIdx].pt for m in good])
    F, inl = cv2.findFundamentalMat(pz, ps, cv2.USAC_MAGSAC, 1.5, 0.999, 20000)
    if F is None:
        print(slot, f, 'no F'); continue
    inl = inl.ravel().astype(bool)
    pz, ps = pz[inl], ps[inl]
    # source pixel (1280x720) -> processed (672x378), pixel-centre convention
    up = (ps[:, 0] + 0.5) * common.W / 1280 - 0.5
    vp = (ps[:, 1] + 0.5) * common.H / 720 - 0.5
    ui, vi = np.rint(up).astype(int), np.rint(vp).astype(int)
    ok = (ui >= 1) & (ui < common.W - 1) & (vi >= 1) & (vi < common.H - 1)
    keep = []
    for k in np.nonzero(ok)[0]:
        u0, v0 = ui[k], vi[k]
        if not MV[slot, v0, u0]:
            continue
        nb = np.asarray(V[slot, v0 - 1:v0 + 2, u0 - 1:u0 + 2]).reshape(-1, 3)
        valid = MV[slot, v0 - 1:v0 + 2, u0 - 1:u0 + 2].ravel()
        if valid.sum() < 7:
            continue
        spread = np.linalg.norm(nb[valid] - nb[4], axis=1).max()
        if spread > 0.03:  # depth edge
            continue
        keep.append(k)
    keep = np.array(keep, int)
    print(f'slot {slot:2d} frame {f:4d} good {len(good):4d} F-inliers {inl.sum():4d} kept {len(keep):4d}', flush=True)
    for k in keep:
        rows.append((pz[k, 0], pz[k, 1], *V[slot, vi[k], ui[k]], slot, up[k], vp[k]))

rows = np.array(rows)
np.savez(OUT, zed_uv=rows[:, :2], ref_xyz=rows[:, 2:5], slot=rows[:, 5].astype(int), src_uv=rows[:, 6:8], zed_image=ZED_IMG)
print('total correspondences', len(rows))
