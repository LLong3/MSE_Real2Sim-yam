"""ZED scene-camera pose in the aligned reference frame: 2D-3D matches via view-synthesised source frames.

Each Pi3X source frame is forward-warped with its own point map into the ZED view at the current
pose (K from the converted metadata), so the matcher sees one camera's texture at the ZED geometry.
SIFT matches (ratio + F-RANSAC) between the real ZED image and each warped frame give 2D (ZED) -
3D (reference point behind the warped pixel) correspondences; all frames are pooled, then
PnP + RANSAC, LM refinement on the inliers; repeat from the new pose.
Usage: pnp_warp.py IMAGE OUT.npz [init_pose.npz] [iters] [transform.npz with M (ref->metric)]
"""
import sys

import cv2
import numpy as np

import common
import render

IMG = sys.argv[1]
OUT = sys.argv[2]
INIT = sys.argv[3] if len(sys.argv) > 3 else str(common.WORK / 'pose_init.npz')
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
XFORM = np.load(sys.argv[5])['M'] if len(sys.argv) > 5 else np.eye(4)  # reference -> metric (applied to all points)
RATIO = 0.8
PNP_PX = 3.0
EXCL_M = 0.25  # drop reference points this close to the camera centre (pole, bracket, cable)

_, _, K = common.zed_frame(0)
real = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
V, MV, Cols = common.points()
c2w, Kref, fidx = common.cameras()
p0 = np.load(INIT)
R, t = p0['R'], p0['t'].ravel()

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.01)
g_real = clahe.apply(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY))
kr, dr = sift.detectAndCompute(g_real, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=128))

hist = []
for it in range(ITERS):
    C = -R.T @ t
    uv_all, X_all, s_all = [], [], []
    per_frame = []
    for s in range(64):
        P = np.asarray(V[s]).reshape(-1, 3).astype(np.float64) @ XFORM[:3, :3].T + XFORM[:3, 3]
        m = MV[s].ravel() & (np.linalg.norm(P - C, axis=1) > EXCL_M)
        if m.sum() < 1000:
            continue
        col = np.asarray(Cols[s]).reshape(-1, 3)[m]
        Pm = P[m]
        rgb, depth, own = render.render(Pm, col, R, t, K, radius=1)
        cover = (own >= 0).mean()
        if cover < 0.15:
            per_frame.append((s, cover, 0)); continue
        g = clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        ks, ds = sift.detectAndCompute(g, (own >= 0).astype(np.uint8) * 255)
        if ds is None or len(ks) < 20:
            per_frame.append((s, cover, 0)); continue
        mm = flann.knnMatch(dr, ds, k=2)
        good = [a for a, b in mm if a.distance < RATIO * b.distance]
        if len(good) < 8:
            per_frame.append((s, cover, 0)); continue
        a = np.float32([kr[g_.queryIdx].pt for g_ in good])
        b = np.float32([ks[g_.trainIdx].pt for g_ in good])
        # after view synthesis the two images should nearly coincide: keep small displacements
        near = np.linalg.norm(a - b, axis=1) < 40
        a, b = a[near], b[near]
        bi = np.rint(b).astype(int)
        keep = []
        for k in range(len(a)):
            u0, v0 = bi[k]
            win = depth[max(v0 - 1, 0):v0 + 2, max(u0 - 1, 0):u0 + 2]
            if own[v0, u0] < 0 or np.isnan(win).any() or np.nanmax(win) - np.nanmin(win) > 0.03:
                continue
            keep.append(k)
        per_frame.append((s, cover, len(keep)))
        for k in keep:
            uv_all.append(a[k]); X_all.append(Pm[own[bi[k, 1], bi[k, 0]]]); s_all.append(s)
    uv = np.float64(uv_all); X = np.float64(X_all); sl = np.array(s_all)
    rv0, _ = cv2.Rodrigues(R)
    okp, rv, tv, pin = cv2.solvePnPRansac(X, uv, K, None, rvec=rv0.copy(), tvec=t.reshape(3, 1).copy(),
                                          useExtrinsicGuess=True, iterationsCount=50000,
                                          reprojectionError=PNP_PX, confidence=0.9999,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
    if pin is None:
        print('PnP failed', len(uv)); break
    pin = pin.ravel()
    rv, tv = cv2.solvePnPRefineLM(X[pin], uv[pin], K, None, rv, tv)
    err = np.linalg.norm(cv2.projectPoints(X[pin], rv, tv, K, None)[0][:, 0] - uv[pin], axis=1)
    R = cv2.Rodrigues(rv)[0]; t = tv.ravel(); C = -R.T @ t
    h = dict(iter=it, candidates=len(uv), pnp_inliers=len(pin),
             frames_with_inliers=int(len(np.unique(sl[pin]))),
             reproj_median_px=float(np.median(err)), reproj_rms_px=float(np.sqrt(np.mean(err ** 2))),
             centre=C.round(4).tolist(), per_frame_kept=[p for p in per_frame if p[2] > 0])
    hist.append(h)
    print({k: v for k, v in h.items() if k != 'per_frame_kept'}, flush=True)

np.savez(OUT, R=R, t=t, C=C, K=K, uv=uv[pin], X=X[pin], slot=sl[pin], err=err, image=IMG)
common.save_json(OUT.replace('.npz', '.json'), dict(image=IMG, iterations=hist))
