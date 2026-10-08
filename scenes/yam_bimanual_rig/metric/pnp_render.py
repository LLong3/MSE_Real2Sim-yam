"""ZED scene-camera pose in the aligned reference frame: render-and-match PnP.

Loop: render the multi-view-consistent reference cloud from the current pose with the ZED K,
match the real image to the render (SIFT + F-RANSAC), look up the 3D reference point behind
each matched render pixel, solve PnP + RANSAC, refine (LM) on inliers, repeat.
Usage: pnp_render.py IMAGE OUT.npz [init_pose.npz]
"""
import sys

import cv2
import numpy as np

import common
import render

IMG = sys.argv[1]
OUT = sys.argv[2]
INIT = sys.argv[3] if len(sys.argv) > 3 else str(common.WORK / 'pose_init.npz')
ITERS = 6

_, _, K = common.zed_frame(0)
real = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
d = np.load(common.WORK / 'clean_cloud.npz')
P, col = d['points'], d['colors']
p0 = np.load(INIT)
R, t = p0['R'], p0['t'].ravel()

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.01)


def feats(rgb):
    g = clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return sift.detectAndCompute(g, None)


kr, dr = feats(real)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=128))
hist = []
for it in range(ITERS):
    C = -R.T @ t
    far = np.linalg.norm(P - C, axis=1) > 0.25
    Pf = P[far]
    rgb, depth, own = render.render(Pf, col[far], R, t, K, radius=1)
    ks, ds = feats(rgb)
    mm = flann.knnMatch(dr, ds, k=2)
    good = [m for m, n in mm if m.distance < 0.85 * n.distance]
    a = np.float32([kr[m.queryIdx].pt for m in good])
    b = np.float32([ks[m.trainIdx].pt for m in good])
    # small viewpoint change: loose geometric prefilter on displacement consistency
    F, inl = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, 3.0, 0.999, 20000)
    inl = inl.ravel().astype(bool) if F is not None else np.ones(len(a), bool)
    a, b = a[inl], b[inl]
    bi = np.rint(b).astype(int)
    o = own[bi[:, 1], bi[:, 0]]
    # require a stable 3D point: 3x3 depth neighbourhood consistent
    ok = o >= 0
    X = np.zeros((len(a), 3))
    for k in np.nonzero(ok)[0]:
        u0, v0 = bi[k]
        win = depth[max(v0 - 1, 0):v0 + 2, max(u0 - 1, 0):u0 + 2]
        if np.isnan(win).any() or np.nanmax(win) - np.nanmin(win) > 0.03:
            ok[k] = False
            continue
        X[k] = Pf[o[k]]
    a, X = a[ok], X[ok]
    rv0, _ = cv2.Rodrigues(R)
    okp, rv, tv, pin = cv2.solvePnPRansac(X, a, K, None, rvec=rv0.copy(), tvec=t.reshape(3, 1).copy(),
                                          useExtrinsicGuess=True, iterationsCount=20000,
                                          reprojectionError=4.0, confidence=0.9999,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
    pin = pin.ravel()
    rv, tv = cv2.solvePnPRefineLM(X[pin], a[pin], K, None, rv, tv)
    proj = cv2.projectPoints(X[pin], rv, tv, K, None)[0][:, 0]
    err = np.linalg.norm(proj - a[pin], axis=1)
    R = cv2.Rodrigues(rv)[0]
    t = tv.ravel()
    C = -R.T @ t
    hist.append(dict(iter=it, matches=len(good), f_inliers=int(inl.sum()), with_3d=len(a),
                     pnp_inliers=len(pin), reproj_median_px=float(np.median(err)),
                     reproj_rms_px=float(np.sqrt(np.mean(err ** 2))), centre=C.tolist()))
    print(hist[-1], flush=True)

np.savez(OUT, R=R, t=t, C=C, K=K, uv=a[pin], X=X[pin], err=err, image=IMG)
common.save_json(OUT.replace('.npz', '.json'), dict(image=IMG, iterations=hist))
b = (0.5 * real + 0.5 * render.render(P[far], col[far], R, t, K, radius=1)[0]).astype(np.uint8)
cv2.imwrite(OUT.replace('.npz', '_blend.jpg'), cv2.cvtColor(b, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
