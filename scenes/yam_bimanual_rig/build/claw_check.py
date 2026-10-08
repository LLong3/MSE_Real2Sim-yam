"""Claw colour and silhouette check against the real images (render IDs vs HSV colour masks of the real image).

python claw_check.py RENDER_DIR CONFIG OUT.json
Pad = orange TPU, clamp = yellow block. Real masks: HSV thresholds (OpenCV H 0-179): orange H 3-18, yellow H 19-35,
S > 110, V > 90. Render masks: first-hit object IDs of the claw parts. Colours: mean sRGB inside the render mask
eroded by 2 px (scene) / 4 px (wrists), through the camera model. Whole-image MAE without the finger exclusions.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
import camera_model as cm

rd, cfg, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
res = {}
for view in ('scene', 'left_wrist', 'right_wrist'):
    real = cm.real_u8(view)
    camp = cfg['camera_model'][view]
    rend = cm.to_u8(cm.apply_camera(cm.render_lin(rd, view), camp, camp['K']))
    reg, k = cm.region_masks(rd, view, erode=0)
    hsv = cv2.cvtColor(real, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    real_masks = {'claw_pad': (H >= 3) & (H <= 18) & (S > 110) & (V > 90), 'claw_clamp': (H >= 19) & (H <= 35) & (S > 110) & (V > 90)}
    e = 2 if view == 'scene' else 4
    r = {}
    for part in ('claw_pad', 'claw_clamp'):
        rm = reg[part]; qm = real_masks[part]
        if view == 'scene':  # restrict the real colour mask to the gripper neighbourhood (SAM masks, dilated)
            sam = np.load(cm.SAM)['masks']
            qm &= cv2.dilate((sam[0] | sam[1]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        inter, uni = (rm & qm).sum(), (rm | qm).sum()
        er = cv2.erode(rm.astype(np.uint8), np.ones((2 * e + 1,) * 2, np.uint8)).astype(bool)
        eq = cv2.erode(qm.astype(np.uint8), np.ones((2 * e + 1,) * 2, np.uint8)).astype(bool)
        r[part] = dict(iou=round(float(inter / max(uni, 1)), 3), render_px=int(rm.sum()), real_px=int(qm.sum()),
                       render_srgb_in_render_mask=np.round(rend[er].mean(0), 1).tolist() if er.sum() else None,
                       real_srgb_in_real_mask=np.round(real[eq].mean(0), 1).tolist() if eq.sum() else None,
                       dE76_means=round(float(np.linalg.norm(cm.lab(rend[er][None]).mean((0, 1)) - cm.lab(real[eq][None]).mean((0, 1)))), 2) if er.sum() and eq.sum() else None)
    ex = np.zeros(real.shape[:2], bool)
    for poly in cm.EXCLUDE_POLYS.get(view, [])[2 if view != 'scene' else 0:]:
        m = np.zeros(real.shape[:2], np.uint8); cv2.fillPoly(m, [np.array(poly, np.int32)], 1); ex |= m.astype(bool)
    if view == 'right_wrist' or view == 'scene':
        pass
    r['whole_image_srgb_mae_incl_fingers'] = round(float(np.abs(real[~ex].astype(int) - rend[~ex].astype(int)).mean()), 2)
    res[view] = r
out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
