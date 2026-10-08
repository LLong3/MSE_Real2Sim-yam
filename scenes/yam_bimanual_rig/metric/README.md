# Metric frame of the YAM bimanual rig twin

`metric_frame.json` is the metric world frame for the `yam_bimanual_rig` twin. It holds:
- the transform from the aligned Pi3X reference to metres;
- the desk box, floor, desk-top and back-wall planes;
- both YAM arm base poses (MJCF q=0) and the rail;
- the ZED X scene camera (intrinsics and pose).

Task `yam-metric-20260929`; the back wall and rail were corrected by task `yam-metric-tape-20260929` from the user's tape (`tape_0929_b`, `revision`). Blender Build, the YAM arm asset task and the later MuJoCo bridge read this file. The schema is stable: fields are only added, never renamed.

## Contents of `metric_frame.json`

| Key | Holds |
|---|---|
| `inputs` | reference run, basis `cameras.json`, ZED episode and images, tape values (`tape_0929`, `tape_0929_b`), MJCF |
| `conventions` | world axes, arm base frame |
| `transform_reference_to_metric` | `M`, `S`, `t`, form, derivation, uncertainty |
| `residuals_final` | residual at every anchor after the transform |
| `alternatives` | candidates A (per axis, chosen), B (uniform 1.06), C (scale 1), D (uniform 1.09), each with residuals and arm ICP |
| `desk` | box, corners, top z, gap to the wall (+ bounds) |
| `planes` | floor, desk top, back wall (`n . p + d = 0`) |
| `arms` | per arm: `base_position_m`, `yaw_deg`, `world_from_base`, ICP stats, free-z / with-fingers / ZED-silhouette checks |
| `arm_spacing`, `rail` | base spacing; rail extents, top height, profiles, pole x/y |
| `scene_camera` | K, FOV, distortion, pose (OpenCV + Blender), checks |
| `back_wall_check` | back-wall verdict (35 mm gap by tape), why the earlier flush reading was wrong, options, evidence |
| `revision` | the latest change to this file: what changed, what did not, the pre-change copy |

## Conventions

- **World:** metres, right-handed, Z up.
  - Floor z = 0; desk top z = 0.72.
  - +X runs along the back wall. -X is the frosted-glass end; +X is the monitor-desk end.
  - +Y points toward the back wall.
  - Origin and axes are those of the aligned reference (`cameras.json`).
- **Transform:** `p_metric = diag(S) p_ref + t`, with no rotation.
  - Camera poses: `c2w_metric = M c2w_ref`, then re-orthonormalise the rotation.
- **Arm base:** the frame of the MJCF `yam.xml` world. Origin at the base mounting face, Z up; at q=0 the arm folds along +x.
  - `world_from_base = [Rz(yaw) | base]`. At yaw 90 deg the grippers point at the wall.
- **Camera:** `world_from_camera_opencv` (x right, y down, z forward).
  - `world_from_camera_blender` = OpenCV @ diag(1,-1,-1). Use it as Blender `matrix_world`.
  - Lens (36 mm sensor, horizontal fit): focal 13.7258 mm, shift_x -0.01399, shift_y 0.00143, render 960x600.
- **Planes:** `n . p + d = 0`, with n a unit vector.

## Key values

| | |
|---|---|
| Transform | S = (1.06, 1.00, 1.09); t = (0.1203, -0.0150, 0.0077) m |
| Desk box | x -3.0488..-1.5288, y 1.3239..2.0839, top z 0.72 (tape 1.52 x 0.76 x 0.72) |
| Back wall | vertical plane y = 2.1189: desk back edge (y 2.0839) + 0.035 m gap (tape) |
| Left arm | base (-2.5752, 1.3798, 0.7426), yaw 90.25 deg |
| Right arm | base (-1.9542, 1.3922, 0.7426), yaw 92.68 deg |
| Base spacing | 0.621 m (ZED silhouettes 0.628); base face 22.6 mm above the desk top |
| Rail | x -2.7647..-1.7647 (1.00 m, tape; centred on the arm bases, each end 0.1895 m outside its base); y 1.314..1.434; top z 0.738 (18 mm plate) |
| Pole | x, y = (-2.276, 1.375) |
| Camera | ZED X 41925345, left eye, rectified 960x600 |
| Camera K | fx = fy = 366.02, cx 492.93, cy 300.87, no distortion |
| Camera pose | position (-2.2848, 1.4969, 1.6057); pitch 47.2 deg down, yaw -2.3 deg, roll -3.1 deg |

## Regenerate

Run the commands in `scenes/yam_bimanual_rig/metric`, one at a time. Only `zed_stereo_pair.py` uses the GPU (ZED SDK). Run it alone, with no other GPU job.

```bash
source ../../../kimodo_blender/env.sh
PY=../../../.runtime/pi3x-mesh/venv/bin/python   # numpy, scipy, OpenCV, matplotlib (same versions as the Pi3X env)
RPY=~/robot/raiden/.venv/bin/python              # mujoco + i2rt, pyzed 5.5
L=logs
$PY extract_frames.py        > $L/extract_frames.log   # 64 full-res source frames -> work/frames_full (keeps existing)
$PY clean_cloud.py           > $L/clean_cloud.log      # multi-view consistent reference cloud -> work/clean_cloud.npz
$PY floor_check.py           > $L/floor_check.log      # floor near the desk -> work/floor_check.json, review 03
$PY zed_metric.py                                      # ZED desk plane, edges, crease frame -> work/zed_metric.{json,npz}
for o in 0 0.035 0.0475 0.055 0.06; do $RPY yam_mesh.py $o work/yam_q0_open$o.npz; done   # MJCF q=0 surfaces
$PY arm_icp.py                                         # MJCF vs reference arms (per-axis scale diagnosis)
$PY fit_metric.py            > $L/fit_metric.log       # anchors, candidates A-D, arm ICP -> work/fit_metric.json, xform_*, pose_zed_direct_*
$PY arm_zed_fit2.py          > $L/arm_zed_fit2.log     # MJCF gripper silhouettes vs the ZED images
$PY pnp_warp.py /home/frank/robot/bimanual_bringup/pi05/scene_camera.jpg work/pnp_A_0929.npz \
    work/pose_zed_direct_A_per_axis.npz 4 work/xform_A_per_axis.npz > $L/pnp_A_0929.log   # PnP cross-check (arguments inferred, not logged)
$PY build_metric.py          > $L/build_metric.log     # -> metric_frame.json
# back-wall check (09-29 evening)
$PY wall_check.py            > $L/wall_check.log       # ZED neural depth on the wall -> work/wall_check.json
$RPY zed_stereo_pair.py 60   > $L/zed_stereo_pair.log  # GPU: rectified L/R from the SVO2, DEPTH_MODE.NONE
$PY stereo_edges.py          > $L/stereo_edges.log     # classical stereo: desk plane, wall-end edge
$PY wall_gap_reference.py    > $L/wall_gap_reference.log
$PY video_slot_check.py      > $L/video_slot_check.log
$PY back_wall.py             > $L/back_wall.log        # merges the 09-29 flush verdict (superseded by tape_b.py; keep the order)
$PY tape_b.py                > $L/tape_b.log           # user tape 09-29 (b): 35 mm gap, 1.00 m rail; evidence -> work/tape_b/
$PY verify_metric.py         > $L/verify_metric.log    # self-consistency; exit 1 on any failure
$PY review_images.py         > $L/review_images.log    # 01, 02, 04, 05 (or a subset, e.g. 01 02 05) -> ~/robot/markdowns/bimanual/1_media/alignment/
```

`tape_b.py` replaces `metric_frame.json` atomically and keeps the pre-change copy in `work/metric_frame.before_tape_b.json`. Its rail-end pixel picks were made on crops from `work/tape_b/rail_overlay.py` and `rail_grid.py` (frame numbers as arguments; `PYTHONPATH=.`).

Superseded or diagnostic scripts, not needed: `match_zed.py` (plain SIFT failed), `pnp_render.py`, `arm_zed_fit.py`. Shared modules: `common.py`, `metric_frame.py`, `render.py`.

`work/ref_*.npy` (250 MB) are caches of the reference `layers.npz`. You can delete them; `common.points()` rebuilds them.

## Known limits

- **Scale is anisotropic.** The Pi3X shape error differs per axis: x 1.06, y 1.00, z 1.09.
  - y is 1.00-1.04: the desk wood depth gives 1.03, the upper wall 1.01, the MJCF arm length 1.00-1.04. Near the desk the choice moves nothing by more than 3 mm; far from it, reference y distances may read up to 4 % short.
  - Above the desk, the z scale is uncertain (1.05-1.12). Arm tops, the pole and the camera body are uncertain by 1-3 cm in z.
  - Near the desk, the local error is 2-3 cm.
- **Back wall.** There is a 35 mm gap between the desk back edge and the wall (user tape, 09-29). Build the wall vertical at y = 2.1189.
  - The earlier conclusion "flush to ~1 cm" was wrong. Why:
    - Pi3X flattened the wall seen down the gap into a ~2 cm desk-height strip of wall-paint points behind the wood, and pulled the lower wall onto it. The "desk-top points that reach the wall" were that strip. The wall "lean" of 1.8-2.2 deg is the same merge.
    - From above, the gap hides. The ZED ray past the edge drops 56 deg, so the 35 mm gap spans ~10 px and shows beige wall in shadow, continuous with the wall above. There is no black slot. The source video shows the same darker band (24-38 px, at least the 19-24 px of a 35 mm slot); it was read as lighting, and only the thin line inside it was measured.
    - The ZED readings next to the desk showed the gap and were explained away: neural depth 20-43 mm at 0-10 cm above the desk, and classical stereo 41 mm at the painted wall end (read as a reveal).
  - The desk back edge (y 2.0839) is the real wood edge in the metric frame. The ZED wood/wall hue boundary lies on it (+0.1 / -0.5 px, 1 px ~ 3.5 mm). The camera pose is anchored on it and does not depend on the wall, so the pose is unchanged.
    - The box value itself came from the reference's merged desk/wall line. The reference's own wood edge sits 22 mm in front of that line, 35 mm in front of y 2.0839 after the -15 mm y offset. The desk depth still matches the tape because this 22 mm offsets the ~3 % short Pi3X y scale.
  - The ZED depth now agrees next to the desk: -1.8 mm median behind the plane at 0-10 cm above the desk top. Higher up, NEURAL_LIGHT on the textureless wall reads a median 46-51 mm (p95 ~200 mm) behind the plane. Do not use the ZED depth on the wall above ~10 cm.
  - The Pi3X wall sits 17-54 mm in front of the plane (the merge). Do not author the wall from the reference.
  - At the glass end, the painted wall ends 0.045-0.06 m right of the desk left end. Its end edge is on the wall plane (41 mm behind the desk edge by ZED stereo). The frosted glass to its left is recessed further: about 0.07 m behind the desk edge in the reference, possibly up to ~0.12 m.
- **Arms.** Both bases share one height (the rail top).
  - The right base is 12.7 mm further +Y than the ZED silhouette fit.
  - The ICP yaws are 90.2 / 92.7 deg; the ZED silhouette fits give 89.6 / 91.1 deg.
- **Rail.** Length 1.00 m by tape, centred on the arm-base midpoint (x -2.2647); ends ±1 cm.
  - Evidence: rail ends picked in source frames 341/381/401/421 and read through each frame's own Pi3X point map put the centre at -2.269 (0.4 cm from the arm midpoint; front photo 401: equal overhangs). The same picks read the rail 0.94-0.98 m long, so Pi3X is 2-3 cm short at each end. The earlier 1.048 m extent (right end mirrored) is replaced.
  - C-clamps hold both ends to the desk front edge.
- **Camera.** The pose comes from the ZED alone: desk edges plus the desk plane from depth. The ZED depth is scaled by 0.9912 so the desk length matches the tape.
  - The PnP against the reference differs by 3-5 cm and 4 deg; this is the reference shape error.
  - The distortion is none because the image is the rectified left-eye image.
- **Floor.** The floor is set to z = 0 near the desk. Away from the desk it is +8 mm median (p90 +31 mm).
- **Validity.** Valid for the rig as set up on 09-28/09-29. The desk edges in the 09-28 frame and the 09-29 image agree to within 0.2 px.

## Measurements (user)

1. **Desk back edge to wall:** 0.035 m (tape, 09-29, `inputs.tape_0929_b`). Settled.
2. **Rail length:** 1.00 m (tape, 09-29). The end positions are not taped; they are placed symmetric about the arm bases (see Known limits).
