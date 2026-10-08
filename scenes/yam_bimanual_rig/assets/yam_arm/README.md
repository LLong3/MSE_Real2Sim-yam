# YAM arm asset (i2rt YAM + linear_4310 + UMI-FT claws), scene-local

One poseable YAM arm with the linear_4310 gripper and the rig's UMI-FT claws, for the `yam_bimanual_rig` twin. Scene-local; not a registered RoomKit library item. Append it twice for the two arms.

- **Asset:** `yam_arm.blend`, collection `YAM arm (i2rt yam + linear_4310)`, root Empty `yam_arm` (asset version 2).
- **Asset version 4:** `yam_arm_v4.blend` (09-30, `source/umi_ft/claw_v4.json`): the same claws plus a per-arm claw mount. See "Version 4". `yam_arm.blend` is unchanged.
- **Build:** `build/` (reproducible; see "Rebuild").

## Version 4 (09-30, per-finger follow-up, task `yam-claw-finger-fit-20260930`)

Run `runs/yam_bimanual_rig/20260930t181852z-claw-finger-fit-6deaf8d1/` (RUN): photos IMG_5412-5415, clamp screws in the wrist images, `scripts/`, `stages/v4_final/fit.json`, `check_v4/`.

- **Claw mount:** an Empty `yam_claw_mount` under link_6 holds both finger bodies (carriage + claw). It rolls about the link_6 z axis (the gripper centre line) from the root property `claw_mount_roll`. Per-arm values are in `root['claw_mount_roll_by_arm']`: left arm -4.27 deg, right arm 0. The saved default is 0, which is the MJCF gripper, so the FK check is unchanged (PASS, `fk/fk_check_v4.json`).
- **Per arm:** call `yam_rig.setup_arm(root, 'left_arm' | 'right_arm')` after appending. It sets the tags and the roll. `build_scene.py` still sets only the tags.
- **Why a roll:** in the left wrist image the tip_left clamp screws are 36 px off and tip_right fits. The fixes that were rejected:
  - a per-finger offset: rejected by the scene camera and by IMG_5414 (the two claws are aligned along z6 within 1.3 mm);
  - a wrist camera roll: rejected by the background (1.46 -> 6.05 px).

  The claw-pair roll plus a 9 mm camera translation fits the four screws to 2.7 px.
- **Geometry:** schema 2 unchanged (1 mm placeholder, user answer). No pad trim: IMG_5413 matches the pad back within 0.5-2 mm, and the proximal overshoot came from the camera distance. No per-finger offsets.
- **Open:** IMG_5413 shows a 3.4-3.8 mm yellow strip next to the clamp (`claw_v4.json` `placeholder_observed_0930`).
- **Wrist cameras:** `evidence/wrist_fit_v4/wrist_cameras.json`. The translation moved 9.0 mm (left) and 3.4 mm (right) within the stage-2 prior; the rotation is unchanged. `evidence/wrist_fit/` is unchanged.
- **MuJoCo:** `source/umi_ft/mjcf_v4/`. The meshes, inertial and boxes are the same as `mjcf/`. The snippet adds the left-arm `umi_claw_mount` body, and `yam_linear_4310_umi_ft_left_arm.xml` is a test copy with the mount.
- **Rebuild:** `build_yam_arm.py`, then `add_umi_ft_claws.py -- --claw source/umi_ft/claw_v4.json --out yam_arm_v4.blend`, then `fk_check.py`; `umi_ft_mjcf.py --claw source/umi_ft/claw_v4.json --out source/umi_ft/mjcf_v4`. The schema-2 defaults reproduce `yam_arm.blend` and `mjcf/` exactly (checked).

## Source

- raiden `a8316968` (branch `bimanual`), `raiden/_xml_paths.get_yam_4310_linear_xml_path()` = i2rt `combine_arm_and_gripper_xml(yam.xml, linear_4310.xml)`. This is the model raiden's IK/FK, calibration and the MESA twin's link offsets use.
- i2rt submodule `third_party/i2rt` @ `7b6d5016` (github i2rt-robotics/i2rt). The files were read from raiden's `.venv` copy (`site-packages/i2rt/robot_models/...`), byte-identical (md5) to the submodule:
  - `arm/yam/yam.xml` md5 `81203a99…`, meshes `arm/yam/assets/{base,link1..link5}.stl`
  - `gripper/linear_4310/linear_4310.xml` md5 `85fd2aaa…`, meshes `gripper/linear_4310/assets/{gripper,tip_left,tip_right}.stl`
  - the STLs are i2rt's decimated meshes (8000 faces or fewer each), the same ones MuJoCo loads.
- Combined MJCF copy: `source/yam_linear_4310.xml`. Structure dump: `source/mjcf_model.json` (MuJoCo 3.3.5, raiden venv, read-only).
- Shell/motor colour split: `source/face_labels.npz`. Transferred from the Menagerie YAM (MESA copy `vla-benchmark/mesa/sim/envs/robots/assets/yam/yam_robot.xml`, which splits links 2 and 3 into black and white parts) by a per-link rigid ICP fit to the i2rt meshes (median residual 0.3-1.0 mm; `source/face_labels.json`). Geometry is the i2rt mesh, unchanged.
- UMI-FT claws: see "Fingers (UMI-FT claws)".

## Hierarchy

Root `yam_arm` = MJCF world frame = arm base frame: origin at the bottom of the arm foot, +Z up, +X toward the gripper at home. One Empty per MJCF body, parented as in the MJCF. Each Empty's world frame equals MuJoCo's body frame (`xpos`/`xquat`). Mesh geoms are children at their XML `pos`/`quat` with the raw STL vertices.

```
yam_arm (world)            geom base            [joint controls on this object]
└─ yam_link1   joint1 hinge z   [-2.618, 3.054]   geom link1
   └─ yam_link2   joint2 hinge z   [0, 3.65]      geom link2
      └─ yam_link3   joint3 hinge z   [0, 3.665]  geom link3
         └─ yam_link4   joint4 hinge z   [-1.571, 1.571]   geom link4
            └─ yam_link5   joint5 hinge z   [-1.571, 1.571]   geom link5
               └─ yam_link_6   joint6 hinge -z   [-2.094, 2.094]   geom gripper, sites tcp_site, grasp_site
                  ├─ yam_tip_left    joint7 slide -z  [0, 0.0475]   geom tip_left (hidden), carriage, UMI-FT claw
                  └─ yam_tip_right   joint8 slide -z  [0, 0.0475]   geom tip_right (hidden), carriage, UMI-FT claw   (joint8 = joint7)
```

Every link Empty carries `mjcf_body`, `mjcf_parent`, `body_pos`, `body_quat_wxyz`, `joint_name`, `joint_type`, `joint_axis` (in its own body frame), `joint_range` and `joint_control`. Resolve links by `mjcf_body`, not by object name; appended copies get `.001` names.

Each finger body has these children:
- the stock MJCF geom `yam_tip_*_geom`: hidden from render, drawn as wire; kept for `fk_check.py` and the MJCF;
- `yam_tip_*_carriage`;
- the claw parts `yam_tip_*_umi_{pad,clamp,placeholder,holder_white,holder_black,adapter_black,adapter_white,tag}` (claw.json schema 2, 09-30).

Claw objects carry `claw_part`, `claw_finger` and `asset_part_key`. Their vertices are in the body frame (identity transform), so they follow the finger joints.

## Set joints and tags

The root has 7 custom properties: `joint1`..`joint6` (radians, UI shows degrees, clamped to the MJCF ranges) and `gripper` (metres, per-finger stroke = MJCF joint7 = joint8; 0 closed, 0.0475 open). Each link follows through simple-expression drivers (`sin`/`cos` only). Python auto-run is not needed.

- **Saved default = raiden home** (`FOLLOWER_HOME_POS`): arm joints 0, gripper 1.0 (open) = 0.0475 m. MJCF q=0 has the gripper closed.
- **In the UI:** select `yam_arm` and edit Object Properties > Custom Properties.
- **In Python:** use `build/yam_rig.py` (also stored in the file as the text block `yam_rig.py`):

```python
import sys; sys.path.insert(0, '<asset dir>/build'); import yam_rig
root = yam_rig.roots()[0]
yam_rig.set_joints(root, [0.3, 1.0, 1.2, 0.0, 0.2, 0.0], gripper=0.02)   # or q with 7 values
yam_rig.set_finger_tags(root, *yam_rig.ARM_TAG_IDS['right_arm'])        # left_arm (7, 6) = saved default, right_arm (1, 0)
yam_rig.links(root)['link_6'].matrix_world
```

- **Two arms:** append the collection twice (`bpy.data.libraries.load(..., link=False)`, or File > Append). Each copy has its own root, and its drivers point to that root (checked). Move only the root: its pose is the arm base pose in the scene. Set each copy's tags with `set_finger_tags`: the tag mesh holds all four tag materials, and the helper picks one.

## FK check (Blender vs MuJoCo)

`build/fk_check.py` → `fk/fk_check.json`. MuJoCo reference: `build/mjcf_reference.py` → `fk/mujoco_poses.json`, `fk/mujoco_geom_verts.npz`. Five configurations: MJCF q=0 (gripper closed), raiden home (gripper open), and 3 uniform random configurations in the joint ranges (seed 20260929, fingers coupled). The check runs twice: root at the identity, and root moved to (-2.3, 1.39, 0.654) and turned -90° about Z. Poses are compared in the root frame.

| Check | Worst over all configs | Target |
|---|---|---|
| Link frame position (9 bodies) | 0.0014 mm | < 1 mm |
| Link frame rotation | 0.0002° | < 0.1° |
| Sites `tcp_site`, `grasp_site` | included in the above | |
| Mesh vertices (every MuJoCo compiled vertex vs the Blender vertex; stock tip geoms included) | 0.0018 mm | < 1 mm |
| Two appended copies pose independently | pass | |

Result: PASS on the UMI-FT asset (`fk/fk_check.json`, 09-29 23:41Z), with the same numbers as before the claws. The residual is float32 round-off. `add_umi_ft_claws.py` asserts that the claw parts are children of the finger bodies with body-frame vertices.

## Fingers (UMI-FT claws)

The rig's claw is a modified UMI-FT claw (UMI-FT, https://umi-ft.github.io/, arXiv 2601.09988). The linear_4310 motor and carriage are unchanged (user, 09-29). So the MJCF bodies `tip_left`/`tip_right`, their slide joints and the actuation stay; only the claw geometry changes.

### Sources

User exports from the public Onshape document "UMI-FT Public", copied read-only from `~/Downloads` into `source/umi_ft/`. Units: metres.

| File (Onshape element) | md5 | Content |
|---|---|---|
| `umift_finger_no_contact_mic.stl` ("UMIFT finger no contact mic") | `217d8bef2cde83e082376b1232337bc2` | the fin-ray finger alone: 1732 faces, watertight, 25.0 x 17.2 x 96.7 mm, 12.75 cm³. Flat contact face, 12 diagonal ribs, curved back, two lugs 19 mm apart. |
| `umift_assembled.stl` ("UMIFT assembled") | `d90b9fbe02e4bf08089f5d51b2548f09` | the handheld UMI-FT: 143,564 faces, 204 x 219 x 238 mm, 2349 connected parts (phone holder, handle, gripper, two fingers, holders, clamps, CoinFT plates, screws). |
| `umift_wsg50_gripper.stl` ("UMIFT WSG50 gripper") | `bf4f92bfe415dcd80f1c5ff7b9ff5534` | UMI-FT on a WSG50: 49,406 faces, 166 x 196 x 84 mm, 15 parts: WSG50 body, two holders (43 x 109 x 38 mm), two jaw adapters (18 x 37 x 30 mm), two clamps (14.2 x 31 x 29 mm), top mount, connectors, pins. No fin-ray fingers. |

- Onshape: document `b31120a070e70ac6a98354f5`, workspace `3ec325c31eb83364aeb8c9d2` (owner Hojung Choi, modified 2026-02-17). URL: https://cad.onshape.com/documents/b31120a070e70ac6a98354f5/w/3ec325c31eb83364aeb8c9d2/e/ef873b4e8cfb682e63e2f473. Linked from the UMI-FT hardware instructions. Elements:
  - assembly "UMIFT assembled";
  - part studio "UMIFT finger no contact mic" `d994188ebbe136e88378c8a6`;
  - part studio "UMIFT finger with contact mic" `50bc7117eb45a1de5cfcb682`;
  - part studio "UMIFT WSG50 gripper" `04037fd9c7b23ac2d4152b18`;
  - part studio "UMIFT gripper" `fca90b48cb46ce9166d7f506`.
- Licence: none stated on the Onshape document. The UMI-FT code repo (github.com/real-stanford/UMI-FT, HEAD `5e41969881`) and the UMI repo are MIT. Keep the attribution; check before redistributing the meshes.
- Anonymous Onshape API access gives thumbnails (`evidence/umi/`) and the feature list, but no geometry; hence the user exports. The feature list is `source/umi_ft/features_finger_no_mic.json` (sourceMicroversion `b9b1c1bfeae66181bbb2a080`). Its finger_width 25 mm and finger_thickness 17.2 mm match the STL. Its finger_length 84 mm is a sketch dimension; the STL is 96.7 mm tip to tip (not traced further).
- `build/umi_ft_parts.py` (raiden venv) → `source/umi_ft/parts/` (`parts.json`: inventory, transforms, volumes).
  - The finger STL matches one assembled finger exactly (axis-permutation fit, residual 8e-9 m). The other assembled finger is the with-mic variant (4.1 mm off).
  - The clamp and holder that touch that finger are taken from the assembly and expressed in the finger frame, so the finger-clamp-holder relation is UMI-FT's.
  - The WSG50 file shows the same holder and clamp on a parallel jaw through an adapter: the pattern the team followed on the 4310 carriage.

### What the asset has, per finger (`build/add_umi_ft_claws.py`)

Finger frame F = the finger STL frame: x toward the contact face, y = extrusion, z = length, lugs toward +z.

| Part | Source | Material |
|---|---|---|
| pad | UMI-FT finger STL, unchanged. Its flat contact face (26 triangles at x_F = 0) is a second material slot: black, appearance only (user 09-30: the friction strip is thin, no thickness) | orange TPU, black contact face |
| clamp | UMI-FT clamp from the assembly, unchanged | yellow PLA |
| holder_white / holder_black | UMI-FT holder from the assembly, clipped at z_F = -61.5 mm (the UMI rack plate is dropped). The top 17 mm (y_F < -6 mm) behind z_F = -17.8 mm is the rig's black tag block. Shifted +1.53 mm along x_F with the tag and adapter, so that the CoinFT slot closes to the placeholder. | white / black PLA |
| placeholder | 1.0 mm box in the CoinFT slot between holder and clamp (user 09-30: no CoinFT, a thin yellow placeholder) | yellow PLA |
| adapter_black / adapter_white | 5.1 mm box from the holder to the carriage block (the rig's adapter detail is not visible) | black / white PLA |
| tag | 24.6 mm square on the tag block top face: ArUco DICT_4X4_50, 21.2 mm marker, 0.8 mm white outline (IMG_5411 f1256). Image up = toward the fingertip. | tag image |
| carriage | stock `tip_*` faces above z6 = -70 mm (rack slider and end block; linear_4310, unchanged) | black gripper housing |

The right claw is the left one mirrored in y6 (normals checked); the tag images are not mirrored. Placement, from `source/umi_ft/claw.json` (link_6 at raiden home):

- The pad contact face is at y6 = ∓51.6 mm (`open_half`). The gap is 103.2 mm at home and 8.2 mm at q = 0 (MuJoCo check). The pads do not touch when closed (user 09-30).
- The finger-frame origin is at z6 = -135.6 mm:
  - the pad runs from z6 -93.6 to -190.3 mm (stock claw tip: -144.7);
  - the clamp's distal end is at -166.8;
  - the claw's proximal end is at -69.0, at the bottom of the carriage block.
- The pad mid-plane is at x6 = 0: the claw is centred on the carriage, like the stock claw.
- The claw's outer face (holder) is at y6 = ∓92.7 mm; the housing sides are at ±51 mm.

Tags on the rig (UMI convention 6·gripper + finger): left arm tip_left 7, tip_right 6; right arm tip_left 1, tip_right 0.
- Ids 0 and 6 were decoded with cv2.aruco on IMG_5411 (frames 1058-1256 and 910-1170; the video's near arm is the right arm).
- Ids 1 and 7 are matched by glyph in the ZED scene image.
- The textures are generated from the dictionary (`source/umi_ft/tags/`, each checked to decode) and packed in the .blend.

### User decisions (09-30, user origin, via the orchestrator)

In `source/umi_ft/claw.json` (`user_answers_0930`), applied by `add_umi_ft_claws.py` and `umi_ft_mjcf.py`:
- The pads do not touch when fully closed: the gap at q = 0 is kept (8.2 mm with the fitted opening).
- The friction strip is thin: no thickness. The pad's contact face is black (appearance only).
- No CoinFT sensor: a thin yellow placeholder (modelled 1.0 mm); the holder group moves +1.53 mm along x_F to close the slot.
- The holder and adapter are custom: they only need to look right from the scene and wrist cameras.

### How the placement was set (09-30: stage 1 and stage 2, current)

Run `runs/yam_bimanual_rig/20260930t125032z-wrist-claw-fit-efcb76a0/` (RUN). Scripts in `build/`: `wrist_bg_fit.py`, `wrist_capture_check.py`, `claw_model.py` (numpy label render; IoU 0.95-0.98 against the Blender object IDs), `claw_stage2_fit.py`, `stage2_summary.py`, `write_wrist_records.py`, `make_twin_config.py`, `twin_check.sh`, `compose_wrist_fit_review.py`. Data: the 09-29 16:00 images, arms at raiden home.

- **Stage 1** (RUN/stage1): each wrist camera's world rotation from named background edges (desk, wall, monitor), the claws masked. Named-edge median 0.72 / 1.07 px (left / right; v1 5.3 / 1.75, v2 7.4 / 3.7). The background does not fix the camera translation (several cm).
- **Stage 2** (RUN/stage2): the claw placement and the arm pose at home, with the stage-1 rotations fixed. Loss: IoU of the wrist pad / clamp / black face, IoU of the scene-camera silhouette / pad / clamp, background edges, weak priors. Variants a, a_mount, b, b_capX, b_joints, b_mount, c, c_mount, c_bg0.045, c_bg0.15; table in RUN/stage2/summary_all.json, rail forms in RUN/stage2/rail_forms.json.
  - Metric bases (a, a_mount) cannot fit the scene camera (silhouette IoU 0.54-0.71).
  - b (camera on raiden's hand-eye translation) needs 7-11° of base pitch, or 11-18° of joint offsets: rejected.
  - c (camera translation free, 15 mm prior, background term) needs 4.4 / 2.4° of pitch: plausible as joint offsets.
  - Higher background weights (c_bg*, stopped after 2 rounds) reach 0.7 / 1.1 px, but the wrist IoU falls to 0.41-0.48.
- **Chosen: c.** `open_half` 51.6 mm and `pad_z6` -135.6 mm from c_mount (its 0.8° mount rotation is not significant and is not applied). `pad_x6` 0 (not identifiable). The final arm poses and cameras: c re-fitted with this placement fixed (RUN/stage2/final_c). Across c and c_mount the placement spreads 0.4 / 1.4 mm.
- **Only the link_6 pose at home is observable**: links 1-5 are outside the three views. A base tilt and joint 2/3/4 zero offsets (or sag) give the same images. The proposal is therefore the rail form: base at the metric height, no tilt; base yaw and x, y fitted; home offsets j2 - j3 = 4.4 / 2.4° and j6 = -0.4 / 2.9° (`evidence/wrist_fit/arm_bases_proposed.json`).
- **Result vs v2** (Blender check render RUN/check_final, `claw_check.py`):
  - scene: pad IoU 0.57 (v2 0.23), clamp 0.58 (0.07); silhouette centroid 3-4 px (v2 13-14 px);
  - wrist background: named-edge median 1.5 / 3.1 px (v2 7.4 / 3.7);
  - wrist claws: pad 0.52 / 0.51 (v2 0.68 / 0.61), clamp 0.62 / 0.76 (v2 0.70 / 0.77). This is worse than v2. In the left wrist, one claw is about 35 px off along the pad length; the other claw fits (see `evidence/wrist_fit/`).
- **Board square size:** 35.56 mm (user tape, 09-30) puts the calibration board 10.5-12 mm above the desk, parallel to it. The captures at 35.56 and 36 mm are both kept in the records.

### How the placement was set (09-29, history; superseded by the section above)

- **Wrist images** (`build/fit_claw_wrist.py` → `evidence/claw_fit/fit.json`, CPU). The pad, clamp and strip silhouettes are matched to colour masks of the 09-29 wrist images.
  - Cameras: the raiden hand-eye result and the HD1200 K/2. The live 960x600 stream K is the same, with fx ≠ fy.
  - Each camera gets a small correction (prior 2°, 10 mm), because the hand-eye translation is weakly constrained (7 captures).
  - The chosen placement fits best: loss 0.438. A free fit of the opening and the reach (c, d) stopped at 47.9 / -133.3 mm with loss 0.457.
  - The corrections are at most 2.7° and 7.9 mm.
  - IoU with the corrected cameras: pad 0.65 / 0.56, clamp 0.77 / 0.81 (left / right). With the raw hand-eye cameras: pad 0.47 / 0.32, clamp 0.33 / 0.10.
- **ZED scene image, pad-to-pad gap.** The gap does not depend on the arm-pose error. Real 64.8 / 65.4 px; model 59.1 / 61.1 at c = 47.5 mm, 63.3 / 65.3 at 51 mm, 66.1 / 68.0 at 53 mm. So c = 51-52 mm.
- **x6 of the pad** cannot be identified: it trades off against the hand-eye translation. A fit with free x6 gives +17 mm and c = 57 mm, which the scene gap rejects (about 6 px too wide). 0 is chosen.
- **IMG_5411 f1256**, with the clamp as a ruler (31 mm = 135 px):
  - the pad extends 22 mm past the clamp (CAD: 23.5) and 75 mm behind it (CAD: 73);
  - the claw is 95 mm from the clamp end to the carriage (model: 97.8);
  - the tag is 24.6 mm.
- The scene camera's own optimum for the reach (z6 = -113 mm) would start the claw inside the housing. The metric arm pose is 12-14 px off at the grippers (metric_frame `zed_image_check_final_pose`), so the scene image is not used for the reach.

## Materials

The materials are editable Principled node graphs. Each has an object-space noise that drives a small bump and a ±0.04 roughness variation. The colours are set by eye from IMG_5411, the ZED frames and the wrist images, not measured.

| Material | sRGB base | Roughness | Used on |
|---|---|---|---|
| `YAM \| Light grey shell` | 0.70, 0.71, 0.71 | 0.50 | link2/link3 twin-tube covers (Menagerie "white" faces) |
| `YAM \| Black motor housing` | 0.06, 0.062, 0.066 | 0.42 | link1, link4, link5, the black faces of link2/link3 |
| `YAM \| Black base` | 0.06, 0.062, 0.065 | 0.50 | arm foot (MJCF `base` mesh) |
| `YAM \| Black gripper housing` | 0.07, 0.072, 0.075 | 0.45 | linear_4310 housing, carriage |
| `YAM \| White printed finger` | 0.90, 0.90, 0.88 | 0.55 | stock finger meshes (hidden from render) |
| `UMI-FT \| Orange TPU` | 0.90, 0.24, 0.02 | 0.38 | pad. Subsurface 0.1, radius 1.5 mm: satin, slightly translucent. |
| `UMI-FT \| Yellow PLA` | 0.96, 0.80, 0.00 | 0.45 | clamp, placeholder |
| `UMI-FT \| White PLA` | 0.90, 0.90, 0.88 | 0.50 | holder, adapter bottom |
| `UMI-FT \| Black PLA` | 0.04, 0.04, 0.045 | 0.50 | tag block, adapter top |
| `UMI-FT \| Black contact face` | 0.025, 0.025, 0.028 | 0.85 | pad contact face (second slot of the pad) |
| `UMI-FT \| ArUco 4x4_50 id 0/1/6/7` | image (packed) | 0.45 | tag |

At normal exposure, AgX renders the saturated orange and yellow pale (salmon, cream). The Standard view transform keeps them, so the review renders use Standard: -1.8 EV for the scene and wrist cameras, -1.0 EV for the asset views.

## Known gaps vs the real rig (not modelled; no geometry invented)

- **Claw details:** the team's holder and adapter have no CAD.
  - The holder is UMI-FT's: its distal half matches the rig's white block in all views. The adapter is a box.
  - Screws and any pad modification are not modelled. The CoinFT slot holds the 1 mm yellow placeholder (user 09-30).
  - In the wrist view, near the image bottom, the model pad covers more pixels than the real orange (pad precision 0.55-0.64, recall 0.96-0.97). The cause is either the hand-eye error or a difference in the real pad's proximal part.
- **Opening at q = 0:** 8.2 mm in the model (open_half 51.6 ± 2 mm). The user confirmed that the pads do not touch when closed (09-30); the gap was not measured.
- **Wrist camera (out of scope):** the rig has a ZED X One on a light-grey printed bracket on top of the gripper housing, with a teal GMSL connector and a black cable. None of this is in the MJCF.
  - Stereolabs publishes ZED X One CAD; the bracket has no CAD.
  - raiden's `zed_mounter_{left,right}.STL` are ZED Mini brackets (83 x 119 x 73 mm) and do not match.
  - The wrist hand-eye calibration of the new rig is in `~/.config/raiden/calibration_results.json` (09-29).
- **Cables and zip ties** along the arm are not modelled.
- **Foot plate:** the MJCF foot is a 70 x 200 mm slotted plate. The rig plate under each arm sits on the extrusion rail. Not checked: the scene camera does not see the feet.
- raiden's fin-ray CAD (`raiden/docs/assets/finray_{short,long,adapter}.STL`) is a different fin-ray design; not used.

## MuJoCo (proposal only)

`source/umi_ft/mjcf/PROPOSAL.md`, from `build/umi_ft_mjcf.py`. raiden and MESA are not edited. It contains:
- visual meshes in the MJCF body frames;
- a pad convex hull and two boxes for collision;
- a paste-in snippet;
- a test copy of the combined MJCF, which compiles in MuJoCo 3.3.5;
- `check.json`: contact surfaces, gap and masses.

The proposed finger mass is 101.6 g (stock 71.0 g; range 99.0-111.5 g by print density).

## Rebuild

```bash
cd ~/robot/aha-3d/scenes/yam_bimanual_rig/assets/yam_arm; RPY=~/robot/raiden/.venv/bin/python
$RPY build/mjcf_reference.py      # source/*.xml|json, fk/mujoco_*
$RPY build/material_labels.py     # source/face_labels.*
$RPY build/umi_ft_parts.py        # source/umi_ft/parts/*, source/umi_ft/tags/*
source ~/robot/aha-3d/kimodo_blender/env.sh
$BLENDER_BIN -b --factory-startup --python-exit-code 1 --python build/build_yam_arm.py -- --out <tmp>/stock.blend
$BLENDER_BIN -b <tmp>/stock.blend --python-exit-code 1 --python build/add_umi_ft_claws.py -- --out <new>.blend
$BLENDER_BIN -b <new>.blend --python-exit-code 1 --python build/fk_check.py -- --out fk/fk_check.json
$RPY build/umi_ft_mjcf.py                              # source/umi_ft/mjcf/* (PROPOSAL.md is hand-written: update its numbers)
# placement: constants in source/umi_ft/claw.json. Evidence (CPU; RUN = runs/yam_bimanual_rig/20260930t125032z-wrist-claw-fit-efcb76a0):
$RPY -u build/wrist_bg_fit.py --out RUN/stage1
$RPY -u build/wrist_capture_check.py --square 0.03556 --out RUN/captures_sq0.03556
$RPY -u build/claw_stage2_fit.py --stage1 RUN/stage1/cameras_stage1.json --captures RUN/captures_sq0.03556/captures.json \
    --variant c --start RUN/stage2/c/fit.json --C 0.0516 -0.1356 0.0 --fix-shared --rounds 3 --out RUN/stage2/final_c
$RPY -u build/write_wrist_records.py --fit RUN/stage2/final_c/fit.json --stage1 RUN/stage1/stage1_background_fit.json \
    --captures RUN/captures_sq0.03556/captures.json --captures-alt RUN/captures_sq0.036/captures.json \
    --check-render RUN/check_final --out evidence/wrist_fit
$RPY build/make_twin_config.py --cameras evidence/wrist_fit/cameras_world.json --bases evidence/wrist_fit/bases_proposed_4x4.json --out RUN/check_final
build/twin_check.sh RUN/check_final $PWD/yam_arm.blend        # GPU, one job; claw_check.json + compare
```

The 09-29 placement script `build/fit_claw_wrist.py` (`evidence/claw_fit/`) is kept as history; it assumed schema 1 (strip, spacer).

## Evidence

Media copies for the user: `~/robot/markdowns/bimanual/1_media/yam_asset/`.

- **Final asset.** `yam_arm.blend` (09-30 14:10Z, md5 `1f671785…`) = `build_yam_arm.py` + `add_umi_ft_claws.py` with claw.json schema 2 and the stage-2 placement. FK check PASS (same numbers as 09-29). The 09-29 file (md5 `be66c3fe…`, schema 1, used by the v2 build) is kept in RUN/asset_v2/.
- **Wrist fit (09-30).** `evidence/wrist_fit/wrist_cameras.json` (per wrist camera: K, world / link_6 / grasp_site poses, the base they are relative to, residuals), `evidence/wrist_fit/arm_bases_proposed.json` (per arm: the equivalent base, the rail form with joint offsets, what is identifiable). Media: `~/robot/markdowns/bimanual/1_media/wrist_fit/` (01 stage 1 background, 02 wrist claws real / v2 / final, 03 scene grippers real / v2 / final).
  - State of the new file: `logs/state_yam_arm.json`.
  - Logs: `logs/{build_yam_arm,add_umi_ft_claws,fk_check,umi_ft_parts,umi_ft_mjcf}.log`.
- **`00_asset_views.jpg`**: the asset alone: three views at home, a random pose, the gripper, clay (`evidence/asset_views/`, CPU).
- **`01_scene_camera_side_by_side.jpg`**: the real ZED scene image of 09-29 (arms at home), both arms rendered at raiden home from the metric ZED camera with their tags, and the overlay.
  - No fitting: the arm poses are `arms.*.world_from_base`, the camera is `scene_camera.K` and `pose.world_from_camera_blender` of `../../metric/metric_frame.json`.
  - The metric frame is read-only, and only those two sections are read (created 21:55Z).
- **`02_scene_camera_grippers_zoom.jpg`**: 3x zoom on each gripper. Asset outline magenta, SAM 3.1 mask green.
- **`04_fingers_wrist_camera_vs_raiden_cad.jpg`**: the real fingers and wrist camera next to raiden's fin-ray and ZED Mini mount CAD (made before the rebuild).
- **`05_wrist_view_vs_real.jpg`**: each arm's wrist image next to the asset rendered from that wrist camera.
  - The camera is the hand-eye result plus the fitted correction, so it is approximate.
  - Real pad and clamp outlines are red and magenta; the asset's are orange and yellow.
  - `evidence/wrist_camera/metrics.json`: pad IoU 0.63 / 0.54, clamp 0.72 / 0.75 (left / right, with all parts occluding). The raw hand-eye numbers are in `evidence/claw_fit/fit.json`.

Silhouette agreement in the scene camera: `evidence/scene_camera/metrics.json` (before: `metrics_before_stock_fingers.json`).
- The ZED sees only the grippers, in the bottom 100 px of the image.
- Real mask: SAM 3.1, prompt "robot arm", one instance per gripper (scores 0.55 / 0.54). The masks are reused; no new SAM run.
- Asset mask: exact triangle coverage of the render-visible meshes with the metric K and pose. It agrees with the Cycles alpha at IoU 0.98.

| Gripper | IoU before (stock fingers) | IoU after (UMI-FT) | Asset pixels inside the real outline, before → after | Real pixels covered, before → after |
|---|---|---|---|---|
| left arm | 0.42 | 0.57 | 0.77 → 0.69 | 0.48 → 0.77 |
| right arm | 0.46 | 0.53 | 0.75 → 0.64 | 0.54 → 0.76 |
| both | 0.44 | 0.55 | 0.76 → 0.66 | 0.51 → 0.76 |
| left / right, without fingers (claws, carriage) | 0.14 / 0.20 | 0.14 / 0.20 | 0.94 / 0.83 | 0.14 / 0.21 |

- The claws now cover about 3/4 of the real gripper pixels, against 1/2 before. The unchanged "without fingers" rows show that the rest of the pipeline did not change.
- Precision fell because the model claws sit about 8 px forward of the real ones in this image. The metric arm pose is 12-14 px off at the grippers (metric_frame `zed_image_check_final_pose`), and the reach comes from the wrist images and IMG_5411 (see "How the placement was set").

Video close-up (IMG_5411): **not done.** The camera fit (`build/fit_review_camera.py`) was killed by machine crashes at 21:55Z and 22:28Z. After that the task was restricted to the CPU and the close-up was skipped. Inputs and partial logs: `evidence/closeup/`; the fit script docstring has the status.

Re-run (CPU):

```bash
cd ~/robot/aha-3d/scenes/yam_bimanual_rig/assets/yam_arm; PY=~/robot/aha-3d/.runtime/pi3x-inference/venv/bin/python
source ~/robot/aha-3d/kimodo_blender/env.sh
$BLENDER_BIN -b --factory-startup --python-exit-code 1 --python build/render_scene_camera.py -- \
    --metric ../../metric/metric_frame.json --out evidence/scene_camera --samples 64
$PY build/compose_scene_camera.py --real ~/robot/bimanual_bringup/pi05/scene_camera.jpg --dir evidence/scene_camera \
    --sam-instances left_arm=1 right_arm=0 --media ~/robot/markdowns/bimanual/1_media/yam_asset
for arm in left_arm right_arm; do $BLENDER_BIN -b --factory-startup --python-exit-code 1 --python build/render_wrist_camera.py -- \
    --arm $arm --out evidence/wrist_camera --samples 32 --correction evidence/claw_fit/fit.json; done
$PY build/compose_wrist_camera.py --dir evidence/wrist_camera --real-dir ~/robot/bimanual_bringup/pi05 \
    --media ~/robot/markdowns/bimanual/1_media/yam_asset
$BLENDER_BIN -b yam_arm.blend --python-exit-code 1 --python build/render_views.py -- \
    --views evidence/asset_views/views_asset.json --outdir evidence/asset_views --samples 48 --device CPU
(cd evidence/asset_views && $PY ../../build/montage.py sheet.jpg 3 a1_front_left.png a2_side.png a3_rear_right.png \
    a4_pose_rand1.png a5_gripper.png a6_clay_front_left.png)     # media 00 = this sheet at 0.6x
$PY build/cad_views.py --out evidence/cad_parts
$PY build/compose_parts_sheet.py --out ~/robot/markdowns/bimanual/1_media/yam_asset/04_fingers_wrist_camera_vs_raiden_cad.jpg
# SAM masks (GPU, only when the GPU is free; not re-run 09-29):
$SAM3_PYTHON build/sam_arm_mask.py --image ~/robot/bimanual_bringup/pi05/scene_camera.jpg \
    --out evidence/scene_camera/sam --width 1008 --phrases 'robotic gripper' 'robot gripper' 'robot arm'
```
