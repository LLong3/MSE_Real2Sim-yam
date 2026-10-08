# yam_bimanual_rig Build (Blender twin in the metric frame)

Task `yam-rig-build-20260929` (acceptance scope `yam-rig-static`).
- **Current: `../blender/yam_rig_twin_v4.blend`** (config `configs/delivery_v4.json`, 09-30) = v3.1 + the v4 arm asset (`yam_arm_v4.blend`: left claw pair rolled -4.27 deg) + the v4 wrist cameras. Run with evidence: `runs/yam_bimanual_rig/20260930t195845z-twin-v4-36225d75/`.
- v3.1: `yam_rig_twin_v3_1.blend` (`configs/delivery_v3_1.json`, 09-30) = v3 with the desk texture repainted around the home grippers. Run: `runs/yam_bimanual_rig/20260930t181637z-desk-repaint-86335e99/`.
- v3: `yam_rig_twin_v3.blend` (`configs/delivery_v3.json`, 09-30). Run: `runs/yam_bimanual_rig/20260930t142059z-build-v3-45764863/`.
- Kept for comparison: `yam_rig_twin_v2.blend` (final claws at the metric bases; `configs/delivery_v2.json`) and `yam_rig_twin_v1.blend` (placeholder arms). Their run: `runs/yam_bimanual_rig/20260929t225835z-build-1de8ab34/`.

## Frame and inputs

- Blender world = `metric/metric_frame.json` (read only): metres, Z up, floor z 0, desk top z 0.72, +Y to the back wall.
- Tape 09-29 (user): back wall at the desk back edge + 0.035 m (y 2.1189); rail 1.00 m, x -2.7647..-1.7647; desk 1.52 x 0.76 m, top 0.72. The build reads them from `metric_frame.json` and checks them against the config.
- Target image: `bimanual_bringup/pi05/scene_camera.jpg` (ZED X left eye, 09-29 16:00, arms at home). Wrist images `left_/right_wrist_camera.jpg`.
- Arms: two appended copies of `assets/yam_arm/yam_arm.blend` (v4: `yam_arm_v4.blend` via `--arms`; read only). Each copy is set with the asset's `yam_rig.setup_arm`: tags left 7/6, right 1/0 and, on the v4 asset, the claw mount roll (left -4.27 deg about link_6 z, right 0; `claw_mount_roll_rad` in the `.build.json`). Builds up to v3.1 used `yam_rig.set_finger_tags` (same result on the v3 asset).
  - v1/v2: at the metric `arms.*.world_from_base`, raiden home (joints 0, gripper 0.0475).
  - **v3: proposed correction, not the metric frame** (`arm_poses` in the config, from `assets/yam_arm/evidence/wrist_fit/arm_bases_proposed.json` `rail_form`; 09-30 wrist/claw fit). Each base is flat on the rail at the metric height, with the fitted yaw and x, y (left -0.66 deg, (-19.5, -14.2) mm; right -2.61 deg, (-5.5, -23.3) mm). The home joint offsets are the displayed pose: left j2 +5.26, j3 +0.84, j6 -0.41 deg; right j2 +0.17, j3 -2.24, j6 +2.94 deg.
  - raiden home (q = 0) is shown with these offsets. Set the joints to 0 for the uncorrected MJCF home.
  - Right j3 is below the MJCF limit 0, so this scene copy's joint-3 control range is widened to -0.04 rad (`joint_range_widened` on the root; the drivers do not clamp).
  - Each root carries `arm_pose_status`, `arm_pose_source`, `home_joint_offsets_deg` and `metric_world_from_base`. `metric/` is unchanged.

## Scripts

| File | Does |
|---|---|
| `build_scene.py` | Blender: builds the scene from `configs/<v>.json` (+ metric frame, RoomKit materials, arm asset) |
| `render_views.py` | Blender: Cycles GPU renders of the scene / wrist / overview cameras; linear EXR, PNG, object-ID EXR |
| `camera_model.py` | numpy: regions from IDs, exclusions, post-render camera model (vignette + RGB gain) |
| `calibrate.py` | fits flat albedos, emission and daylight per region, the scene-camera vignette, wrist gains/vignette |
| `light_search.py` | grid search of the ceiling-panel row against the scene image |
| `surface_texture.py` | image-derived albedo maps (desk 1 mm, wall 5 mm low-pass) |
| `wrist_refine.py` | wrist-camera rotation fitted to real-image lines (`wrist_lines.json`, v1) |
| `compare.py` | side-by-side, overlay, difference, edges, metrics |
| `review_images.py` | user review JPEGs (v1/v2; v3: run `scripts/review_images_v3.py`) |
| `claw_check.py` | claw pad/clamp IoU and colour against HSV masks of the real images |
| `configs/` | one config per iteration; `delivery_v{1,2,3,3_1,4}.json` = the delivered scenes (`delivery_v3` = v38 with its delivery note; `delivery_v3_1` = v3 + desk texture; `delivery_v4` = v3.1 + v4 wrist cameras and arm note) |

v3-only scripts live in the run (`scripts/`): `make_v3_config.py` (v34 from delivery_v2 + the 09-30 inputs), `verify_v3.py` (v1/v2/v3 metrics), `render_face_mask.py` (pad contact-face mask), `review_images_v3.py`, `check_blend.py`. v3.1 and v4 scripts are in their runs (`make_v3_1_config.py`; `make_v4_config.py`, `build_check_v4.sh`).

Rebuild v3 (background Blender, one GPU job at a time):
```bash
$BLENDER_BIN -b --factory-startup --python-exit-code 1 --python build_scene.py -- --config configs/delivery_v3.json --out ../blender/<new>.blend --arms-status final
$BLENDER_BIN -b ../blender/<new>.blend --python-exit-code 1 --python render_views.py -- --out RUN/renders/<dir> --views scene,left_wrist,right_wrist --samples 256
```
Rebuild v4: the same with `--config configs/delivery_v4.json --arms ../assets/yam_arm/yam_arm_v4.blend`.
v3 chain: v34 (new arms, cameras, room) → white-desk render → `surface_texture.py --arm-dilate 22` → `desk_albedo_v34.exr` → v35 → `calibrate.py --steps wrist_vignette,wrist` → v36 → v37 (glass B band light) → v38 (carpet albedo) = delivery_v3.

## Geometry (metric frame)

- Rig desk: box from the metric frame, 25 mm laminate top, edge band, dark frame, two thin brackets at the left end.
- Rail: T-slot extrusion 1.00 m on the desk front edge, two C-clamps, black adapter plates under the arms (v3: plates follow the proposed base x).
- Camera pole: 40 x 80 mm extrusion at `rail.pole_xy_m`, top 1.70 m, ZED X body at the metric camera pose.
- Back wall: vertical plane y 2.1189; painted end at x -2.975 (ZED edge back-projected onto the tape plane; metric `wall_end_x_m` -3.004 is the stereo value from before the gap update).
- Glass partition: y 2.155, mullion post x -3.874 (60 x 90 mm), floor channel right of the post. Glass A (right of the post): frosted band 0.83-2.05 m, clear below and above. Glass B (left): v1/v2 like A; **v3 clear over the full height** (`glass.frosted_panels`).
- Left end, v1/v2: white wall stub at x -4.9. **v3** (`side_door`, from the layout check `runs/yam_bimanual_rig/20260930t125155z-layout-metric-f9dc6ff9/v3_room_corrections.json`):
  - a wood door leaf about 8 deg ajar, hinge at the glass end; room face at floor level (-4.833, 2.098) → (-4.725, 1.33), 45 mm thick, with a lever handle and card reader;
  - a white jamb x -4.87..-4.82, y 2.10..2.20 (in the glass-partition group);
  - a beige wall left of the door at x -4.98 (y 0.50..1.25), a header above the door and a latch-side jamb.
  - Assumed: leaf height 2.13 m (>= 2.0 m observed), the header and the latch jamb. None of it is seen by the rig cameras.
- Monitor desk: x -1.415..0.093 (**v3: -1.406**, ZED edge), top 0.71; privacy panel (front face y 2.065, x from -1.50, z 0.804-1.233), monitor arm, outlet box, four small items (v3: black device +0.023 m in x, so it no longer overhangs).
- Monitor: v1/v2 by hand at x -0.55..0.05. **v3** (`monitor_desk.monitor`): x -0.84..-0.32, front face y 1.845 at the centre, yaw -6 deg, z 0.81..1.18; the stand and foot move with it.
- Carpet floor continues behind the glass; ceiling 2.75 m with three LED panels.
- Not modelled: tripod camera and clutter behind the camera (not seen by the rig cameras), cables, croissant and task objects (user: empty table).

## Materials

All colours are linear RGB in the ZED camera colour space (the ZED white balance makes the beige wall near neutral). Flat values are fitted per region by `calibrate.py` (median real / median render).

| Surface | Material |
|---|---|
| Desk top | image-derived albedo: ZED image / rendered irradiance (desk base colour 1), rectified to 1 mm texels; grippers, arms, croissant and vial inpainted; unobserved front strip = median; plus a fine procedural grain along x (3 %). Texture per version: v1 `desk_albedo_v27.exr`; v2 `desk_albedo_v30.exr` (final arms); v3 `desk_albedo_v34.exr` (re-baked under the v3 arms and claws; 26 % of texels changed by > 2 %, all near the grippers); v3.1 and v4 `desk_albedo_v3_1.exr` (around the home grippers from `bimanual_smoke` scene frames with the arms away, each divided by a white-desk render posed as in the frame; 25 % of texels changed by > 2 %). |
| Back wall | procedural matte paint + low-frequency image-derived albedo `textures/wall_albedo_v27.exr` (5 mm texels, 0.12 m low-pass) over x -2.975..-0.9, z 0.6-2.75 |
| Carpet | procedural navy speckle, fitted albedo. v3: refitted, carpet only, x1.04/1.06/1.07. The v2 white stub bounced light onto the floor at the image lower left; the door returns less. A diagnostic with the stub restored gives the v2 carpet. |
| Frosted band | emission (fitted) + weak diffuse; non-camera rays see 35 % of the emission |
| Clear glass | glass BSDF; blocks direct light (the two rooms are lit separately) |
| Glass B band (v3) | clear glass for camera rays. Other rays keep the v2 band light (emission x 0.35 + diffuse), because the v2 light fit counted that light entering the room through glass B. |
| Door (v3) | RoomKit pale cut firewood (leaf), RoomKit brushed steel (handle), white paint (jambs), beige matte paint (side wall, header) |
| Glass frame | light anodised aluminium + small blue glow (fitted by hand) |
| Claw pad, clamp (v2, v3) | asset materials with a scene-only ZED camera-space base colour (`arm_material_overrides`): pad linear (0.80, 0.22, 0.018), clamp (0.66, 0.56, 0.043), fitted on the wrist and scene images; the asset file keeps its own colours. v3: the 1 mm CoinFT-slot placeholder uses the clamp material and gets the clamp colour; the pad's black contact face (second material) is not overridden. |
| Rail, pole | RoomKit brushed stainless steel; desk frames RoomKit satin blackened iron; privacy panel RoomKit charcoal woven upholstery (tinted, matte); monitor RoomKit dark television glass |

## Lighting and camera model

- Three ceiling LED area panels (0.6 x 1.2 m, x -3.0 / -1.3 / 0.4, y 0.3, z 2.74), row position from `light_search.py`; ambient world; a downward blue daylight area light over the area behind the glass.
- Scene camera: metric K and pose; Standard view transform; exposure log2(gain) = 0.67 stop; radial vignette V(r) = 1 + k1 r^2 + k2 r^4 (k1 -0.50, k2 0.10; V 0.60 at r = 1) applied in post (`camera_model.py`). Values on the camera object (`camera_model_json`).
- Wrist cameras: parented to each arm's `grasp_site`; intrinsics from the ZED SDK calibration scaled to 960x600 (fx != fy, render pixel aspect on the camera); own RGB gain and vignette (v3: refitted on the v35 render, as v2 did on v31).
  - v1: raiden hand-eye 09-29 with a rotation fitted to image lines.
  - v2: hand-eye + the asset's 09-29 claw-fit correction.
  - **v3: the 09-30 wrist/claw fit, `T_grasp_site_cam_opencv` from `assets/yam_arm/evidence/wrist_fit/wrist_cameras.json`.** Valid only with the proposed bases: with the rail-form bases and joint offsets, the rendered cameras match the fit's world poses to 0.001 mm and 0.0 deg (`check_vs_fit_T_world_cam` on the camera, `wrist_cameras` in the `.build.json`).
  - **v4: `assets/yam_arm/evidence/wrist_fit_v4/wrist_cameras.json`** (claw fit v4): v3 rotations, translation moved left (-4.0, -7.0, 4.0) mm, right (0.5, -3.0, 1.5) mm (metric frame). Rendered poses match to 0.001 mm and 0.0 deg. Wrist camera models not refitted.

## Limits

- v3 arm bases, home joint offsets and wrist cameras are a proposal from one arm pose (home). The fit cannot separate base tilt from joint offsets or sag (asset README).
- v3 wrist claws: the model pad overshoots the real pad at the proximal end. In the left wrist, one claw is about 35 px off along the pad. claw_check pad IoU is 0.52 / 0.51, against v2 0.68 / 0.61 (v2 moved its camera to fit the claws, at the cost of the background).
- v4 (64 spp check, run `20260930t195845z-twin-v4-36225d75` `analysis/blender_v4.json`): wrist pad IoU 0.57 / 0.53, clamp 0.61 / 0.65 (right clamp 0.76 in v3.1: the real yellow strip is 3.6 mm, the placeholder 1 mm). Scene left gripper IoU 0.842 → 0.818 from the left claw roll (accepted for the wrist gain). Left wrist background: all-sample edge median 1.46 → 1.37 px, but the desk left edge 2.5 → 7.3 px and the wall end 6.0 → 10.6 px (the desk back edge has most samples).
- v3.1 / v4 desk: the model claw shadows are too dark and fall toward the wall; the real ones are faint and fall toward the camera. v3's texture hid them, so near the claws Blender is worse than v3 (scene repaint zone MAE 1.90 → 4.62). MuJoCo renders no shadows and gains.
- The texture and lighting fits are tuned to one image; other lighting states of the room are not modelled. The desk texture has faint smudges near the grippers.
- The layout-inspection gate needs a rigid model-to-reference transform; the metric frame is an anisotropic rescale of the Pi3X reference. The layout check of 09-30 uses a converted reference (run 20260930t125155z-layout-metric-f9dc6ff9).
