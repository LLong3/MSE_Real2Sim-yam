# Proposal: UMI-FT claws in raiden's yam_linear_4310 MJCF

Status: proposal only. raiden, i2rt and MESA are not edited. Made by `build/umi_ft_mjcf.py` (checked with MuJoCo 3.3.5).

## What to replace

In the combined model (`raiden/_xml_paths.get_yam_4310_linear_xml_path()`, from i2rt `gripper/linear_4310/linear_4310.xml`), bodies `tip_left` and `tip_right`:

- replace `<geom ... mesh="tip_left" />` / `mesh="tip_right"` (one geom, visual and collision) and the body `<inertial>` with the lines in `umi_ft_fingers_snippet.xml`;
- keep everything else: bodies and frames, `joint7`/`joint8` (slide -z, 0 to 0.0475 m), the `joint7 = joint8` equality, the actuator, and the contact excludes (link_6 vs tips, tip vs tip).

`yam_linear_4310_umi_ft.xml` is a test copy of the combined model with this change (absolute mesh paths). It compiles. `check.json` holds the results.

## Geometry

- Meshes are in the body frames (geom pos/quat = identity). Visual meshes (group 2, no contact): pad, black contact face, clamp (with the 1 mm yellow placeholder), white holder, black tag block, carriage. The carriage is the stock tip above z6 = -70 mm (rack slider), unchanged.
- claw.json schema 2 (user answers 09-30): no friction-strip thickness (the black face is appearance only), no CoinFT (1 mm yellow placeholder), the pads do not touch when closed.
- The contact surface (the pad's flat face) sits at y6 = ∓(0.0516 - 0.0475 + q). The gap is 103.2 mm at q = 0.0475 (open) and 8.2 mm at q = 0. The opening is fitted from images (stage 2, 09-30; about ±2 mm, README). The user confirmed that the pads do not touch when closed.
- Reach: the pad tip is at z6 = -190.3 mm (stock claw tip: -144.7). The flat contact face spans z6 -93.6 to -190.3 mm, centre -142 mm. `grasp_site` stays at -134.7 mm; IK and action semantics are unchanged.

## Collision and contact

- Pad: convex hull of the fin-ray. The fin-ray outline is convex, so the hull equals the pad: a flat 96.7 x 17.2 mm contact face.
- Holder and clamp: two boxes, `body_outer` and `body_inner`. The carriage has no collision geom (it is inside the housing footprint and excluded against link_6).
- The real fin-ray (TPU 95A) bends around objects. It gives an area contact and several mm of passive give. As a rigid hull it gives edge or point contacts and no give, so expect more slip and less tolerance to position error in sim.
- Starting contact values on the pad (untested, tune against real grasp outcomes in MESA): `friction="1.2 0.02 0.002" condim="4" solref="0.01 1" solimp="0.9 0.95 0.002"`. condim 4 adds torsional friction for the flat patch.
- If rigid pads are not enough: put each pad on a child body with a spring slide joint along the closing axis (a few hundred N/m, damped) for passive compliance.

## Mass

- The stock tip is 71.0 g (MJCF inertial, 52.3 cm³; includes carriage metal).
- Proposed tip: 101.6 g (+31 g per finger), from:
  - the stock mass minus the removed stock claw (28.7 cm³ printed);
  - pad 12.75 cm³ TPU at 1.21 g/cm³;
  - printed PLA parts (48.9 cm³) at 0.75 g/cm³ (about 60 % fill).
- Range 99.0 to 111.5 g (fill 50 % to solid). Screws are not counted.
- The COM moves forward. The full inertia, from the part meshes, is in the snippet.

## Not covered

- MESA's YAM twin uses its own model; it needs the same meshes in its finger frames.
- The tags are not in the MJCF (rgba only).
