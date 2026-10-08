# MuJoCo twin of the bimanual YAM rig

The rig scanned in `scenes/yam_bimanual_rig` (Blender twin v3), bridged to MuJoCo in MESA: two i2rt YAM arms on the rail, stock linear_4310 grippers (UMI-FT claws selectable), and the three rig cameras `scene_camera`, `left_wrist_camera`, `right_wrist_camera`. The room is baked Cycles lightmaps, rendered unlit.

These are MESA files at their MESA paths. They run inside a MESA checkout, not on their own.

| Path | What |
|---|---|
| `mesa/task_suites/rigs/raiden_bimanual/` | `rig.json` (arm bases, home pose, dynamics, gripper, cameras, `sim_render`), `layout.json`, `room/` (meshes, baked textures) |
| `mesa/sim/envs/robots/raiden_yam_bimanual.py` | `RaidenYamBimanualLeft` / `RaidenYamBimanualRight` |
| `mesa/sim/envs/robots/assets/yam/` | YAM arm and gripper MJCF and meshes, including the generated per-arm XMLs and the UMI-FT claw meshes |
| `mesa/sim/envs/{arenas,objects,problems}/raiden_bimanual*.py` | arena, task objects (vial, cap), problem |
| `mesa/task_suites/bddl_files/raiden_bimanual/` | tasks `empty_rig` and `vial_capping` |
| `scripts/raiden_bimanual/` | `render_home.py` (renders the three cameras at home), `test_vial_capping.py` |
| `mesa_fb9c203f9.patch` | changes to existing MESA files: registrations, `raiden_yam.py`, two-arm `raiden_sim_server.py` |

## Install

MESA is `pairlab/vla-benchmark`; the base commit `fb9c203f9` is on branch `new-rig-sim`. MESA's own assets (`mesa/sim/assets/`) come from its `scripts/setup/download_assets.py`.

```bash
cd <mesa checkout>
git checkout fb9c203f9
cp -r <this repo>/sim_twin/mesa <this repo>/sim_twin/scripts .
git apply <this repo>/sim_twin/mesa_fb9c203f9.patch
```

## Run

From the MESA checkout, in MESA's venv:

```bash
MUJOCO_GL=egl python scripts/raiden_bimanual/render_home.py --out <dir>
python scripts/raiden_sim_server.py --robots RaidenYamBimanualLeft RaidenYamBimanualRight \
    --task-json mesa/task_suites/bddl_files/raiden_bimanual/empty_rig/source/000.json --port 5599
```

The server is the sim side of raiden's `rd record --sim 127.0.0.1:5599` and `rd serve --sim 127.0.0.1:5599` (raiden branch `bimanual-sim`, not in this repo).

## Notes

- `rig.json` is hand-edited (`sim_render`, friction, gripper, cameras, home pose). Its `source` fields name the fit files they came from. Paths under `~/robot/aha-3d/scenes/yam_bimanual_rig` are `scenes/yam_bimanual_rig` here; `runs/` is not included.
- Home joint offsets are MuJoCo joint `ref = -offset`.
- The per-geom emission (`layout.json` `room_emission`) must be recomputed whenever the MuJoCo lights change.
