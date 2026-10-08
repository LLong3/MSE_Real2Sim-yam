"""Export world triangles (arm root frame) of every geom at a pose: Blender, -- --out X.npz [--gripper G]."""
import argparse, sys
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yam_rig
argv = sys.argv[sys.argv.index('--') + 1:]
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--gripper', type=float, default=0.0475)
a = ap.parse_args(argv)
(root,) = yam_rig.roots()
yam_rig.set_joints(root, [0.0] * 6, gripper=a.gripper)
out = {}
inv = root.matrix_world.inverted()
for o in root.children_recursive:
    if o.type != 'MESH':
        continue
    M = np.array(inv @ o.matrix_world)
    v = np.array([vv.co[:] for vv in o.data.vertices]) @ M[:3, :3].T + M[:3, 3]
    f = np.array([p.vertices[:] for p in o.data.polygons])
    out[o['mjcf_geom_mesh']] = v[f].astype(np.float32)
np.savez_compressed(a.out, **out)
print({k: v.shape for k, v in out.items()})
