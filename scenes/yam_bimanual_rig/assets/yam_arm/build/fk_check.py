"""Compare the Blender asset's link/site/mesh world poses with MuJoCo FK (fk/mujoco_poses.json).

    $BLENDER_BIN -b <asset.blend> --python-exit-code 1 --python build/fk_check.py -- --out fk/fk_check.json

Poses are compared in the asset root frame (root placed anywhere). Mesh check: every MuJoCo
compiled vertex (fk/mujoco_geom_verts.npz) against the nearest vertex of the Blender geom.
Also checks that two appended copies of the asset collection pose independently.
Fails (exit 1) above 1 mm / 0.1 deg.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'build'))
import yam_rig  # noqa: E402

POS_TOL_MM, ANG_TOL_DEG = 1.0, 0.1


def rel(root, obj):
    return root.matrix_world.inverted() @ obj.matrix_world


def ang_deg(Ra, Rb):
    # atan2 form: well conditioned near zero, unlike acos((tr - 1) / 2) on float32 matrices
    R = np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)
    s = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return math.degrees(math.atan2(s, (np.trace(R) - 1) / 2))


def check_instance(root, ref, verts):
    links = yam_rig.links(root)
    sites = {o['mjcf_site']: o for o in root.children_recursive if o.get('mjcf_site')}
    geoms = {o['mjcf_geom_mesh']: o for o in root.children_recursive if o.get('mjcf_geom_mesh')}
    kd = {}
    for name, g in geoms.items():
        me = g.data
        tree = KDTree(len(me.vertices))
        for i, v in enumerate(me.vertices):
            tree.insert(v.co, i)
        tree.balance()
        kd[name] = tree
    saved = yam_rig.get_joints(root)
    results = {}
    for key, cfg in ref['configs'].items():
        q = cfg['qpos']
        yam_rig.set_joints(root, q[:6], gripper=q[6])
        rows, worst_p, worst_a = {}, 0.0, 0.0
        for bname, mj in cfg['bodies'].items():
            M = rel(root, links[bname])
            dp = (M.to_translation() - Vector(mj['pos'])).length * 1e3
            da = ang_deg(M.to_3x3(), mj['mat'])
            rows[bname] = dict(pos_err_mm=dp, ang_err_deg=da)
            worst_p, worst_a = max(worst_p, dp), max(worst_a, da)
        site_rows = {}
        for sname, mj in cfg['sites'].items():
            M = rel(root, sites[sname])
            dp = (M.to_translation() - Vector(mj['pos'])).length * 1e3
            da = ang_deg(M.to_3x3(), mj['mat'])
            site_rows[sname] = dict(pos_err_mm=dp, ang_err_deg=da)
            worst_p, worst_a = max(worst_p, dp), max(worst_a, da)
        mesh_rows = {}
        for mname, g in geoms.items():
            v_mj = verts[f'{key}|{mname}']
            # MuJoCo world (= root frame) -> geom local, then nearest Blender vertex
            to_local = (root.matrix_world.inverted() @ g.matrix_world).inverted()
            d = max(kd[mname].find(to_local @ Vector(v))[2] for v in v_mj)
            mesh_rows[mname] = dict(max_vertex_err_mm=d * 1e3, n=len(v_mj))
        results[key] = dict(qpos=q, bodies=rows, sites=site_rows, meshes=mesh_rows,
                            max_link_pos_err_mm=worst_p, max_link_ang_err_deg=worst_a,
                            max_mesh_vertex_err_mm=max(r['max_vertex_err_mm'] for r in mesh_rows.values()))
    yam_rig.set_joints(root, saved)
    return results


def independence_test(blend_path, collection_name):
    """Append the asset collection twice into a new scene; posing one must not move the other."""
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    roots = []
    for _ in range(2):
        with bpy.data.libraries.load(str(blend_path), link=False) as (src, dst):
            dst.collections = [collection_name]
        coll = dst.collections[0]
        scene.collection.children.link(coll)
        roots.append(next(o for o in coll.objects if o.get('yam_arm_root')))
    roots[1].location = (0.0, 0.59, 0.0)
    dg = bpy.context.evaluated_depsgraph_get()
    before = [yam_rig.links(r)['tip_left'].matrix_world.copy() for r in roots]
    roots[0]['joint1'] = 1.0; roots[0]['joint2'] = 1.2; roots[0]['gripper'] = 0.03
    roots[0].update_tag(); dg.update(); bpy.context.view_layer.update()
    after = [yam_rig.links(r)['tip_left'].matrix_world.copy() for r in roots]
    moved0 = (after[0].to_translation() - before[0].to_translation()).length
    moved1 = (after[1].to_translation() - before[1].to_translation()).length
    targets = set()
    for o in roots[1].children_recursive:
        if o.animation_data:
            for fc in o.animation_data.drivers:
                targets.update(v.targets[0].id.name for v in fc.driver.variables)
    return dict(copy0_tip_moved_m=moved0, copy1_tip_moved_m=moved1, copy1_driver_targets=sorted(targets),
                copy1_root=roots[1].name, ok=bool(moved0 > 0.05 and moved1 < 1e-9 and targets == {roots[1].name}))


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    ref = json.loads((HERE / 'fk' / 'mujoco_poses.json').read_text())
    verts = dict(np.load(HERE / 'fk' / 'mujoco_geom_verts.npz'))
    (root,) = yam_rig.roots()
    coll_name = root.users_collection[0].name
    res = dict(blend=bpy.data.filepath, tolerance=dict(pos_mm=POS_TOL_MM, ang_deg=ANG_TOL_DEG), seed=ref['seed'])
    res['placed_at_identity'] = check_instance(root, ref, verts)
    # same check with the root moved and turned (placement must not matter)
    root.matrix_world = Matrix.Translation((-2.3, 1.39, 0.654)) @ Matrix.Rotation(math.radians(-90), 4, 'Z')
    bpy.context.view_layer.update()
    res['placed_in_scene'] = check_instance(root, ref, verts)
    root.matrix_world = Matrix.Identity(4)
    blend = Path(bpy.data.filepath)
    res['independence'] = independence_test(blend, coll_name)
    worst = dict(pos_mm=max(c['max_link_pos_err_mm'] for p in ('placed_at_identity', 'placed_in_scene') for c in res[p].values()),
                 ang_deg=max(c['max_link_ang_err_deg'] for p in ('placed_at_identity', 'placed_in_scene') for c in res[p].values()),
                 mesh_mm=max(c['max_mesh_vertex_err_mm'] for p in ('placed_at_identity', 'placed_in_scene') for c in res[p].values()))
    res['worst'] = worst
    res['pass'] = bool(worst['pos_mm'] < POS_TOL_MM and worst['ang_deg'] < ANG_TOL_DEG and worst['mesh_mm'] < POS_TOL_MM
                       and res['independence']['ok'])
    Path(args.out).write_text(json.dumps(res, indent=1))
    for p in ('placed_at_identity', 'placed_in_scene'):
        for k, c in res[p].items():
            print(f'{p:20s} {k:6s} link pos {c["max_link_pos_err_mm"]:.2e} mm  ang {c["max_link_ang_err_deg"]:.2e} deg  '
                  f'mesh {c["max_mesh_vertex_err_mm"]:.2e} mm')
    print('independence', res['independence'])
    print('FK_CHECK', 'PASS' if res['pass'] else 'FAIL', worst)
    if not res['pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
