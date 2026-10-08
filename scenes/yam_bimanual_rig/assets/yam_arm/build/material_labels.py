"""Transfer the shell/motor colour split of the MuJoCo Menagerie YAM onto raiden's i2rt link meshes.

The i2rt STLs are one mesh per link (no colour split). The Menagerie model (copy in MESA,
vla-benchmark mesa/sim/envs/robots/assets/yam/yam_robot.xml) splits links 2 and 3 into black and
white parts. Each Menagerie link is rigidly aligned (24 axis rotations + ICP) to the i2rt link mesh,
then every i2rt face takes the material of the nearest Menagerie part. Geometry is not changed.

Run with raiden's venv (read-only use): ~/robot/raiden/.venv/bin/python build/material_labels.py
Writes source/face_labels.npz (per link: int8 per STL face, 0 = motor/black, 1 = shell/grey) and
source/face_labels.json (alignment residuals and label counts).
"""
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]
MENAGERIE_DIR = Path('/home/frank/robot/vla-benchmark/mesa/sim/envs/robots/assets/yam')
MENAGERIE_XML = MENAGERIE_DIR / 'yam_robot.xml'
# i2rt body -> Menagerie body whose visual parts cover the same link
LINKS = {'world': 'arm', 'link1': 'link_1', 'link2': 'link_2', 'link3': 'link_3', 'link4': 'link_4', 'link5': 'link_5'}


def quat_mat(q):
    return trimesh.transformations.quaternion_matrix(np.asarray(q, float) / np.linalg.norm(q))


def menagerie_parts():
    root = ET.parse(MENAGERIE_XML).getroot()
    root.find('compiler').set('meshdir', str(MENAGERIE_DIR))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    parts = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 1:
            continue
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        mat = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[g])
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        f = m.mesh_face[fa:fa + fn]
        # body-local coordinates: undo the Menagerie body world pose
        R, p = d.xmat[m.geom_bodyid[g]].reshape(3, 3), d.xpos[m.geom_bodyid[g]]
        parts.setdefault(body, []).append(dict(mesh=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid), material=mat,
                                                verts=(v - p) @ R, faces=f))
    return parts


def rotations24():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                R[i, j] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def icp(src, tree, dst, R, t, iters=60):
    for _ in range(iters):
        moved = src @ R.T + t
        dist, idx = tree.query(moved)
        keep = dist < max(np.percentile(dist, 90), 1e-4)
        a, b = moved[keep], dst[idx[keep]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        dR = Vt.T @ D @ U.T
        R, t = dR @ R, dR @ (t - ca) + cb
    dist, _ = tree.query(src @ R.T + t)
    return R, t, dist


def main():
    model = json.loads((HERE / 'source' / 'mjcf_model.json').read_text())
    parts = menagerie_parts()
    labels, report = {}, {'menagerie_xml': str(MENAGERIE_XML),
                          'menagerie_xml_md5': hashlib.md5(MENAGERIE_XML.read_bytes()).hexdigest(), 'links': {}}
    for body in model['bodies']:
        if body['name'] not in LINKS:
            continue
        (geom,) = body['geoms']
        mesh = trimesh.load(geom['file'], process=False)
        # raw STL -> i2rt body frame (XML geom pos/quat)
        T = quat_mat(geom['quat']); T[:3, 3] = geom['pos']
        src_v = trimesh.transform_points(mesh.vertices, T)
        src_c = trimesh.transform_points(mesh.triangles_center, T)
        mparts = parts[LINKS[body['name']]]
        dst_meshes = [trimesh.Trimesh(p['verts'], p['faces'], process=False) for p in mparts]
        samples, owner = [], []
        for k, dm in enumerate(dst_meshes):
            pts, _ = trimesh.sample.sample_surface_even(dm, 40000, seed=1) if dm.area > 0 else (dm.vertices, None)
            pts = np.vstack([pts, dm.vertices])
            samples.append(pts); owner.append(np.full(len(pts), k))
        dst = np.vstack(samples); owner = np.concatenate(owner)
        tree = cKDTree(dst)
        sub = src_v[np.random.default_rng(0).choice(len(src_v), min(4000, len(src_v)), replace=False)]
        best = None
        for R0 in rotations24():
            t0 = dst.mean(0) - sub.mean(0) @ R0.T
            R, t, dist = icp(sub, tree, dst, R0, t0, iters=25)
            if best is None or np.median(dist) < np.median(best[2]):
                best = (R, t, dist)
        R, t, _ = icp(src_v, tree, dst, best[0], best[1], iters=60)
        dv, _ = tree.query(src_v @ R.T + t)
        dc, idx = tree.query(src_c @ R.T + t)
        part_of_face = owner[idx]
        is_shell = np.array([mparts[k]['material'] == 'yam_white' for k in range(len(mparts))])
        lab = is_shell[part_of_face].astype(np.int8)
        labels[body['name']] = lab
        report['links'][body['name']] = dict(
            menagerie_body=LINKS[body['name']], menagerie_parts=[(p['mesh'], p['material']) for p in mparts],
            i2rt_mesh=geom['file'], faces=int(len(lab)), shell_faces=int(lab.sum()),
            vertex_residual_mm=dict(median=float(np.median(dv) * 1e3), p95=float(np.percentile(dv, 95) * 1e3),
                                    max=float(dv.max() * 1e3)),
            face_centre_residual_mm_p95=float(np.percentile(dc, 95) * 1e3),
            faces_over_2mm=int((dc > 0.002).sum()),
            rotation=np.round(R, 4).tolist(), translation_m=np.round(t, 5).tolist())
        print(body['name'], report['links'][body['name']]['vertex_residual_mm'], 'shell', int(lab.sum()), '/', len(lab))
    np.savez_compressed(HERE / 'source' / 'face_labels.npz', **labels)
    (HERE / 'source' / 'face_labels.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
