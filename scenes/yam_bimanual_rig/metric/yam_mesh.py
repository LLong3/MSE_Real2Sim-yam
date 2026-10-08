"""Export the YAM + linear_4310 MJCF surface at q=0 (arm) in the arm-base frame, for a finger opening.

Read-only on raiden/i2rt; run with raiden's venv (mujoco + i2rt):
  /home/frank/robot/raiden/.venv/bin/python yam_mesh.py OPENING_M OUT.npz
Base frame = MJCF world frame of yam.xml: origin at the base mounting face, Z up; at q=0 the arm
folds along +X (grippers at +X, upper-arm rear at -X).
"""
import sys

import mujoco
import numpy as np
from i2rt.robots.utils import combine_arm_and_gripper_xml

R = '/home/frank/robot/raiden/third_party/i2rt/i2rt/robot_models'
opening = float(sys.argv[1])
out = sys.argv[2]
xml = combine_arm_and_gripper_xml(f'{R}/arm/yam/yam.xml', f'{R}/gripper/linear_4310/linear_4310.xml')
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
q = np.zeros(m.nq)
for jn in ('joint7', 'joint8'):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    if j >= 0:
        q[m.jnt_qposadr[j]] = opening
d.qpos[:] = q
mujoco.mj_forward(m, d)
V, F, G, names = [], [], [], []
off = 0
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    f = m.mesh_face[fa:fa + fn] + off
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or 'world'
    V.append(v); F.append(f); G.append(np.full(len(f), len(names))); names.append(body)
    off += n
sites = {}
for s in ('tcp_site', 'grasp_site'):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
    if i >= 0:
        sites[s] = d.site_xpos[i].copy()
np.savez(out, vertices=np.concatenate(V), faces=np.concatenate(F), face_geom=np.concatenate(G),
         geom_body=np.array(names), qpos=q, opening=opening,
         tcp_site=sites.get('tcp_site'), grasp_site=sites.get('grasp_site'))
v = np.concatenate(V)
print('opening', opening, 'bbox min', v.min(0).round(4), 'max', v.max(0).round(4), 'nq', m.nq)
