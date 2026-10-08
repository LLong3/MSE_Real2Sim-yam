"""Dump raiden's YAM + linear_4310 MJCF (structure and MuJoCo FK) for the Blender asset.

Run with raiden's venv (read-only use; it has mujoco and i2rt):
    ~/robot/raiden/.venv/bin/python build/mjcf_reference.py
Writes source/yam_linear_4310.xml, source/mjcf_model.json, fk/mujoco_poses.json, fk/mujoco_geom_verts.npz.
"""
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from i2rt.robots.utils import ARM_YAM_XML_PATH, GRIPPER_LINEAR_4310_PATH, combine_arm_and_gripper_xml

HERE = Path(__file__).resolve().parents[1]
RAIDEN = Path('/home/frank/robot/raiden')
SEED = 20260929
N_RANDOM = 3
HOME_STROKE = 0.0475  # MJCF joint7/8 upper limit = raiden gripper 1.0 (open)


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def git(*args, cwd=RAIDEN):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def floats(text, n):
    v = [float(x) for x in text.split()] if text else None
    assert v is None or len(v) == n, text
    return v


def main():
    combined = combine_arm_and_gripper_xml(ARM_YAM_XML_PATH, GRIPPER_LINEAR_4310_PATH)
    out_xml = HERE / 'source' / 'yam_linear_4310.xml'
    shutil.copy(combined, out_xml)
    m = mujoco.MjModel.from_xml_path(str(out_xml))
    d = mujoco.MjData(m)
    root = ET.parse(out_xml).getroot()
    mesh_files = {e.get('name'): e.get('file') for e in root.find('asset') if e.tag == 'mesh'}

    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    bodies = []
    for b in range(m.nbody):
        bname = name(mujoco.mjtObj.mjOBJ_BODY, b) or 'world'
        el = root.find('worldbody') if b == 0 else root.find(f".//body[@name='{bname}']")
        joints = []
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            joints.append(dict(name=name(mujoco.mjtObj.mjOBJ_JOINT, j),
                               type={0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}[int(m.jnt_type[j])],
                               axis=m.jnt_axis[j].tolist(), pos=m.jnt_pos[j].tolist(),
                               range=m.jnt_range[j].tolist(), qposadr=int(m.jnt_qposadr[j])))
        geoms = []
        # geoms directly under this body element, in XML order (raw STL frame: XML pos/quat)
        for g in el.findall('geom'):
            geoms.append(dict(mesh=g.get('mesh'), file=mesh_files[g.get('mesh')],
                              file_md5=md5(mesh_files[g.get('mesh')]),
                              pos=floats(g.get('pos'), 3) or [0, 0, 0],
                              quat=floats(g.get('quat'), 4) or [1, 0, 0, 0],
                              rgba=floats(g.get('rgba'), 4)))
        sites = [dict(name=s.get('name'), pos=floats(s.get('pos'), 3) or [0, 0, 0],
                      quat=floats(s.get('quat'), 4) or [1, 0, 0, 0]) for s in el.findall('site')]
        bodies.append(dict(name=bname, id=b,
                           parent=(name(mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[b])) or 'world') if b else None,
                           pos=m.body_pos[b].tolist(), quat=m.body_quat[b].tolist(),
                           xml_pos=floats(el.get('pos'), 3) if b else None,
                           xml_quat=floats(el.get('quat'), 4) if b else None,
                           joints=joints, geoms=geoms, sites=sites))
    eq = [dict(joint1=name(mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj1id[e])),
               joint2=name(mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj2id[e])),
               polycoef=m.eq_data[e][:5].tolist()) for e in range(m.neq)]
    model = dict(
        source=dict(
            raiden_repo=str(RAIDEN), raiden_head=git('rev-parse', 'HEAD'), raiden_branch=git('branch', '--show-current'),
            i2rt_submodule=str(RAIDEN / 'third_party/i2rt'), i2rt_commit=git('rev-parse', 'HEAD', cwd=RAIDEN / 'third_party/i2rt'),
            i2rt_describe=git('describe', '--tags', '--always', cwd=RAIDEN / 'third_party/i2rt'),
            i2rt_origin=git('remote', 'get-url', 'origin', cwd=RAIDEN / 'third_party/i2rt'),
            arm_xml=ARM_YAM_XML_PATH, arm_xml_md5=md5(ARM_YAM_XML_PATH),
            gripper_xml=GRIPPER_LINEAR_4310_PATH, gripper_xml_md5=md5(GRIPPER_LINEAR_4310_PATH),
            combine='i2rt.robots.utils.combine_arm_and_gripper_xml (as raiden/_xml_paths.get_yam_4310_linear_xml_path)',
            mujoco_version=mujoco.__version__),
        nq=int(m.nq), bodies=bodies, equality=eq)
    (HERE / 'source' / 'mjcf_model.json').write_text(json.dumps(model, indent=1))

    # FK reference: q=0 and N random configs within joint ranges; fingers coupled (joint8 = joint7).
    rng = np.random.default_rng(SEED)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    # q0: all MJCF qpos zero (gripper closed); home: raiden FOLLOWER_HOME_POS (arm zeros, gripper 1.0 = open)
    configs = {'q0': np.zeros(m.nq), 'home': np.r_[np.zeros(6), HOME_STROKE, HOME_STROKE]}
    for k in range(N_RANDOM):
        q = rng.uniform(lo, hi)
        q[7] = q[6]
        configs[f'rand{k}'] = q
    poses, verts = {}, {}
    for key, q in configs.items():
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        poses[key] = dict(
            qpos=q.tolist(),
            bodies={(name(mujoco.mjtObj.mjOBJ_BODY, b) or 'world'): dict(pos=d.xpos[b].tolist(), quat=d.xquat[b].tolist(),
                                                                         mat=d.xmat[b].reshape(3, 3).tolist())
                    for b in range(m.nbody)},
            sites={name(mujoco.mjtObj.mjOBJ_SITE, s): dict(pos=d.site_xpos[s].tolist(), mat=d.site_xmat[s].reshape(3, 3).tolist())
                   for s in range(m.nsite)})
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            verts[f'{key}|{name(mujoco.mjtObj.mjOBJ_MESH, mid)}'] = v
    (HERE / 'fk' / 'mujoco_poses.json').write_text(json.dumps(dict(seed=SEED, configs=poses), indent=1))
    np.savez_compressed(HERE / 'fk' / 'mujoco_geom_verts.npz', **verts)
    print('bodies', [b['name'] for b in bodies])
    print('joints', [(j['name'], j['type'], j['axis'], j['range']) for b in bodies for j in b['joints']])
    for key, p in poses.items():
        print(key, np.round(p['qpos'], 4).tolist())


if __name__ == '__main__':
    main()
