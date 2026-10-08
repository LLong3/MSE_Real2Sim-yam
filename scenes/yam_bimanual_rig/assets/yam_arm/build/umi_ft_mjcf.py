"""Export the UMI-FT claws for MuJoCo (proposal only; raiden and MESA are not edited).

    ~/robot/raiden/.venv/bin/python build/umi_ft_mjcf.py [--claw source/umi_ft/claw_v4.json --out source/umi_ft/mjcf_v4]

Writes source/umi_ft/mjcf/:
- <finger>_{pad,face,clamp,white,black,carriage}.stl: visual meshes in the MJCF body frames tip_left/tip_right
  (geom pos/quat = identity), same placement as the Blender asset (source/umi_ft/claw.json schema 2: no strip
  thickness, the pad contact face as a black 0.2 mm visual slab, 1.0 mm yellow placeholder in the clamp group,
  holder group shifted by holder_shift_F_m);
- <finger>_pad_collision.stl: convex hull of the pad; two boxes approximate holder + clamp (in the snippet);
- umi_ft_fingers_snippet.xml: <asset> meshes and the <geom> lines that replace the stock tip geoms;
- yam_linear_4310_umi_ft.xml: test copy of source/yam_linear_4310.xml with those geoms and the proposed <inertial>;
- check.json: MuJoCo 3.3.5 compile, contact-surface positions and gap vs gripper q, masses.
Schema 3 (claw_v4.json): per-arm claw mount ('arm_mount': roll about the link_6 z axis): the
snippet names the wrapper body, and a second test copy yam_linear_4310_umi_ft_<arm>.xml wraps tip_left/tip_right in
it for each arm with a non-zero roll.
"""
import argparse
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parents[1]
UMI = HERE / 'source' / 'umi_ft'
_AP = argparse.ArgumentParser()
_AP.add_argument('--claw', help='claw config (default source/umi_ft/claw.json)')
_AP.add_argument('--out', help='output folder (default source/umi_ft/mjcf)')
_A, _ = _AP.parse_known_args()
_rel = lambda p: p if p.is_absolute() else HERE / p  # noqa: E731
CLAW_PATH = _rel(Path(_A.claw)) if _A.claw else UMI / 'claw.json'
OUT = _rel(Path(_A.out)) if _A.out else UMI / 'mjcf'
CLAW = json.loads(CLAW_PATH.read_text())
MOUNT_ROLL = {arm: float(v['roll_rad']) for arm, v in CLAW.get('arm_mount', {}).items() if isinstance(v, dict)}
MODEL = json.loads((HERE / 'source' / 'mjcf_model.json').read_text())
R_LEFT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
DENS = dict(tpu=1210.0, rubber=1100.0, pla_solid=1240.0, pla_printed=750.0)   # kg/m3; printed = ~60 % fill
RGBA = dict(pad='0.9 0.24 0.02 1', face='0.03 0.03 0.03 1', clamp='0.96 0.8 0 1', white='0.9 0.9 0.88 1',
            black='0.04 0.04 0.045 1', carriage='0.07 0.072 0.075 1')
assert CLAW.get('schema') in (2, 3), 'claw.json schema 2 or 3 expected'
SH = CLAW['holder_shift_F_m']
# collision boxes in the finger frame (m): holder + clamp (outer), proximal holder + adapter (inner); the holder
# group is shifted by SH along x_F (schema 2), so the proximal holder reaches SH past the pad plane (x_F = 0)
COLLISION_BOXES_F = dict(body_outer=([-0.04261 + SH, -0.01502], [-0.0231, 0.0059], [-0.0666, 0.03124]),
                         body_inner=([-0.01502, SH], [-0.0231, 0.0059], [-0.0666, -0.042]))
CONTACT = 'friction="1.2 0.02 0.002" condim="4" solref="0.01 1" solimp="0.9 0.95 0.002"'


def T(pos, quat_wxyz):
    M = np.eye(4); M[:3, :3] = Rot.from_quat([*quat_wxyz[1:], quat_wxyz[0]]).as_matrix(); M[:3, 3] = pos
    return M


def body_T_l6(name, q):
    b = next(b for b in MODEL['bodies'] if b['name'] == name)
    M = T(b['pos'], b['quat']); M[:3, 3] += M[:3, :3] @ (np.array(b['joints'][0]['axis']) * q)
    return M, b


def finger_T():
    M = np.eye(4); M[:3, :3] = R_LEFT
    M[:3, 3] = [CLAW['pad_x6_m'] + CLAW['pad_thickness_m'] / 2, -CLAW['open_half_m'], CLAW['pad_z6_m']]
    return M


def to_body(mesh, finger, T_l6_body):
    m = mesh.copy(); m.apply_transform(finger_T())
    if finger == 'tip_right':
        m.apply_transform(np.diag([1, -1, 1, 1.0]))   # trimesh flips the winding of a reflection
    m.apply_transform(np.linalg.inv(T_l6_body))
    return m


def box_F(x, y, z):
    return trimesh.creation.box(bounds=[[x[0], y[0], z[0]], [x[1], y[1], z[1]]])


def mount_wrapped(xml, roll):
    """the test MJCF with tip_left and tip_right moved into a body rolled about the link_6 z axis."""
    i0 = xml.index('<body name="tip_left"')
    ind = xml[xml.rindex('\n', 0, i0) + 1:i0]
    # end of tip_right: the closing tag that balances its <body
    j = xml.index('<body name="tip_right"')
    depth, k = 0, j
    for mt in re.finditer(r'<body\b|</body>', xml[j:]):
        depth += 1 if mt.group(0) == '<body' else -1
        if depth == 0:
            k = j + mt.end(); break
    assert xml[i0:j].count('<body') == xml[i0:j].count('</body>'), 'tip_left and tip_right must be adjacent'
    inner = xml[i0:k]
    q = f'{math.cos(roll / 2):.8f} 0 0 {math.sin(roll / 2):.8f}'
    return xml[:i0] + f'<body name="umi_claw_mount" quat="{q}">\n{ind}  ' + inner.replace('\n', '\n  ') + f'\n{ind}</body>' + xml[k:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    home = CLAW['home_gripper_m']
    shift = trimesh.transformations.translation_matrix([SH, 0, 0])
    pad = trimesh.load(UMI / 'umift_finger_no_contact_mic.stl')
    face = (pad.face_normals[:, 0] > 0.99) & (pad.triangles_center[:, 0] > -0.0005)
    assert face.sum() == 26, face.sum()
    holder = [trimesh.load(UMI / 'parts' / f'holder_{c}.stl') for c in ('white', 'black')]
    for h in holder:
        h.apply_transform(shift)
    fb = pad.submesh([np.nonzero(face)[0]], append=True).bounds     # the face is the full 17.2 x 96.7 mm rectangle
    face_mesh = box_F([0.0, 0.0002], fb[:, 1], fb[:, 2])            # visual 0.2 mm slab (a flat mesh has no volume in MuJoCo)
    parts_F = dict(pad=[pad], face=[face_mesh],
                   clamp=[trimesh.load(UMI / 'parts' / 'clamp.stl')], white=[holder[0]], black=[holder[1]])
    for name, b in CLAW['boxes_F'].items():
        bx = box_F(b['x'], b['y'], b['z'])
        if name in CLAW['holder_group']:
            bx.apply_transform(shift)
        parts_F[b['material']].append(bx)
    dens = dict(pad=DENS['tpu'], clamp=DENS['pla_printed'], white=DENS['pla_printed'], black=DENS['pla_printed'])
    report = dict(claw=CLAW, claw_json=str(CLAW_PATH), densities_kg_m3=DENS, fingers={})
    asset_lines, geom_lines, inertials = [], {}, {}
    for finger in ('tip_left', 'tip_right'):
        T_l6_b, body = body_T_l6(finger, home)
        stock = trimesh.load(body['geoms'][0]['file'])
        stock.apply_transform(T(body['geoms'][0]['pos'], body['geoms'][0]['quat']))          # body frame
        # carriage = stock above z6 cut (plane expressed in the body frame), capped for mass properties
        n_b = np.linalg.inv(T_l6_b)[:3, :3] @ [0, 0, 1.0]
        o_b = (np.linalg.inv(T_l6_b) @ [0, 0, CLAW['carriage_cut_z6_m'], 1])[:3]
        carriage = stock.slice_plane(o_b, n_b, cap=True)
        removed = stock.slice_plane(o_b, -n_b, cap=True)
        m_removed = removed.volume * DENS['pla_printed']
        m_carriage = body_mass = None
        stock_mass = 0.0710042
        m_carriage = stock_mass - m_removed
        groups = {k: trimesh.util.concatenate([to_body(p, finger, T_l6_b) for p in v]) for k, v in parts_F.items() if v}
        groups['carriage'] = carriage
        geoms = []
        for g, m in groups.items():
            f = OUT / f'{finger}_{g}.stl'; m.export(f)
            asset_lines.append(f'    <mesh name="umi_{finger}_{g}" file="{f.name}" />')
            geoms.append(f'<geom type="mesh" mesh="umi_{finger}_{g}" rgba="{RGBA[g]}" contype="0" conaffinity="0" group="2" />')
        hull = groups['pad'].convex_hull
        f = OUT / f'{finger}_pad_collision.stl'; hull.export(f)
        asset_lines.append(f'    <mesh name="umi_{finger}_pad_collision" file="{f.name}" />')
        geoms.append(f'<geom name="umi_{finger}_pad" type="mesh" mesh="umi_{finger}_pad_collision" group="3" {CONTACT} />')
        for bname, (x, y, z) in COLLISION_BOXES_F.items():
            bx = to_body(box_F(x, y, z), finger, T_l6_b)
            c, h = bx.bounds.mean(0), (bx.bounds[1] - bx.bounds[0]) / 2
            geoms.append(f'<geom name="umi_{finger}_{bname}" type="box" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
                         f'size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}" group="3" />')
        geom_lines[finger] = geoms
        # proposed inertial: carriage (stock mass minus the removed stock claw) + UMI parts
        comps = [(carriage, m_carriage / carriage.volume)] + [(groups[g], dens[g]) for g in ('pad', 'clamp', 'white', 'black')]
        mass, com, I = 0.0, np.zeros(3), np.zeros((3, 3))
        props = []
        for m, rho in comps:
            m = m.copy(); m.density = rho
            props.append((m.mass, m.center_mass, m.moment_inertia))
        mass = sum(p[0] for p in props)
        com = sum(p[0] * p[1] for p in props) / mass
        for mi, ci, Ii in props:
            d = ci - com
            I += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        inertials[finger] = (f'<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" mass="{mass:.4f}" '
                             f'fullinertia="{I[0,0]:.4e} {I[1,1]:.4e} {I[2,2]:.4e} {I[0,1]:.4e} {I[0,2]:.4e} {I[1,2]:.4e}" />')
        vol = {g: float(groups[g].volume * 1e6) for g in ('pad', 'clamp', 'white', 'black')}
        printed = vol['clamp'] + vol['white'] + vol['black']
        report['fingers'][finger] = dict(
            stock_mass_kg=stock_mass, stock_volume_cm3=float(stock.volume * 1e6), stock_claw_removed_cm3=float(removed.volume * 1e6),
            carriage_volume_cm3=float(carriage.volume * 1e6), umi_volumes_cm3=vol,
            new_mass_kg=float(mass), new_com_body_m=com.tolist(),
            mass_range_kg=[float(stock_mass - removed.volume * d + vol['pad'] * 1.21e-3 + printed * d * 1e-6)
                           for d in (DENS['pla_printed'] * 0.62 / 0.75, DENS['pla_solid'])],
            inertial=inertials[finger])
    snippet = [f'<!-- UMI-FT claws for raiden yam_linear_4310 (proposal; placement source/umi_ft/{CLAW_PATH.name}). -->',
               '<!-- asset: add to <asset> (paths relative to this folder) -->', *asset_lines]
    for finger in ('tip_left', 'tip_right'):
        snippet += [f'<!-- body {finger}: replace <geom ... mesh="{finger}" /> and its <inertial> with -->',
                    f'    {inertials[finger]}', *[f'    {g}' for g in geom_lines[finger]]]
    for arm, roll in MOUNT_ROLL.items():
        if roll:
            snippet += [f'<!-- {arm}: claw pair rolled {math.degrees(roll):.2f} deg about the link_6 z axis (claw_v4.json arm_mount); wrap '
                        f'bodies tip_left and tip_right in: -->',
                        f'    <body name="umi_claw_mount" quat="{math.cos(roll / 2):.8f} 0 0 {math.sin(roll / 2):.8f}">']
    (OUT / 'umi_ft_fingers_snippet.xml').write_text('\n'.join(snippet) + '\n')
    # test copy of the combined MJCF
    xml = (HERE / 'source' / 'yam_linear_4310.xml').read_text()
    xml = xml.replace('  </asset>', '\n'.join(f'  {l.strip()}'.replace('file="', f'file="{OUT}/') for l in asset_lines) + '\n  </asset>', 1)
    for finger in ('tip_left', 'tip_right'):
        xml, n = re.subn(rf'<geom [^>]*mesh="{finger}" />', '\n        '.join(geom_lines[finger]), xml)
        assert n == 1, finger
        xml, n = re.subn(rf'(<body name="{finger}"[^>]*>\s*)<inertial [^>]*/>', lambda m: m.group(1) + inertials[finger], xml)
        assert n == 1, finger
    test = OUT / 'yam_linear_4310_umi_ft.xml'; test.write_text(xml)
    tests = {'default': test}
    for arm, roll in MOUNT_ROLL.items():
        if roll:
            tests[arm] = OUT / f'yam_linear_4310_umi_ft_{arm}.xml'
            tests[arm].write_text(mount_wrapped(xml, roll))
    # MuJoCo check
    mdl = mujoco.MjModel.from_xml_path(str(test)); data = mujoco.MjData(mdl)
    stock_mdl = mujoco.MjModel.from_xml_path(str(HERE / 'source' / 'yam_linear_4310.xml'))
    def contact_y(q):
        data.qpos[:] = 0; data.qpos[6] = data.qpos[7] = q; mujoco.mj_kinematics(mdl, data)
        l6 = mdl.body('link_6').id
        R6, p6 = data.xmat[l6].reshape(3, 3), data.xpos[l6]
        out = {}
        for finger in ('tip_left', 'tip_right'):
            g = mdl.geom(f'umi_{finger}_pad').id
            mid = mdl.geom_dataid[g]
            v = mdl.mesh_vert[mdl.mesh_vertadr[mid]:mdl.mesh_vertadr[mid] + mdl.mesh_vertnum[mid]]
            w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            y6 = (w - p6) @ R6[:, 1]
            out[finger] = float(y6.max() if finger == 'tip_left' else y6.min())
        return out
    checks = {}
    for q in (0.0, 0.01, home):
        cy = contact_y(q)
        checks[f'q={q}'] = dict(contact_y6_m=cy, gap_m=cy['tip_right'] - cy['tip_left'])
    for arm, path in tests.items():
        if arm == 'default':
            continue
        m2 = mujoco.MjModel.from_xml_path(str(path)); d2 = mujoco.MjData(m2)
        d2.qpos[:] = 0; d2.qpos[6] = d2.qpos[7] = home; mujoco.mj_kinematics(m2, d2)
        l6 = m2.body('link_6').id
        R6 = d2.xmat[l6].reshape(3, 3)
        data.qpos[:] = 0; data.qpos[6] = data.qpos[7] = home; mujoco.mj_kinematics(mdl, data)
        R0 = data.xmat[mdl.body('link_6').id].reshape(3, 3).T @ data.xmat[mdl.body('tip_left').id].reshape(3, 3)
        R1 = R6.T @ d2.xmat[m2.body('tip_left').id].reshape(3, 3)
        dR = R1 @ R0.T                                  # the mount roll, measured in link_6
        got = math.atan2(dR[1, 0], dR[0, 0])
        assert abs(got - MOUNT_ROLL[arm]) < 1e-6 and abs(dR[2, 2] - 1) < 1e-9, (arm, got)
        checks[f'{arm}_mount'] = dict(test_model=str(path), compiled=True, roll_rad=MOUNT_ROLL[arm],
                                      body_mass_kg={b: float(m2.body(b).mass[0]) for b in ('tip_left', 'tip_right')})
    report['mujoco'] = dict(version=mujoco.__version__, compiled=True, test_model=str(test), contact_surfaces=checks,
                            body_mass_kg={b: float(mdl.body(b).mass[0]) for b in ('tip_left', 'tip_right')},
                            stock_body_mass_kg={b: float(stock_mdl.body(b).mass[0]) for b in ('tip_left', 'tip_right')},
                            ngeom=int(mdl.ngeom), stock_ngeom=int(stock_mdl.ngeom))
    (OUT / 'check.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(fingers={k: {kk: v[kk] for kk in ('new_mass_kg', 'mass_range_kg', 'stock_claw_removed_cm3', 'umi_volumes_cm3')} for k, v in report['fingers'].items()},
                          mujoco=report['mujoco']), indent=1))


if __name__ == '__main__':
    main()
