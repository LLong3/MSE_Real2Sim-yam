"""UMI-FT claw geometry in link_6 at raiden home (numpy), shared by the 09-30 claw fit and the asset build.

Parts in the finger frame F (the UMI-FT finger STL frame: x toward the contact face, y extrusion, z length, lugs
toward +z): pad (fin-ray; its flat contact face x_F = 0 is labelled separately), clamp, placeholder (thin yellow
part in the CoinFT slot), holder (white, black tag block), adapter boxes, tag. Constants and boxes come from
source/umi_ft/claw.json (schema 2, 09-30). Placement of the left claw (tip_left) in link_6:
    x6 = y_F + pad_x6 + pad_thickness / 2,  y6 = x_F - open_half,  z6 = pad_z6 - z_F
so the contact face is at y6 = -open_half, the pad mid-plane at x6 = pad_x6; the right claw is the left one
mirrored in y6. The holder group (holder, tag, adapter) is shifted by holder_shift_F_m along x_F relative to the
pad and clamp (it closes the CoinFT slot down to the placeholder thickness).
Also: the stock arm (MuJoCo geoms: housing = link_6 gripper mesh, carriage = stock tip faces above
carriage_cut_z6_m, other links) as labelled occluders, and a painter's-algorithm label rasterizer (cv2) for fits.
Large triangles are subdivided (SPLIT: 8 mm on the fitted parts, 15 mm on the holder group, 4 mm otherwise) so the
per-triangle depth order holds, and back faces are culled (all meshes are outward-wound; the mirrored claw is
re-wound). Validated against the Blender v2 object IDs by build/claw_model_check.py (09-30).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parents[1]
UMI = HERE / 'source' / 'umi_ft'
R_LEFT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
LABELS = dict(pad=1, pad_face=2, clamp=3, placeholder=4, holder_white=5, holder_black=6, tag=7, adapter_black=8,
              adapter_white=9, carriage=10, housing=11, strip=12, spacer=13, arm=14)
MAX_EDGE = 0.004                  # default split size (m); per part in SPLIT (fit speed vs painter's accuracy)
SPLIT = dict(pad=0.008, pad_face=0.008, clamp=0.008, holder_white=0.015, holder_black=0.015, adapter_black=0.015,
             adapter_white=0.015)
HOLDER_GROUP = ('holder_white', 'holder_black', 'tag', 'adapter_black', 'adapter_white')
TAG_CENTRE_F = np.array([-0.02885, -0.0231 - 0.00015, -0.0523])   # tag block top face (add_umi_ft_claws.py)
TAG_SIZE = 0.0246


def load_cfg(path=None):
    return json.loads(Path(path or UMI / 'claw.json').read_text())


def box_tris(b):
    (x0, x1), (y0, y1), (z0, z1) = b['x'], b['y'], b['z']
    m = trimesh.creation.box(bounds=[[x0, y0, z0], [x1, y1, z1]])
    return m.triangles


def subdivide(tris, max_edge):
    """split triangles until every edge is shorter than max_edge (triangle soup; for the painter's order)."""
    tris = np.asarray(tris, float)
    if not len(tris):
        return tris
    v, f = trimesh.remesh.subdivide_to_size(tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3), max_edge)
    return v[f]


def parts_F(cfg):
    """{part: triangles (N,3,3) in F}, with the holder group already shifted by holder_shift_F_m."""
    pad = trimesh.load(UMI / 'umift_finger_no_contact_mic.stl')
    n, c = pad.face_normals, pad.triangles_center
    face = (n[:, 0] > 0.99) & (c[:, 0] > -0.0005)
    out = dict(pad=pad.triangles[~face], pad_face=pad.triangles[face],
               clamp=trimesh.load(UMI / 'parts' / 'clamp.stl').triangles,
               holder_white=trimesh.load(UMI / 'parts' / 'holder_white.stl').triangles,
               holder_black=trimesh.load(UMI / 'parts' / 'holder_black.stl').triangles)
    for k, b in cfg['boxes_F'].items():
        out[k] = box_tris(b)
    u, v = np.array([1.0, 0, 0]) * TAG_SIZE / 2, np.array([0, 0, 1.0]) * TAG_SIZE / 2   # top face y_F = const: x_F, z_F
    cF = TAG_CENTRE_F
    q = np.array([cF - u - v, cF + u - v, cF + u + v, cF - u + v])
    out['tag'] = np.array([[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
    sh = cfg.get('holder_shift_F_m', 0.0)
    for k in cfg.get('holder_group', HOLDER_GROUP):
        if k in out:
            out[k] = out[k] + np.array([sh, 0, 0])
    return {k: subdivide(v, SPLIT.get(k, MAX_EDGE)) for k, v in out.items()}


def arm_tris(fk, q6=None, gripper=0.0475, cut_z6=-0.070, links=('link_6',)):
    """[(label, tris in the arm base frame)] of the stock arm at q6 (default home): housing (link_6 mesh), carriage
    (stock tip faces whose centre is above cut_z6 in link_6, as add_umi_ft_claws.py), other links in `links` as 'arm'."""
    m, d, mj = fk.m, fk.d, fk.mj
    q = np.zeros(m.nq); q[:6] = np.zeros(6) if q6 is None else q6
    if m.nq >= 8:
        q[6] = q[7] = gripper
    d.qpos[:] = q
    mj.mj_kinematics(m, d)
    b6 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, 'link_6')
    T6 = np.eye(4); T6[:3, :3] = d.xmat[b6].reshape(3, 3); T6[:3, 3] = d.xpos[b6]
    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if body not in ('tip_left', 'tip_right') and body not in links:
            continue
        mid = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        tri = (V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[F]
        if body in ('tip_left', 'tip_right'):
            z6 = ((tri.reshape(-1, 3) - T6[:3, 3]) @ T6[:3, :3])[:, 2].reshape(-1, 3).mean(1)
            out.append((LABELS['carriage'], tri[z6 > cut_z6]))
        else:
            out.append((LABELS['housing'] if body == 'link_6' else LABELS['arm'], tri))
    return out, T6


def T_l6_F(open_half, pad_z6, pad_x6, pad_thick=0.0172):
    M = np.eye(4); M[:3, :3] = R_LEFT
    M[:3, 3] = [pad_x6 + pad_thick / 2, -open_half, pad_z6]
    return M


def claw_tris_l6(pF, open_half, pad_z6, pad_x6, parts=None):
    """[(label, tris in link_6)] for both claws at raiden home."""
    T = T_l6_F(open_half, pad_z6, pad_x6)
    out = []
    for k, tri in pF.items():
        if parts is not None and k not in parts:
            continue
        t = tri @ T[:3, :3].T + T[:3, 3]
        out.append((LABELS[k], t))
        tr = t[:, ::-1].copy(); tr[..., 1] *= -1          # mirror in y6, re-wound (outward normals kept)
        out.append((LABELS[k], tr))
    return out


# the tag is 0.15 mm above the tag block top face (coplanar for the painter's order): sort it
# slightly nearer, as Blender's first hit shows it
DEPTH_BIAS = {LABELS['tag']: -0.002}


def render_labels(items, K, T_wc, W=960, H=600, scale=1.0, bias=None, cull=True):
    """items: [(label, tris (N,3,3) in the world, outward-wound)]; painter's algorithm by mean depth (far first),
    back faces culled. Returns (H,W) uint8."""
    bias = DEPTH_BIAS if bias is None else bias
    tris = np.concatenate([t for _, t in items]); labs = np.concatenate([np.full(len(t), l, np.uint8) for l, t in items])
    P = tris.reshape(-1, 3)
    pc = (P - T_wc[:3, 3]) @ T_wc[:3, :3]
    z = pc[:, 2].reshape(-1, 3)
    ok = (z > 0.01).all(1)
    if cull:
        t3 = pc.reshape(-1, 3, 3)
        ok &= np.einsum('ij,ij->i', np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0]), t3[:, 0]) < 0
    uv = (pc[:, :2] / np.clip(pc[:, 2:3], 1e-6, None) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).reshape(-1, 3, 2) * scale
    ok &= (np.abs(uv) < 8000).all((1, 2))
    uv, z, labs = uv[ok], z[ok].mean(1), labs[ok]
    for lab, b in bias.items():
        z[labs == lab] += b
    Hs, Ws = int(round(H * scale)), int(round(W * scale))
    inb = (uv[..., 0].max(1) >= 0) & (uv[..., 0].min(1) < Ws) & (uv[..., 1].max(1) >= 0) & (uv[..., 1].min(1) < Hs)
    uv, z, labs = uv[inb], z[inb], labs[inb]
    img = np.zeros((Hs, Ws), np.uint8)
    pts = np.round(uv * 16).astype(np.int32)
    for i in np.argsort(-z):
        cv2.fillConvexPoly(img, pts[i], int(labs[i]), cv2.LINE_8, 4)
    return img


def transform(items, T):
    return [(l, t @ T[:3, :3].T + T[:3, 3]) for l, t in items]
