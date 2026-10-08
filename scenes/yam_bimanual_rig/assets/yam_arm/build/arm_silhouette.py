"""Stock-arm silhouettes from the MJCF (MuJoCo geoms at any joint vector) projected into a camera (numpy/OpenCV).

Used by the 09-30 wrist/claw fit to check arm base poses in the scene camera: the calibration captures show the
arms at 7 poses, the 09-29 16:00 image shows the grippers at home.
"""
import cv2
import numpy as np

import wristfit_common as W


class ArmMesh:
    def __init__(self, fk=None):
        self.fk = fk or W.FK()
        m = self.fk.m
        mj = self.fk.mj
        self.geoms = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH or m.geom_group[g] > 2:
                continue
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            V = m.mesh_vert[v0:v0 + nv].copy(); F = m.mesh_face[f0:f0 + nf].copy()
            body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            self.geoms.append(dict(g=g, body=body, V=V, F=F))

    def tris(self, q6, gripper=0.0475, skip=('tip_left', 'tip_right')):
        """world(base)-frame triangles (N,3,3) and a body label per triangle."""
        d, m = self.fk.d, self.fk.m
        q = np.zeros(m.nq); q[:6] = q6
        if m.nq >= 8:
            q[6] = q[7] = gripper
        d.qpos[:] = q
        self.fk.mj.mj_kinematics(m, d)
        out, lab = [], []
        for ge in self.geoms:
            if ge['body'] in skip:
                continue
            R = d.geom_xmat[ge['g']].reshape(3, 3); t = d.geom_xpos[ge['g']]
            V = ge['V'] @ R.T + t
            out.append(V[ge['F']]); lab += [ge['body']] * len(ge['F'])
        return np.concatenate(out), np.array(lab)


def silhouette(tris_world, K, T_wc, W_=960, H=600, scale=1):
    """binary mask of the projected triangles (no z-buffer needed for a silhouette)."""
    P = tris_world.reshape(-1, 3)
    pc = (P - T_wc[:3, 3]) @ T_wc[:3, :3]
    ok = (pc[:, 2] > 0.02).reshape(-1, 3).all(1)
    uv = (pc[:, :2] / np.clip(pc[:, 2:3], 1e-6, None) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).reshape(-1, 3, 2)[ok]
    img = np.zeros((H * scale, W_ * scale), np.uint8)
    uv = uv[(np.abs(uv) < 4000).all((1, 2))]
    pts = np.round(uv * scale * 4).astype(np.int32)
    for t in pts:        # one call per triangle: cv2.fillPoly on a list is even-odd filled (overlaps become holes)
        cv2.fillConvexPoly(img, t, 1, cv2.LINE_8, 2)
    if scale > 1:
        img = cv2.resize(img, (W_, H), interpolation=cv2.INTER_AREA)
    return img > 0
