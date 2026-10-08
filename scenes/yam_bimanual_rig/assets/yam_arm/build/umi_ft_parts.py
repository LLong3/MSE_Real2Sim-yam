"""Extract the UMI-FT claw parts from the Onshape STL exports into the finger frame (source prep).

    ~/robot/raiden/.venv/bin/python build/umi_ft_parts.py

Inputs (user exports of the public Onshape document "UMI-FT Public", copied read-only into source/umi_ft/):
umift_finger_no_contact_mic.stl, umift_assembled.stl, umift_wsg50_gripper.stl (metres).

Finger frame F = the frame of umift_finger_no_contact_mic.stl: x toward the flat contact face (face at x=0,
back at x=-25 mm), y = extrusion (-17.2..0 mm), z = length (-42..54.7 mm, lugs toward +z).
The finger STL matches one assembled finger exactly (axis-permutation fit); the clamp and holder that
touch that finger are taken from the assembly and expressed in F, so their relation is the UMI-FT CAD's.
Rig modifications (README): holder clipped where its full-height block starts (the UMI rack plate is
dropped); its proximal top part painted black (the rig's tag block).
Writes source/umi_ft/parts/{clamp,holder_white,holder_black}.stl (F frame, metres), parts.json
(inventory of all three STLs, transforms, residuals) and the finger tag textures source/umi_ft/tags/*.png.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / 'source' / 'umi_ft'
OUT = SRC / 'parts'
HOLDER_CLIP_Z = -0.0615    # F z: the UMI holder is full height (29 mm) only above this; below it is an 8 mm rack plate
CAP_Z, CAP_Y = -0.0178, -0.006   # black tag block: F z < CAP_Z and F y < CAP_Y (top 17 mm), from IMG_5411 f1256


# Finger tags (UMI-style ArUco, DICT_4X4_50), read from IMG_5411 (ids 0, 6 decoded) and the ZED scene image (glyphs):
# left arm tip_left 7, tip_right 6; right arm tip_left 1, tip_right 0. Layout measured on IMG_5411 f1256 (mm):
TAG_IDS = (0, 1, 6, 7)
TAG = dict(outer_mm=24.6, white_line_mm=0.8, black_gap_mm=0.9, px_per_mm=20)


def write_tags(out_dir):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    n = round(TAG['outer_mm'] * TAG['px_per_mm'])
    line, gap = round(TAG['white_line_mm'] * TAG['px_per_mm']), round(TAG['black_gap_mm'] * TAG['px_per_mm'])
    inner = n - 2 * (line + gap)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    files = {}
    for i in TAG_IDS:
        img = np.zeros((n, n), np.uint8)
        img[:, :] = 255
        img[line:n - line, line:n - line] = 0
        m = cv2.aruco.generateImageMarker(d, i, inner, borderBits=1)
        img[line + gap:line + gap + inner, line + gap:line + gap + inner] = m
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        _, ids, _ = det.detectMarkers(cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255))
        assert ids is not None and ids.ravel().tolist() == [i], ids
        p = out_dir / f'aruco_4x4_50_id{i}.png'
        cv2.imwrite(str(p), img)
        files[i] = f'source/umi_ft/tags/{p.name}'
    return dict(TAG, dictionary='DICT_4X4_50', size_px=n, aruco_mm=inner / TAG['px_per_mm'], files=files,
                image_up='toward the fingertip', assignment=dict(left_arm=dict(tip_left=7, tip_right=6),
                                                                  right_arm=dict(tip_left=1, tip_right=0)))


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def axis_fit(src, dst):
    """Best proper axis-permutation rotation + translation mapping src vertices onto dst (congruent meshes)."""
    tree = cKDTree(dst.vertices)
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[p, i] = signs[i]
            if np.linalg.det(R) < 0:
                continue
            v = src.vertices @ R.T
            t = dst.bounds.mean(0) - (v.min(0) + v.max(0)) / 2
            err = tree.query(v + t)[0].max()
            if best is None or err < best[0]:
                best = (err, R, t)
    T = np.eye(4); T[:3, :3] = best[1]; T[:3, 3] = best[2]
    return T, float(best[0])


def inventory(mesh):
    parts = mesh.split(only_watertight=False)
    big = sorted([p for p in parts if len(p.faces) >= 500], key=lambda p: -len(p.faces))
    return parts, dict(faces=int(len(mesh.faces)), extents_mm=(mesh.extents * 1e3).round(1).tolist(),
                       components=len(parts), components_ge_500_faces=[
                           dict(faces=int(len(p.faces)), extents_mm=(p.extents * 1e3).round(1).tolist(),
                                centre_mm=(p.bounds.mean(0) * 1e3).round(1).tolist(), watertight=bool(p.is_watertight))
                           for p in big])


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.maximum(a.bounds[0] - b.bounds[1], b.bounds[0] - a.bounds[1]))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = dict(finger='umift_finger_no_contact_mic.stl', assembled='umift_assembled.stl', wsg50='umift_wsg50_gripper.stl')
    meshes = {k: trimesh.load(SRC / f) for k, f in files.items()}
    report = dict(inputs={k: dict(file=f'source/umi_ft/{f}', md5=md5(SRC / f)) for k, f in files.items()},
                  units='metres (finger extents 25.0 x 17.2 x 96.7 mm = published finger_width, finger_thickness; '
                        'finger_length 84 mm is the fin-ray body without the lug ends)')
    asm_parts, report['inventory_assembled'] = inventory(meshes['assembled'])
    _, report['inventory_wsg50'] = inventory(meshes['wsg50'])
    report['inventory_finger'] = dict(faces=int(len(meshes['finger'].faces)), watertight=bool(meshes['finger'].is_watertight),
                                      extents_mm=(meshes['finger'].extents * 1e3).round(2).tolist(),
                                      volume_cm3=float(meshes['finger'].volume * 1e6))
    fin = meshes['finger']
    # assembled finger instances: congruent extents; the exact match is the no-mic finger
    cands = [p for p in asm_parts if np.allclose(np.sort(p.extents), np.sort(fin.extents), atol=2e-4)]
    fits = sorted((axis_fit(fin, p) + (p,) for p in cands), key=lambda r: r[1])
    T_asm_F, err, asm_fin = fits[0]
    assert err < 1e-6, err
    F_from_asm = np.linalg.inv(T_asm_F)
    touching = [p for p in asm_parts if p is not asm_fin and len(p.faces) > 1000 and gap(p, asm_fin) < 0.004]
    clamp = min(touching, key=lambda p: abs(p.extents - [0.0142, 0.031, 0.029]).sum()).copy()
    holder = max(touching, key=lambda p: len(p.faces)).copy()
    clamp.apply_transform(F_from_asm); holder.apply_transform(F_from_asm)
    assert np.allclose(clamp.extents, [0.01415, 0.029, 0.031], atol=5e-4), clamp.extents
    report['finger_to_assembly'] = dict(T_assembly_from_finger=T_asm_F.tolist(), max_vertex_residual_m=err,
                                        other_finger_instance_residual_m=[float(r[1]) for r in fits[1:]],
                                        note='the second assembled finger is the with-contact-mic variant (4.1 mm off)')
    # holder: drop the rack plate, split off the black tag block
    h = holder.slice_plane([0, 0, HOLDER_CLIP_Z], [0, 0, 1], cap=True)
    white_distal = h.slice_plane([0, 0, CAP_Z], [0, 0, 1], cap=True)
    prox = h.slice_plane([0, 0, CAP_Z], [0, 0, -1], cap=True)
    black = prox.slice_plane([0, CAP_Y, 0], [0, -1, 0], cap=True)
    white_prox = prox.slice_plane([0, CAP_Y, 0], [0, 1, 0], cap=True)
    white = trimesh.util.concatenate([white_distal, white_prox])
    for name, m in dict(clamp=clamp, holder_white=white, holder_black=black).items():
        m.export(OUT / f'{name}.stl')
    report['parts_in_finger_frame'] = {
        name: dict(file=f'source/umi_ft/parts/{name}.stl', faces=int(len(m.faces)), watertight=bool(m.is_watertight),
                   volume_cm3=float(m.volume * 1e6) if m.is_watertight else None,
                   bounds_mm=(m.bounds * 1e3).round(2).tolist())
        for name, m in dict(pad=fin, clamp=clamp, holder_white=white, holder_black=black).items()}
    report['holder_full_before_clip'] = dict(faces=int(len(holder.faces)), volume_cm3=float(holder.volume * 1e6),
                                             bounds_mm=(holder.bounds * 1e3).round(2).tolist())
    report['modifications'] = dict(holder_clip_z_m=HOLDER_CLIP_Z, tag_block=dict(z_below_m=CAP_Z, y_below_m=CAP_Y))
    report['tags'] = write_tags(SRC / 'tags')
    (OUT / 'parts.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('units', 'parts_in_finger_frame')}, indent=1))


if __name__ == '__main__':
    main()
