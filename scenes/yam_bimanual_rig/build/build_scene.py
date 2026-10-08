"""Author the yam_bimanual_rig twin in the metric frame (Blender 5.2, background).

blender -b --factory-startup --python-exit-code 1 --python build_scene.py -- \
    --config scene_config.json --out ../blender/rig_vNN.blend [--arms PATH|none] [--arms-status placeholder|final]

Blender world = metric frame (metres, Z up, floor z=0, desk top z=0.72, +Y toward the back wall).
Reads metric/metric_frame.json (read-only) and the RoomKit v1 material library (read-only).
Arms are appended from the scene-local yam_arm asset (read-only).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
ROOT = SCENE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aha3d.blender import semantics  # noqa: E402

ROOMKIT = ROOT / 'assets/roomkit/v1/roomkit_furniture_materials.blend'
ARM_COLLECTION = 'YAM arm (i2rt yam + linear_4310)'


def parse():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--metric', type=Path, default=SCENE / 'metric/metric_frame.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', default=str(SCENE / 'assets/yam_arm/yam_arm.blend'))
    p.add_argument('--arms-status', choices=['placeholder', 'final'], default='placeholder')
    return p.parse_args(argv)


# ---------------------------------------------------------------- geometry

def link(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def mesh_object(name, bm, col, material=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    for poly in me.polygons:
        poly.use_smooth = False
    return obj


def planar_uv(bm):
    """UVs in metres from world coordinates (vertices are authored in world space)."""
    uv = bm.loops.layers.uv.new('UVMap')
    for f in bm.faces:
        n = f.normal
        a = max(range(3), key=lambda i: abs(n[i]))
        for loop in f.loops:
            co = loop.vert.co
            loop[uv].uv = (co.x, co.y) if a == 2 else (co.y, co.z) if a == 0 else (co.x, co.z)


def box(name, x, y, z, col, material=None, bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x = x[0] + (v.co.x + 0.5) * (x[1] - x[0])
        v.co.y = y[0] + (v.co.y + 0.5) * (y[1] - y[0])
        v.co.z = z[0] + (v.co.z + 0.5) * (z[1] - z[0])
    bm.normal_update()
    planar_uv(bm)
    obj = mesh_object(name, bm, col, material)
    if bevel:
        m = obj.modifiers.new('Edge radius', 'BEVEL')
        m.width, m.segments, m.limit_method = bevel, 2, 'ANGLE'
    return obj


def prism(name, pts2, axis, span, col, material=None):
    """Extrude a closed 2D profile. axis 'x': pts are (y, z); 'y': (x, z); 'z': (x, y)."""
    bm = bmesh.new()
    def co(p, s):
        if axis == 'x':
            return (s, p[0], p[1])
        if axis == 'y':
            return (p[0], s, p[1])
        return (p[0], p[1], s)
    a = [bm.verts.new(co(p, span[0])) for p in pts2]
    b = [bm.verts.new(co(p, span[1])) for p in pts2]
    n = len(pts2)
    bm.faces.new(a[::-1])
    bm.faces.new(b)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new([a[i], a[j], b[j], b[i]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    planar_uv(bm)
    return mesh_object(name, bm, col, material)


def rounded_rect(x0, x1, z0, z1, r, round_bottom=False, seg=8):
    pts = []
    def arc(cx, cz, a0, a1):
        for i in range(seg + 1):
            a = a0 + (a1 - a0) * i / seg
            pts.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    if round_bottom:
        arc(x0 + r, z0 + r, math.pi, 1.5 * math.pi); arc(x1 - r, z0 + r, 1.5 * math.pi, 2 * math.pi)
    else:
        pts += [(x0, z0), (x1, z0)]
    arc(x1 - r, z1 - r, 0, 0.5 * math.pi)
    arc(x0 + r, z1 - r, 0.5 * math.pi, math.pi)
    return pts


def set_origin(obj, point):
    """Move the object origin to a world point without moving geometry."""
    off = Vector(point) - obj.matrix_world.translation
    obj.data.transform(Matrix.Translation(-off))
    obj.matrix_world.translation += off


# ---------------------------------------------------------------- materials

def roomkit_material(name, new_name):
    with bpy.data.libraries.load(str(ROOMKIT), link=False) as (src, dst):
        if name not in src.materials:
            raise ValueError('RoomKit material missing: ' + name)
        dst.materials = [name]
    mat = dst.materials[0]
    mat.name = new_name
    mat['roomkit_source'] = 'roomkit-v1::' + name
    return mat


def tint(mat, rgb):
    """Multiply the Base Color input of every Principled node by a linear RGB tint (keeps the RoomKit graph)."""
    if list(rgb) == [1, 1, 1]:
        return mat
    nt = mat.node_tree
    for n in [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED']:
        inp = n.inputs['Base Color']
        mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
        mul.inputs['Factor'].default_value = 1.0
        mul.inputs['B'].default_value = (*rgb, 1)
        if inp.links:
            nt.links.new(inp.links[0].from_socket, mul.inputs['A'])
        else:
            mul.inputs['A'].default_value = inp.default_value[:]
        nt.links.new(mul.outputs['Result'], inp)
    mat['tint_linear'] = list(rgb)
    return mat


def matte(mat, specular):
    """Lower the specular level of every Principled node (matte fabric seen at grazing angles)."""
    if specular is None:
        return mat
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            n.inputs['Specular IOR Level'].default_value = specular
            if 'Sheen Weight' in n.inputs:
                n.inputs['Sheen Weight'].default_value = 0.0
    mat['specular_override'] = specular
    return mat


def new_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    return mat, nt, out


def principled(name, albedo, roughness=0.5, metallic=0.0, specular=0.5, bump=0.0, bump_scale=400.0):
    mat, nt, out = new_material(name)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (*albedo, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Specular IOR Level'].default_value = specular
    nt.links.new(bsdf.outputs[0], out.inputs['Surface'])
    if bump:
        tc = nt.nodes.new('ShaderNodeTexCoord')
        noise = nt.nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = bump_scale
        nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
        b = nt.nodes.new('ShaderNodeBump')
        b.inputs['Strength'].default_value = bump
        b.inputs['Distance'].default_value = 0.0005
        nt.links.new(noise.outputs['Fac'], b.inputs['Height'])
        nt.links.new(b.outputs['Normal'], bsdf.inputs['Normal'])
    mat['albedo_linear'] = list(albedo)
    return mat, nt, bsdf


def image_node(nt, path, colorspace='Linear Rec.709'):
    img = bpy.data.images.load(str(path), check_existing=True)
    try:
        img.colorspace_settings.name = colorspace
    except TypeError:
        img.colorspace_settings.name = 'Non-Color'
    img.pack()
    node = nt.nodes.new('ShaderNodeTexImage')
    node.image = img
    node.interpolation = 'Cubic'
    node.extension = 'EXTEND'
    return node


def uv_rect_mapping(nt, x0, y0, lx, ly, uv_name='UVMap'):
    uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = uv_name
    mp = nt.nodes.new('ShaderNodeMapping'); mp.vector_type = 'POINT'
    mp.inputs['Location'].default_value = (-x0 / lx, -y0 / ly, 0)
    mp.inputs['Scale'].default_value = (1 / lx, 1 / ly, 1)
    nt.links.new(uvn.outputs['UV'], mp.inputs['Vector'])
    return uvn, mp


def desk_top_material(cfg, desk):
    c = cfg['materials']['desk_top']
    mat, nt, bsdf = principled('Rig desk | pale oak laminate top', c['albedo'], c['roughness'], 0, c['specular'])
    x0, x1 = desk['box_x_m']; y0, y1 = desk['box_y_m']
    base = None
    if c.get('texture'):
        uvn, mp = uv_rect_mapping(nt, x0, y0, x1 - x0, y1 - y0)
        tex = image_node(nt, HERE / c['texture'])
        nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
        base = tex.outputs['Color']
        mat['albedo_texture'] = c['texture']
    # fine grain streaks along X (the laminate grain runs along the desk length), mean 1
    uvg = nt.nodes.new('ShaderNodeUVMap'); uvg.uv_map = 'UVMap'
    sep = nt.nodes.new('ShaderNodeMapping'); sep.vector_type = 'POINT'
    sep.inputs['Scale'].default_value = (6.0, 420.0, 1.0)
    nt.links.new(uvg.outputs['UV'], sep.inputs['Vector'])
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 1.0
    noise.inputs['Detail'].default_value = 6.0
    noise.inputs['Distortion'].default_value = 0.6
    nt.links.new(sep.outputs['Vector'], noise.inputs['Vector'])
    grain = nt.nodes.new('ShaderNodeMath'); grain.operation = 'MULTIPLY_ADD'
    grain.inputs[1].default_value = 2 * c.get('grain_strength', 0.05)
    grain.inputs[2].default_value = 1 - c.get('grain_strength', 0.05)
    nt.links.new(noise.outputs['Fac'], grain.inputs[0])
    mul = nt.nodes.new('ShaderNodeVectorMath'); mul.operation = 'MULTIPLY'
    if base is None:
        mul.inputs[0].default_value = c['albedo']
    else:
        nt.links.new(base, mul.inputs[0])
    nt.links.new(grain.outputs[0], mul.inputs[1])
    nt.links.new(mul.outputs[0], bsdf.inputs['Base Color'])
    return mat


def wall_material(cfg, wall_x, z_top, desk):
    c = cfg['materials']['wall']
    mat, nt, bsdf = principled('Back wall | beige matte paint', c['albedo'], c['roughness'], 0, 0.3, bump=0.03)
    if c.get('texture'):
        # low-frequency albedo baked from the ZED view: UV in metres (x, z) on the wall face
        t = c['texture_rect']  # [x0, x1, z0, z1]
        uvn, mp = uv_rect_mapping(nt, t[0], t[2], t[1] - t[0], t[3] - t[2])
        tex = image_node(nt, HERE / c['texture'])
        tex.interpolation = 'Linear'
        nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        mat['albedo_texture'] = c['texture']
    return mat


def carpet_material(cfg):
    c = cfg['materials']['carpet']
    mat, nt, bsdf = principled('Floor | dark navy office carpet', c['albedo'], c['roughness'], 0, 0.2)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = 90.0; n1.inputs['Detail'].default_value = 8.0
    nt.links.new(tc.outputs['Object'], n1.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[1].position = 0.75
    nt.links.new(n1.outputs['Fac'], ramp.inputs['Fac'])
    base = Vector(c['albedo'])
    fleck = base * (1 + c.get('speckle', 0.5) * 2.5)
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (*(base * 0.8), 1)
    mix.inputs['B'].default_value = (*fleck, 1)
    nt.links.new(ramp.outputs['Color'], mix.inputs['Factor'])
    nt.links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
    b = nt.nodes.new('ShaderNodeBump'); b.inputs['Strength'].default_value = 0.3; b.inputs['Distance'].default_value = 0.002
    n2 = nt.nodes.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = 900.0
    nt.links.new(tc.outputs['Object'], n2.inputs['Vector'])
    nt.links.new(n2.outputs['Fac'], b.inputs['Height'])
    nt.links.new(b.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def frosted_material(cfg):
    c = cfg['materials']['frosted']
    mat, nt, out = new_material('Glass partition | frosted blue backlit band')
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (*c['emission'], 1)
    em.inputs['Strength'].default_value = c['strength']
    dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
    dif.inputs['Color'].default_value = (*c['diffuse'], 1)
    # camera rays see the full backlit radiance; the light the band throws into the room is scaled by
    # light_fraction (forward-scattering diffuser: most of the transmitted light leaves near the incident direction)
    lp = nt.nodes.new('ShaderNodeLightPath')
    fr = nt.nodes.new('ShaderNodeMath'); fr.operation = 'MULTIPLY_ADD'
    lf = c.get('light_fraction', 1.0)
    nt.links.new(lp.outputs['Is Camera Ray'], fr.inputs[0])
    fr.inputs[1].default_value = c['strength'] * (1 - lf); fr.inputs[2].default_value = c['strength'] * lf
    nt.links.new(fr.outputs[0], em.inputs['Strength'])
    add = nt.nodes.new('ShaderNodeAddShader')
    nt.links.new(em.outputs[0], add.inputs[0]); nt.links.new(dif.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs['Surface'])
    mat['light_fraction'] = lf
    mat['note'] = ('Backlit frosted band modelled as emission (light from the daylit area behind the partition) plus a weak '
                   'diffuse term; non-camera rays see light_fraction of the emission.')
    return mat


def clear_band_light_material(cfg):
    """v3 glass B band: clear glass for camera rays; for all other rays the light of the v2 frosted band (emission x
    light_fraction + weak diffuse). The v2 lights and albedos were fitted with this light entering the room through
    glass B; through clear glass the daylight of the other room still enters (the model blocks direct light through glass)."""
    c = cfg['materials']['frosted']; cg = cfg['materials']['clear_glass']
    mat, nt, out = new_material('Glass partition | clear glass B band (passes the band light)')
    glass = nt.nodes.new('ShaderNodeBsdfGlass')
    glass.inputs['Color'].default_value = (*cg['tint'], 1)
    glass.inputs['Roughness'].default_value = cg['roughness']
    glass.inputs['IOR'].default_value = 1.5
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (*c['emission'], 1)
    em.inputs['Strength'].default_value = c['strength'] * c.get('light_fraction', 1.0)
    dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
    dif.inputs['Color'].default_value = (*c['diffuse'], 1)
    add = nt.nodes.new('ShaderNodeAddShader')
    nt.links.new(em.outputs[0], add.inputs[0]); nt.links.new(dif.outputs[0], add.inputs[1])
    lp = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs['Fac'])
    nt.links.new(add.outputs[0], mix.inputs[1]); nt.links.new(glass.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    mat['note'] = clear_band_light_material.__doc__.strip()
    return mat


def clear_glass_material(cfg):
    c = cfg['materials']['clear_glass']
    mat, nt, out = new_material('Glass partition | clear glass')
    glass = nt.nodes.new('ShaderNodeBsdfGlass')
    glass.inputs['Color'].default_value = (*c['tint'], 1)
    glass.inputs['Roughness'].default_value = c['roughness']
    glass.inputs['IOR'].default_value = 1.5
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    tr.inputs['Color'].default_value = (*c['tint'], 1)
    if c.get('shadow_transparent', True):
        lp = nt.nodes.new('ShaderNodeLightPath')
        mix = nt.nodes.new('ShaderNodeMixShader')
        nt.links.new(lp.outputs['Is Shadow Ray'], mix.inputs['Fac'])
        nt.links.new(glass.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs['Surface'])
    else:
        # plain glass: camera rays see through it; direct light does not pass (the rooms are lit separately)
        nt.links.new(glass.outputs[0], out.inputs['Surface'])
    mat['shadow_transparent'] = c.get('shadow_transparent', True)
    return mat


def emission_material(name, color, strength):
    mat, nt, out = new_material(name)
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (*color, 1); em.inputs['Strength'].default_value = strength
    nt.links.new(em.outputs[0], out.inputs['Surface'])
    return mat


# ---------------------------------------------------------------- scene parts

def build(cfg, m, args):
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    cols = {k: semantics_collection(k) for k in ('Room', 'Rig', 'Background', 'Lighting', 'Cameras', 'Arms')}
    report = {'objects': {}, 'materials': {}}
    desk = m['desk']; x0, x1 = desk['box_x_m']; y0, y1 = desk['box_y_m']; zt = desk['top_z_m']
    meas = cfg['measurements']
    wall_y = -m['planes']['back_wall']['d'] * m['planes']['back_wall']['n'][1]
    if abs(wall_y - (y1 + meas['wall_gap_m'])) > 1e-3:
        raise SystemExit('metric_frame back wall plane disagrees with the tape gap in the config')
    if abs(m['rail']['length_recommended_m'] - meas['rail_length_m']) > 1e-3 or max(abs(a - b) for a, b in zip(m['rail']['x_extent_recommended_m'], meas['rail_x_m'])) > 1e-3:
        raise SystemExit('metric_frame rail extent disagrees with the config')
    room = cfg['room']; mats = cfg['materials']

    # materials
    M = {}
    M['desk_top'] = desk_top_material(cfg, desk)
    M['desk_edge'] = principled('Rig desk | laminate edge band', mats['desk_edge']['albedo'], mats['desk_edge']['roughness'])[0]
    M['mdesk_top'] = principled('Monitor desk | laminate top', mats['monitor_desk_top']['albedo'], mats['monitor_desk_top']['roughness'], 0, 0.35)[0]
    M['wall'] = wall_material(cfg, wall_y, room['ceiling_z'], desk)
    M['carpet'] = carpet_material(cfg)
    M['frosted'] = frosted_material(cfg)
    M['clear_glass'] = clear_glass_material(cfg)
    fa = mats['frame_alu']
    M['frame'], fnt, fbsdf = principled('Glass partition | anodised aluminium frame', fa['albedo'], fa['roughness'], fa['metallic'])
    if fa.get('glow'):
        # light from the adjacent backlit frosted panes, carried along the glass edges into the white frame
        fbsdf.inputs['Emission Color'].default_value = (*fa['glow'], 1)
        fbsdf.inputs['Emission Strength'].default_value = 1.0
        M['frame']['glow_linear'] = list(fa['glow'])
    M['bracket'] = principled('Brackets | dark grey powder coat', mats.get('bracket', {}).get('albedo', [0.05, 0.05, 0.05]), 0.5)[0]
    M['ceiling'] = principled('Ceiling | white acoustic tile', mats['ceiling']['albedo'], 0.95)[0]
    M['white_plastic'] = principled('Power unit | white plastic', mats['white_plastic']['albedo'], mats['white_plastic']['roughness'])[0]
    M['stub_wall'] = principled('Side wall | white paint', mats['stub_wall']['albedo'], 0.9)[0]
    M['alu'] = roomkit_material('RK | Brushed stainless steel', 'Rail and pole | aluminium extrusion (RoomKit brushed steel)')
    M['dark_metal'] = roomkit_material('RK | Satin blackened iron', 'Desk frame and brackets | dark metal (RoomKit satin blackened iron)')
    M['fabric'] = matte(tint(roomkit_material('RK | Charcoal woven upholstery', 'Privacy panel | charcoal fabric (RoomKit charcoal woven upholstery)'), cfg['materials'].get('panel', {}).get('tint', [1, 1, 1])), cfg['materials'].get('panel', {}).get('specular'))
    M['frame_grey'] = principled('Desk frame | grey powder coat', mats.get('frame_grey', {}).get('albedo', [0.2, 0.2, 0.2]), 0.5)[0]
    M['black_plastic'] = principled('Camera body | black plastic', (0.02, 0.02, 0.022), 0.4)[0]
    M['screen'] = roomkit_material('RK | Dark television glass', 'Monitor | dark screen (RoomKit dark television glass)')
    M['fixture'] = emission_material('Ceiling fixture | LED panel diffuser', (1, 0.97, 0.93), 1.0)

    # floor (whole visible area, continues behind the glass partition)
    fl = box('Floor | carpet', room['floor_x'], room['floor_y'], (-0.02, 0.0), cols['Room'], M['carpet'])
    semantics.tag_surface(fl, 'floor')
    set_origin(fl, (0, 0, 0))
    r = semantics.tag_root(fl, 'architecture/floor', 'floor')
    link(r, cols['Room'])

    # ceiling
    ce = box('Ceiling | tiles', room['floor_x'], room['floor_y'], (room['ceiling_z'], room['ceiling_z'] + 0.02), cols['Room'], M['ceiling'])
    semantics.tag_surface(ce, 'ceiling')
    parts = [ce]
    for i, p in enumerate(cfg['lighting']['ceiling_panels']):
        cx, cy, cz = p['center']; sx, sy = p['size']
        parts.append(box(f'Ceiling | LED panel {i}', (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2),
                         (room['ceiling_z'] - 0.01, room['ceiling_z']), cols['Room'], M['fixture']))
    r = semantics.adopt_group(parts, 'architecture/ceiling', 'ceiling',
                              placement_matrix=Matrix.Translation((x0 + (x1 - x0) / 2, y0, room['ceiling_z'])))
    link(r, cols['Room'])

    # back wall: vertical plane at the tape-measured gap behind the desk back edge
    we = cfg.get('wall_end_x_override', {}).get('x', m['planes']['back_wall']['wall_end_x_m'])
    wl = box('Back wall | painted', (we, room['wall_x_right']), (wall_y, wall_y + room['wall_thickness']), (0, room['ceiling_z']), cols['Room'], M['wall'])
    semantics.tag_surface(wl, 'wall')
    set_origin(wl, ((we + room['wall_x_right']) / 2, wall_y, 0))
    r = semantics.tag_root(wl, 'architecture/wall', 'back_wall')
    link(r, cols['Room'])
    report['objects']['back_wall'] = dict(plane_y=wall_y, x=[we, room['wall_x_right']], z=[0, room['ceiling_z']],
                                          source='desk back edge + user tape gap %.3f m' % meas['wall_gap_m'])

    # glass partition (in line with the wall, stepped back), aluminium frame, stub wall
    g = cfg['glass']; gy = g['plane_y']; gt = g['thickness']; ch = g['channel_h']
    parts = []
    px0, px1 = g['post_x'] - g['post_size'][0] / 2, g['post_x'] + g['post_size'][0] / 2
    fz0, fz1 = g['frosted_z']; ztop = room['ceiling_z']
    frosted_panels = g.get('frosted_panels', ['A', 'B'])
    for name, (a, b) in (('A', (px1, we - 0.02)), ('B', (g['x_left'], px0))):
        parts.append(box(f'Glass {name} | clear lower', (a, b), (gy - gt / 2, gy + gt / 2), (ch, fz0), cols['Background'], M['clear_glass']))
        if name in frosted_panels:
            parts.append(box(f'Glass {name} | frosted band', (a, b), (gy - gt / 2, gy + gt / 2), (fz0, fz1), cols['Background'], M['frosted']))
        else:  # v3: glass B is clear over its full height (layout check 09-30, v3_room_corrections.json glass_B)
            mb = clear_band_light_material(cfg) if g.get('clear_band_passes_light') else M['clear_glass']
            parts.append(box(f'Glass {name} | clear middle (no frosted band)', (a, b), (gy - gt / 2, gy + gt / 2), (fz0, fz1), cols['Background'], mb))
        parts.append(box(f'Glass {name} | clear upper', (a, b), (gy - gt / 2, gy + gt / 2), (fz1, ztop - ch), cols['Background'], M['clear_glass']))
    door = cfg.get('side_door')
    if door:  # white jamb between the door and glass B (holds the glass end; same placement root as the partition)
        parts.append(box('Door | white jamb at the glass end', tuple(door['jamb_x']), tuple(door['jamb_y']), (0, ztop), cols['Background'], M['stub_wall']))
    py = (gy - g['post_size'][1] / 2, gy + g['post_size'][1] / 2)
    parts.append(box('Glass frame | mullion post', (px0, px1), py, (0, ztop), cols['Background'], M['frame'], bevel=0.003))
    parts.append(box('Glass frame | wall-end jamb', (we - 0.03, we), (gy + gt / 2 + 0.002, gy + 0.03), (0, ztop), cols['Background'], M['frame']))
    parts.append(box('Back wall | end return', (we - 0.001, we), (wall_y, gy + gt / 2), (0, ztop), cols['Background'], M['wall']))
    parts.append(box('Glass frame | floor channel', (px1, we), (gy - 0.02, gy + 0.02), (0, ch), cols['Background'], M['frame']))
    parts.append(box('Glass frame | ceiling channel', (g['x_left'], we), (gy - 0.02, gy + 0.02), (ztop - ch, ztop), cols['Background'], M['frame']))
    r = semantics.adopt_group(parts, 'architecture/wall/glass_partition', 'glass_partition',
                              placement_matrix=Matrix.Translation(((g['x_left'] + we) / 2, gy, 0)))
    link(r, cols['Background'])
    report['objects']['glass_partition'] = dict(plane_y=gy, x=[g['x_left'], we], frosted_z=g['frosted_z'], post_x=g['post_x'],
                                                frosted_panels=frosted_panels)
    if not door:
        sw = box('Side wall stub | white', g['stub_wall_x'], g['stub_wall_y'], (0, ztop), cols['Background'], M['stub_wall'])
        semantics.tag_surface(sw, 'wall')
        set_origin(sw, (sum(g['stub_wall_x']) / 2, sum(g['stub_wall_y']) / 2, 0))
        link(semantics.tag_root(sw, 'architecture/wall', 'side_wall_left'), cols['Background'])
    else:
        report['objects'].update(side_door_and_wall(door, ztop, cols['Background'], M, mats))

    # rig desk: laminate top 25 mm, edge band, height-adjustable frame, two panel brackets at the left end
    parts = []
    top = box('Rig desk | top', (x0, x1), (y0, y1), (zt - 0.025, zt), cols['Rig'], M['desk_top'])
    top.data.materials.append(M['desk_edge'])
    for poly in top.data.polygons:
        if abs(poly.normal.z) < 0.5:
            poly.material_index = 1
    parts.append(top)
    for i, lx in enumerate((x0 + 0.12, x1 - 0.12)):
        parts.append(box(f'Rig desk | leg column {i}', (lx - 0.035, lx + 0.035), (1.68, 1.75), (0.03, zt - 0.06), cols['Rig'], M['dark_metal'], bevel=0.004))
        parts.append(box(f'Rig desk | foot {i}', (lx - 0.03, lx + 0.03), (y0 + 0.04, y1 - 0.04), (0.0, 0.03), cols['Rig'], M['dark_metal'], bevel=0.004))
        parts.append(box(f'Rig desk | top bracket {i}', (lx - 0.025, lx + 0.025), (y0 + 0.06, y1 - 0.06), (zt - 0.06, zt - 0.025), cols['Rig'], M['dark_metal']))
    parts.append(box('Rig desk | cross beam', (x0 + 0.12, x1 - 0.12), (1.69, 1.74), (zt - 0.10, zt - 0.06), cols['Rig'], M['dark_metal']))
    for i, (bx, by) in enumerate(cfg['rig_desk_brackets_xy']):
        parts.append(box(f'Rig desk | panel bracket {i}', (bx - 0.004, bx + 0.004), (by - 0.01, by + 0.01), (zt - 0.10, cfg.get('rig_desk_bracket_top_z', 0.99)), cols['Rig'], M['bracket']))
    r = semantics.adopt_group(parts, 'furniture/tables/desk', 'desk', placement_matrix=Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, 0)))
    semantics.tag_support(r, -(x1 - x0) / 2, (x1 - x0) / 2, -(y1 - y0) / 2, (y1 - y0) / 2, zt)
    link(r, cols['Rig'])
    report['objects']['desk'] = dict(x=[x0, x1], y=[y0, y1], top_z=zt, source='metric_frame.json desk (tape 1.52 x 0.76 x 0.72)')

    # rail: aluminium T-slot extrusion on the desk front edge, arm adapter plates
    ry0, ry1 = m['rail']['y_extent_m']; rz1 = m['rail']['top_z_m']; rx0, rx1 = m['rail']['x_extent_recommended_m']
    prof = [(ry0, zt), (ry1, zt), (ry1, rz1)]
    slots = [ry0 + 0.015 + i * (ry1 - ry0 - 0.03) / 4 for i in range(4, -1, -1)]
    for sc in slots:
        prof += [(sc + 0.003, rz1), (sc + 0.003, rz1 - 0.006), (sc - 0.003, rz1 - 0.006), (sc - 0.003, rz1)]
    prof += [(ry0, rz1)]
    parts = [prism('Rail | T-slot extrusion', prof, 'x', (rx0, rx1), cols['Rig'], M['alu'])]
    for arm in ('left_arm', 'right_arm'):
        bx, by, bz = arm_base(cfg, m, arm).translation  # v3: the proposed rail-form base when the config has arm_poses
        parts.append(box(f'Rail | {arm} adapter plate', (bx - 0.12, bx + 0.12), (ry0 + 0.005, ry1 - 0.005), (rz1, bz), cols['Rig'], M['dark_metal']))
    for i, cx in enumerate((rx0 + 0.03, rx1 - 0.03)):  # C-clamps holding the rail ends to the desk front edge
        parts.append(box(f'Rail | C-clamp {i} spine', (cx - 0.012, cx + 0.012), (ry0 - 0.03, ry0 - 0.012), (zt - 0.09, rz1 + 0.02), cols['Rig'], M['dark_metal']))
        parts.append(box(f'Rail | C-clamp {i} top jaw', (cx - 0.012, cx + 0.012), (ry0 - 0.03, ry0 + 0.05), (rz1 + 0.008, rz1 + 0.02), cols['Rig'], M['dark_metal']))
        parts.append(box(f'Rail | C-clamp {i} bottom jaw', (cx - 0.012, cx + 0.012), (ry0 - 0.03, ry0 + 0.05), (zt - 0.09, zt - 0.078), cols['Rig'], M['dark_metal']))
        parts.append(box(f'Rail | C-clamp {i} screw', (cx - 0.005, cx + 0.005), (ry0 + 0.02, ry0 + 0.03), (rz1, rz1 + 0.008), cols['Rig'], M['dark_metal']))
    r = semantics.adopt_group(parts, 'fixture/robot_rail', 'rail', support_id='desk',
                              placement_matrix=Matrix.Translation(((rx0 + rx1) / 2, (ry0 + ry1) / 2, zt)))
    link(r, cols['Rig'])
    report['objects']['rail'] = dict(x=[rx0, rx1], y=[ry0, ry1], top_z=rz1, source=meas['rail_source'])

    # camera pole + ZED X body at the metric scene-camera pose
    pc = cfg['pole']; (px, py_), (sx, sy) = pc['xy'], pc['size_xy']
    parts = [box('Camera pole | aluminium extrusion', (px - sx / 2, px + sx / 2), (py_ - sy / 2, py_ + sy / 2), (rz1, pc['top_z']), cols['Rig'], M['alu'], bevel=0.002)]
    gus = [(py_ + sy / 2, rz1), (py_ + sy / 2 + 0.05, rz1), (py_ + sy / 2, rz1 + 0.05)]
    parts.append(prism('Camera pole | base gusset', gus, 'x', (px - 0.004, px + 0.004), cols['Rig'], M['alu']))
    T = Matrix(m['scene_camera']['pose']['world_from_camera_opencv'])
    zb = cfg['zed_body']; off = Vector(zb['left_eye_offset_cam'])
    body = box('ZED X scene camera | body', (-zb['size'][0] / 2, zb['size'][0] / 2), (-zb['size'][1] / 2, zb['size'][1] / 2), (-zb['size'][2] / 2, zb['size'][2] / 2), cols['Rig'], M['black_plastic'], bevel=0.006)
    body.matrix_world = T @ Matrix.Translation(off)
    parts.append(body)
    cam_center = (T @ Matrix.Translation(off)).translation
    parts.append(box('ZED X scene camera | mount bracket', (px - 0.02, px + 0.02), (py_ + sy / 2, cam_center.y - 0.015), (cam_center.z + 0.03, pc['top_z']), cols['Rig'], M['alu']))
    r = semantics.adopt_group(parts, 'fixture/camera_pole', 'camera_pole', support_id='rail',
                              placement_matrix=Matrix.Translation((px, py_, rz1)))
    link(r, cols['Rig'])
    report['objects']['camera_pole'] = dict(xy=pc['xy'], size_xy=pc['size_xy'], top_z=pc['top_z'])

    # monitor desk corner (separate desk abutting at the +X end), privacy panel, power unit
    md = cfg['monitor_desk']; mx0, mx1 = md['x']; my0, my1 = md['y']; mz = md['top_z']
    parts = []
    mtop = box('Monitor desk | top', (mx0, mx1), (my0, my1), (mz - 0.025, mz), cols['Background'], M['mdesk_top'])
    mtop.data.materials.append(M['desk_edge'])
    for poly in mtop.data.polygons:
        if abs(poly.normal.z) < 0.5:
            poly.material_index = 1
    parts.append(mtop)
    for i, lx in enumerate((mx0 + 0.035, mx1 - 0.035)):
        parts.append(box(f'Monitor desk | side rail {i}', (lx - 0.02, lx + 0.02), (my0 + 0.05, my1 - 0.05), (mz - 0.085, mz - 0.025), cols['Background'], M['frame_grey']))
    for i, lx in enumerate((mx0 + 0.12, mx1 - 0.12)):
        parts.append(box(f'Monitor desk | leg column {i}', (lx - 0.035, lx + 0.035), (1.68, 1.75), (0.03, mz - 0.06), cols['Background'], M['dark_metal']))
        parts.append(box(f'Monitor desk | foot {i}', (lx - 0.03, lx + 0.03), (my0 + 0.04, my1 - 0.04), (0.0, 0.03), cols['Background'], M['dark_metal']))
    r = semantics.adopt_group(parts, 'furniture/tables/desk', 'monitor_desk', placement_matrix=Matrix.Translation(((mx0 + mx1) / 2, (my0 + my1) / 2, 0)))
    semantics.tag_support(r, -(mx1 - mx0) / 2, (mx1 - mx0) / 2, -(my1 - my0) / 2, (my1 - my0) / 2, mz)
    link(r, cols['Background'])
    parts = []
    pts = rounded_rect(md['panel_x'][0], md['panel_x'][1], md['panel_z'][0], md['panel_z'][1], md['panel_corner_r'])
    parts.append(prism('Monitors | privacy panel', pts, 'y', tuple(md['panel_y']), cols['Background'], M['fabric']))
    for i, (a, b) in enumerate(md['arm_segments']):
        a, b = Vector(a), Vector(b)
        hw = md.get('arm_width', 0.03) / 2
        seg = box(f'Monitors | monitor arm segment {i}', (-0.006, 0.006), (-hw, hw), (0, (b - a).length), cols['Background'], M['bracket'], bevel=0.003)
        seg.matrix_world = Matrix.Translation(a) @ (b - a).to_track_quat('Z', 'Y').to_matrix().to_4x4()
        parts.append(seg)
    # one monitor in front of the panel (right of the ZED field of view; from the video)
    mon = md.get('monitor')
    if not mon:  # v1/v2: placed by hand
        parts.append(box('Monitors | monitor screen', (-0.55, 0.05), (1.93, 1.96), (0.86, 1.23), cols['Background'], M['screen'], bevel=0.004))
        parts.append(box('Monitors | monitor stand', (-0.28, -0.22), (1.96, 1.99), (mz, 0.95), cols['Background'], M['dark_metal']))
        parts.append(box('Monitors | monitor foot', (-0.36, -0.14), (1.86, 2.0), (mz, mz + 0.01), cols['Background'], M['dark_metal']))
    else:  # v3: from the layout check (v3_room_corrections.json monitor_screen); local x along the screen, y from the front face
        sx0, sx1 = mon['screen_x']; hw = (sx1 - sx0) / 2; dep = mon['depth']
        piv = Matrix.Translation(((sx0 + sx1) / 2, mon['front_y'], 0)) @ Matrix.Rotation(math.radians(mon['yaw_deg']), 4, 'Z')
        st, ft = mon['stand'], mon['foot']
        for name, bx_, by_, bz_, mat, bev in (
                ('Monitors | monitor screen', (-hw, hw), (0, dep), tuple(mon['screen_z']), M['screen'], 0.004),
                ('Monitors | monitor stand', (-st['half_width'], st['half_width']), (dep, dep + st['depth']), (mz, st['top_z']), M['dark_metal'], 0.0),
                ('Monitors | monitor foot', (-ft['half_width'], ft['half_width']), tuple(ft['y']), (mz, mz + ft['height']), M['dark_metal'], 0.0)):
            o = box(name, bx_, by_, bz_, cols['Background'], mat, bevel=bev)
            o.matrix_world = piv
            parts.append(o)
        report['objects']['monitor'] = dict(mon, source=mon.get('source', ''))
    r = semantics.adopt_group(parts, 'furniture/desk_accessories/privacy_panel_and_monitor', 'monitors', support_id='monitor_desk',
                              placement_matrix=Matrix.Translation(((md['panel_x'][0] + md['panel_x'][1]) / 2, md['panel_y'][0], mz)))
    link(r, cols['Background'])
    items = cfg.get('monitor_desk_items', [])
    if items:
        parts = []
        for it in items:
            (cx, cy), (sx, sy, sz) = it['xy'], it['size']
            o = box('Monitor desk items | ' + it['name'], (-sx / 2, sx / 2), (-sy / 2, sy / 2), (0, sz), cols['Background'], M[it['material']], bevel=min(0.004, sz / 3))
            o.matrix_world = Matrix.Translation((cx, cy, mz)) @ Matrix.Rotation(math.radians(it.get('yaw_deg', 0)), 4, 'Z')
            parts.append(o)
        r = semantics.adopt_group(parts, 'prop/desk_clutter', 'monitor_desk_items', support_id='monitor_desk',
                                  placement_matrix=Matrix.Translation((items[0]['xy'][0], items[0]['xy'][1], mz)))
        link(r, cols['Background'])
    pu = md['power_unit']
    o = box('Power unit | desk outlet box', pu['x'], pu['y'], pu['z'], cols['Background'], M['white_plastic'], bevel=0.004)
    set_origin(o, (sum(pu['x']) / 2, sum(pu['y']) / 2, pu['z'][0]))
    o = semantics.tag_root(o, 'prop/power_outlet', 'power_unit', support_id=pu.get('support', 'monitor_desk'))
    o['placement_fixed'] = bool(pu.get('mounted', False))
    link(o, cols['Background'])
    report['objects']['monitor_desk'] = dict(x=md['x'], y=md['y'], top_z=mz, gap_to_rig_desk=round(mx0 - x1, 4))

    # lighting: ceiling LED panels (area lights just below the fixtures), daylight area behind the glass, world ambient
    L = cfg['lighting']
    for i, p in enumerate(L['ceiling_panels']):
        d = bpy.data.lights.new(f'Ceiling LED panel {i}', 'AREA')
        d.shape = 'RECTANGLE'; d.size, d.size_y = p['size']
        d.energy = p['strength'] * p['size'][0] * p['size'][1] * math.pi
        d.color = p['color']
        o = bpy.data.objects.new(d.name, d); cols['Lighting'].objects.link(o)
        o.location = (p['center'][0], p['center'][1], room['ceiling_z'] - 0.012)
    dl = L['daylight']
    d = bpy.data.lights.new('Daylight behind glass partition', 'AREA')
    d.shape = 'RECTANGLE'; d.size, d.size_y = dl['size']; d.energy = dl['power']; d.color = dl['color']
    o = bpy.data.objects.new(d.name, d); cols['Lighting'].objects.link(o)
    o.location = dl['center']
    aim = Vector(dl.get('aim', (dl['center'][0], 0.0, dl['center'][2])))
    o.rotation_euler = (aim - Vector(dl['center'])).to_track_quat('-Z', 'Y').to_euler()
    w = bpy.data.worlds.new('Office ambient'); scene.world = w; w.use_nodes = True
    bg = w.node_tree.nodes['Background']; bg.inputs['Color'].default_value = (*L['world']['color'], 1); bg.inputs['Strength'].default_value = L['world']['strength']

    # scene camera (metric K and pose)
    sc = m['scene_camera']
    cd = bpy.data.cameras.new('ZED scene camera'); cam = bpy.data.objects.new('ZED scene camera', cd)
    cols['Cameras'].objects.link(cam)
    cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0
    W, H = sc['image_size_wh']; fx, fy, cx, cy = sc['fx'], sc['fy'], sc['cx'], sc['cy']
    cd.lens = fx / W * 36.0
    cd.shift_x = -(cx - (W - 1) / 2) / W
    cd.shift_y = (cy - (H - 1) / 2) / W
    cd.clip_start, cd.clip_end = 0.02, 50
    cam.matrix_world = Matrix(sc['pose']['world_from_camera_blender'])
    cam['intrinsics_K'] = [list(r) for r in sc['K']]
    cam['image_size_wh'] = [W, H]
    cam['source'] = 'metric_frame.json scene_camera (ZED X 41925345 left eye, rectified)'
    scene.camera = cam
    cmod = cfg.get('camera_model', {})
    if 'scene' in cmod:
        cam['camera_model_json'] = json.dumps(cmod['scene'])
        # ZED look in a plain render: Standard view transform + the fitted exposure; the radial vignette
        # V(r) = 1 + k1 r^2 + k2 r^4 (r = pixel distance to the principal point / f) is applied in post (build/camera_model.py)
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = math.log2(sum(cmod['scene']['gain']) / 3)
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
    report['camera'] = dict(lens=cd.lens, shift_x=cd.shift_x, shift_y=cd.shift_y)

    # arms: two appended copies of the scene-local YAM asset at the metric base poses, raiden home
    ARM_OVERRIDES.update(cfg.get('arm_material_overrides', {}))
    if args.arms and args.arms != 'none':
        report['arms'] = append_arms(Path(args.arms), m, cols['Arms'], args.arms_status, cfg)
        report['wrist_cameras'] = add_wrist_cameras(cols['Cameras'], cfg)
    report['materials'] = {k: v.name for k, v in M.items()}
    return report


def arm_base(cfg, m, arm):
    """Arm base pose: the metric frame (v1/v2) or, with cfg['arm_poses'] (v3), the proposed correction of the 09-30 fit."""
    ap = cfg.get('arm_poses', {}).get(arm)
    return Matrix(ap['world_from_base'] if ap else m['arms'][arm]['world_from_base'])


def side_door_and_wall(d, ztop, col, M, mats):
    """v3 (v3_room_corrections.json side_wall_stub): the v2 white wall stub is a wood door, about 8 deg ajar, observed in
    the source video. Door leaf modelled as observed (room face along the floor line), white jamb at the glass end (in the
    glass partition group), beige wall left of the door at x ~ -4.98 and a header above the door. None of this is seen
    by the rig cameras."""
    wood = tint(roomkit_material(d['leaf_material'], 'Door | pale wood leaf (RoomKit %s)' % d['leaf_material'].replace('RK | ', '')), d.get('leaf_tint', [1, 1, 1]))
    steel = roomkit_material('RK | Brushed stainless steel', 'Door | lever handle and card reader (RoomKit brushed steel)')
    beige = principled('Side wall | beige matte paint', mats['wall']['albedo'], 0.92, 0, 0.3)[0]
    (fx0, fy0), (fx1, fy1) = d['face_floor_line']            # room face of the leaf at floor level, hinge end -> free end
    s = (fy0 - d['hinge_end_y']) / (fy0 - fy1)                 # leaf starts in front of the jamb
    hx, hy = fx0 + s * (fx1 - fx0), d['hinge_end_y']
    dx, dy = fx1 - hx, fy1 - hy
    width = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)                                   # local +x = along the leaf toward the free end
    # local frame: x along the leaf (hinge -> free end), +y = room side (rotated +x), z up; leaf body behind the face (y < 0)
    T = Matrix.Translation((hx, hy, 0)) @ Matrix.Rotation(yaw, 4, 'Z')
    t, H = d['thickness'], d['height']
    parts = []
    o = box('Door | wood leaf', (0, width), (-t, 0), (d['bottom_gap'], H), col, wood, bevel=0.002)
    o.matrix_world = T
    parts.append(o)
    hz = d['handle_z']; e = d['handle_from_free_edge']
    o = box('Door | card reader lock body', (width - e - 0.025, width - e + 0.025), (0, 0.022), (hz - 0.07, hz + 0.12), col, steel, bevel=0.003)
    o.matrix_world = T; parts.append(o)
    o = box('Door | lever handle', (width - e - 0.13, width - e), (0.05, 0.07), (hz - 0.009, hz + 0.009), col, steel, bevel=0.004)
    o.matrix_world = T; parts.append(o)
    o = box('Door | lever handle neck', (width - e - 0.012, width - e + 0.012), (0.022, 0.05), (hz - 0.009, hz + 0.009), col, steel)
    o.matrix_world = T; parts.append(o)
    r = semantics.adopt_group(parts, 'architecture/wall/door', 'side_door', placement_matrix=T.copy())
    r['placement_fixed'] = True
    r['observed'] = ('wood door ajar about %.1f deg (hinge at the glass end), room face at floor level from %s to %s; '
                     'height %.2f m assumed (>= 2.0 m observed)' % (math.degrees(abs(math.atan2(fx1 - fx0, fy0 - fy1))), d['face_floor_line'][0], d['face_floor_line'][1], H))
    link(r, col)
    wx, wy = d['wall_x'], d['wall_y']
    wall = box('Side wall | beige, left of the door', tuple(wx), tuple(wy), (0, ztop), col, beige)
    semantics.tag_surface(wall, 'wall')
    parts = [wall,
             box('Side wall | header above the door', tuple(d['header_x']), (wy[1], d['jamb_y'][0]), (H + 0.03, ztop), col, beige),
             box('Door | latch-side jamb', tuple(d['header_x']), (wy[1], wy[1] + 0.04), (0, H + 0.03), col, M['stub_wall'])]
    r = semantics.adopt_group(parts, 'architecture/wall', 'side_wall_left',
                              placement_matrix=Matrix.Translation(((wx[0] + wx[1]) / 2, (wy[0] + wy[1]) / 2, 0)))
    link(r, col)
    return {'side_door': dict(hinge_end=[round(hx, 4), round(hy, 4)], free_end=[fx1, fy1], width=round(width, 4), yaw_deg=round(math.degrees(yaw), 2),
                              ajar_deg=round(math.degrees(abs(math.atan2(fx1 - fx0, fy0 - fy1))), 2), thickness=t, height=H,
                              source=d.get('source', '')),
            'side_wall_left': dict(x=wx, y=wy, header_x=d['header_x'], jamb=dict(x=d['jamb_x'], y=d['jamb_y']))}


def semantics_collection(name):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    return col


def append_arms(path, m, col, status, cfg=None):
    import hashlib
    cfg = cfg or {}
    out = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'status': status, 'arms': {}}
    for arm in ('left_arm', 'right_arm'):
        with bpy.data.libraries.load(str(path), link=False) as (src, dst):
            dst.collections = [ARM_COLLECTION]
        c = dst.collections[0]
        c.name = f'YAM {arm} ({status})'
        col.children.link(c)
        root = next(o for o in c.all_objects if o.get('yam_arm_root') or (o.parent is None and o.type == 'EMPTY' and 'joint1' in o))
        ap = cfg.get('arm_poses', {}).get(arm)
        root.matrix_world = arm_base(cfg, m, arm)
        q = list(ap['joints_rad']) if ap else [0.0] * 6
        widened = {}
        for j in range(1, 7):
            key = f'joint{j}'
            ui = root.id_properties_ui(key).as_dict()
            if q[j - 1] < ui['min'] or q[j - 1] > ui['max']:
                # v3 right arm: the home offset of joint 3 (-2.24 deg) is below the MJCF lower limit 0. Widen this scene
                # copy's control range (UI only; the drivers do not clamp) so that the displayed pose is accepted.
                lo = min(ui['min'], math.floor(q[j - 1] * 1000) / 1000); hi = max(ui['max'], math.ceil(q[j - 1] * 1000) / 1000)
                root.id_properties_ui(key).update(min=lo, max=hi, soft_min=lo, soft_max=hi)
                widened[key] = dict(mjcf_range=[ui['min'], ui['max']], scene_range=[lo, hi], value=q[j - 1])
            root[key] = float(q[j - 1])
        root['gripper'] = 0.0475
        tags = None
        rig_py = path.parent / 'build' / 'yam_rig.py'
        if rig_py.exists():  # final asset: claw ArUco tags per arm through the asset's own helper (read only, no bytecode)
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(rig_py.parent))
            import yam_rig
            if hasattr(yam_rig, 'setup_arm'):  # tags and, on asset version 4, the arm's claw mount roll
                tags = yam_rig.ARM_TAG_IDS[arm]
                yam_rig.setup_arm(root, arm)
            elif hasattr(yam_rig, 'set_finger_tags'):
                tags = yam_rig.ARM_TAG_IDS[arm]
                yam_rig.set_finger_tags(root, *tags)
            sys.path.remove(str(rig_py.parent))
        root = semantics.tag_root(root, 'fixture/robot_arm/yam', arm, asset_id='scene-local/yam_arm', support_id='rail')
        out.setdefault('material_overrides', {}).update(override_arm_materials(root, ARM_OVERRIDES))
        root['finger_tag_ids'] = list(tags) if tags else 'none'
        root['arm_asset_status'] = status
        root['arm_asset_note'] = ('placeholder: stock linear_4310 fingers; the UMI-FT finger rebuild (task yam-arm-asset-20260929) '
                                  'was still active when this scene was built' if status == 'placeholder' else
                                  cfg.get('arm_asset_note', 'final asset (UMI-FT claws, task yam-arm-asset-20260929 completed)'))
        entry = dict(root=root.name, world_from_base=[list(r) for r in root.matrix_world], joints=q, gripper=0.0475,
                     finger_tags=list(tags) if tags else None)
        if 'claw_mount_roll' in root:
            entry['claw_mount_roll_rad'] = float(root['claw_mount_roll'])
        if ap:
            root['arm_pose_status'] = ap['status']
            root['arm_pose_source'] = ap['source']
            root['home_joint_offsets_deg'] = json.dumps(ap['home_joint_offsets_deg'])
            root['metric_world_from_base'] = json.dumps(m['arms'][arm]['world_from_base'])
            root['raiden_home_note'] = ('raiden home (FOLLOWER_HOME_POS, q = 0) is displayed here with the fitted home joint offsets '
                                        '(model q = raiden q + offset); set the joints back to 0 for the uncorrected MJCF home')
            if widened:
                root['joint_range_widened'] = json.dumps(widened)
            entry.update(status=ap['status'], source=ap['source'], home_joint_offsets_deg=ap['home_joint_offsets_deg'],
                         metric_world_from_base=m['arms'][arm]['world_from_base'], widened_controls=widened)
        out['arms'][arm] = entry
    return out


ARM_OVERRIDES = {}


def override_arm_materials(root, overrides):
    """Scene-only colour calibration of appended arm materials (the asset file is not changed).

    overrides: {material name prefix: {'base_color_linear': [r, g, b], 'source': str}}; the appended copy of the
    material (per arm, materials are per-scene datablocks) gets the ZED camera-space base colour."""
    done = {}
    mats = {s.material for o in semantics.descendants(root) if o.type == 'MESH' for s in o.material_slots if s.material}
    for mat in mats:
        for prefix, ov in overrides.items():
            if not mat.name.startswith(prefix) or not mat.use_nodes:
                continue
            for n in mat.node_tree.nodes:
                if n.type != 'BSDF_PRINCIPLED':
                    continue
                inp = n.inputs['Base Color']
                old = list(inp.default_value[:3])
                for l in list(inp.links):
                    mat.node_tree.links.remove(l)
                inp.default_value = (*ov['base_color_linear'], 1)
                mat['asset_base_color_linear'] = old
                mat['scene_base_color_linear'] = list(ov['base_color_linear'])
                mat['scene_override_source'] = ov.get('source', '')
                done[mat.name] = dict(old=[round(x, 4) for x in old], new=ov['base_color_linear'])
    return done


def add_wrist_cameras(col, cfg):
    """Approximate wrist ZED X One cameras from raiden's 09-29 hand-eye calibration (grasp_site frame, OpenCV axes)."""
    calib = json.loads(Path('/home/frank/.config/raiden/calibration_results.json').read_text())
    bpy.context.view_layer.update()
    out = {}
    for arm, key in (('left_arm', 'left_wrist_camera'), ('right_arm', 'right_wrist_camera')):
        root = next(o for o in bpy.data.objects if o.get('instance_id') == arm)
        site = next(o for o in semantics.descendants(root) if o.get('mjcf_site') == 'grasp_site')
        he = calib['cameras'][key]['hand_eye_calibration']
        T = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                T[i][j] = he['rotation_matrix'][i][j]
            T[i][3] = he['translation_vector'][i]
        view = key.replace('_camera', '')
        ref = None
        source = cfg.get('wrist_camera_source', 'wrist_refinement')
        wf = None
        if source == 'wrist_fit_0930':
            # 09-30 wrist/claw fit (task yam-wrist-claw-fit-20260930): camera in the grasp_site frame at raiden home, valid only
            # with the proposed arm bases (cfg arm_poses; the rail form gives the same link_6 pose as the fit's equivalent base)
            wf = json.loads((SCENE / cfg['wrist_fit_cameras']).read_text())['cameras'][view]
            T = Matrix(wf['T_grasp_site_cam_opencv'])
            ref = dict(source=cfg['wrist_fit_cameras'] + ' cameras.%s.T_grasp_site_cam_opencv' % view,
                       relative_to=wf['relative_to']['base'], vs_raiden_hand_eye=wf['vs']['raiden_hand_eye'])
        elif source == 'asset_claw_fit':
            # the arm asset's claw fit (fit_claw_wrist.py): small correction of the hand-eye camera in the link_6 frame
            fit = json.loads((SCENE / cfg['wrist_claw_fit']).read_text())
            corr = fit['asset_placement']['camera_corrections'][view.replace('_wrist', '')]
            l6 = next(o for o in semantics.descendants(root) if o.get('mjcf_body') == 'link_6')
            T_l6_gs = l6.matrix_world.inverted() @ site.matrix_world
            T_l6_cam = T_l6_gs @ T
            rv = Vector(corr['rotvec_rad'])
            R = Matrix.Rotation(rv.length, 3, rv.normalized()) if rv.length > 0 else Matrix.Identity(3)
            Tn = (T_l6_cam.to_3x3() @ R).to_4x4()
            Tn.translation = T_l6_cam.translation + Vector(corr['translation_l6_m'])
            T = T_l6_gs.inverted() @ Tn
            ref = dict(rotation_correction_deg=corr['rotvec_deg'], translation_l6_mm=corr['translation_l6_mm'],
                       source=cfg['wrist_claw_fit'] + ' asset_placement.camera_corrections')
        else:
            ref = cfg.get('wrist_refinement', {}).get(view)
            if ref:
                T = Matrix(ref['T_grasp_site_cam_opencv'])
        K = calib['cameras'][key]['intrinsics']['camera_matrix']
        cd = bpy.data.cameras.new(f'{key} (approx)'); cam = bpy.data.objects.new(cd.name, cd)
        col.objects.link(cam)
        cam.parent = site
        cam.matrix_parent_inverse = Matrix.Identity(4)
        cam.matrix_basis = T @ Matrix.Diagonal((1, -1, -1, 1))
        W, H = 960, 600; s = W / calib['cameras'][key]['intrinsics']['image_size'][0]
        fx, fy, cx, cy = K[0][0] * s, K[1][1] * s, K[0][2] * s, K[1][2] * s
        cd.sensor_fit = 'HORIZONTAL'; cd.sensor_width = 36.0
        cd.lens = fx / W * 36.0
        cd.shift_x = -(cx - (W - 1) / 2) / W
        cd.shift_y = (cy - (H - 1) / 2) / W * (fx / fy)
        cd.clip_start, cd.clip_end = 0.01, 50
        cam['intrinsics_K_960x600'] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        cam['pixel_aspect_xy'] = [fy / fx, 1.0] if fy >= fx else [1.0, fx / fy]
        cam['wrist_refinement'] = json.dumps({k: v for k, v in ref.items() if k != 'T_grasp_site_cam_opencv'}) if ref else 'none'
        cam['T_grasp_site_cam_opencv'] = json.dumps([list(r) for r in T])
        if view in cfg.get('camera_model', {}):
            cam['camera_model_json'] = json.dumps(cfg['camera_model'][view])
        cam['approximate'] = ('Pose = arm FK grasp_site x raiden hand-eye (09-29, Tsai, 7 poses) T_grasp_site_cam with the correction in wrist_refinement; '
                              'intrinsics = ZED SDK factory calibration scaled to 960x600; fx != fy needs the render pixel aspect '
                              'in pixel_aspect_xy. Approximate: bracket not modelled, calibration not checked here beyond the image comparison.')
        if wf is not None:
            cam['approximate'] = ('Pose = 09-30 wrist/claw fit (world rotation from the background, translation and arm base from the '
                                  'wrist claws and the scene camera); proposed correction, not raiden calibration. Intrinsics = ZED SDK '
                                  'factory calibration scaled to 960x600 (pixel_aspect_xy). Bracket not modelled.')
            bpy.context.view_layer.update()
            Tw = cam.matrix_world @ Matrix.Diagonal((1, -1, -1, 1))
            Tf = Matrix(wf['T_world_cam_opencv'])
            dR = (Tf.to_3x3().transposed() @ Tw.to_3x3()).to_quaternion()
            chk = dict(rot_deg=round(math.degrees(dR.angle), 4), trans_mm=round((Tw.translation - Tf.translation).length * 1e3, 3))
            cam['check_vs_fit_T_world_cam'] = json.dumps(chk)
            out[view] = dict(source=ref['source'], check_vs_fit_world=chk, T_world_cam_opencv=[list(r) for r in Tw])
    return out


def main():
    args = parse()
    cfg = json.loads(args.config.read_text())
    m = json.loads(args.metric.read_text())
    report = build(cfg, m, args)
    bpy.context.view_layer.update()
    report['instances'] = [dict(id=o['instance_id'], cls=o['semantic_class'], name=o.name) for o in semantics.roots()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit('refusing to overwrite ' + str(args.out))
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out), compress=True)
    report['out'] = str(args.out)
    report['config'] = str(args.config)
    report['metric_frame'] = str(args.metric)
    import hashlib
    report['metric_frame_sha256'] = hashlib.sha256(args.metric.read_bytes()).hexdigest()
    report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    Path(str(args.out) + '.build.json').write_text(json.dumps(report, indent=1))
    print('BUILD_OK', args.out)


main()
