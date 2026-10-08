# Dump a comparable state of a YAM asset .blend (objects, props, drivers, materials, mesh hashes).
import bpy, json, sys, hashlib
import numpy as np
out = sys.argv[sys.argv.index('--') + 1]
def h(a): return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
st = {'objects': {}, 'materials': {}, 'collections': sorted(c.name for c in bpy.data.collections),
      'texts': sorted(t.name for t in bpy.data.texts), 'scene_objects': len(bpy.context.scene.objects)}
for o in bpy.data.objects:
    d = dict(type=o.type, parent=o.parent.name if o.parent else None,
             mw=[round(v, 9) for row in o.matrix_world for v in row],
             props={k: (list(v) if hasattr(v, '__len__') and not isinstance(v, str) else v) for k, v in o.items()},
             drivers=sorted((fc.data_path, fc.array_index, fc.driver.expression) for fc in (o.animation_data.drivers if o.animation_data else [])),
             collections=sorted(c.name for c in o.users_collection))
    if o.type == 'MESH':
        co = np.zeros(len(o.data.vertices) * 3, np.float32); o.data.vertices.foreach_get('co', co)
        mi = np.zeros(len(o.data.polygons), np.int32); o.data.polygons.foreach_get('material_index', mi)
        d.update(verts=h(co), mat_index=h(mi), mats=[m.name for m in o.data.materials], nf=len(o.data.polygons))
    st['objects'][o.name] = d
for m in bpy.data.materials:
    b = m.node_tree.nodes.get('Principled BSDF') if m.node_tree else None
    st['materials'][m.name] = dict(nodes=sorted(n.bl_idname for n in m.node_tree.nodes) if m.node_tree else None,
                                   base=[round(v, 6) for v in b.inputs['Base Color'].default_value] if b else None,
                                   props={k: (list(v) if hasattr(v, '__len__') and not isinstance(v, str) else v) for k, v in m.items()})
json.dump(st, open(out, 'w'), indent=0, sort_keys=True, default=str)
print('DUMPED', out, len(st['objects']))
