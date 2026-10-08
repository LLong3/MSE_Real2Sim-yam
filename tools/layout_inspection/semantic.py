"""SAM3 semantic masks in the exact cached Pi3X image grid; no video resampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

DEFAULT_PROMPTS = {'person': ['person'], 'glass': ['glass window', 'glass door', 'glass table'], 'mirror': ['mirror']}
STRUCTURAL_PROMPTS = {'floor': ['floor'], 'wall': ['wall']}
LAYERS = {'static': 0, 'person': 1, 'glass': 2, 'mirror': 3, 'geometry_invalid': 4, 'semantic_unknown': 5}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_cache(cache, bundle, frame_ids, shape, *, require_structure=False):
    """Validate source path, frame mapping and raster; absent frames remain unknown."""
    cache, bundle = Path(cache), Path(bundle)
    meta = json.loads((cache / 'manifest.json').read_text())
    if meta.get('status') != 'complete' or Path(meta.get('bundle', '')).resolve() != bundle.resolve():
        raise ValueError('Incomplete semantic cache or different source bundle')
    with np.load(cache / 'masks.npz', allow_pickle=False) as z:
        ids = z['frame_indices']
        if len(set(ids.tolist())) != len(ids):
            raise ValueError('Duplicate semantic frame IDs')
        keys = list(DEFAULT_PROMPTS)
        if require_structure and not set(STRUCTURAL_PROMPTS).issubset(z.files):
            raise ValueError('Structural alignment requires floor and wall masks; rebuild masks with --structure or preserve the existing basis')
        keys += [k for k in STRUCTURAL_PROMPTS if k in z.files]
        masks = {k: z[k].astype(bool) for k in keys}
    for value in masks.values():
        if value.shape != (len(ids), *shape):
            raise ValueError('Semantic mask pixel grid mismatch; resizing is forbidden')
    if not set(ids.tolist()).issubset(set(frame_ids.tolist())):
        raise ValueError('Semantic cache contains foreign frame IDs')
    known = np.zeros(len(frame_ids), bool)
    full = {k: np.zeros((len(frame_ids), *shape), bool) for k in masks}
    for i, fid in enumerate(frame_ids):
        ix = np.flatnonzero(ids == fid)
        if len(ix):
            known[i] = True
            for k in full:
                full[k][i] = masks[k][ix[0]]
    return full, known, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path('.runtime/sam3-segmentation'))
    p.add_argument('--frames', type=int, nargs='+', help='Exact source frame IDs; default all cached frames')
    p.add_argument('--prompts', type=Path, help='JSON with person/glass/mirror lists of text prompts')
    p.add_argument('--threshold', type=float, default=.3)
    p.add_argument('--structure', action='store_true', help='Also segment floor and wall for shared room alignment')
    a = p.parse_args()
    if not 0 < a.threshold < 1:
        p.error('threshold must be between zero and one')
    expected = {**DEFAULT_PROMPTS, **(STRUCTURAL_PROMPTS if a.structure else {})}
    prompts = json.loads(a.prompts.read_text()) if a.prompts else expected
    if set(prompts) != set(expected) or any(not isinstance(v, list) or not v or any(not isinstance(s, str) or not s.strip() for s in v) for v in prompts.values()):
        p.error('prompts must contain nonempty string lists for ' + ', '.join(expected))
    with np.load(a.bundle / 'inputs.npz', allow_pickle=False) as z:
        rgb, ids = z['rgb'], z['frame_indices']
    selected = ids.tolist() if a.frames is None else a.frames
    if len(set(selected)) != len(selected) or not set(selected).issubset(set(ids.tolist())):
        p.error('frames must be distinct IDs present in Pi3X inputs')
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    runtime = json.loads((a.runtime / 'runtime.json').read_text())
    model = build_sam3_image_model(checkpoint_path=str(a.runtime / runtime.get('checkpoint', 'checkpoints/sam3.pt')), load_from_HF=False, device='cuda', compile=False)
    processor = Sam3Processor(model, confidence_threshold=a.threshold)
    outputs = {k: [] for k in prompts}
    records = []
    for fid in selected:
        img = rgb[int(np.flatnonzero(ids == fid)[0])]
        counts = {}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(img))
            for category, phrases in prompts.items():
                union = np.zeros(img.shape[:2], bool)
                count = 0
                for phrase in phrases:
                    processor.reset_all_prompts(state)
                    result = processor.set_text_prompt(prompt=phrase, state=state)
                    masks = result['masks'].detach().float().cpu().numpy()
                    for mask in masks:
                        mask = np.squeeze(mask) > .5
                        if mask.shape != union.shape:
                            raise ValueError('SAM3 output grid differs from Pi3X input grid')
                        union |= mask
                        count += 1
                outputs[category].append(union)
                counts[category] = {'instances': count, 'pixels': int(union.sum())}
                overlay = img.copy()
                overlay[union] = (overlay[union] * .5 + np.array([255, 160, 0]) * .5).astype(np.uint8)
                Image.fromarray(overlay).save(a.out / f'{fid:06d}_{category}.jpg')
        records.append({'source_frame': int(fid), 'detections': counts})
        print(json.dumps(records[-1]), flush=True)
    np.savez_compressed(a.out / 'masks.npz', frame_indices=np.array(selected), **{k: np.stack(v) for k, v in outputs.items()})
    meta = {'schema_version': 2, 'status': 'complete', 'review_status': 'pending_visual_review', 'bundle': str(a.bundle.resolve()), 'prompts': prompts, 'threshold': a.threshold, 'model': 'SAM3', 'runtime': runtime, 'frames': records, 'run_id': os.environ.get('INDOOR_RUN_ID'), 'policy': 'Union all detected instances per text class; zero detections are observed model output, not proof of absence. No temporal propagation. Masks do not establish geometric reliability or missing-space occupancy.'}
    (a.out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')

if __name__ == '__main__':
    main()
