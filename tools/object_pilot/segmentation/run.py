"""Generate native-resolution visible-object masks with official SAM3.

Text finds candidates; an explicit source bounding box selects the intended
instance. Box-only masks are retained as alternatives for visual review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    inter = np.maximum(hi-lo, 0).prod()
    return float(inter / max(1, np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - inter))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path('.runtime/sam3-segmentation'))
    p.add_argument('--prompts', type=Path, help='JSON mapping object ID to text phrase; otherwise use label')
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.manifest.read_text())
    prompts = json.loads(args.prompts.read_text()) if args.prompts else {}
    started = time.perf_counter()
    runtime = json.loads((args.runtime/'runtime.json').read_text())
    model = build_sam3_image_model(checkpoint_path=str(args.runtime/runtime.get('checkpoint', 'checkpoints/sam3.pt')),
                                  load_from_HF=False, device='cuda', compile=False)
    processor = Sam3Processor(model, confidence_threshold=.3)
    torch.cuda.synchronize()
    report = {'schema_version': 1, 'model': 'SAM3', 'model_load_seconds': time.perf_counter()-started,
              'runtime': runtime,
              'run_id': os.environ.get('INDOOR_RUN_ID'), 'gpu': torch.cuda.get_device_name(),
              'source_manifest': str(args.manifest.resolve()), 'code_sha256': digest(__file__),
              'status': 'pending_visual_review', 'objects': []}
    for obj in source['objects']:
        image = Image.open(obj['image']).convert('RGB')
        width, height = image.size
        bbox = None
        if 'selection_bbox_xyxy' in obj:
            bbox = np.asarray(obj['selection_bbox_xyxy'], dtype=float)
        elif 'polygons' in obj:
            # Compatibility for the original pilot's rejected manual masks.
            # Only their bounding box is used; polygon pixels never guide SAM3.
            refw, refh = obj['polygon_reference_size']
            points = np.asarray([xy for poly in obj['polygons'] for xy in poly], dtype=float)
            bbox = np.r_[points.min(0), points.max(0)] * [width/refw, height/refh, width/refw, height/refh]
        if bbox is not None:
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or (bbox[2:] <= bbox[:2]).any():
                raise ValueError('selection_bbox_xyxy must be finite native-pixel [left,top,right,bottom]')
            box_prompt = [(bbox[0]+bbox[2])/2/width, (bbox[1]+bbox[3])/2/height,
                          (bbox[2]-bbox[0])/width, (bbox[3]-bbox[1])/height]
        text = prompts.get(obj['id'], obj['label'])
        target = args.out / obj['id']
        target.mkdir()
        variants = []
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(image)
            for mode in (['text', 'box'] if bbox is not None else ['text']):
                processor.reset_all_prompts(state)
                result = processor.set_text_prompt(prompt=text, state=state) if mode == 'text' else processor.add_geometric_prompt(box_prompt, True, state)
                masks = result['masks'].detach().float().cpu().numpy()
                boxes = result['boxes'].detach().float().cpu().numpy()
                scores = result['scores'].detach().float().cpu().numpy()
                for index, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                    mask = np.squeeze(mask) > .5
                    if mask.shape != (height, width) or not mask.any():
                        continue
                    name = f'{mode}_{index}'
                    path = target / f'{name}_mask.png'
                    Image.fromarray(mask.astype(np.uint8)*255).save(path)
                    rgb = np.asarray(image).astype(float)
                    rgb[mask] = rgb[mask]*.55 + np.array([30,220,120])*.45
                    overlay = Image.fromarray(rgb.astype(np.uint8))
                    overlay.thumbnail((1008,567))
                    overlay.save(target/f'{name}_overlay.jpg', quality=95)
                    white = np.asarray(image).copy()
                    white[~mask] = 235
                    masked = Image.fromarray(white)
                    ys,xs = np.where(mask)
                    crop = masked.crop((max(0,int(xs.min())-10),max(0,int(ys.min())-10),
                                        min(width,int(xs.max())+11),min(height,int(ys.max())+11)))
                    crop.save(target/f'{name}_cutout.png')
                    variants.append({'variant': name, 'mode': mode, 'mask': str(path),
                                     'score': float(score), 'box_xyxy': box.tolist(),
                                     'selection_box_iou': overlap(np.asarray(box), bbox) if bbox is not None else None,
                                     'mask_pixels': int(mask.sum())})
        torch.cuda.synchronize()
        if not variants:
            raise RuntimeError(f"No SAM3 candidate for {obj['id']}")
        text_matches = [v for v in variants if v['mode']=='text' and (bbox is None or v['selection_box_iou'] > .2)]
        best = max(text_matches or variants, key=lambda x: x['selection_box_iou'] if bbox is not None else x['score'])
        record = {'id': obj['id'], 'source_frame': obj['source_frame'], 'image': obj['image'],
                  'image_sha256': digest(obj['image']), 'mask': best['mask'],
                  'mask_method': 'SAM3 visible-instance segmentation with text or positive-box prompting',
                  'prompt': text, 'selection_bbox_xyxy': bbox.tolist() if bbox is not None else None, 'selected_variant': best['variant'],
                  'selection_policy': 'Prefer text candidate overlapping intended bbox; maximize bbox IoU, or confidence without a box. Pending visual review.',
                  'variants': variants, 'elapsed_seconds_including_variant_export': time.perf_counter()-start,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        report['objects'].append(record)
        (args.out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'id': obj['id'], 'selected': best, 'seconds': record['elapsed_seconds_including_variant_export']}), flush=True)


if __name__ == '__main__':
    main()
