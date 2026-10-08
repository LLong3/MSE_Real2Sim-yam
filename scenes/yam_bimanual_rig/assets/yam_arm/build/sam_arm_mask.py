"""SAM 3.1 text-prompted instance masks of the arms on one video frame (for the review camera fit).

    $SAM3_PYTHON build/sam_arm_mask.py --image evidence/video_frames/f1140_t38.0.jpg --out evidence/closeup/sam_f1140
    $SAM3_PYTHON build/sam_arm_mask.py --image .../pi05/scene_camera.jpg --out evidence/scene_camera/sam --width 1008 \
        --phrases 'robotic gripper' 'robot gripper' 'robot arm'   # only 'robot arm' returned masks

Same builder/processor/bf16 autocast as tools/layout_inspection/semantic.py; checkpoint from the
runtime.json selection. Writes <out>/masks.npz (per-instance masks at image resolution, scores,
phrase) and an overlay JPEG.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RUNTIME = Path('/home/frank/robot/aha-3d/.runtime/sam3-segmentation')
PHRASES = ['robot arm', 'robotic gripper']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--width', type=int, default=1344)
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--phrases', nargs='+', default=PHRASES)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rt = json.loads((RUNTIME / 'runtime.json').read_text())
    ckpt = RUNTIME / rt['checkpoint']
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    full = Image.open(a.image).convert('RGB')
    W, H = full.size
    w = a.width - a.width % 14
    h = round(H * w / W / 14) * 14
    img = full.resize((w, h), Image.Resampling.LANCZOS)
    model = build_sam3_image_model(checkpoint_path=str(ckpt), load_from_HF=False, device='cuda', compile=False)
    model.requires_grad_(False)
    proc = Sam3Processor(model, confidence_threshold=a.threshold)
    masks, scores, phrases = [], [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = proc.set_image(img)
        for phrase in a.phrases:
            proc.reset_all_prompts(state)
            res = proc.set_text_prompt(prompt=phrase, state=state)
            for m, s in zip(res['masks'].detach().float().cpu().numpy(), res['scores'].float().cpu().numpy()):
                mm = Image.fromarray((np.squeeze(m) > 0.5).astype(np.uint8) * 255).resize((W, H), Image.Resampling.NEAREST)
                masks.append(np.asarray(mm) > 127); scores.append(float(s)); phrases.append(phrase)
    np.savez_compressed(a.out / 'masks.npz', masks=np.stack(masks), scores=np.array(scores), phrases=np.array(phrases))
    base = np.asarray(full).astype(np.float32)
    rng = np.random.default_rng(0)
    for i, m in enumerate(masks):
        c = rng.uniform(60, 255, 3)
        base[m] = 0.5 * base[m] + 0.5 * c
    Image.fromarray(base.astype(np.uint8)).resize((960, round(960 * H / W))).save(a.out / 'overlay.jpg', quality=88)
    info = [dict(i=i, phrase=p, score=round(s, 3), pixels=int(m.sum()),
                 bbox=[int(v) for v in (np.where(m.any(0))[0].min(), np.where(m.any(1))[0].min(),
                                         np.where(m.any(0))[0].max(), np.where(m.any(1))[0].max())] if m.any() else None)
            for i, (m, s, p) in enumerate(zip(masks, scores, phrases))]
    (a.out / 'masks.json').write_text(json.dumps(dict(image=str(a.image), checkpoint=str(ckpt), grid=[w, h], instances=info), indent=1))
    print(json.dumps(info, indent=1))
    print('peak GiB', torch.cuda.max_memory_allocated() / 2**30)


if __name__ == '__main__':
    main()
