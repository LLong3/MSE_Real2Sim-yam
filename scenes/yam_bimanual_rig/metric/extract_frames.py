"""Full-resolution source frames (1920x1080) of the 64 Pi3X samples -> work/frames_full/<frame>.jpg.

Reads the video named by the reference run (read-only), sequential decode (no seeking), frame numbers from
cameras.json. Existing files are kept. Usage: extract_frames.py [OUT_DIR]
"""
import json
import sys
from pathlib import Path

import cv2

import common

out = Path(sys.argv[1]) if len(sys.argv) > 1 else common.WORK / 'frames_full'
out.mkdir(parents=True, exist_ok=True)
want = {f['source_frame'] for f in json.load(open(common.REF / 'alignment/cameras.json'))['frames']}
video = str(common.HERE.parent / 'references/IMG_5411.MOV')  # scene symlink to ~/Downloads/IMG_5411.MOV (read-only)
cap = cv2.VideoCapture(video)
i, n = 0, 0
while want:
    ok, img = cap.read()
    if not ok:
        break
    if i in want:
        p = out / f'{i:06d}.jpg'
        if not p.exists():
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
        want.discard(i)
    i += 1
print('written', n, 'missing', sorted(want))
