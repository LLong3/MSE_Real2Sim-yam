"""Tile RGBA PNGs over a grey background: montage.py out.jpg cols img1.png img2.png ... (pi3x venv, OpenCV)."""
import sys
import cv2
import numpy as np
out, cols, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
tiles = []
for p in paths:
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        a = im[:, :, 3:4].astype(np.float32) / 255
        im = (im[:, :, :3] * a + 150 * (1 - a)).astype(np.uint8)
    cv2.putText(im, p.split('/')[-1][:-4], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    tiles.append(im)
h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(150, 150, 150)) for t in tiles]
while len(tiles) % cols: tiles.append(np.full_like(tiles[0], 150))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
