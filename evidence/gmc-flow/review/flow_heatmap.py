"""Heatmap of the NVOFA cell flow minus the frame's G1, with the 0.45 mask and L2 track regions.

usage: flow_heatmap.py SCDR SCFL L2_DIAG.csv IMG_DIR OUT_DIR SEQ FRAME[,FRAME...]
Frame k uses flow k-1 -> k on the k-1 grid, so the background image is frame k-1
(file k, 1-based) and the mask comes from SCDR frame k-1, as in PLAN §4.
"""
import csv
import struct
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

scdr, scfl, diag_path, img_dir, out_dir, seq = sys.argv[1:7]
frames = [int(x) for x in sys.argv[7].split(',')]

raw = open(scdr, 'rb').read()
_, size = struct.unpack_from('<II', raw, 4)
base = 4 + 8 + 16
def kept_boxes(k):
    off = base + k * size
    n = struct.unpack_from('<QQ7I5I', raw, off)[3]
    rows = np.frombuffer(raw, dtype=np.dtype([('b', '<f4', 5), ('u', '<u4', 3)]), count=n, offset=off + 64)
    keep = (rows['u'][:, 2] & 1).astype(bool) & (rows['u'][:, 0] == 0) & (rows['b'][:, 4] >= 0.45)
    return rows['b'][keep]

fl = open(scfl, 'rb').read()
_, W, H, cell, cols, rows_, _ = struct.unpack_from('<I6i', fl, 4)
fsize = 12 + 4 * rows_ * cols * 2
def flow(k):
    return np.frombuffer(fl, dtype='<f4', count=rows_ * cols * 2, offset=40 + k * fsize + 12).reshape(rows_, cols, 2)

diag = {}
for r in csv.DictReader(open(diag_path)):
    diag.setdefault(int(r['frame']), []).append(r)

cx = np.arange(cols) * cell + cell / 2
cy = np.arange(rows_) * cell + cell / 2
for k in frames:
    v = flow(k)
    boxes = kept_boxes(k - 1)
    mask = np.zeros((rows_, cols), bool)
    for b in boxes:
        mask |= (cy[:, None] >= b[1]) & (cy[:, None] < b[3]) & (cx[None, :] >= b[0]) & (cx[None, :] < b[2])
    g1 = np.median(v[~mask], axis=0)
    dev = np.linalg.norm(v - g1, axis=2)
    img = cv2.cvtColor(cv2.imread(f'{img_dir}/{k:06d}.jpg'), cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    extent = (0, cols * cell, rows_ * cell, 0)
    for ax in axes:
        ax.imshow(img, cmap='gray', alpha=0.55, extent=(0, W, H, 0))
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    hm = axes[0].imshow(np.ma.masked_less(dev, 0.5), cmap='inferno', vmin=0, vmax=12, alpha=0.75, extent=extent)
    for b in boxes:
        axes[0].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='cyan', lw=1.2))
    axes[0].set_title(f'{seq} k={k}: |cell flow - G1| (px, clipped 12), cyan = mask boxes score>=0.45 (frame k-1). '
                      f'G1=({g1[0]:+.2f},{g1[1]:+.2f})', fontsize=10)
    fig.colorbar(hm, ax=axes[0], fraction=0.025, pad=0.01)

    leak = (~mask) & (dev > 3)
    axes[1].imshow(np.where(mask, 1.0, np.nan), cmap='Greys', vmin=0, vmax=2, alpha=0.5, extent=extent)
    axes[1].imshow(np.ma.masked_where(~leak, dev), cmap='autumn_r', vmin=3, vmax=12, alpha=0.9, extent=extent)
    pushed = 0
    for r in diag.get(k, []):
        x1, y1, x2, y2 = (float(r[c]) for c in ('x1', 'y1', 'x2', 'y2'))
        dx, dy = float(r['dx']) - g1[0], float(r['dy']) - g1[1]
        if np.hypot(dx, dy) <= 2:
            continue
        pushed += 1
        w, h = x2 - x1, y2 - y1
        axes[1].add_patch(Rectangle((x1, y1), w, h, fill=False, ec='lime', lw=1.2))
        axes[1].add_patch(Rectangle(((x1 + x2) / 2 - 1.5 * w, y2), 3 * w, 0.3 * h, fill=False, ec='lime', lw=1, ls='--'))
        axes[1].annotate('', xy=((x1 + x2) / 2 + 5 * dx, y2 + 5 * dy), xytext=((x1 + x2) / 2, y2),
                         arrowprops=dict(color='lime', width=1.5, headwidth=6))
    tracks = len(diag.get(k, []))
    axes[1].set_title(f'red = unmasked cells with |flow - G1| > 3 px ({leak.sum()} cells); grey = masked; '
                      f'green = L2 tracks whose control differs from G1 by > 2 px ({pushed}/{tracks}), '
                      f'band dashed, arrow = deviation x5', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/{seq}-k{k}.png', dpi=80)
    plt.close(fig)
    print(f'{seq} k={k}: G1=({g1[0]:+.2f},{g1[1]:+.2f}) unmasked>3px={leak.sum()} masked={mask.sum()} '
          f'L2 tracks pushed>2px={pushed}/{tracks}')
