"""Scatter of the L2 foot-band cell vectors for chosen tracks, with mode shares.

usage: band_distribution.py OUT.png PANEL [PANEL ...]
PANEL = LABEL:SCDR:SCFL:L2_DIAG:FRAME:N  plots the N tracks of FRAME whose L2 control is
farthest from G1 (pushed) or, with N negative, |N| tracks closest to G1.
Region, mask and own-box rules copy src/gmc_flow.cpp (PLAN §4).
"""
import csv
import struct
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load(scdr, scfl):
    raw = open(scdr, 'rb').read()
    _, size = struct.unpack_from('<II', raw, 4)
    fl = open(scfl, 'rb').read()
    _, W, H, cell, cols, rows, _ = struct.unpack_from('<I6i', fl, 4)
    def boxes(k):
        off = 28 + k * size
        n = struct.unpack_from('<QQ7I5I', raw, off)[3]
        r = np.frombuffer(raw, dtype=np.dtype([('b', '<f4', 5), ('u', '<u4', 3)]), count=n, offset=off + 64)
        keep = (r['u'][:, 2] & 1).astype(bool) & (r['u'][:, 0] == 0) & (r['b'][:, 4] >= 0.45)
        return r['b'][keep]
    def flow(k):
        return np.frombuffer(fl, '<f4', rows * cols * 2, 40 + k * (12 + 8 * rows * cols) + 12).reshape(rows, cols, 2)
    cx = np.arange(cols) * 8 + 4.0
    cy = np.arange(rows) * 8 + 4.0
    return boxes, flow, cx, cy

def modes(v):
    """Greedy 1.5 px-radius mode peeling: (centre, share) of each mode holding >= 10 % of cells."""
    left, out = v.copy(), []
    while len(left) >= max(4, 0.1 * len(v)):
        d = np.linalg.norm(left[:, None] - left[None], axis=2)
        c = left[np.argmax((d < 1.5).sum(1))]
        for _ in range(5):
            c = left[np.linalg.norm(left - c, axis=1) < 1.5].mean(0)
        near = np.linalg.norm(left - c, axis=1) < 1.5
        if near.sum() < 0.1 * len(v):
            break
        out.append((c, near.sum() / len(v)))
        left = left[~near]
    return out

out_png, panels = sys.argv[1], sys.argv[2:]
figs = []
for p in panels:
    label, scdr, scfl, diag, k, n = p.split(':')
    k, n = int(k), int(n)
    boxes, flow, cx, cy = load(scdr, scfl)
    v = flow(k)
    mask = np.zeros(v.shape[:2], bool)
    for b in boxes(k - 1):
        mask |= (cy[:, None] >= b[1]) & (cy[:, None] < b[3]) & (cx[None] >= b[0]) & (cx[None] < b[2])
    g1 = np.median(v[~mask], axis=0)
    rows = [r for r in csv.DictReader(open(diag)) if int(r['frame']) == k and r['fallback'] == '0']
    dev = lambda r: np.hypot(float(r['dx']) - g1[0], float(r['dy']) - g1[1])
    rows.sort(key=dev, reverse=n > 0)
    for r in rows[:abs(n)]:
        x1, y1, x2, y2 = (float(r[c]) for c in ('x1', 'y1', 'x2', 'y2'))
        w, h, xc = x2 - x1, y2 - y1, (x1 + x2) / 2
        inside = ((cx[None] >= xc - 1.5 * w) & (cx[None] < xc + 1.5 * w) & (cy[:, None] >= y2) & (cy[:, None] < y2 + 0.3 * h)
                  & ~mask & ~((cx[None] >= x1) & (cx[None] < x2) & (cy[:, None] >= y1) & (cy[:, None] < y2)))
        cells = v[inside]
        figs.append((f'{label} k={k} uid={r["uid"]} h={h:.0f}px', cells, g1,
                     np.array([float(r['dx']), float(r['dy'])]), modes(cells)))

fig, axes = plt.subplots((len(figs) + 2) // 3, 3, figsize=(15, 4.6 * ((len(figs) + 2) // 3)), squeeze=False)
for ax in axes.flat[len(figs):]:
    ax.axis('off')
for ax, (title, cells, g1, ctrl, ms) in zip(axes.flat, figs):
    ax.scatter(cells[:, 0], cells[:, 1], s=8, alpha=0.5, color='tab:blue', label=f'band cells n={len(cells)}')
    ax.plot(*g1, 'k+', ms=16, mew=2, label=f'G1 ({g1[0]:+.1f},{g1[1]:+.1f})')
    ax.plot(*ctrl, 'rx', ms=14, mew=2, label=f'L2 median ({ctrl[0]:+.1f},{ctrl[1]:+.1f})')
    for i, (c, s) in enumerate(ms):
        ax.add_patch(plt.Circle(c, 1.5, fill=False, ec='tab:green', lw=1.5))
        ax.annotate(f'{100 * s:.0f}%', c, xytext=(6, 6), textcoords='offset points', color='tab:green', fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('dx px'); ax.set_ylabel('dy px'); ax.invert_yaxis(); ax.set_aspect('equal', 'datalim')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower left')
    print(title, 'n', len(cells), 'modes', [(np.round(c, 1).tolist(), round(float(s), 2)) for c, s in ms])
fig.tight_layout()
fig.savefig(out_png, dpi=75)
